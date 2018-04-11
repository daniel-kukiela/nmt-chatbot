import os
__file__ = os.path.realpath(__file__)
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import sys
sys.path.insert(0, os.getcwd())
from core.tokenizer import tokenize


# Prepare all files
def prepare():
    global vocab, written_lines

    # Files to be prepared
    files = {
        '{}.{}'.format(hparams['train_prefix'].replace('.bpe', ''), hparams['src']).replace(preprocessing['train_folder'], '').lstrip('\\/'): {'amount': 1, 'up_to': -1}, # copy all of data (up to "samples")
        '{}.{}'.format(hparams['dev_prefix'].replace('.bpe', ''),   hparams['src']).replace(preprocessing['train_folder'], '').lstrip('\\/'): {'amount': .1, 'up_to': preprocessing['test_size']},  # copy 1/10th but up to 'test_size'
        '{}.{}'.format(hparams['test_prefix'].replace('.bpe', ''),  hparams['src']).replace(preprocessing['train_folder'], '').lstrip('\\/'): {'amount': .1, 'up_to': preprocessing['test_size']},
        '{}.{}'.format(hparams['train_prefix'].replace('.bpe', ''), hparams['tgt']).replace(preprocessing['train_folder'], '').lstrip('\\/'): {'amount': 1, 'up_to': -1},
        '{}.{}'.format(hparams['dev_prefix'].replace('.bpe', ''),   hparams['tgt']).replace(preprocessing['train_folder'], '').lstrip('\\/'): {'amount': .1, 'up_to': preprocessing['test_size']},
        '{}.{}'.format(hparams['test_prefix'].replace('.bpe', ''),  hparams['tgt']).replace(preprocessing['train_folder'], '').lstrip('\\/'): {'amount': .1, 'up_to': preprocessing['test_size']},
    }

    print(colorama.Fore.GREEN + "\nPreparing training set from raw set" + colorama.Fore.RESET)

    # Ensure that train folder exists
    try:
        os.makedirs(preprocessing['train_folder'])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Ensure that model/log folder exists
    train_log_dir = hparams['out_dir'] + 'train_log/'
    try:
        os.makedirs(train_log_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    corpus_size = 0

    if not preprocessing['cache_preparation'] or not Path('{}/cache_data_vocab.pickle'.format(preprocessing['train_folder'])).exists() or not Path('{}/cache_data_vocab.pickle'.format(preprocessing['train_folder'])).is_file():

        data_vocab = Counter()

        # Iterate thru files and prepare them
        for file_name, amounts in files.items():

            vocab = Counter()

            print("File: {}{}{}".format(colorama.Fore.GREEN, file_name, colorama.Fore.RESET))

            # Output file handler
            out_file = open('{}{}'.format(preprocessing['train_folder'], file_name), 'w', encoding='utf-8', buffering=131072)

            # Maximum number of lines
            read = 0
            amount = int(min(amounts['amount'] * preprocessing['samples'] if preprocessing['samples'] > 0 else 10 ** 20, amounts['up_to'] if amounts['up_to'] > 0 else 10 ** 20))

            # Prepare thread variables
            write_thread = None
            vocab_thread = None
            written_lines = 0

            # We are going to use multiprocessing for tokenization, as it's cpu intensive
            with Pool(processes=preprocessing['cpu_count']) as pool:

                # Count number of lines in file
                number_of_records = min(amount, sum(1 for _ in open('{}{}'.format(preprocessing['source_folder'], file_name), 'r', encoding='utf-8', buffering=131072)))
                if file_name == '{}.{}'.format(hparams['train_prefix'].replace('.bpe', ''), hparams['src']).replace(preprocessing['train_folder'], '').lstrip('\\/'):
                    corpus_size = number_of_records
                    with open('{}/corpus_size'.format(preprocessing['train_folder']), 'w') as f:
                        f.write(str(corpus_size))
                elif file_name == '{}.{}'.format(hparams['train_prefix'].replace('.bpe', ''), hparams['tgt']).replace(preprocessing['train_folder'], '').lstrip('\\/'):
                    number_of_records = corpus_size
                progress = tqdm(ascii=True, unit=' lines', total=number_of_records)

                # Open input file
                with open('{}{}'.format(preprocessing['source_folder'], file_name), 'r', encoding='utf-8', buffering=131072) as in_file:

                    last_batch = False

                    # Iterate every 10k lines
                    for rows in read_lines(in_file, 30000, ''):

                        # If number of lines is greater than limit - break
                        read += len(rows)
                        if read >= amount:
                            rows = rows[:amount-read+len(rows)]
                            last_batch = True

                        # Process using multiprocessing
                        rows = pool.map(tokenize, rows, 500)

                        # Process vocab using multiprocessing
                        vocab_part = pool.map(sentence_split, rows, 500)

                        # Join running threads from previous loop
                        if write_thread is not None:
                            write_thread.join()
                            vocab_thread.join()
                            progress.update(written_lines)

                        # Thread for vocab update
                        vocab_thread = Thread(target=append_vocab, args=(vocab_part,))
                        vocab_thread.start()

                        # And thread for saving tokenized data to output file
                        write_thread = Thread(target=write_lines, args=(out_file, rows, written_lines == 0))
                        write_thread.start()

                        # Last batch - break / exit loop
                        if last_batch:
                            break

                    # Join running threads and update progress bar
                    write_thread.join()
                    vocab_thread.join()
                    progress.update(written_lines)
                    progress.close()

            # If it's train file, save vocab
            if file_name == '{}.{}'.format(hparams['train_prefix'].replace('.bpe', ''), hparams['src']).replace(preprocessing['train_folder'], '').lstrip('\\/'):
                data_vocab[hparams['src']] = vocab
            elif file_name == '{}.{}'.format(hparams['train_prefix'].replace('.bpe', ''), hparams['tgt']).replace(preprocessing['train_folder'], '').lstrip('\\/'):
                data_vocab[hparams['tgt']] = vocab

        # If joined vocab - add counters
        if preprocessing['joined_vocab']:
            data_vocab[hparams['src']] += data_vocab[hparams['tgt']]
            del data_vocab[hparams['tgt']]

        with open('{}/cache_data_vocab.pickle'.format(preprocessing['train_folder']), 'wb') as f:
            pickle.dump(data_vocab, f)

    else:
        print('Using cached data')
        with open('{}/cache_data_vocab.pickle'.format(preprocessing['train_folder']), 'rb') as f:
            data_vocab = pickle.load(f)

    # BPE/WPM-like tokenization
    # inspired by and based on https://github.com/rsennrich/subword-nmt
    if preprocessing['use_bpe']:

        print(colorama.Fore.GREEN + "\nLearning BPE" + colorama.Fore.RESET)

        # List of subword joins to be applied to training data
        joins = {}

        # Final train vocab for NMT
        train_vocab = {}

        # Learn BPE for both vocabs (or common vocab)
        for source, raw_vocab in data_vocab.items():

            if not preprocessing['cache_preparation'] or not Path('{}/cache_temp_vocab.pickle'.format(preprocessing['train_folder'])).exists() or not Path('{}/cache_temp_vocab.pickle'.format(preprocessing['train_folder'])).is_file():

                # Pair stats
                stats = Counter()

                # Pair indexes
                indices = defaultdict(lambda: defaultdict(int))

                # Build 'new' vocab used for BPE learning (train_vocab will be a final vocab for NMT)
                vocab = []
                train_vocab[source] = Counter()

                # Build vocab for BPE learning purpose
                print("Building temporary vocab ({})".format(hparams['src'] if preprocessing['joined_vocab'] else source))
                for i, (entity, freq) in tqdm(enumerate(raw_vocab.most_common()), ascii=True, unit=' tokens'):

                    # Split vocab token
                    entity = tuple(entity.split())

                    # Make pairs ("ABCD" -> (A, B), (B, C), (C, D)), stats, indexes and train vocab
                    prev_char = entity[0]
                    train_vocab[source][prev_char] += freq
                    for char in entity[1:]:
                        stats[prev_char, char] += freq
                        indices[prev_char, char][i] += 1
                        train_vocab[source][char] += freq
                        prev_char = char
                    vocab.append((entity, freq))

                with open('{}/cache_temp_vocab.pickle'.format(preprocessing['train_folder']), 'wb') as f:
                    pickle.dump((stats, dict(indices), train_vocab, vocab), f)

            else:
                print('Using cached data')
                with open('{}/cache_temp_vocab.pickle'.format(preprocessing['train_folder']), 'rb') as f:
                    stats, indices, train_vocab, vocab = pickle.load(f)
                    indices = defaultdict(lambda: defaultdict(int), indices)

            print("Learning BPE for vocab of {} tokens".format(preprocessing['vocab_size']))

            # List of joins per vocab
            joins[source] = []

            # Partial stats speeds up learning process - optimization for 'max' above
            partial_stats = Counter(['', -1])
            partial_stats_min = -1
            update_partial_stats = True

            # Current number of vocab tokens
            train_vocab_len = prev_train_vocab_len = len(train_vocab[source])

            # Progress bar
            progress = tqdm(ascii=True, unit=' tokens', total=preprocessing['vocab_size'], maxinterval=0.1, miniters=10)
            progress.monitor_interval = 1
            progress.update(prev_train_vocab_len)

            # Learn until vocab will contain desired number of tokens
            while train_vocab_len < preprocessing['vocab_size']:

                clean_train_vocab = False

                # Get most frequent pair
                most_frequent, freq = partial_stats.most_common(1)[0]

                # Update partial stats or frequency of most frequent pair is less than saved minimum for partial stats
                if update_partial_stats or freq < partial_stats_min:
                    partial_stats_min = stats.most_common(500)[-1][1]
                    partial_stats = Counter()
                    for k, v in stats.most_common():
                        if v < partial_stats_min:
                            break
                        partial_stats[k] = v
                    update_partial_stats = False

                    # Get most frequent pair (again, proper one this time)
                    most_frequent, _ = partial_stats.most_common(1)[0]

                # If frequency is lower than 2 - exit
                if stats[most_frequent] < 2:
                    print('No pair has frequency greater than 1. Stopping earlier, your vocab file will include less tokens.\n')
                    break

                # Replace pair "A B" with new entity "AB"

                # Changes made
                changes = []

                # Replace regex
                pattern = re.compile(r'(?<!\S)' + re.escape(' '.join(most_frequent)) + r'(?!\S)')

                # Loop through indices
                for j, freq in indices[most_frequent].items():

                    # Do not touch not existent pairs
                    if freq < 1:
                        continue

                    # Get entity and frequency
                    entity, freq = vocab[j]

                    # Replace "A B" with "AB" in entity
                    new_entity = pattern.sub(''.join(most_frequent), ' '.join(entity))
                    new_entity = tuple(new_entity.split())

                    # Update entity
                    vocab[j] = (new_entity, freq)

                    changes.append((j, new_entity, entity, freq))

                # Update indices and pair stats
                # Merged pair doesn't exist anymore
                stats[most_frequent] = 0
                partial_stats[most_frequent] = 0
                indices[most_frequent] = defaultdict(int)

                # Get entities and a new pair
                first, second = most_frequent
                new_pair = first + second

                # Iterate through all changes
                for j, entity, old_entity, freq in changes:

                    # Find all occurences of first pair entity
                    prev = -2
                    for i in iter([i for i, entity in enumerate(old_entity) if entity == first]):

                        # Do not touch second "B B" if "B B B"
                        if i == prev + 1:
                            continue

                        # Check if second pair entity follows first one
                        if i < len(old_entity) - 1 and old_entity[i + 1] == second:

                            # Reduce frequency of "A B" in "A B C D" where "B C" is a merged pair
                            if i:
                                prev = old_entity[i - 1:i + 1]
                                stats[prev] -= freq
                                partial_stats[prev] = stats[prev]
                                indices[prev][j] -= 1

                            # Reduce frequency of "C D" in "A B C D" where "B C" is a merged pair
                            if i < len(old_entity) - 2:

                                # But do not touch "C B" if "A B C B C" as values will be adjusted with next occurence of "B C" pair
                                if old_entity[i + 2] != first or i >= len(old_entity) - 3 or old_entity[i + 3] != second:
                                    next = old_entity[i + 1:i + 3]
                                    stats[next] -= freq
                                    partial_stats[next] = stats[next]
                                    indices[next][j] -= 1

                            prev = i

                            if train_vocab[source][first] <= freq or train_vocab[source][second] <= freq:
                                clean_train_vocab = True
                            train_vocab[source][first] -= freq
                            train_vocab[source][second] -= freq

                    # Find all occurences of first pair entity
                    for i in [i for i, entity in enumerate(entity) if entity == new_pair]:

                        # Increase frequency of (new pair) "A BC" in "A BC D"
                        if i:
                            prev = entity[i - 1:i + 1]
                            stats[prev] += freq
                            #if stats[prev] >= partial_stats_min:
                            #    update_partial_stats = True
                            partial_stats[prev] = stats[prev]
                            indices[prev][j] += 1

                        # Increase frequency of (new pair) "BC D" in "A BC D", but do not touch if "A BC BC" as stats for "BC BC" will be adjusted win next occurence of "BC" pair
                        if i < len(entity) - 1 and entity[i + 1] != new_pair:
                            next = entity[i:i + 2]
                            stats[next] += freq
                            #if stats[next] >= partial_stats_min:
                            #    update_partial_stats = True
                            partial_stats[prev] = stats[prev]
                            indices[next][j] += 1

                        # Set frequency of a new pair
                        train_vocab[source][new_pair] += freq

                # Current pair is merged - is not a pair anymore, so has frequency of 0
                stats[most_frequent] = 0
                partial_stats[most_frequent] = 0

                # Remove (from training vocab) tokens with frequency of 0
                if clean_train_vocab:
                    train_vocab[source] = +train_vocab[source]

                # Calculate current number of train vocab entities
                prev_train_vocab_len = train_vocab_len
                train_vocab_len = len(train_vocab[source])
                train_vocab_len_diff = train_vocab_len - prev_train_vocab_len

                # Update progress bar
                if train_vocab_len_diff >= 0:
                    progress.update(train_vocab_len_diff)

                # For a negative number set new value directly - tqdm doesn't support negative updates
                else:
                    progress.n += train_vocab_len_diff
                    progress.refresh()

                # Add new join pair
                joins[source].append(most_frequent)

            # Save list of joins for train vocab
            joins[source] = dict(reversed([(v, i) for i, v in enumerate(joins[source])]))

            # Done
            progress.close()

        # Save list of joins to a file (joined vocab) and replace main vocabs
        if preprocessing['joined_vocab']:
            with open('{}{}'.format(preprocessing['train_folder'], 'bpe_joins.common.json'), 'w', encoding='utf-8', buffering=131072) as bpe_file:
                json.dump({json.dumps(k):v for k,v in joins[hparams['src']].items()}, bpe_file)
            data_vocab[hparams['src']] = train_vocab[hparams['src']]

        # Save list of joins to files (separated vocab)
        else:
            with open('{}{}'.format(preprocessing['train_folder'], 'bpe_joins.{}.json'.format(hparams['src'])), 'w', encoding='utf-8', buffering=131072) as bpe_file:
                json.dump({json.dumps(k):v for k,v in joins[hparams['src']].items()}, bpe_file)
            with open('{}{}'.format(preprocessing['train_folder'], 'bpe_joins.{}.json'.format(hparams['tgt'])), 'w', encoding='utf-8', buffering=131072) as bpe_file:
                json.dump({json.dumps(k):v for k,v in joins[hparams['tgt']].items()}, bpe_file)
            data_vocab[hparams['src']] = train_vocab[hparams['src']]
            data_vocab[hparams['tgt']] = train_vocab[hparams['tgt']]

        print(colorama.Fore.GREEN + "\nApplying BPE" + colorama.Fore.RESET)

        # BPE files to be prepared
        bpe_files = [
            '{}.{}'.format(hparams['train_prefix'], hparams['src']).replace(preprocessing['train_folder'], '').lstrip('\\/'),
            '{}.{}'.format(hparams['dev_prefix'],   hparams['src']).replace(preprocessing['train_folder'], '').lstrip('\\/'),
            '{}.{}'.format(hparams['test_prefix'],  hparams['src']).replace(preprocessing['train_folder'], '').lstrip('\\/'),
            '{}.{}'.format(hparams['train_prefix'], hparams['tgt']).replace(preprocessing['train_folder'], '').lstrip('\\/'),
            '{}.{}'.format(hparams['dev_prefix'],   hparams['tgt']).replace(preprocessing['train_folder'], '').lstrip('\\/'),
            '{}.{}'.format(hparams['test_prefix'],  hparams['tgt']).replace(preprocessing['train_folder'], '').lstrip('\\/'),
        ]

        # Iterate thru files and apply BPE
        for i, file_name in enumerate(bpe_files):

            # Current train vocab
            source = hparams['src'] if preprocessing['joined_vocab'] else file_name.split('.')[-1]

            print("File: {}{}{}".format(colorama.Fore.GREEN, file_name, colorama.Fore.RESET))

            # Output file handler
            out_file = open('{}{}'.format(preprocessing['train_folder'], file_name), 'w', encoding='utf-8', buffering=131072)

            # Prepare thread variables
            write_thread = None
            written_lines = 0

            # We are going to use multiprocessing for joins, as it's cpu intensive
            with Pool(processes=preprocessing['cpu_count'], initializer=apply_bpe_init, initargs=(joins[source],)) as pool:

                # Progress bar
                if file_name == '{}.{}'.format(hparams['train_prefix'], hparams['src']).replace(preprocessing['train_folder'], '').lstrip('\\/'):
                    if not corpus_size:
                        with open('{}/corpus_size'.format(preprocessing['train_folder']), 'r') as f:
                            number_of_records = corpus_size = int(f.read())
                    else:
                        number_of_records = corpus_size
                elif file_name == '{}.{}'.format(hparams['train_prefix'], hparams['tgt']).replace(preprocessing['train_folder'], '').lstrip('\\/'):
                    number_of_records = corpus_size
                else:
                    number_of_records = sum(1 for _ in open('{}{}'.format(preprocessing['train_folder'], file_name.replace('.bpe.', '.')), 'r', encoding='utf-8', buffering=131072))
                progress = tqdm(ascii=True, unit=' lines', total=number_of_records)

                # Open input file
                with open('{}{}'.format(preprocessing['train_folder'], file_name.replace('.bpe.', '.')), 'r', encoding='utf-8', buffering=131072) as in_file:

                    # Iterate every 10k lines
                    for rows in read_lines(in_file, 10000, ''):

                        # Process using multiprocessing
                        rows = pool.map(apply_bpe, rows, 100)

                        # Join running threads from previous loop
                        if write_thread is not None:
                            write_thread.join()
                            #vocab_thread.join()
                            #print('+')
                            progress.update(written_lines)
                            #vocab_thread2.join()

                        # Thread for saving tokenized data to output BPE file
                        write_thread = Thread(target=write_lines, args=(out_file, rows, written_lines == 0))
                        write_thread.start()

                    # Join running threads and update progress bar
                    write_thread.join()
                    progress.update(written_lines)
                    progress.close()

            # Remove unnecessary train file (BPE one will be used by NMT)
            if not preprocessing['cache_preparation']:
                os.remove('{}{}'.format(preprocessing['train_folder'], file_name.replace('.bpe.', '.')))

    print(colorama.Fore.GREEN + "\nPostprocessing and saving vocabs" + colorama.Fore.RESET)

    # Vocab files to be prepared
    # Joined vocab
    if preprocessing['joined_vocab']:
        vocab_files = [
            '{}.{}'.format(hparams['train_prefix'].replace('train', 'vocab'), hparams['src']).replace(preprocessing['train_folder'], '').lstrip('\\/'),
        ]

    # Separated vocabs
    else:
        vocab_files = [
            '{}.{}'.format(hparams['train_prefix'].replace('train', 'vocab'), hparams['src']).replace(preprocessing['train_folder'], '').lstrip('\\/'),
            '{}.{}'.format(hparams['train_prefix'].replace('train', 'vocab'), hparams['tgt']).replace(preprocessing['train_folder'], '').lstrip('\\/'),
        ]

    for vocab_file_name in vocab_files:

        print("File: {}{}{}".format(colorama.Fore.GREEN, vocab_file_name, colorama.Fore.RESET))

        # Get most common entities
        source = vocab_file_name.split('.')[-1]
        data_vocab[source] = [entity for entity, _ in data_vocab[source].most_common()]

        # Write entities to a file
        with open('{}{}'.format(preprocessing['train_folder'], vocab_file_name), 'w', encoding='utf-8', buffering=131072) as vocab_file:
            vocab_file.write("<unk>\n<s>\n</s>\n" + "\n".join(data_vocab[source][:preprocessing['vocab_size']]))
        with open('{}{}'.format(preprocessing['train_folder'], vocab_file_name.replace('vocab', 'vocab_unused')), 'w', encoding='utf-8', buffering=131072) as vocab_file:
            vocab_file.write("\n".join(data_vocab[source][preprocessing['vocab_size']:]))

    print(colorama.Fore.GREEN + "\nWriting pbtxt file" + colorama.Fore.RESET)

    # Write pbtxt file for metadata for embeddings
    with open(train_log_dir + 'projector_config.pbtxt', 'w', encoding='utf-8', buffering=131072) as pbtxt_file:
        pbtxt_file.write(('''embeddings {{\n    tensor_name: 'embeddings/decoder/embedding_decoder'\n    '''+
                         '''metadata_path: '{}'\n}}\nembeddings {{\n    '''+
                         '''tensor_name: 'embeddings/encoder/embedding_encoder'\n    metadata_path: '{}'\n}}''').format(
            '{}{}'.format(preprocessing['train_folder'], vocab_files[0].replace('train', 'vocab')),
            '{}{}'.format(preprocessing['train_folder'], vocab_files[0 if preprocessing['joined_vocab'] else 1].replace('train', 'vocab'))
        ))

    print(colorama.Fore.GREEN + "\nAll done" + colorama.Fore.RESET)

# Helper function, reads 'amount' number of lines from file handler
def read_lines(file, amount, fillvalue=None):

    args = [iter(file)] * amount
    return zip_longest(*args, fillvalue=fillvalue)

## use first instead of last, do not include \n on first, than include  at the beginning of string, or use last_batch above
# Writle batch of lines to a file
def write_lines(file, lines, first_batch):

    global written_lines

    # Handling empty lines (described above)
    if not len(lines) or lines[-1] == '' or lines[-1] == '▁':
        lines = list(filter(lambda line: False if line == '' or line == '▁' else True, list(lines)))

    file.write(('' if first_batch else '\n') + '\n'.join(lines))

    written_lines = len(lines)

# Append tokens to vocab
def append_vocab(lines):
    global vocab

    # Split lines for that vocab thread
    local_vocab = []

    # Add entities
    for line in lines:
        local_vocab.extend(line)

    # Add entities to vocab
    vocab.update(local_vocab)


# Prepare training data set
if __name__ == "__main__":
    import errno
    from collections import Counter, defaultdict
    from setup.settings import preprocessing, hparams
    from core.tokenizer import apply_bpe_init, apply_bpe, sentence_split
    from tqdm import tqdm
    from itertools import zip_longest
    from multiprocessing import Pool
    from threading import Thread
    import regex as re
    import json
    import colorama
    import pickle
    from pathlib import Path

    colorama.init()
    vocab = Counter()

    prepare()
