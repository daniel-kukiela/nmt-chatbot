import sys

sys.path.insert(0, '../')
import os
import errno
from collections import Counter
from setup.settings import preprocessing, hparams
from core.tokenizer import tokenize
from core.sentence import score_answers, replace_in_answers
from tqdm import tqdm
from itertools import zip_longest
from multiprocessing import Pool
from threading import Thread
import time

# Files to be prepared
files = {
    'train.from': {'amount': 1, 'up_to': -1},  # copy all of data (up to "samples")
    'tst2012.from': {'amount': .1, 'up_to': preprocessing['test_size']},  # copy 1/10th but up to 'test_size'
    'tst2013.from': {'amount': .1, 'up_to': preprocessing['test_size']},
    'train.to': {'amount': 1, 'up_to': -1},
    'tst2012.to': {'amount': .1, 'up_to': preprocessing['test_size']},
    'tst2013.to': {'amount': .1, 'up_to': preprocessing['test_size']},
}

vocab = Counter([])


# Prepare all files
def prepare():
    global vocab

    print("\nPreparing training set from raw set")

    # Ensure that train folder exists
    try:
        os.makedirs(preprocessing['train_folder'])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Ensure that model/log folder exists
    train_log_dir = os.path.join(hparams['out_dir'], 'train_log')
    try:
        os.makedirs(train_log_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Iterate thru files and prepare them
    for file_name, amounts in files.items():

        vocab = Counter([])

        print("\nFile: {} (iteration = 10k lines)".format(file_name))

        # Output file handler
        out_file = open('{}/{}'.format(preprocessing['train_folder'], file_name), 'w', encoding='utf-8',
                        buffering=131072)

        # Maximum number of lines
        read = 0
        amount = int(min(amounts['amount'] * preprocessing['samples'] if preprocessing['samples'] > 0 else 10 ** 20,
                         amounts['up_to'] if amounts['up_to'] > 0 else 10 ** 20))

        # Prepare thread variables
        write_thread = None
        vocab_thread1 = None
        vocab_thread2 = None

        # We are going to use multiprocessing for tokenization, as it's cpu intensive
        with Pool(processes=preprocessing['cpu_count']) as pool:

            # Open input file
            with open('{}/{}'.format(preprocessing['source_folder'], file_name), 'r', encoding='utf-8',
                      buffering=131072) as in_file:

                # Iterate every 10k lines
                for rows in tqdm(read_lines(in_file, 10000, '')):

                    # Process using multiprocessing
                    rows = pool.map_async(tokenize, rows, 100).get()

                    # Join running threads from previous loop
                    if write_thread is not None:
                        write_thread.join()
                        vocab_thread1.join()
                        vocab_thread2.join()

                    # If number of lines greater than limit or EOF - break
                    # We are leaving before last save as we have to handle last batch diffrently
                    # zip_longest in read_lines adds extra empty lines up to batch size and we need to remove them
                    # but only for last batch - no need to do that for every batch
                    read += len(rows)
                    if read >= amount:
                        rows = rows[:amount-read+len(rows)]
                        break
                    assert len(rows) == 10000

                    # We are going to process vocab in two threads - a bit faster than one and we need shared memory
                    # Also multiprocessing is slower here
                    vocab_thread1 = Thread(target=append_vocab, args=(rows, 1))
                    vocab_thread1.start()
                    vocab_thread2 = Thread(target=append_vocab, args=(rows, 2))
                    vocab_thread2.start()

                    # And thread for saving tokenized data to putput file
                    write_thread = Thread(target=write_lines, args=(out_file, rows))
                    write_thread.start()

                    rows = []

        # Last vocab parts and last lines to write
        vocab_thread1 = Thread(target=append_vocab, args=(rows, 1))
        vocab_thread1.start()
        vocab_thread2 = Thread(target=append_vocab, args=(rows, 2))
        vocab_thread2.start()
        write_thread = Thread(target=write_lines, args=(out_file, rows))
        write_thread.start()
        vocab_thread1.join()
        vocab_thread2.join()
        write_thread.join()
        out_file.close()

        # If it's train file, make vocab
        if file_name == 'train.from' or file_name == 'train.to':
            print("\nFile: {} (saving vocab)".format(file_name.replace('train', 'vocab')))

            # Get most common entities
            vocab = [entity for entity, v in vocab.most_common()]

            # Do replacements
            new_vocab = [replace_in_answers([entity], 'vocab')[0] for entity in vocab]

            # Filter out duplicates and empty entities
            vocab = set()
            vocab = [entity for entity in new_vocab if not (entity in vocab or vocab.add(entity)) and entity]

            # Write entities to a file
            with open('{}/{}'.format(preprocessing['train_folder'], file_name.replace('train', 'vocab')), 'w',
                      encoding='utf-8', buffering=131072) as vocab_file:
                vocab_file.write("<unk>\n<s>\n</s>\n" + "\n".join(vocab[:preprocessing['vocab_size']]))
            with open('{}/{}'.format(preprocessing['train_folder'], file_name.replace('train', 'vocab_unused')), 'w',
                      encoding='utf-8', buffering=131072) as vocab_file:
                vocab_file.write("\n".join(vocab[preprocessing['vocab_size']:]))

            # Write metadata for embeddings
            with open('{}/{}'.format(os.path.join(train_log_dir), 'decoder.tsv' if file_name == 'train.to' else 'encoder.tsv'), 'w',
                      encoding='utf-8', buffering=131072) as metadata_file:
                metadata_file.write("<unk>\n<s>\n</s>\n" + "\n".join(vocab[:preprocessing['vocab_size']]))

    # Write pbtxt file for metadata for embeddings
    with open('{}/{}'.format(os.path.join(train_log_dir), 'projector_config.pbtxt'), 'w',
              encoding='utf-8', buffering=131072) as pbtxt_file:
        pbtxt_file.write('''embeddings {\n    tensor_name: 'embeddings/decoder/embedding_decoder'\n    '''+
                         '''metadata_path: 'decoder.tsv'\n}\nembeddings {\n    '''+
                         '''tensor_name: 'embeddings/encoder/embedding_encoder'\n    metadata_path: 'encoder.tsv'\n}''')


# Helper function, reads 'amount' number of lines from file handler
def read_lines(file, amount, fillvalue=None):

    args = [iter(file)] * amount
    return zip_longest(*args, fillvalue=fillvalue)


# Writle batch of lines to a file
def write_lines(file, lines):

    # Handling empty lines (described above)
    last = False
    if not len(lines) or lines[-1] == '':
        lines = list(filter(None, list(lines)))
        last = True

    file.write('\n'.join(lines) + ('' if last else '\n'))


# Append tokens to vocab
def append_vocab(lines, thread):
    global vocab

    # Split lines for that vocab thread
    local_vocab = []
    if thread == 1:
        lines = lines[:5000]
    else:
        lines = lines[5000:]

    # Add entities
    for line in lines:
        local_vocab.extend(line.split(' '))

    # Add entities to vocab
    vocab.update(local_vocab)


# Prepare training data set
if __name__ == "__main__":
    prepare()
