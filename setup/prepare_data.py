import sys
sys.path.insert(0, '../')
import os
import errno
from collections import Counter
from setup.settings import preprocessing
from core.tokenizer import tokenize
from tqdm import tqdm


# Files to be prepared
files = {
    'train.from': {'amount': 1, 'up_to': -1}, # copy all of data (up to "samples")
    'tst2012.from': {'amount': .1, 'up_to': preprocessing['test_size']},  # copy 1/10th but up to 'test_size'
    'tst2013.from': {'amount': .1, 'up_to': preprocessing['test_size']},
    'train.to': {'amount': 1, 'up_to': -1},
    'tst2012.to': {'amount': .1, 'up_to': preprocessing['test_size']},
    'tst2013.to': {'amount': .1, 'up_to': preprocessing['test_size']},
}

# Prepare all files
def prepare():

    print("\nPreparing training set from raw set")

    # Ensure that destination folder exists
    try:
        os.makedirs(preprocessing['train_folder'])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Iterate thru files and prepare them
    for file_name, amounts in files.items():

        print("\nFile: {}".format(file_name))

        train_data = None

        # Read a file
        with open('{}/{}'.format(preprocessing['source_folder'], file_name), 'r', encoding='utf-8') as train_file:
            train_data = train_file.read()

        # Split sentences by lines and get appropriate amount according to "samples" variable
        train_data = train_data.split("\n")
        amount = int(min(amounts['amount'] * preprocessing['samples'] if preprocessing['samples'] > 0 else len(train_data), amounts['up_to'] if amounts['up_to'] > 0 else len(train_data)))
        train_data = train_data[:amount]

        # Tokenize every sentence
        train_data = [tokenize(sentence) for sentence in tqdm(train_data)]

        # Wtite sentences to a file
        with open('{}/{}'.format(preprocessing['train_folder'], file_name), 'w', encoding='utf-8') as vocab_file:
            vocab_file.write("\n".join(train_data))

        # If it's file with samples, make vocab
        if file_name == 'train.from' or file_name == 'train.to':

            # Get up to 50k most common tokens
            vocab = " ".join(train_data).split()

            # Limit max length of vocab entity?
            if preprocessing['vocab_entity_len'] > 0:
                vocab = [entity for entity in vocab if len(entity) <= preprocessing['vocab_entity_len']]

            # Get nn most common entities
            vocab = Counter(vocab).most_common(preprocessing['vocab_size'])

            # Write them to a file
            with open('{}/{}'.format(preprocessing['train_folder'], file_name.replace('train', 'vocab')), 'w', encoding='utf-8') as vocab_file:
                vocab_file.write("<unk>\n<s>\n</s>\n" + "\n".join([k for k, v in vocab]))


# Prepare training data set
if __name__ == "__main__":
    prepare()



