import os

## Internal path settings

# Root package path (leave as is, or type full path)
package_path = os.path.realpath(os.path.dirname(__file__) + '/..')

# Model path
out_dir = os.path.join(package_path, "model")

# Training data path
train_dir = os.path.join(package_path, "data")

# Raw data path (data to be prepared and tokenized)
source_dir = os.path.join(package_path, "new_data")


## Settings you can adjust

# Preprocessing settings
preprocessing = {

    # Number of samples to save in training data set
    # -1 means all available in source data set
    'samples': -1,

    # Vocab max size
    'vocab_size': 15000,

    # Whether to use joined (common) vocab for both source and destination
    # (should work well with BPE/WPM-like tokenization for our chatbot - english-english translation)
    'joined_vocab': True,

    # Whether to use BPE/WPM-like tokenization, or standard one
    'use_bpe': True,
    
    # Whether to use:
    # - embedded detokenizer (increases number of vocab tokens, but is more accurate)
    # - external/rule-based detokenizer (based of a bunch of rules and regular expressions -
    #   doesn't increase number of tokens in vocab, but it's hard to make a rule for every case)
    # Note, that embedded detokenizer is forced to True while using BPE-like tokenizer
    'embedded_detokenizer': True,

    # Test sets' max size
    'test_size': 100,


    ## You don't need to change anything below (internal settings)

    # Source (raw) data folder
    'source_folder': source_dir,

    # Place to save preprocessed and tokenized training set
    'train_folder': train_dir,

    # File with protected phrases for standard tokenizer
    'protected_phrases_standard_file': os.path.join(package_path, 'setup/protected_phrases_standard.txt'),

    # File with protected phrases for BPE/WPM-like tokenizer
    'protected_phrases_bpe_file': os.path.join(package_path, 'setup/protected_phrases_bpe.txt'),

    # File with blacklisted answers
    'answers_blacklist_file': os.path.join(package_path, 'setup/answers_blacklist.txt'),

    # File with detokenizer rules
    'answers_detokenize_file': os.path.join(package_path, 'setup/answers_detokenize.txt'),

    # File with replace rules for answers
    'answers_replace_file': os.path.join(package_path, 'setup/answers_replace.txt'),

    # File with blacklisted answers
    'vocab_blacklist_file': os.path.join(package_path, 'setup/vocab_blacklist.txt'),

    # File with replace rules for vocab
    'vocab_replace_file': os.path.join(package_path, 'setup/vocab_replace.txt'),

    # Number of processes to be spawned during tokenization (leave None for os.cpu_count())
    'cpu_count': None,
}

# hparams
hparams = {
    'attention': 'scaled_luong',
    'num_train_steps': 10000000,
    'num_layers': 2,
    'num_units': 512,
#    'batch_size': 128,
#    'override_loaded_hparams': True,
#    'decay_scheme': 'luong234'
#    'residual': True,
    'optimizer': 'adam',
    'encoder_type': 'bi',
    'learning_rate':0.001,
    'beam_width': 20,
    'length_penalty_weight': 1.0,
    'num_translations_per_input': 20,
#    'num_keep_ckpts': 5,

    ## You don't need to change anything below (internal settings)
    'src': 'from',
    'tgt': 'to',
    'vocab_prefix': os.path.join(train_dir, "vocab"),
    'train_prefix': os.path.join(train_dir, "train"),
    'dev_prefix': os.path.join(train_dir, "tst2012"),
    'test_prefix': os.path.join(train_dir, "tst2013"),
    'out_dir': out_dir,
    'share_vocab': preprocessing['joined_vocab'],
}


######## DO NOT TOUCH ANYTHING BELOW ########

if preprocessing['use_bpe']:
    preprocessing['embedded_detokenizer'] = True
    hparams['subword_option'] = 'spm'

preprocessing['protected_phrases_file'] = preprocessing['protected_phrases_bpe_file'] if preprocessing['use_bpe'] else preprocessing['protected_phrases_standard_file']

if preprocessing['use_bpe']:
    hparams['vocab_prefix'] += '.bpe'
    hparams['train_prefix'] += '.bpe'
    hparams['dev_prefix'] += '.bpe'
    hparams['test_prefix'] += '.bpe'

if preprocessing['joined_vocab']:
    hparams['share_vocab'] = True
