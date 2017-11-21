
# model path
out_dir = 'd:/seq2seq_test/model'

# training data path
train_dir = 'd:/seq2seq_test2/data'

# raw data path (data to be prepared and tokenized)
source_dir = 'd:/seq2seq_test2/new_data'

# preprocessing settings
preprocessing = {
    # number of samples to save in training data set
    # -1 means all available in source data set
    'samples': -1,

    # vocab max size
    'vocab_size': 18000,

    # test sets' max size
    'test_size': 100,

    # vocab max entity length (-1 for no limit)
    'vocab_entity_len': -1,

    # source (raw) data folder
    'source_folder': source_dir,

    # place to save preprocessed and tokenized training set
    'train_folder': train_dir,

    # file with protected phrases
    'protected_phrases_file': 'protected_phrases.txt',

}

# hparams
hparams = {
    'attention': 'scaled_luong',
    'src': 'from',
    'tgt': 'to',
    'vocab_prefix': train_dir + '/vocab',
    'train_prefix': train_dir + '/train',
    'dev_prefix': train_dir + '/tst2012',
    'test_prefix': train_dir + '/tst2013',
    'out_dir': out_dir,
    'num_train_steps': 500000,
    'num_layers': 3,
    'num_units': 256,
#    'override_loaded_hparams': True,
    'decay_factor': 0.99995,
    'decay_steps': 1,
#    'residual': True,
    'start_decay_step': 1,
    'beam_width': 10,
    'length_penalty_weight': 1.0,
    'num_translations_per_input': 10
}
