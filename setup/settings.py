import os

## Internal path settings

# Root package path (do not change if you're unsure about that change)
package_path = ''

# Model path
out_dir = os.path.join(package_path, "model/")

# Training data path
train_dir = os.path.join(package_path, "data/")

# Raw data path (data to be prepared and tokenized)
source_dir = os.path.join(package_path, "new_data/")


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
    # - external/rule-based detokenizer (based of a bunch of rules and regular expressions)
    #   doesn't increase number of tokens in vocab, but it's hard to make a rule for every case)
    # Note, that embedded detokenizer is forced to True while using BPE-like tokenizer
    'embedded_detokenizer': True,

    # Test sets' max size
    'test_size': 100,

    # Custom decaying scheme and training duration:
    # - trains model for certain number of epochs (number of list entries)
    # - applies learning rate for every epoch
    # - default: [0.001, 0.0001, 0.00001] - means: train for 3 epochs using choosen learing rates for corresponding epoch
    # - automatically sets number of steps and restarts training every epoch with changed learning rate
    # - ends training after set number of epochs
    # Set to None to disable
    'epochs': [0.001, 0.0001, 0.00001],


    ## You don't normally need to change anything below (internal settings)

    # Cache 'prepairing training set' and 'building temporary vocab' steps
    'cache_preparation': False,

    # Source (raw) data folder
    'source_folder': source_dir,

    # Place to save preprocessed and tokenized training set
    'train_folder': train_dir,

    # File with protected phrases for standard tokenizer
    'protected_phrases_standard_file': os.path.join(package_path, 'setup/protected_phrases_standard.txt'),

    # File with protected phrases for BPE/WPM-like tokenizer
    'protected_phrases_bpe_file': os.path.join(package_path, 'setup/protected_phrases_bpe.txt'),

    # File with detokenizer rules (for standard detokenizer)
    'answers_detokenize_file': os.path.join(package_path, 'setup/answers_detokenize.txt'),

    # File with replace rules for answers
    'answers_replace_file': os.path.join(package_path, 'setup/answers_replace.txt'),

    # Number of processes to be spawned during tokenization (leave None for os.cpu_count())
    'cpu_count': None,
}

# hparams
hparams = {
    'attention': 'scaled_luong',
    'num_train_steps': 10000000,
    'num_layers': 2,
#    'num_encoder_layers': 2,
#    'num_decoder_layers': 2,
    'num_units': 512,
#    'batch_size': 128,
#    'override_loaded_hparams': True,
#    'decay_scheme': 'luong234'
#    'residual': True,
    'optimizer': 'adam',
    'encoder_type': 'bi',
    'learning_rate': 0.001,
    'beam_width': 20,
    'length_penalty_weight': 1.0,
    'num_translations_per_input': 20,
#    'num_keep_ckpts': 5,

    ## You don't normally need to change anything below (internal settings)
    'src': 'from',
    'tgt': 'to',
    'vocab_prefix': os.path.join(train_dir, "vocab"),
    'train_prefix': os.path.join(train_dir, "train"),
    'dev_prefix': os.path.join(train_dir, "tst2012"),
    'test_prefix': os.path.join(train_dir, "tst2013"),
    'out_dir': out_dir,
    'share_vocab': preprocessing['joined_vocab'],
}

# response score settings
score = {

    # Either to use scoring or score responses by order only
    'use_scoring': True,

    # File with blacklisted answers
    'answers_subsentence_score_file': os.path.join(package_path, 'setup/answers_subsentence_score.txt'),

    # Starting score of every output sentence (response)
    'starting_score': 10,

    # How to choose response to return
    # - None - first response with best score
    # - 'best_score' - random response with best score
    # - 'above_threshold' - random response above or equal threshold
    'pick_random': 'best_score',

    # Threshold of good response
    # 'above_threshold':
    # - zero or greater - that value
    # - negative value - difference from best score
    # 'best_score' and None - returns sentence only if above threshold (including negative value)
    'bad_response_threshold': 0,

    ## Answer to question similarity (using Levenshtein ratio)

    # 0..1
    'question_answer_similarity_threshold': 0.75,

    # Minimum sentence length to run test (allow very short responses similar to questions)
    'question_answer_similarity_sentence_len': 10,

    # Modifier type:
    # - value - add given value
    # - multiplier - diffrence between ratio and threshold divided by (1 - ratio) multiplied by modifier value
    'question_answer_similarity_modifier': 'value',  # 'multiplier'

    # Modifier value (None to disable)
    'question_answer_similarity_modifier_value': -100,

    ## Subsentence similarity

    # Regular expression of subsentence dividers
    'subsentence_dividers': "[\.,!\?;]|but",

    # 0..1
    'answer_subsentence_similarity_threshold': 0.5,

    # Minimum sentence length to run test (allow very short responses similar to questions)
    'answer_subsentence_similarity_sentence_len': 10,

    # Modifier type:
    # - value - add given value
    # - multiplier - diffrence between ratio and threshold divided by (1 - ratio) multiplied by modifier value
    'answer_subsentence_similarity_modifier': 'multiplier',  # 'value'

    # Modifier value (None to disable)
    'answer_subsentence_similarity_modifier_value': -10,

    # Regular expression for where url ends
    'url_delimiters': ' )',

    # Modifier value for incorrect url (None to disable)
    'incorrect_url_modifier_value': -100,

    # Regular expression of sentence endings
    'sentence_ending': '[\.!\?;]|FTFY',  # FTFY doesn;t end with dot

    # Sentence length threshhold
    'sentence_ending_sentence_len': 20,

    # Tuple of two modifiers: (None to disable)
    # - modifier if sentence is longer than length above
    # - modifier if sentence is shorter (only if sententence ends with letter of digit), else threated as long sentence
    'no_ending_modifier_value': (-100, -5),

    # Unk modifier value (None to disable)
    'unk_modifier_value': -100,

    # Weather to use subsentence score file or not (modifier value included inside that file)
    'use_subsentence_score': True,

    # Response number modidier (key - starting index, value - modifier value)
    'position_modifier': {1: 1.5, 2: 1, 4: 0.5, 8: 0},

    # Ascii emoticon detector - ratio of non-word chars to all chars in word
    'ascii_emoticon_non_char_to_all_chars_ratio': 0.7,

    # Modifier value if asci detected (None to disable)
    'ascii_emoticon_modifier_value': 1,

    # Score reward for every word in sentence (None to disable)
    'reward_long_sentence_value': 0.15,

    # 'question_answer_diffrence_threshold': 0.7,
    # 'question_answer_diffrence_sentence_len': 10,
    # 'question_answer_diffrence_modifier': 'value',  # 'multiplier'
    # 'question_answer_diffrence_modifier_value': None,

    # To be removed, doesnt really work as expected
    # 'answer_subsentence_diffrence_threshold': 0.5,
    # 'answer_subsentence_diffrence_sentence_len': 10,
    # 'answer_subsentence_diffrence_modifier': 'multiplier',  # 'value'
    # 'answer_subsentence_diffrence_modifier_value': None,

    'show_score_modifiers': False,
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

if not score['use_scoring']:
    score['bad_response_threshold'] = 0