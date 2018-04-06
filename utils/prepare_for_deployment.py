import os
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import sys
sys.path.insert(0, os.getcwd())
from setup.settings import hparams, preprocessing
import errno
import shutil
import time
import colorama


colorama.init()

# Copy file or folder recursively
def copy(path):

    print('{}Copying:{} {}'.format(colorama.Fore.GREEN, colorama.Fore.RESET, path))

    while True:
        try:

            # If it's file - copy it
            if os.path.isfile(path):

                # Create folder(s) first
                try:
                    os.mkdir(os.path.dirname('_deployment/' + path))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                # Copy file
                shutil.copy2(path, '_deployment/' + path)

            # Folder - copy it's content recursively
            else:
                shutil.copytree(path, '_deployment/' + path, ignore=copy_ignore)

        # In case of error sleep for a while, print error mesage and retry
        except Exception as e:
            print("{}{}{}".format(colorama.Fore.RED, str(e), colorama.Fore.RESET))
            time.sleep(1)
            continue
        return

# Filter function for folder copy (prints filenames and returns list of files/folders to ignore)
def copy_ignore(path, content):

    blacklist = [path for path in content if path.startswith('.git') or path == '__pycache__']
    print('\n'.join(['{}Copying:{} {}/{}'.format(colorama.Fore.GREEN, colorama.Fore.RESET, path, file) for file in content if file not in blacklist]))
    return blacklist

# Find all available checkpoints
checkpoints = [file[:-6] for file in os.listdir(hparams['out_dir']) if os.path.isfile(hparams['out_dir'] + file) and file[-6:] == '.index']

# If there are no any - print error message and quit
if not checkpoints:
    print('{}There are no model checkpoints ready for deployment{}'.format(colorama.Fore.RED, colorama.Fore.RESET))
    sys.exit()

# Read default checkpoint for model
try:
    default_checkpoint = open(hparams['out_dir'] + 'checkpoint').readline()
except:
    default_checkpoint = ''

# Create deployment folder - wrror if folder exists
try:
    os.makedirs(os.getcwd() + '/_deployment')
except OSError as e:
    if e.errno == errno.EEXIST:
        print('{}Deployment folder already exists, (re)move it to continue{}'.format(colorama.Fore.RED, colorama.Fore.RESET))
        sys.exit()
    else:
        raise

# Print list of checkpoints
default_index = len(checkpoints)
print("\n\n{}List of available checkpoints:{}".format(colorama.Fore.GREEN, colorama.Fore.RESET))
for index, checkpoint in enumerate(checkpoints):
    print("{}{}.{} {}{}".format(colorama.Fore.GREEN, index + 1, colorama.Fore.RESET, '*' if checkpoint in default_checkpoint else '', checkpoint))
    if checkpoint in default_checkpoint:
        default_index = index + 1
print("{}quit{}. Quit tool".format(colorama.Fore.GREEN, colorama.Fore.RESET))

# Ask for checkpoint to include in a copy
while True:

    print("\n{}Choose checkpoint [1-{}], quit or empty for default ({}):{} ".format(colorama.Fore.GREEN, index + 1, default_index, colorama.Fore.RESET), end='')
    choice = input()

    # Empty - default checkpoint
    if not choice:
        choice = default_index

    # Quit
    if choice == 'quit':
        print('{}Quitting{}'.format(colorama.Fore.GREEN,colorama.Fore.RESET))
        sys.exit()

    # Check if number and if in range of available checkpoints
    try:
        choice = int(choice)
        assert choice in range(1, index + 2)
    except:
        print('{}Incorrect choice{}'.format(colorama.Fore.RED, colorama.Fore.RESET))
        continue
    break
print()

# Static list of files to be copied for any settings
paths = ['core', 'nmt', 'setup/settings.py', 'setup/answers_replace.txt', 'setup/answers_subsentence_score.txt', 'utils/run_tensorboard.py', 'inference.py', 'requirements.txt', hparams['out_dir'] + 'hparams']

# Append source vocab
paths.append(hparams['vocab_prefix'] + '.' + hparams['src'])

# Append target vocab (if shared vocab is not set)
if not hparams['share_vocab']:
    paths.append(hparams['vocab_prefix'] + '.' + hparams['tgt'])

# If model is using our BPE/WPM-like tokenizer
if preprocessing['use_bpe']:

    # And shared vocab - copy json file with list of joins
    if hparams['share_vocab']:
        paths.append(preprocessing['train_folder'] + 'bpe_joins.common.json')

    # Else copy source and target files as above
    else:
        paths.append(preprocessing['train_folder'] + 'bpe_joins.{}.json'.format(hparams['src']))
        paths.append(preprocessing['train_folder'] + 'bpe_joins.{}.json'.format(hparams['tgt']))

    # Protected phrases for BPE/WMP-like tokenizer
    paths.append('setup/protected_phrases_bpe.txt')

else:

    # Protected phrases for standard tokenizer
    paths.append('setup/protected_phrases_standard.txt')

# Append rules for standard tokenizer if used
if not preprocessing['embedded_detokenizer']:
    paths.append('setup/answers_detokenize.txt')

# Finally append choosen model files
paths.extend([hparams['out_dir'] + file for file in os.listdir(hparams['out_dir']) if file.startswith(checkpoints[choice - 1])])

# Copy all files
[copy(path) for path in paths]

# Write checkpoint file for TensorFlow with choosen model
print('{}Writing:{} {}checkpoint'.format(colorama.Fore.GREEN, colorama.Fore.RESET, hparams['out_dir']))
while True:
    try:
        with open('_deployment/' + hparams['out_dir'] + 'checkpoint', 'w', encoding='utf-8', newline='') as checkpoint_file:
            checkpoint_file.write('model_checkpoint_path: "{}"'.format(checkpoints[choice - 1]))
    except Exception as e:
        print("{}{}{}".format(colorama.Fore.RED, str(e), colorama.Fore.RESET))
        time.sleep(1)
        continue
    break

# Create best_bleu folder (necessary)
print('{}Creating:{} {}best_bleu/checkpoint'.format(colorama.Fore.GREEN, colorama.Fore.RESET, hparams['out_dir']))
while True:
    try:
        os.mkdir('_deployment/' + hparams['out_dir'] + 'best_bleu')
    except Exception as e:
        if e.errno != errno.EEXIST:
            print("{}{}{}".format(colorama.Fore.RED, str(e), colorama.Fore.RESET))
            time.sleep(1)
            continue
    break

print('\n{}Done. You can find deployment-ready copy of chatbot in _deployment folder{}\n\n'.format(colorama.Fore.GREEN, colorama.Fore.RESET))