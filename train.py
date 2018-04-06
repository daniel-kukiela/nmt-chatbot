import sys
import os
import argparse
from setup.settings import hparams, preprocessing
import math
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/nmt")
from nmt import nmt
import tensorflow as tf
import colorama
from threading import Thread
from setup.custom_summary import custom_summary

colorama.init()


def train():

    print('\n\n{}Training model...{}\n'.format(colorama.Fore.GREEN, colorama.Fore.RESET))

    # Custom epoch training and decaying
    if preprocessing['epochs'] is not None:

        # Load corpus size, calculate number of steps
        with open('{}/corpus_size'.format(preprocessing['train_folder']), 'r') as f:
            corpus_size = int(f.read())

        # Load current train progress
        try:
            with open('{}epochs_passed'.format(hparams['out_dir']), 'r') as f:
                initial_epoch = int(f.read())
        except:
            initial_epoch = 0

        # Iterate thru epochs
        for epoch, learning_rate in enumerate(preprocessing['epochs']):

            # Check if model already passed that epoch
            if epoch < initial_epoch:
                print('{}Epoch: {}, learning rate: {} - already passed{}'.format(colorama.Fore.GREEN, epoch + 1, learning_rate, colorama.Fore.RESET))
                continue

            # Calculate new number of training steps - up to the end of current epoch
            num_train_steps = math.ceil((epoch + 1) * corpus_size / (hparams['batch_size'] if 'batch_size' in hparams else 128))
            print("\n{}Epoch: {}, steps per epoch: {}, epoch ends at {} steps, learning rate: {} - training{}\n".format(
                colorama.Fore.GREEN,
                epoch + 1,
                math.ceil(corpus_size / (hparams['batch_size'] if 'batch_size' in hparams else 128)),
                num_train_steps,
                learning_rate,
                colorama.Fore.RESET
            ))

            # Override hparams
            hparams['num_train_steps'] = num_train_steps
            hparams['learning_rate'] = learning_rate
            hparams['override_loaded_hparams'] = True

            # Run TensorFlow threaded (exits on finished training, but we want to train more)
            thread = Thread(target=nmt_train)
            thread.start()
            thread.join()

            # Save epoch progress
            with open('{}epochs_passed'.format(hparams['out_dir']), 'w') as f:
                f.write(str(epoch + 1))

    # Standard training
    else:
        nmt_train()

    print('\n\n{}Training finished{}\n'.format(colorama.Fore.GREEN, colorama.Fore.RESET))


def nmt_train():
    # Modified autorun from nmt.py (bottom of the file)
    # We want to use original argument parser (for validation, etc)
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)

    # But we have to hack settings from our config in there instead of commandline options
    nmt.FLAGS, unparsed = nmt_parser.parse_known_args(['--'+k+'='+str(v) for k,v in hparams.items()])

    # Add custom summary function (hook)
    nmt.summary_callback = custom_summary

    # And now we can run TF with modified arguments
    tf.app.run(main=nmt.main, argv=[os.getcwd() + '\nmt\nmt\nmt.py'] + unparsed)

train()
