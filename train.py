import sys
import os
import argparse
from setup.settings import hparams
sys.path.append(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(os.path.realpath(os.path.dirname(__file__)) + "/nmt")
from nmt import nmt
import tensorflow as tf


# Modified autorun from nmt.py (bottom of the file)
# We want to use original argument parser (for validation, etc)
nmt_parser = argparse.ArgumentParser()
nmt.add_arguments(nmt_parser)
# But we have to hack settings from our config in there instead of commandline options
nmt.FLAGS, unparsed = nmt_parser.parse_known_args(['--'+k+'='+str(v) for k,v in hparams.items()])
# And now we can run TF with modified arguments
tf.app.run(main=nmt.main, argv=[os.getcwd() + '\nmt\nmt\nmt.py'] + unparsed)
