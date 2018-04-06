import os
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import sys
sys.path.insert(0, os.getcwd())
from setup.settings import hparams
import subprocess


p = subprocess.Popen('tensorboard --port 22222 --logdir {}'.format(hparams['out_dir']), stdout=subprocess.PIPE, shell=True, bufsize=1)
for line in iter(p.stdout.readline, b''):
    print(line),
p.stdout.close()
p.wait()
