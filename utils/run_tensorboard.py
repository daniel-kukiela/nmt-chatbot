import sys
sys.path.insert(0, '../')
from setup.settings import hparams
import subprocess


p = subprocess.Popen('tensorboard --port 22222 --logdir {}'.format(hparams['out_dir']), stdout=subprocess.PIPE, shell=True, bufsize=1)
for line in iter(p.stdout.readline, b''):
    print(line),
p.stdout.close()
p.wait()
