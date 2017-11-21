import sys
sys.path.insert(0, '../')
from setup.settings import hparams
import os


# Quick file to pair epoch outputs w/ original test file
if __name__ == '__main__':
    with open(os.path.join(hparams['out_dir'], 'output_dev'), 'r') as f:
        content = f.read()
        to_data = content.split('\n')

    with open(hparams['dev_prefix'] + '.' + hparams['src'], 'r') as f:
        content = f.read()
        from_data = content.split('\n')

    for n, _ in enumerate(to_data[:-1]):
        print(30*'_')
        print('>', from_data[n])
        print()
        print('Reply:', to_data[n])
