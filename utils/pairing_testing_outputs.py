import os
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import sys
sys.path.insert(0, os.getcwd())
from setup.settings import hparams, preprocessing


# Quick file to pair epoch outputs w/ original test file
if __name__ == '__main__':
    with open(os.path.join(hparams['out_dir'], 'output_dev'), 'r', encoding='utf-8') as f:
        content = f.read()
        to_data = content.split('\n')

    with open(hparams['dev_prefix'] + '.' + hparams['src'], 'r', encoding='utf-8') as f:
        content = f.read()
        from_data = content.split('\n')
        if preprocessing['use_bpe']:
            from_data = [answer.replace(' ', '').replace('▁', ' ') for answer in from_data]

    for n, _ in enumerate(to_data[:-1]):
        print(30*'_')
        print('>', from_data[n])
        print()
        print('Reply:', to_data[n])
