import sys
sys.path.insert(0, '../')
from core.sentence import score_answers
from colorama import Fore, init


tests = [
    ['<unk>', -1],
    ['Word <unk> word', -1],
    ['[ ]', 0],
    ['I \'', 0],
    ['It \'', 0],
    ['You \' re right , I', 0],
    ['That \' s what I \'', 0],
    ['Thank you', 1],
    ['You', 0],
    ['What \' s your', 0],
]

init()

for test in tests:
    scored = score_answers([test[0]])
    print('[{}]  {}  ->  {}{}'.format(Fore.GREEN + 'PASS' + Fore.RESET if scored[0] == test[1] else Fore.RED + 'FAIL' + Fore.RESET, test[0], test[1], '' if scored[0] == test[1] else '  Result: {}'.format(scored[0])))
