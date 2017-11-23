import sys
sys.path.insert(0, '../')
from core.sentence import score_answers
from colorama import Fore, init


tests = [
    ['', 1],
]

init()

for test in tests:
    scored = score_answers([test[0]])
    print('[{}]  {}  ->  {}{}'.format(Fore.GREEN + 'PASS' + Fore.RESET if scored[0] == test[1] else Fore.RED + 'FAIL' + Fore.RESET, test[0], test[1], '' if scored[0] == test[1] else '  Result: {}'.format(scored[0])))
