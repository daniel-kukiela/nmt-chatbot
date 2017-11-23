import sys
sys.path.insert(0, '../')
from core.sentence import replace_in_answers
from colorama import Fore, init


tests = [
    ['', ''],
]

init()

for test in tests:
    replaced_answers = replace_in_answers([test[0]])
    print('[{}]  {}  ->  {}{}'.format(Fore.GREEN + 'PASS' + Fore.RESET if replaced_answers[0] == test[1] else Fore.RED + 'FAIL' + Fore.RESET, test[0], test[1], '' if replaced_answers[0] == test[1] else '  Result: {}'.format(replaced_answers[0])))
