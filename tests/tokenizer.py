import sys
sys.path.insert(0, '../')
from core.tokenizer import tokenize
from colorama import Fore, init


tests = [
    ['M.O.R.E.', 'M.O.R.E.'],
    ['No.25', 'No. 2 5'],
    ['no.25', 'no. 2 5'],
    ['No. 25', 'No. 2 5'],
    ['no. 25', 'no. 2 5'],
    ['No.AA', 'No. AA'],
    ['no.AA', 'no. AA'],
    ['Mr. Daniel', 'Mr. Daniel'],
    ['mr. Daniel', 'mr. Daniel'],
    ['Mr.Daniel', 'Mr. Daniel'],
    ['mr.Daniel', 'mr. Daniel'],
    ['mrr. Daniel', 'mrr . Daniel'],
    ['test No.25 test No. 25 test mr. Daniel test', 'test No. 2 5 test No. 2 5 test mr. Daniel test'],
    ['https://www.youtube.com/watch?v=r8b0PWR1qxI', 'https : / / www.youtube.com / watch ? v = r 8 b 0 PWR 1 qxI'],
    ['www.example.com', 'www.example.com'],
    [':)', ': )'],
    ['word...', 'word ...'],
    ['360,678', '3 6 0 , 6 7 8'],
    ['360.678', '3 6 0 . 6 7 8'],
    ['Test phrase. Test phrase.', 'Test phrase . Test phrase .'],
    ['<unk>', ''],
    ['you\'re', 'you \' re'],
    ['you \'re', 'you \' re'],
    ['you\' re', 'you \' re'],
    ['1950\'s', '1 9 5 0 \' s'],
    ['`', '\''],
    ['\'\'', '"'],
    [':/', ': /'],
    ['^^^^^', '^ ^ ^ ^ ^'],
]

init()

for test in tests:
    tokenized = tokenize(test[0])
    print('[{}]  {}  ->  {}{}'.format(Fore.GREEN + 'PASS' + Fore.RESET if tokenized == test[1] else Fore.RED + 'FAIL' + Fore.RESET, test[0], test[1], '' if tokenized == test[1] else '  Result: {}'.format(tokenized)))
