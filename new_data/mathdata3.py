import random
from collections import defaultdict


hm_test = 500
hm_samples = 10000000+hm_test

max_val = 100000
max_number_of_nums = 10
operators = ["+", "-", "*", "/"]

equations = {}

while len(equations) < hm_samples:
    nums = [random.randrange(1,max_val) for _ in range(random.randrange(2,max_number_of_nums))]

    number_of_parenthesis = random.randrange(0, min(4, len(nums)-2)) if len(nums) > 2 else 0
    opening_parenthesis = defaultdict(lambda: 0)
    closing_parenthesis = defaultdict(lambda: 0)
    for _ in range(number_of_parenthesis):
        opening_parenthesis_position = random.randrange(0, len(nums)-1)
        if opening_parenthesis[opening_parenthesis_position] > 0 and opening_parenthesis[opening_parenthesis_position] + 1 in closing_parenthesis.values():
            continue
        opening_parenthesis[opening_parenthesis_position] += 1
        closing_parenthesis_position = random.randrange(opening_parenthesis_position + 1, len(nums))
        if closing_parenthesis[closing_parenthesis_position] > 0 and closing_parenthesis[closing_parenthesis_position] + 1 in opening_parenthesis.values():
            opening_parenthesis[opening_parenthesis_position] -= 1
            continue
        closing_parenthesis[closing_parenthesis_position] += 1

    init_str = ''

    while opening_parenthesis[0] > 0 and closing_parenthesis[len(nums)-1] > 0:
        opening_parenthesis[0] -= 1
        closing_parenthesis[len(nums)-1] -= 1

    for index, num in enumerate(nums):

        while opening_parenthesis[index] > 0 and closing_parenthesis[index] > 0:
            opening_parenthesis[index] -= 1
            closing_parenthesis[index] -= 1

        operator = random.choice(operators) if init_str != '' else ''
        init_str += "{}{}{}{}".format(operator, '('*opening_parenthesis[index], str(num), ')'*closing_parenthesis[index])

    try:
        equations[init_str] = eval(init_str)
    except:
        pass

#print('\n'.join([k + ' = ' + str(v) for k, v in equations.items()]))
with open("train.from", "a") as fin:
    with open("train.to", "a") as fout:
        for k, v in list(equations.items())[:-hm_test]:
            fin.write(k)
            fin.write('\n')
            fout.write(str(v))
            fout.write('\n')


with open("tst2012.from", "a") as fin1:
    with open("tst2013.from", "a") as fin2:
        with open("tst2012.to", "a") as fout1:
            with open("tst2013.to", "a") as fout2:

                for k, v in list(equations.items())[-hm_test:]:
                    fin1.write(k)
                    fin1.write('\n')
                    fin2.write(k)
                    fin2.write('\n')
                    fout1.write(str(v))
                    fout1.write('\n')
                    fout2.write(str(v))
                    fout2.write('\n')

