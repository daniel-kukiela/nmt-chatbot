import re
from setup.settings import preprocessing


# List with regex-based blacklisted phrases
answers_blacklist = None
vocab_blacklist = None

# Load blacklisted answers
with open(preprocessing['answers_blacklist_file'], 'r', encoding='utf-8') as answers_blacklist_file:
    answers_blacklist = list(filter(lambda word: False if word[0] == '#' else True, filter(None, answers_blacklist_file.read().split("\n"))))
with open(preprocessing['vocab_blacklist_file'], 'r', encoding='utf-8') as vocab_blacklist_file:
    vocab_blacklist = list(filter(lambda word: False if word[0] == '#' else True, filter(None, vocab_blacklist_file.read().split("\n"))))

# Returns index of best answer, 0 if not found
def score_answers(answers, name):

    answers_rate = []

    # Rate each answer
    for answer in answers:
        if re.search('<unk>', answer):
            answers_rate.append(-1)
        elif any(re.search(regex, answer) for regex in eval(name + '_blacklist')):
            answers_rate.append(0)
        else:
            answers_rate.append(1)

    return answers_rate

# List with regex-based replace phrases
answers_replace = None
vocab_replace = None

# Load blacklisted answers
with open(preprocessing['answers_replace_file'], 'r', encoding='utf-8') as answers_replace_file:
    answers_replace = list(filter(lambda word: False if word[0] == '#' else True, filter(None, answers_replace_file.read().split("\n"))))
with open(preprocessing['vocab_replace_file'], 'r', encoding='utf-8') as vocab_replace_file:
    vocab_replace = list(filter(lambda word: False if word[0] == '#' else True, filter(None, vocab_replace_file.read().split("\n"))))

# Replaces phrases in answers
def replace_in_answers(answers, name):

    replaces_answers = []

    # For every answer
    for answer in answers:

        # And every regex rule
        for replace in eval(name + '_replace'):

            diffrence = 0
            replace = replace.split('##->##')
            replace_from = replace[0].strip()
            replace_to = replace[1].strip()

            # If replace regex was found in answer
            if re.search(replace_from, answer):

                # Search for all occurrences and iterate thru them again
                regex = re.compile(replace_from)
                for p in regex.finditer(answer):

                    # Calculate data
                    replace_from = p.groups()[0]
                    position = p.start(1) + diffrence
                    diffrence += -len(replace_from) + len(replace_to)

                    # Remove spaces
                    answer = answer[:position] + answer[position:].replace(replace_from, replace_to, 1)

        replaces_answers.append(answer)

    return replaces_answers