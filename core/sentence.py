import re
from setup.settings import preprocessing


# List with regex-based blacklisted phrases in answers
answers_blacklist = None

# Load blacklisted answers
with open(preprocessing['answers_blacklist_file'], 'r', encoding='utf-8') as answers_blacklist_file:
    answers_blacklist = list(filter(lambda word: False if word[0] == '#' else True, filter(None, answers_blacklist_file.read().split("\n"))))

# Returns index of best answer, 0 if not found
def score_answers(answers):

    answers_rate = []

    # Rate each answer
    for answer in answers:
        if re.search('<unk>', answer):
            answers_rate.append(-1)
        elif any(re.search(regex, answer) for regex in answers_blacklist):
            answers_rate.append(0)
        else:
            answers_rate.append(1)

    return answers_rate

