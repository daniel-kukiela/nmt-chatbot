import re
from setup.settings import preprocessing
from urllib.parse import urlparse

# List with regex-based blacklisted phrases
answers_blacklist = None
vocab_blacklist = None

# Load blacklisted answers
with open(preprocessing['answers_blacklist_file'], 'r', encoding='utf-8') as answers_blacklist_file:
    answers_blacklist = list(filter(lambda word: False if word[0] == '#' else True, filter(None, answers_blacklist_file.read().split("\n"))))
with open(preprocessing['vocab_blacklist_file'], 'r', encoding='utf-8') as vocab_blacklist_file:
    vocab_blacklist = list(filter(lambda word: False if word[0] == '#' else True, filter(None, vocab_blacklist_file.read().split("\n"))))

# Check for URL
def is_complete_url(answer,complete = True):
    if '\n' in answer:
        return False
    parsed = urlparse(answer)
    if parsed.path == answer:
        return False
    elif not parsed.netloc:
        return False
    elif complete and not parsed.path :
        return False
    return True

# Check for repetition in question/answer or in answer itself
def check_repetition(answer, threshhold=2, question=None):
    answer = answer.split(' ')
    if not question:
        question = answer
    else:
        question = question.split(' ')

    size_max = max(len(answer), len(question))
    size_min = 2
    repetition_freq = 0
    if len(answer) < 4 or len(question) < 4:
        threshhold = 1
        size_min = 1

    for size in range(size_min, int(size_max / 2) + 1):
        for pos1 in range(len(answer) - size + 1):
            if question is answer:
                for pos2 in range(pos1 + size, len(question) - size + 1):

                    if answer[pos1:pos1 + size] == question[pos2:pos2 + size] and size >= threshhold:
                        repetition_freq += 1
            else:
                for pos2 in range(0, len(question) - size + 1):

                    if answer[pos1:pos1 + size] == question[pos2:pos2 + size] and size >= threshhold:
                        repetition_freq += 1

    return repetition_freq


# Returns index of best answer, 0 if not found
def score_answers(question,answers, name):

    answers_rate = []

    # Rate each answer
    for answer in answers:
        if re.search('<unk>', answer):
            answers_rate.append('invalid')
        elif any(re.search(regex, answer) for regex in eval(name + '_blacklist')):
            answers_rate.append('blacklist')
        elif is_complete_url(answer):
            answers_rate.append('url_complete')
        elif is_complete_url(answer,complete=False):
            answers_rate.append('url_incomplete')
        elif check_repetition(answer) is not 0:
            freq = check_repetition(answer)
            if freq > 3:
                answers_rate.append('big_repeat')
            else:
                answers_rate.append('small_repeat')
        elif check_repetition(answer,question=question) is not 0:
            answers_rate.append('similar_to_question')
        elif answer[-1] != '.' or '!' or '?':
            answers_rate.append('no_punctuation')
        elif answer[-1] == '.' or '!' or '?':
            answers_rate.append('finished_thought')
        else:
            answers_rate.append('unknown_condition')
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

        # replace newlinechar
        if 'newlinechar' in answer:
            answer = answer.replace('newlinechar', '\n')

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