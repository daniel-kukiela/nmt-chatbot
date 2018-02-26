import Levenshtein
from setup.settings import score as score_settings, preprocessing, hparams
import regex as re
from collections import defaultdict
import time
import requests

# Whether entence is a valid url
full_sentence_valid_url = False

# Check if sentence ends properly
def ending(index, question, answer):
    global full_sentence_valid_url

    # Disabled
    if score_settings['no_ending_modifier_value'] is None:
        return 0

    # Valid url can end with any char
    if full_sentence_valid_url:
        return 0

    # Short sentence and ends with letter or digit
    if len(answer.strip()) < score_settings['sentence_ending_sentence_len'] and not re.search('[^a-zA-Z0-9 ]', answer.strip()):
        return score_settings['no_ending_modifier_value'][1]

    # Ends with ascii emoticon
    last_token = answer.split()[-1]
    ascii_emoticon_score = len(re.findall('[^a-zA-Z0-9]', last_token)) / len(last_token) if len(last_token) > 1 else 0

    # Ends with valid character
    if re.search('(' + score_settings['sentence_ending'] + ')$', answer.strip()) or ascii_emoticon_score > score_settings['ascii_emoticon_non_char_to_all_chars_ratio']:
        return 0

    # Inproper ending
    return score_settings['no_ending_modifier_value'][0]

# Whether sentence ends with emoticon
valid_emoticon = False

# Check if sentence is an ascii emoticon
def ascii_emoticons(index, question, answer):
    global valid_emoticon

    valid_emoticon = False

    # Disabled
    if score_settings['ascii_emoticon_modifier_value'] is None:
        return 0

    # Split by words (tokens)
    tokens = answer.split()

    # Calculate emoticon score
    score = [1 if len(token) > 1 and len(re.findall('[^a-zA-Z0-9]', token)) / len(token) > score_settings['ascii_emoticon_non_char_to_all_chars_ratio'] else 0 for token in tokens]
    score = sum([1 if (index > 0 and score[index - 1] == 0 and value == 1) or (index == 0 and value == 1) else 0 for index, value in enumerate(score)]) * score_settings['ascii_emoticon_modifier_value']

    if score:
        valid_emoticon = True

    return score

# Check if sentence includes 'unk' token
def unk(index, question, answer):

    # Disable
    if score_settings['unk_modifier_value'] is None:
        return 0

    # Includes 'unk'
    if '<unk>' in answer or '_unk' in answer:
        return score_settings['unk_modifier_value']

    return 0

# Load sunsentence scoring
with open(score_settings['answers_subsentence_score_file'], 'r', encoding='utf-8') as answers_subsentence_score_file:
    response_subsentence_score = list(filter(lambda word: False if word[0] == '#' else True, filter(None, answers_subsentence_score_file.read().split("\n"))))
    response_subsentence_score = {(question_regex.strip(), answer_regex.strip()): float(score.strip()) for score, question_regex, answer_regex in [line.split('##->##') for line in response_subsentence_score if '##->##' in line]}

# Score subsentences (part of sentence can modify it's score)
def subsentence_score(index, question, answer):

    # Disabled
    if not score_settings['use_subsentence_score']:
        return 0

    # Apply score
    return sum([response_subsentence_score[(question_regex.re.pattern, answer_regex.re.pattern)] for question_regex, answer_regex in [(re.search(question_regex, question), re.search(answer_regex, answer)) for question_regex, answer_regex in response_subsentence_score] if question_regex is not None and answer_regex is not None])


position_modifiers = None

# Score by position in answers list
def position(index, question, answer):
    global position_modifiers

    # Disabled
    if score_settings['position_modifier'] is None:
        return 0

    # Generate scoring list
    if position_modifiers is None:
        position_modifiers = {}
        last_score_modifier = 0
        for i in range(1, hparams['num_translations_per_input']+1):
            if i in score_settings['position_modifier']:
                last_score_modifier = score_settings['position_modifier'][i]
            position_modifiers[i] = last_score_modifier

    # Return score
    return position_modifiers[index]

# Url cache
url_cache = defaultdict(lambda: [0, 0])

# Check if url is valid
def check_urls(index, question, answer):
    global full_sentence_valid_url

    full_sentence_valid_url = False
    valid_url = False

    # Disabled
    if score_settings['incorrect_url_modifier_value'] is None:
        return 0

    # Find all utls in sentence
    for url in re.finditer('http(?:s?):(//([^/]*?)/(?:[^ ])*?(?=$|[' + re.escape(score_settings['url_delimiters']) + ']))?', answer):

        # Check if result is in cache already and return it
        if url_cache[url.group(0)][1] > time.time():
            if url_cache[url.group(0)][0] == 0:
                return score_settings['incorrect_url_modifier_value']

        # Url not in cache - check it
        else:

            # Send HEAD request and check HTTP response code
            try:
                request = requests.head(url.group(0))
                code = request.status_code
            except Exception as e:
                code = 0

            # Add to cache
            url_cache[url.group(0)] = [1 if code == 200 else 0, time.time() + 86400]

            # If code is diffrent than 200 - return modifier value
            if code != 200:
                return score_settings['incorrect_url_modifier_value']

        # Check if it's full sentence url
        valid_url = (len(url.group(0)) == len(answer))

    # Everyting ok, set if full sentence url and return 0
    full_sentence_valid_url = valid_url
    return 0

# Add score by sentence length
def reward_longer_sentences(index, question, answer):

    # Disabled
    if score_settings['reward_long_sentence_value'] is None:
        return 0

    # Return score multiplied by number of chars
    return len(answer) * score_settings['reward_long_sentence_value']

# Chceck for answer similarity to question
def question_answer_similarity_by_ratio(index, question, answer):
    global valid_emoticon

    # Disabled or short or char emoticon
    if score_settings['question_answer_similarity_modifier_value'] is None or len(answer) < score_settings['question_answer_similarity_sentence_len'] or valid_emoticon:
        return 0

    # Divide response into subsentences
    answer = list(filter(None, re.split(score_settings['subsentence_dividers'], answer))) + [answer]

    # Calculate similarity for every subsentence, gext maximum one
    ratio = max([Levenshtein.ratio(question, s) for s in answer])

    # Not similar
    if ratio < score_settings['question_answer_similarity_threshold']:
        return 0

    # Apply value
    if score_settings['question_answer_similarity_modifier'] == 'value':
        return score_settings['question_answer_similarity_modifier_value']

    # Apply multiplier
    if score_settings['question_answer_similarity_modifier'] == 'multiplier':
        return (ratio - score_settings['question_answer_similarity_threshold']) / (1 - score_settings['question_answer_similarity_threshold']) * score_settings['question_answer_similarity_modifier_value']

    return 0

'''
def question_answer_diffrence_by_distance(index, question, answer):
    global valid_emoticon

    if score_settings['question_answer_diffrence_modifier_value'] is None or len(answer) < score_settings['question_answer_diffrence_sentence_len'] or valid_emoticon:
        return 0

    answer_len = len(answer)
    answer = list(filter(None, re.split(score_settings['subsentence_dividers'], answer)))
    distance = sum([Levenshtein.distance(question, s) for s in answer])/max(len(question), answer_len)
    #print(distance, score_settings['question_answer_diffrence_threshold'], [Levenshtein.distance(question, s) for s in answer], len(answer), len(question))

    if distance > score_settings['question_answer_diffrence_threshold']:
        return 0

    if score_settings['question_answer_diffrence_modifier'] == 'value':
        return score_settings['question_answer_diffrence_modifier_value']

    if score_settings['question_answer_diffrence_modifier'] == 'multiplier':
        return (score_settings['question_answer_diffrence_threshold'] - distance) / score_settings['question_answer_diffrence_threshold'] * score_settings['question_answer_diffrence_modifier_value']

    return 0
'''

# Check subsentences in answer for similarity
def answer_subsentence_similarity_by_ratio(index, question, answer):
    global valid_emoticon

    # Disabled or short or char emoticon
    if score_settings['answer_subsentence_similarity_modifier_value'] is None or len(answer) < score_settings['answer_subsentence_similarity_sentence_len'] or valid_emoticon:
        return 0

    # Split response into subsentences
    answer = list(filter(None, re.split(score_settings['subsentence_dividers'], answer)))

    # Find max similarity
    max_ratio = 0
    for num, subsentence in enumerate(answer):
        for sunsentence2 in answer[num+1:]:
            max_ratio = max(max_ratio, Levenshtein.ratio(subsentence, sunsentence2))

    # Not similar
    if max_ratio < score_settings['answer_subsentence_similarity_threshold']:
        return 0

    # Apply value
    if score_settings['answer_subsentence_similarity_modifier'] == 'value':
        return score_settings['answer_subsentence_similarity_modifier_value']

    # Apply multiplier
    if score_settings['answer_subsentence_similarity_modifier'] == 'multiplier':
        return (max_ratio - score_settings['answer_subsentence_similarity_threshold']) / (1 - score_settings['answer_subsentence_similarity_threshold']) * score_settings['answer_subsentence_similarity_modifier_value']

    return 0

'''
def answer_subsentence_diffrence_by_distance(index, question, answer):
    global valid_emoticon

    if score_settings['answer_subsentence_diffrence_modifier_value'] is None or len(answer) < score_settings['answer_subsentence_diffrence_sentence_len'] or valid_emoticon:
        return 0

    answer = list(filter(None, re.split(score_settings['subsentence_dividers'], answer)))

    min_distance = 1
    for num, subsentence in enumerate(answer):
        for sunsentence2 in answer[num+1:]:
            min_distance = min(min_distance / max(len(subsentence), len(sunsentence2)), Levenshtein.distance(subsentence, sunsentence2))

    if min_distance > score_settings['answer_subsentence_diffrence_threshold']:
        return 0

    if score_settings['answer_subsentence_diffrence_modifier'] == 'value':
        return score_settings['answer_subsentence_diffrence_modifier_value']

    if score_settings['answer_subsentence_diffrence_modifier'] == 'multiplier':
        return (score_settings['answer_subsentence_diffrence_threshold'] - min_distance) / score_settings['answer_subsentence_diffrence_threshold'] * score_settings['answer_subsentence_diffrence_modifier_value']

    return 0
'''

# Score response
def score_answers(question, answers):

    # Functions to run
    functions = [
        check_urls,
        ending,
        ascii_emoticons,
        unk,
        subsentence_score,
        position,
        reward_longer_sentences,
        question_answer_similarity_by_ratio,
        #question_answer_diffrence_by_distance,
        answer_subsentence_similarity_by_ratio,
        #answer_subsentence_diffrence_by_distance,
    ]

    scores = {'score': [], 'score_modifiers': []}

    # Iterate thru answers, apply every scoring function
    for i, answer in enumerate(answers):

        score_modifiers = [function(i+1, question, answer) for function in functions]
        scores['score'].append(score_settings['starting_score'] + sum(score_modifiers))
        scores['score_modifiers'].append(score_modifiers)

    # Return score
    return scores
