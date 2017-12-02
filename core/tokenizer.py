import html
import regex as re
#import re
from setup.settings import preprocessing
import time

# inspired by https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl used in nmt's examples

# Load list of protected words/phrases (those will remain unbreaked, will be not tokenised)
protected_phrases_regex = []
with open(preprocessing['protected_phrases_file'], 'r', encoding='utf-8') as protected_file:
    protected_phrases_regex = list(filter(lambda word: False if word[0] == '#' else True, filter(None, protected_file.read().split("\n"))))

# Prepare regex of protrcted phrases (for speed)
matched_regexes = []
unmatched_regexes = []
phrase = ''
# Join multiple regexes of the same type into big one
for protected_phrase_regex in protected_phrases_regex:
    matched_regex = re.search(r'\(\?:\^\|\\s\)\(\?i:\((.*?) \?\\.\)\)', protected_phrase_regex)
    if matched_regex:
        matched_regexes.append(matched_regex.group(1))
    else:
        unmatched_regexes.append(protected_phrase_regex)
if protected_phrase_regex:
    phrase = ('(?:^|\s)(?i:((?:{}) ?\.))'.format('|'.join(matched_regexes)) if matched_regexes else '')\
             + ('|(?:' + (')|(?:'.join(unmatched_regexes)) + ')' if unmatched_regexes else '')

# Compile regexes
regex = {
    'special': re.compile(r'[\x00-\x1f]+'),
    'protected': re.compile(phrase),
    'periods': re.compile('\.{2,}'),
    'separate': re.compile(r'([^\w\s\.])'),
    'digits': re.compile(r'([\d])'),
    'joined': re.compile(r'[^\w\d_]'),
    'spaces': re.compile(r'\s+'),
    'restorephrases': re.compile(r'PROTECTEDREGEXPHRASE([\d\s]+?)PROTECTEDREGEXPHRASE'),
    'restoreperiods': re.compile(r'PROTECTEDPERIODS([\d\s]+?)PROTECTEDPERIODS'),
}

protected_phrases_replace = []
protected_phrases_counter = 0

# Tokenize sentense
def tokenize(sentence):

    global protected_phrases_replace, protected_phrases_counter, regex
    protected_phrases_replace = []
    protected_phrases_counter = 0
    protected_periods_counter = 0

    # Remove special tokens
    sentence = sentence.replace('<unk>', '').replace('<s>', '').replace('</s>', '')

    # Decode entities
    sentence = html.unescape(sentence)

    # Strip white charactes
    sentence = sentence.strip()

    # Remove white characters inside sentence
    sentence = regex['special'].sub('', sentence)

    # Regex-based protected phrases
    if(regex['protected'].search(sentence)):
        sentence = regex['protected'].sub(replace, sentence)

    # Protect multi-periods
    m = regex['periods'].findall(sentence)
    if m:
        protected_periods_counter += 1
        for dots in list(set(m)):
            sentence = sentence.replace(dots, ' PROTECTEDPERIODS{}PROTECTEDPERIODS '.format(len(dots)))

    # Normalize `->' and '' ->"
    sentence = sentence.replace('`', '\'').replace('\'\'', '"')

    # Separate some special charactes
    sentence = regex['separate'].sub(r' \1 ', sentence)

    # Separate digits in numbers
    sentence = regex['digits'].sub(' \\1 ', sentence)

    # Split sentence into words
    words = sentence.split()
    sentence = []

    # For every word
    for word in words:

        # Find if it ends with period
        if word[-1] == '.':
            m = word.rstrip('.')
            # If string still includes period
            if '.' in m and regex['joined'].search(m):
                pass

            else:
                word = m + ' .'

        # Add word to a sentence
        sentence.append(word)

    # Join words as a sentence again
    sentence = " ".join(sentence)

    # Strip spaces and remove multi-spaces
    sentence = sentence.strip()
    sentence = regex['spaces'].sub(' ', sentence)

    # Restore protected phrases and multidots
    if protected_phrases_counter:
        sentence = regex['restorephrases'].sub(lambda number: protected_phrases_replace[int(number.group(1).replace(" ", ""))], sentence)
    if protected_periods_counter:
        sentence = regex['restoreperiods'].sub(lambda number: ("." * int(number.group(1).replace(" ", ""))), sentence)

    return sentence

# Helper function for re.sub - not only replaces, but also saves replaced entity
def replace(entity):
    global protected_phrases_replace, protected_phrases_counter
    phrase = list(filter(None, list(entity.groups())))[0]
    replacement = entity.group(0).replace(phrase, ' PROTECTEDREGEXPHRASE{}PROTECTEDREGEXPHRASE '.format(protected_phrases_counter))
    protected_phrases_replace.append(phrase)
    protected_phrases_counter += 1
    return replacement

# list with regex-based detokenizer rules
answers_detokenize_regex = None

# Load detokenizer rules
with open(preprocessing['answers_detokenize_file'], 'r', encoding='utf-8') as answers_detokenize_file:
    answers_detokenize_regex = list(filter(lambda word: False if word[0] == '#' else True, filter(None, answers_detokenize_file.read().split("\n"))))

# Returns detokenizes sentences
def detokenize(answers):

    detokenized_answers = []

    # For every answer
    for answer in answers:

        # And every regex rule
        for detokenize_regex in answers_detokenize_regex:

            diffrence = 0

            # If detokenize_regex was found in answer
            if re.search(detokenize_regex, answer):

                # Search for all occurrences and iterate thru them
                regex = re.compile(detokenize_regex)
                for p in regex.finditer(answer):

                    # If there are more groups - process spaces that should stay in response
                    if len(p.groups()) > 1:
                        groups = p.groups()[1:]

                        # Replace spaces that should stay with temporary placeholder
                        for i, group in enumerate(groups):
                            position = p.start(i+2) + (i)*22
                            answer = answer[:position] + answer[position:].replace(" ", "##DONOTTOUCHTHISSPACE##", 1)

                        # Update reges to match placeholders as spaces
                        detokenize_regex = detokenize_regex.replace(' ', '(?: |##DONOTTOUCHTHISSPACE##)')

                # Search for all occurrences and iterate thru them again
                regex = re.compile(detokenize_regex)
                for p in regex.finditer(answer):

                    # Calculate data
                    replace_from = p.groups()[0]
                    replace_to = p.groups()[0].replace(" ", "")
                    position = p.start(1) + diffrence
                    diffrence += -len(replace_from) + len(replace_to)

                    # Remove spaces
                    answer = answer[:position] + answer[position:].replace(replace_from, replace_to, 1)

        # Change placeholders back to spaces
        answer = answer.replace("##DONOTTOUCHTHISSPACE##", ' ')

        detokenized_answers.append(answer)

    return detokenized_answers