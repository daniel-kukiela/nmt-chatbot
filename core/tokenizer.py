import html
import regex as re
#import re
from setup.settings import preprocessing
import time
import json

# inspired by https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl used in nmt's examples
# inspired by and based on https://github.com/rsennrich/subword-nmt

# Load list of protected words/phrases (those will remain unbreaked, will be not tokenized)
with open(preprocessing['protected_phrases_file'], 'r', encoding='utf-8') as protected_file:
    protected_phrases_regex = list(filter(lambda word: False if word[0] == '#' else True, filter(None, protected_file.read().split("\n"))))

# Prepare regex of protrcted phrases (for speed)
matched_regexes = []
unmatched_regexes = []
phrase = None
# Join multiple regexes of the same type into big one
protected_phrase_regex = None
for protected_phrase_regex in protected_phrases_regex:
    matched_regex = re.search(r'\(\?:\^\|\\s\)\(\?i:\((.*?) \?\\.\)\)', protected_phrase_regex)
    if matched_regex:
        matched_regexes.append(matched_regex.group(1))
    else:
        unmatched_regexes.append(protected_phrase_regex)
if protected_phrase_regex:
    phrase = re.compile(('(?:^|\s)(?i:((?:{}) ?\.))'.format('|'.join(matched_regexes)) if matched_regexes else '')\
             + ('|(?:' + (')|(?:'.join(unmatched_regexes)) + ')' if unmatched_regexes else ''))

# Compile regexes
regex = {
    'special': re.compile(r'[\x00-\x1f]+|\u3000'),
    'protected': phrase if phrase else None,
    'periods': re.compile('\.{2,}'),
    'separate': re.compile(r'(?<![▁])([^\w\s\.▁])'),
    'digits': re.compile(r'([\d])'),
    'joined': re.compile(r'[^\w\d_]'),
    'spaces': re.compile(r'[^\S\n]+'),
    'restorephrases': re.compile(r'P▁R([\d\s▁]+?)P▁R'),
    'restoreperiods': re.compile(r'P▁P([\d\s▁]+?)P▁P'),
    'separate_all': re.compile(r'(?<![ ▁])([^ ▁])'),
}

protected_phrases_replace = []
protected_phrases_counter = 0

# Tokenize sentence - standard
def tokenize(sentence):

    global protected_phrases_replace, protected_phrases_counter, regex
    protected_phrases_replace = []
    protected_phrases_counter = 0
    protected_periods_counter = 0

    # Decode entities
    sentence = html.unescape(sentence)

    # Remove special tokens
    sentence = sentence.replace('<unk>', '').replace('<s>', '').replace('</s>', '').replace('▁', '_')

    # Strip white characters
    sentence = sentence.strip()

    # Remove white characters inside sentence
    sentence = regex['special'].sub(' ', sentence)

    # Temporary restore new line
    sentence = sentence.replace('newlinechar', '\n')

    # Regex-based protected phrases
    if regex['protected'] and regex['protected'].search(sentence):
        sentence = regex['protected'].sub(replace, sentence)

    # Protect multi-periods
    m = regex['periods'].findall(sentence)
    if m:
        space = '' if preprocessing['use_bpe'] else ' '
        protected_periods_counter += 1
        for dots in sorted(set(m), reverse=True):
            sentence = sentence.replace(dots, '{}P▁P{}P▁P{}'.format(space, len(dots), space))

    # Normalize `->' and '' ->"
    sentence = sentence.replace('`', '\'').replace('\'\'', '"')

    # Strip spaces and remove multi-spaces
    sentence = sentence.strip()
    sentence = regex['spaces'].sub(' ', sentence)

    # Embedded detokenizer
    if preprocessing['embedded_detokenizer']:
        sentence = '▁' + sentence.replace(' ', ' ▁')

    if not preprocessing['use_bpe']:

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

    else:

        # Separate all characters
        sentence = regex['separate_all'].sub(' \\1', sentence)
        #sentence = ' '.join([(' ' + x) if x not in (' ', '▁') else x for x in list(sentence)])

    # Restore protected phrases and multidots
    if protected_phrases_counter:
        sentence = regex['restorephrases'].sub(lambda number: protected_phrases_replace[int(number.group(1).replace(" ", "").replace("▁", ""))], sentence)
    if protected_periods_counter:
        sentence = regex['restoreperiods'].sub(lambda number: ("." * int(number.group(1).replace(" ", "").replace("▁", ""))), sentence)

    # Replace new line char
    sentence = sentence.replace('\n', 'newlinechar')

    return sentence

# Helper function for re.sub - replaces and saves replaced entity
def replace(entity):
    global protected_phrases_replace, protected_phrases_counter
    phrase = list(filter(None, list(entity.groups())))[0]
    space = '' if preprocessing['use_bpe'] else ' '
    replacement = entity.group(0).replace(phrase, '{}P▁R{}P▁R{}'.format(space, protected_phrases_counter, space))
    protected_phrases_replace.append(phrase)
    protected_phrases_counter += 1
    return replacement

# Load detokenizer rules (for standard detokenizer)
if not preprocessing['embedded_detokenizer']:
    with open(preprocessing['answers_detokenize_file'], 'r', encoding='utf-8') as answers_detokenize_file:
        answers_detokenize_regex = list(filter(lambda word: False if word[0] == '#' else True, filter(None, answers_detokenize_file.read().split("\n"))))

# Returns detokenizes sentences
def detokenize(answers):

    # Embedded detokenizer
    if preprocessing['use_bpe']:
        # return [answer.replace(' ', '').replace('▁', ' ') for answer in answers]
        # Do nothing - sentence is already detokenized thanks to included SPM detokenizer in NMT enabled in setup/settings.py
        return answers

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


# Prepare vocab tokens from line
re_split = re.compile('(?: |^)(?:▁(▁))?([' + re.escape(r'`~!@#$%^&*()-_=+{[}]:;\'",<>?/|\\') + '0-9]|newlinechar|\.+)')
def sentence_split(sentence):

    # If not an embedded detokenizer - split by spaces
    if not preprocessing['embedded_detokenizer']:
        return sentence.split()

    global re_split

    # Prepare for split sentence into a words by ' ▁'
    line = ' ▁▁' + sentence[1:].replace('▁', '▁▁')
    line = re_split.sub(r' ▁\1\2 ▁', line)

    # split, filer and return
    return list(filter(lambda line: False if len(line) == 0 or line == '▁' else True, [token.strip() for token in line.split(' ▁')]))

# Load json file with BPE join pairs
def apply_bpe_load():
    with open('{}/{}'.format(preprocessing['train_folder'], 'bpe_joins.{}.json'.format('common' if preprocessing['joined_vocab'] else 'from')), 'r', encoding='utf-8', buffering=131072) as bpe_file:
        joins = {tuple(json.loads(k)): v for k, v in json.load(bpe_file).items()}

    apply_bpe_init(joins)

# Set BPE join pairs (used mostly by multiprocessing)
joins = []
def apply_bpe_init(joins_data):
    global joins
    joins = joins_data

# Apply BPE
sentence_cache = {}
def apply_bpe(sentence):

    # If BPE tokenization is disabled, return sentence
    if not preprocessing['use_bpe']:
        return sentence

    # Speeds up tokenization
    global sentence_cache

    # Split sentence by ' ▁'
    entities = sentence_split(sentence)
    new_sentence = []

    # For every entity in sentence
    for entity in entities:

        # If entity exists in cache - used cached (computed earlier) result
        original_entity = entity
        if original_entity in sentence_cache:
            new_sentence.append(sentence_cache[original_entity])
            continue

        # Split entity into pieces (mostly chars)
        entity = entity.split()

        # Make pairs of neighboring pieces/chars
        pairs = []
        prev_char = entity[0]
        for char in entity[1:]:
            pairs.append((prev_char, char))
            prev_char = char

        # Single piece/char - nothing to join
        if not pairs:
            new_sentence.append(entity[0])
            continue

        # Make every possible join
        while True:

            # Joins fragment - includes only pairs that exists in current entity
            subjoins = {pair:joins[pair] for pair in pairs if pair in joins}
            
            # Find most common pair
            pair = min(subjoins, key=subjoins.get, default=())

            # If there's no one - entity is joined
            if not pair or pair not in pairs:
                break

            # prepare pieces/chars
            first, second = pair
            new_pair = first + second

            #print(pairs)

            # Replace every occurence of pair with a joied one
            while pair in pairs:

                # Find pair occurence
                index = pairs.index(pair)

                # Remove pair and update neighbour pairs with joined one
                if index > 0:
                    pairs[index - 1] = (pairs[index - 1][0], new_pair)
                if index < len(pairs) - 1:
                    pairs[index + 1] = (new_pair, pairs[index + 1][1])
                if len(pairs) == 1:
                    pairs[0] = (new_pair, '')
                else:
                    del pairs[index]

        # We are going to use first subword from pair to rebuild entity, so we need to add second subword of last entity as a new 'pair'
        # (AB, C), (C, DEF), (DEF, GHIJK) -> AB, C, DEF, GHIJK
        if pairs[-1][1]:
            pairs.append((pairs[-1][1], ''))
        nentity = ' '.join([first for (first, second) in pairs])
        new_sentence.append(nentity)
        sentence_cache[original_entity] = nentity

    # Return joined sentence
    return ' '.join(new_sentence)
