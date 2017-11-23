import html
import re
from setup.settings import preprocessing


# inspired by https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl used in nmt's examples

# Load list of protected words/phrases (those will remain unbreaked, will be not tokenised)
protected_phrases_regex = []
with open(preprocessing['protected_phrases_file'], 'r', encoding='utf-8') as protected_file:
    protected_phrases_regex = list(filter(lambda word: False if word[0] == '#' else True, filter(None, protected_file.read().split("\n"))))

# Tokenize sentense
def tokenize(sentence):
    # Remove special tokens
    sentence = re.sub('<unk>|<s>|</s>', '', sentence, flags=re.IGNORECASE)

    # Decode entities
    sentence = html.unescape(sentence)

    # Strip white charactes
    sentence = sentence.strip()

    # Remove white characters inside sentence
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'[\x00-\x1f]', '', sentence)

    # Regex-based protected phrases
    protected_phrases_regex_replacements = []
    for i, phrase in enumerate(protected_phrases_regex):

        # If phrase was found in sentence
        if re.search(phrase, sentence):

            # Search for all occurrences and iterate thru them
            regex = re.compile(phrase)
            for p in regex.finditer(sentence):

                # Replace with placeholder exactly one occurrence starting with start indice and add to list
                sentence = sentence[:p.start()] + sentence[p.start():].replace(p.groups()[0], ' PROTECTEDREGEXPHRASE{}PROTECTEDREGEXPHRASE '.format(i), 1)
                protected_phrases_regex_replacements.append(p.groups()[0].strip().replace(" ", ""))

    # Strip spaces and remove multi-spaces
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)

    # Protect multi-periods
    m = re.findall('\.{2,}', sentence)
    if m:
        for dots in list(set(m)):
            sentence = sentence.replace(dots, ' PROTECTEDPERIODS{}PROTECTEDPERIODS '.format(len(dots)))

    # Normalize `->' and '' ->"
    sentence = re.sub(r'\`', '\'', sentence)
    sentence = re.sub(r'\'\'', '"', sentence)

    # Separate some special charactes
    sentence = re.sub(r'([^\w\s\.])', r' \1 ', sentence)

    # Separate digits in numbers
    sentence = re.sub(r'([\d])', ' \\1 ', sentence)

    # Split sentence into words
    words = sentence.split()
    sentence = []

    # For every word
    for word in words:

        # Find if it ends with period
        m = re.match('(.+)\.$', word)
        if m:
            m = m.group(1)
            # If string still includes period
            if re.search('\.', m) and re.search(r'[^\w\d_]', m):
                pass

            else:
                word = m + ' .'

        # Add word to a sentence
        sentence.append(word)

    # Join words as a sentence again
    sentence = " ".join(sentence)

    # Strip spaces and remove multi-spaces
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)

    # Restore protected phrases and multidots
    sentence = re.sub(r'PROTECTEDREGEXPHRASE([\d\s]+?)PROTECTEDREGEXPHRASE', lambda number: protected_phrases_regex_replacements[int(number.group(1).replace(" ", ""))] , sentence)
    sentence = re.sub(r'PROTECTEDPERIODS([\d\s]+?)PROTECTEDPERIODS', lambda number: " " + ("." * int(number.group(1).replace(" ", ""))), sentence)

    return sentence
