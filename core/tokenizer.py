import html
import re
from setup.settings import preprocessing


# inspired by https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl used in nmt's examples

# Load list of protected words/phrases (those will remain unbreaked, will be not tokenised)
protected_phrases = []
with open(preprocessing['protected_phrases_file'], 'r', encoding='utf-8') as protected_file:
    protected_phrases = list(filter(None, protected_file.read().split("\n")))
protected_phrases_regex = []
with open(preprocessing['protected_phrases_regex_file'], 'r', encoding='utf-8') as protected_file:
    protected_phrases_regex = list(filter(None, protected_file.read().split("\n")))


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

    # Protect phrases
    i = 0
    for phrase in protected_phrases:
        if phrase in sentence:
            sentence = sentence.replace(phrase, ' PROTECTEDPHRASE{}PROTECTEDPHRASE '.format(i))
        i = i + 1

    # Protected regex-based phrases
    i = 0
    protected_phrases_regex_replacements = []
    for phrase in protected_phrases_regex:

        # If phrase was found in sentence
        if re.search(phrase, sentence):

            # Search for all occurrences and iterate thru them
            regex = re.compile(phrase)
            for p in regex.finditer(sentence):

                # Replace with placeholder exactly one occurrence starting with start indice and add to list
                sentence = sentence[:p.start()] + sentence[p.start():].replace(p.groups()[0], ' PROTECTEDREGEXPHRASE{}PROTECTEDREGEXPHRASE '.format(i), 1)
                protected_phrases_regex_replacements.append(p.groups()[0].strip().replace(" ", ""))
                i = i + 1

    # Strip spaces and remove multi-spaces
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)

    # Protect multi-periods
    m = re.findall('\.{2,}', sentence)
    if m:
        for dots in list(set(m)):
            sentence = sentence.replace(dots, ' PROTECTEDPERIODS{}PROTECTEDPERIODS '.format(len(dots)))

    # Separate some special charactes
    sentence = re.sub(r'([^\w\s\.\'\`\,])', r' \1 ', sentence)

    # Separate string, and ,string (but not number,number except for string at the end of sentence)
    sentence = re.sub(r'([^\d]),', r'\1 , ', sentence)
    sentence = re.sub(r',([^\d])', r' , \1', sentence)
    sentence = re.sub(r'([\d]),$', r'\1 ,', sentence)

    # Normalize `->' and '' ->"
    sentence = re.sub(r'\`', '\'', sentence)
    sentence = re.sub(r'\'\'', '"', sentence)

    # Separate apostrophes in a right way
    sentence = re.sub(r'([^\W\d_])\'([^\W\d_])', '\\1 \' \\2', sentence)
    sentence = re.sub(r'([^\W_])\'([^\w\d_])', '\\1 \' \\2', sentence)
    sentence = re.sub(r'([^\w\d_])\'([^\W\d_])', '\\1 \' \\2', sentence)
    sentence = re.sub(r'([^\w\d_])\'([^\w\d_])', '\\1 \'\\2', sentence)
    sentence = re.sub(r'([\d])\'s', '\\1 \'s', sentence)

    # Separate digits in numbers
    sentence = re.sub(r'([\d])', ' \\1 ', sentence)

    # Split sentence into words
    words = sentence.split()
    i = 1
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

        i = i + 1

    # Join words as a sentence again
    sentence = " ".join(sentence)

    # Strip spaces and remove multi-spaces
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)

    # .' at the end of the sentence
    #sentence = re.sub(r'\.\'$', ' . \' ', sentence)

    # Restore protected phrases and multidots
    sentence = re.sub(r'PROTECTEDPHRASE([\d\s]+?)PROTECTEDPHRASE', lambda number: protected_phrases[int(number.group(1).replace(" ", ""))] , sentence)
    sentence = re.sub(r'PROTECTEDREGEXPHRASE([\d\s]+?)PROTECTEDREGEXPHRASE', lambda number: protected_phrases_regex_replacements[int(number.group(1).replace(" ", ""))] , sentence)
    sentence = re.sub(r'PROTECTEDPERIODS([\d\s]+?)PROTECTEDPERIODS', lambda number: " " + ("." * int(number.group(1).replace(" ", ""))), sentence)

    return sentence
