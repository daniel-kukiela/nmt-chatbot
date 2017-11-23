import html
import re
from setup.settings import preprocessing


# inspired by https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl used in nmt's examples

# List of prefixes that doesnt break period after them
'''nonbreaking_prefixes = {
    'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 1, 'W': 1, 'X': 1, 'Y': 1, 'Z': 1,
    'Adj': 1, 'Adm': 1, 'Adv': 1, 'Asst': 1, 'Bart': 1, 'Bldg': 1, 'Brig': 1, 'Bros': 1, 'Capt': 1, 'Cmdr': 1, 'Col': 1, 'Comdr': 1, 'Con': 1, 'Corp': 1, 'Cpl': 1, 'DR': 1, 'Dr': 1, 'Drs': 1, 'Ens': 1,
    'Gen': 1, 'Gov': 1, 'Hon': 1, 'Hr': 1, 'Hosp': 1, 'Insp': 1, 'Lt': 1, 'MM': 1, 'MR': 1, 'MRS': 1, 'MS': 1, 'Maj': 1, 'Messrs': 1, 'Mlle': 1, 'Mme': 1, 'Mr': 1, 'Mrs': 1, 'Ms': 1, 'Msgr': 1,
    'Op': 1, 'Ord': 1, 'Pfc': 1, 'Ph': 1, 'Prof': 1, 'Pvt': 1, 'Rep': 1, 'Reps': 1, 'Res': 1, 'Rev': 1, 'Rt': 1, 'Sen': 1, 'Sens': 1, 'Sfc': 1, 'Sgt': 1, 'Sr': 1, 'St': 1, 'Supt': 1, 'Surg': 1,
    'v': 1, 'vs': 1, 'i.e': 1, 'rev': 1, 'e.g': 1,
    'No': 2, 'Nos': 1, 'Art': 2, 'Nr': 1, 'pp': 2,
    'Jan': 1, 'Feb': 1, 'Mar': 1, 'Apr': 1, 'Jun': 1, 'Jul': 1, 'Aug': 1, 'Sep': 1, 'Oct': 1, 'Nov': 1, 'Dec': 1
}'''
# Hey, it's reddit, better use lowercased list (and lowercase matched word later in code)
nonbreaking_prefixes = {
    'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1, 'h': 1, 'i': 1, 'j': 1, 'k': 1, 'l': 1, 'm': 1, 'n': 1, 'o': 1, 'p': 1, 'q': 1, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 1, 'w': 1, 'x': 1, 'y': 1, 'z': 1,
    'adj': 1, 'adm': 1, 'adv': 1, 'asst': 1, 'bart': 1, 'bldg': 1, 'brig': 1, 'bros': 1, 'capt': 1, 'cmdr': 1, 'col': 1, 'comdr': 1, 'con': 1, 'corp': 1, 'cpl': 1, 'dR': 1, 'dr': 1, 'drs': 1, 'ens': 1,
    'gen': 1, 'gov': 1, 'hon': 1, 'hr': 1, 'hosp': 1, 'insp': 1, 'lt': 1, 'mm': 1, 'mr': 1, 'mrs': 1, 'ms': 1, 'maj': 1, 'messrs': 1, 'mlle': 1, 'mme': 1, 'msgr': 1,
    'op': 1, 'ord': 1, 'pfc': 1, 'ph': 1, 'prof': 1, 'pvt': 1, 'rep': 1, 'reps': 1, 'res': 1, 'rt': 1, 'sen': 1, 'sens': 1, 'sfc': 1, 'sgt': 1, 'sr': 1, 'st': 1, 'supt': 1, 'surg': 1,
    'vs': 1, 'i.e': 1, 'rev': 1, 'e.g': 1,
    'no': 2, 'nos': 1, 'nrt': 2, 'nr': 1, 'pp': 2,
    'jan': 1, 'feb': 1, 'mar': 1, 'apr': 1, 'jun': 1, 'jul': 1, 'aug': 1, 'sep': 1, 'oct': 1, 'nov': 1, 'dec': 1
}

# Load list of protected words/phrases (those will not be breaked)
protected_phrases = []
with open(preprocessing['protected_phrases_file'], 'r', encoding='utf-8') as protected_file:
    protected_phrases = list(filter(None, protected_file.read().split("\n")))


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
            sentence = sentence.replace(phrase, 'PROTECTEDPHRASE{}PROTECTEDPHRASE '.format(i))
        i = i + 1

    # Strip spaces and remove multi-spaces
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)

    # Separate some special charactes
    sentence = re.sub(r'([^\w\s\.\'\`\,])', r' \1 ', sentence)

    # Protect multi-periods
    m = re.findall('\.{2,}', sentence)
    if m:
        for dots in list(set(m)):
            sentence = sentence.replace(dots, 'PROTECTEDPERIODS{}PROTECTEDPERIODS '.format(len(dots)))

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

            # If string still includes period and includes letter or equals non breaking word or is followed by word starting with lowercase letter (so is not new subsentence)
            if (re.match('\.', m) and re.match(r'[^\w\d_]', m)) or (m.lower() in nonbreaking_prefixes and nonbreaking_prefixes[m.lower()] == 1) or (i < len(words) and re.match('^[a-z]', words[i])):
                pass

            # If string is non breaking word followed by number
            elif m.lower() in nonbreaking_prefixes and nonbreaking_prefixes[m.lower()] == 2 and i < len(words) and re.match('^[0-9]+', words[i]):
                pass

            # Add space between word and period
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
    sentence = re.sub(r'\.\'$', ' . \' ', sentence)

    # Restore protected phrases and multidots
    sentence = re.sub(r'PROTECTEDPHRASE(\d+)PROTECTEDPHRASE', lambda number: protected_phrases[int(number.group(1))], sentence)
    sentence = re.sub(r'PROTECTEDPERIODS(\d+)PROTECTEDPERIODS', lambda number: "." * int(number.group(1)), sentence)

    return sentence
