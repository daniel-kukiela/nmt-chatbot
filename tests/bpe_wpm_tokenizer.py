from tqdm import tqdm

ffrom = open('../data/train.bpe.from', encoding='utf-8').readlines()
fto = open('../data/train.bpe.to', encoding='utf-8').readlines()
vocab = [x.rstrip() for x in open('../data/vocab.bpe.common', encoding='utf-8').readlines()]

for i, _ in tqdm(enumerate(ffrom), ascii=True, unit=' lines', total=len(ffrom)):

    wordsfrom = ffrom[i].split()
    wordsto = fto[i].split()
    for word in wordsfrom:
        if word not in vocab:
            print(word, 'from')
            print(ffrom[i])
            print(fto[i])
            print()
    for word in wordsto:
        if word not in vocab:
            print(word, 'to')
            print(ffrom[i])
            print(fto[i])
            print()
