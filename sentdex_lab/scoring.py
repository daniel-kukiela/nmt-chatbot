import time
from string import punctuation
import random
import re

'''
Maybe do something like "I don't know" if score is less than 0... or -3... or whatever.
emoji detector (multiple characters that arent ASCII? )
'''

sentence_ending_punc = ["'",'"',"!","?",".",")"]
bad_responses = ["http://","https://","http://en.wikipedia.org/wiki/List_of_burn_centers_in_the_United_States"]

# decent string check is [6]:
# emojie check is [-6]


def bad_response(answer,score):
    for br in bad_responses:
        if answer == br:
            score -= 10
    return score


def messedup_link(answer,score):
    badlinks = re.findall(r'\[.*?\]\s?\(',answer)
    goodlinks = re.findall(r'\[.*?\]\s?\(.*?\)',answer)
    if len(badlinks) > len(goodlinks):
        return score-3
    else:
        return score

def remove_punc(txt):
    for punc in punctuation:
        txt = txt.replace(punc,'')
    return txt


def ends_in_equals(answer,score):
    '''
    A lot of links end in = .... so let's penalize this
    '''
    if answer[-1] == "=":
        score -= 3
    return score

def unk_checker(answer,score):
    if "<unk>" in answer:
        score -= 4
    return score
    

def is_answer_identical(question, answer, score):
    '''
    if answer is the same, or very similar to the question, penalize.
    '''
    try:
        question = remove_punc(question)
        answer = remove_punc(answer)
            
        if question == answer:
            return score-4
        if answer in question or question in answer:
            return score-3
        else:
            return score
    except Exception as e:
        print(str(e))
        return score


  
def does_end_in_punctuation(answer,score):
    try:
        end_in_punc = False
        for punc in sentence_ending_punc:
            if answer[-1] == punc:
                end_in_punc = True

        if end_in_punc:
            return score+1
        else:
            return score
            
    except Exception as e:
        print(str(e))
        return score

def answer_echo(answer,score):
    answer = remove_punc(answer)
    tokenized = answer.split(' ')

    toklen = len(tokenized)
    repeats = 0
    for token in tokenized:
        if tokenized.count(token) > 1:
            repeats += 1

    pct = float(repeats)/ float(toklen)
    if pct == 1.0:
        score -= 5
    elif pct >= 0.75:
        score -= 4

    return score


def answer_echo_question(question,answer,score):
    answer = remove_punc(answer)
    question = remove_punc(question)
    
    ans_tokenized = answer.split(' ')
    que_tokenized = question.split(' ')

    toklen = len(ans_tokenized)

    if toklen == 1:
        return score

    else:
        repeats = 0
        for token in ans_tokenized:
            if que_tokenized.count(token) > 0:
                repeats += 1

        pct = float(repeats)/ float(toklen)
        if pct == 1.0:
            score -= 5
        elif pct >= 0.75:
            score -= 4


        #print(answer,repeats,pct)

        return score


def score_based_placement(idx, score):
    how_many_matter = 5
##    scores = [i for i in range(how_many_matter)][::-1]
##    if idx > how_many_matter-1:
##        return score
##    else:
##        return score + scores[idx]

    if idx == 1:
        score += 4
    elif idx < 3:
        score += 2
    elif idx < 6:
        score += 1

    return score

    


def do_scoring(question, answer, score):
    score = is_answer_identical(question, answer, score)
    score = does_end_in_punctuation(answer,score)
    score = answer_echo(answer,score)
    score = answer_echo_question(question,answer,score)
    score = ends_in_equals(answer,score)
    score = unk_checker(answer,score)
    score = messedup_link(answer,score)
    score = bad_response(answer,score)
    return score
    



if __name__ == '__main__':
    name = 'full_some_questions-81k.out'

    with open(name,'r',encoding='utf8') as f:
        contents = f.read().split("\n\n\n")
        for content in contents[0:-1]:
            batches = content.split(">>>")

            ans_score = {}
            
            for idx,batch in enumerate(batches[1:]):
                #print(batch)
                question, answer = batch.split('\n')[0], batch.split('\n')[1].split('::: ')[1]
                score = float(batch.split('\n')[1].split(' ::: ')[0])


                #score = score_based_placement(idx, score)

                score = do_scoring(question, answer, score)

                
                ans_score[answer] = score
                
                '''print(question)
                print(answer)
                print(score)
                print()
                print()'''

            scores = [v for k,v in ans_score.items()]
            
            max_score = max(scores)
            #print('Highest score =',max_score)

            options = [k for k,v in ans_score.items() if v == max_score]
            #print(options)

            choice_answer = random.choice(options)

            print(30*"_")
            print('> ',question)
            print(choice_answer)
            print(30*"_")

            #time.sleep(555)

        

