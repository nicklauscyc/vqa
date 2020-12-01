import json_access 
from collections import Counter
import itertools
from vqa_classes import *
import random
import os
import json

#adapted from: https://github.com/Cyanogenoid/pytorch-vqa/blob/master/preprocess-vocab.py

split = 'train'
annFile='Annotations/%s.json'%(split)
imgDir = 'train'

# initialize VQA api for QA annotations
vqa=json_access.VQA(annFile)

imgs = vqa.getImgs()
anns = vqa.getAnns(imgs=imgs)

def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab

def main():
    questions = anns
    answers = anns
    questions = list(prepare_questions(questions))
    answers = list(prepare_answers(answers))
    question_vocab = extract_vocab(questions, start=1)
    answer_vocab = extract_vocab(answers, top_k=5000) #what should top_k be here?
    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }
    with open("vocab.txt", 'w') as fd:
        json.dump(vocabs, fd)
    
    #v = list(encode_answers(a, answer_vocab) for a in answers)
    #print(v)

if __name__ == '__main__':
    main()






