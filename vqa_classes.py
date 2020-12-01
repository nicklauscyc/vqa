import json
import os
import os.path
import re
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

#adapted from: https://github.com/Cyanogenoid/pytorch-vqa/blob/master/data.py

# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))

def prepare_questions(questions_json):
    #tokenize and normalize questions from given question json
    questions = [q['question'] for q in questions_json]
    for question in questions:
        question = question.lower()[:-1]
        yield question.split(' ')

def prepare_answers(answers_json):
    #normalize answers from given answer json
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json]
    def process_punctuation(s):
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('',s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()
    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))

class VQA(data.Dataset):
    def __init__(self, questions_json, answers_json, vocabulary_path):
        super(VQA, self).__init__()
        #with open(questions_path, 'r') as fd:
        #    questions_json = json.load(fd)
       # with open(answers_path, 'r') as fd:
        #    answers_json = json.load(fd)
        with open(vocabulary_path, 'r') as fd:
            vocab_json = json.load(fd)
        #include self.checkintegrity?

        # vocab
        self.vocab = vocab_json
        self.question_index = self.vocab['question']
        self.answer_index = self.vocab['answer']

        # q and a
        self.questions = list(prepare_questions(questions_json))
        self.answers = list(prepare_answers(answers_json))
        self.questions = [self._encode_question(q) for q in self.questions]
        self.answers = [self._encode_answers(a) for a in self.answers]
    
    def max_question_length(self):
    #get max question length
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions))
        return self._max_length
    
    def num_tokens(self):
    #get no. of question 'tokens'
        return len(self.question_index) + 1  # add 1 for unknown token at index 0

    def encode_question(self, questions):
        #converts question into vector of indices and question length
        #max_length = max(map(len, questions))
        vec = torch.zeros(self.max_question_length).long()
        for i, token in enumerate(question):
            index = self.question_index.get(token, 0)
            vec[i] = index
        return vec, len(question)

    def encode_answers(self, answers):
        #converts answer into vectors
        vec = torch.zeros(len(self.answer_index))
        for answer in answers:
            index = self.answer_index.get(answer)
            if index is not None:
                vec[index] += 1
        return vec
    

