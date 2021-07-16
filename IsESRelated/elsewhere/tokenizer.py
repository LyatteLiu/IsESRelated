# -*- coding: utf-8 -*-
import re as re
from typing import Callable, Iterator, Union, Optional, List
import numpy as np

#import pandas as pd
#from nltk.corpus import stopwords
#print(stopwords.words('english'))

'''
def get_nltk_stopwords_given_separator(separator_pattern):
    result = {}
    for w in stopwords.words('english'):
        tokens = separator_pattern.split(w)
        for t in tokens:
            result[t] = 1
    return result
'''

# need to import it from somewhere else...
class punk_tokenizer(object):
    def __init__(self, vocab_list:List[str], pad_tok_str: str='<PAD>'):
        self.punk_separator = re.compile(r'[^a-zA-Z0-9]+')
        self.vocab_list = list(vocab_list)
        #self.vocab_list.sort()
        self.vocab_list = [pad_tok_str] + self.vocab_list
        # save 
        # the python OrderedDict might not be the old-school thing we were looking for... it's all EDM now.
        self.word2id = {}
        t = 0 # PAD token is index 0
        for v in self.vocab_list:
            self.word2id[v] = t
            t += 1
        pass
    def from_vocab_file(vocab_file_path:str,pad_tok_str: str='<PAD>'):
        with open(vocab_file_path,'r') as fp:
            content = fp.readlines()
        content = [x.strip() for x in content]
        vocab_list = []
        word_counts = {}
        for l in content:
            splits = l.split(' ')
            v = splits[0]
            if v == pad_tok_str: # lame
                continue
            count = int(splits[1])
            word_counts[v] = count
            vocab_list.append(v)
        # 
        result = punk_tokenizer(vocab_list)
        result.word_counts = np.zeros((1+len(vocab_list),1),dtype=np.int64)
        for i in range(len(vocab_list)):
            result.word_counts[i+1] = word_counts[vocab_list[i]]
        return result
    '''
    def from_question_dataset(question_dataset_path:str,
        min_word_occurrence: int = 10,
        min_doc_occurrence: int = 2,
        doc_rat: int = 2,
        pad_tok_str: str='<PAD>',
        save_vocab_filepath='./vocab_file.txt'):
        df_question = pd.read_csv(question_dataset_path)
        punk_separator = re.compile(r'[^a-zA-Z0-9]+') # excessive!
        #stopword_collection = get_nltk_stopwords_given_separator(punk_separator)
        word_counts = {}
        word_doc_counts = {}
        for ds in [df_question['Subject'],df_question['UniqueBody']]:
            for text in ds:
                if not isinstance(text, str): continue
                splited_splinters = punk_separator.split(text)
                #print(splited_splinters)
                word_in_this_doc = {}
                for x in splited_splinters:
                    #print(x)
                    if len(x) == 0: continue
                    #if x in stopword_collection: continue
                    x_lower = x.lower()
                    if x_lower in word_counts:
                        word_counts[x_lower] += 1
                    else:
                        word_counts[x_lower] = 1
                    if not (x_lower in word_in_this_doc):
                        word_in_this_doc[x_lower] = 1
                        if x_lower in word_doc_counts:
                            word_doc_counts[x_lower] += 1
                        else:
                            word_doc_counts[x_lower] = 1
        vocab_list = []
        for v in word_counts:
            if word_counts[v] >= min_word_occurrence and v in word_doc_counts and word_doc_counts[v] >= min_doc_occurrence and word_doc_counts[v] <= len(df_question)//doc_rat:
                vocab_list.append(v)
        vocab_list.sort()
        result = punk_tokenizer(vocab_list)
        # PAD token is index 0
        result.word_counts = np.zeros((1+len(vocab_list),1),dtype=np.int64)
        for i in range(len(vocab_list)):
            result.word_counts[i+1] = word_counts[vocab_list[i]]
        with(open(save_vocab_filepath,'w+')) as fp:
            fp.write('%s %d\n' % (pad_tok_str,result.word_counts[0])) # PAD token is index 0
            for i in range(len(vocab_list)):
                fp.write('%s %d' % (vocab_list[i],result.word_counts[i+1]))
                if i != len(vocab_list)-1:
                    fp.write('\n')
        return result
    '''
    def tokenize(self, text:str) -> List[int]:
        result = []
        splited_splinters = self.punk_separator.split(text)
        for x in splited_splinters:
            x_lower = x.lower()
            if x_lower in self.word2id:
                result.append(self.word2id[x_lower])
        return result