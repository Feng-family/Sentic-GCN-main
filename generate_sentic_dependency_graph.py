# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')


def load_sentic_word():
    """
    load senticNet
    """
    path = './senticNet/senticnet_word.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet
# n*n的零矩阵
# 首先，判断是否为情感词，是，获得词典情感得分，否，情感得分为0。是否为方面词，是，情感得分得1，否，情感得分0。
# 将上面两个判断的结果赋予对角线（ii的值取决于单词i本身）
# 再观察该词的子节点，若子节点为方面词，情感得分基础上再+1（双向）
# （如果情感词是子节点，方面词为父节点，情感词的情感得分怎么向上传播，跟论文好像有点出入）
def dependency_adj_matrix(text, aspect, senticNet):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    #print('='*20+':')
    #print(document)
    #print(senticNet)
    for token in document:
        #print('token:', token)
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)]) + 1 #去除负号，情感范围从【-1，1】到【0，2】
        else:
            sentic = 0
        if str(token) in aspect:
            sentic += 1
        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * sentic
        #     # https://spacy.io/docs/api/token
            for child in token.children:#关系权重建立在父节点的情感得分基础上，子节点为方面词时，额外加1
                if str(child) in aspect:
                    sentic += 1
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1 * sentic
                    matrix[child.i][token.i] = 1 * sentic
    return matrix

# # 实例
# raw_text = 'Food is always fresh. and hot. - ready. to. eat. !'
# aspect = 'Food'
# senticNet = load_sentic_word()
# print(dependency_adj_matrix(raw_text,aspect, senticNet))
# [[1.    1.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#  [1.    0.    1.    1.    0.    0.    0.    0.    0.    0.    1.   ]
#  [0.    1.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#  [0.    1.    0.    1.063 1.063 0.    0.    1.063 0.    0.    0.   ]
#  [0.    0.    0.    1.063 0.    0.    0.    0.    0.    0.    0.   ]
#  [0.    0.    0.    0.    0.    0.15  0.    1.086 0.    0.    0.   ]
#  [0.    0.    0.    0.    0.    0.    0.    1.086 0.    0.    0.   ]
#  [0.    0.    0.    1.063 0.    1.086 1.086 1.086 0.    1.086 0.   ]
#  [0.    0.    0.    0.    0.    0.    0.    0.    1.789 1.148 0.   ]
#  [0.    0.    0.    0.    0.    0.    0.    1.086 1.148 1.148 0.   ]
#  [0.    1.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]]

def process(filename):
    senticNet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph_sdat', 'wb')#写输出
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right, aspect, senticNet)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('done !!!'+filename)
    fout.close()

# if __name__ == '__main__':
#     process('./datasets/acl-14-short-data/train.raw')
#     process('./datasets/acl-14-short-data/test.raw')
    # process('./datasets/semeval14/restaurant_train.raw')
    # process('./datasets/semeval14/restaurant_test.raw')
    # process('./datasets/semeval14/laptop_train.raw')
    # process('./datasets/semeval14/laptop_test.raw')
    # process('./datasets/semeval15/restaurant_train.raw')
    # process('./datasets/semeval15/restaurant_test.raw')
    # process('./datasets/semeval16/restaurant_train.raw')
    # process('./datasets/semeval16/restaurant_test.raw')

#输入'./datasets/acl-14-short-data/train.raw'
#输出'./datasets/acl-14-short-data/train.raw.graph_sdat'