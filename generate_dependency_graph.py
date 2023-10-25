# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
from spacy import displacy
nlp = spacy.load('en_core_web_sm')#使用 spacy 库加载了一个名为 "en_core_web_sm" 的英语语言模型。


#为给定的文本生成依赖关系矩阵。
def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
    return matrix
# #实例：
# raw_text = 'Food is always fresh and hot - ready to eat !'#11个节点
# aspect = 'Food'#方面节点
# print(dependency_adj_matrix(raw_text))#11*11的矩阵，类型是array
# [[1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 1. 1. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 1. 1. 1. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]


# 该函数读取一个文本文件，输入：filename='./datasets/acl-14-short-data/train.raw'
# 从文件中提取特定格式的文本与标记（"T" 标记），输出：filename+'.graph'
# 然后通过调用 dependency_adj_matrix 函数生成依赖关系矩阵
# 并将所有索引与对应的依赖关系矩阵存储到一个新的二进制文件中，以供后续使用。
def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('done !!!', filename)
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
    # process('./datasets/demo/train.raw')


#输入：'./datasets/acl-14-short-data/train.raw'
#输出：'./datasets/acl-14-short-data/train.raw.graph'