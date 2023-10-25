# -*- coding: utf-8 -*-

import numpy as np
#import spacy
import pickle

#nlp = spacy.load('en_core_web_sm')


# 从指定文件中加载 SenticNet（情感词汇资源），将其中的词语与情感值对应关系存储到一个字典中，并返回该字典。
# 在加载完成后，你可以使用返回的字典来获取词语对应的情感值，从而在情感分析等任务中使用。

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

# 根据 1.SenticNet（情感词汇资源）中的情感值和2.目标词（aspect）对矩阵进行更新。
# 将关系两方的情感得分加和赋予关系。
# 一方是方面词，在情感赋值基础上再+1，最后将为零的对角线值全变为1。
def dependency_adj_matrix(text, aspect, senticNet):
    word_list = text.split()
    seq_len = len(word_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for i in range(seq_len):
        word = word_list[i]
        if word in senticNet:
            # 如果当前标记在 SenticNet 中存在，获取其情感值，并将其转换为浮点数。然后将情感值加1，这样可以用于加权依赖关系矩阵。
            sentic = float(senticNet[word]) + 1.0
        else:
            sentic = 0
        if word in aspect:
            #如果是目标词，再将情感值加1，用于加权依赖关系矩阵。
            sentic += 1.0
        for j in range(seq_len):
            matrix[i][j] += sentic
            matrix[j][i] += sentic
    for i in range(seq_len):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
    return matrix

# 实例
# raw_text = 'Food is always fresh and hot - ready to eat !'
# aspect = 'Food'
# senticNet = load_sentic_word()
# print(dependency_adj_matrix(raw_text,aspect, senticNet))
 #[[2.       1.       1.       2.063     1.       1.15      1.     2.086     2.789     2.148     1.      ]
 # [1.       1.       0.       1.063     0.       0.15      0.     1.086     1.789     1.148     0.      ]
 # [1.       0.       1.       1.063     0.       0.15      0.     1.086     1.789     1.148     0.      ]
 # [2.063    1.063    1.063    2.126     1.063    1.2129999 1.063  2.149     2.852     2.211     1.063   ]
 # [1.       0.       0.       1.063     1.       0.15      0.     1.086     1.789     1.148     0.      ]
 # [1.15     0.15     0.15     1.2129999 0.15     0.3       0.15   1.2360001 1.939     1.298     0.15    ]
 # [1.       0.       0.       1.063     0.       0.15      1.     1.086     1.789     1.148     0.      ]
 # [2.086    1.086    1.086    2.149     1.086    1.2360001 1.086  2.172     2.875     2.234     1.086   ]
 # [2.789    1.789    1.789    2.852     1.789    1.939     1.789  2.875     3.578     2.937     1.789   ]
 # [2.148    1.148    1.148    2.211     1.148    1.298     1.148  2.234     2.937     2.296     1.148   ]
 # [1.       0.       0.       1.063     0.       0.15      0.     1.086     1.789     1.148     1.      ]]


def process(filename):
    senticNet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.sentic', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right, aspect, senticNet)
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

#输入：'./datasets/acl-14-short-data/train.raw'
#输出：'./datasets/acl-14-short-data/train.raw.sentic'
