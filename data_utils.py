# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

#从文件中加载预训练的词向量，并将其存储在一个字典中
#输入：word2idx
#输出：word_vec
def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
    return word_vec

#加载预训练的GloVe词向量，并根据词汇表中的词将其填充到一个词嵌入矩阵中。
# 如果词汇表中的词在GloVe中找不到对应的词向量，则该词在词嵌入矩阵中将保持为零向量。
# 如果词嵌入矩阵文件已存在，则直接从文件中加载，否则会进行计算并将结果保存到文件中，以备后续使用。
def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove.42B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

#Tokenizer的Python类，用于文本处理和构建词汇表。
# fit_on_text: 将文本输入Tokenizer对象中，用于根据文本构建词汇表。
# text_to_sequence: 将文本输入Tokenizer对象中，用于将文本转换为对应的序列（索引）。
class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

#构建ABSA的数据集，并通过索引获取其中的样本。
class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


#读取和处理ABSA的数据集。
# __read_text__：从文件中读取文本数据并返回合并后的文本。输入：文件名列表fnames，输出：合并的文本字符串text
# __read_data__：从文件中读取并处理样本数据。输入：文件名fname和一个Tokenizer对象tokenizer，输出：列表all_data，每个元素是一个包含样本信息的字典
# __init__：初始化ABSADatasetReader对象。输入：一个字符串参数dataset（默认为'twitter'）和一个整数参数embed_dim（默认为300）
# 根据dataset的不同选择相应的数据集文件，并调用__read_text__方法读取并合并文本数据。
# 然后，根据是否存在对应的词汇表文件，初始化一个Tokenizer对象。
# 接着，通过调用build_embedding_matrix方法，根据词汇表构建一个嵌入矩阵embedding_matrix。
# 最后，通过调用__read_data__方法读取并处理训练集和测试集的样本数据，
# 将它们转化为ABSADataset对象，并保存在train_data和test_data属性中。
class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'.graph', 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()
        fin = open(fname+'.sentic', 'rb')
        idx2gragh_s = pickle.load(fin)
        fin.close()
        fin = open(fname+'.graph_sdat', 'rb')
        idx2gragh_sdat = pickle.load(fin)
        fin.close()


        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            context = text_left + " " + aspect + " " + text_right

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(text_left)
            polarity = int(polarity)+1
            dependency_graph = idx2gragh[i]
            sentic_graph = idx2gragh_s[i]
            sdat_graph = idx2gragh_sdat[i]

            data = {
                'context': context,
                'aspect': aspect,
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'sentic_graph': sentic_graph,
                'sdat_graph': sdat_graph,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=300):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },

        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        if os.path.exists(dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer))
    
