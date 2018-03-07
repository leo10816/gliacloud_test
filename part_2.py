import re
import numpy as np
from sklearn.decomposition import PCA
def remove_empty(sent_list):
    while '' in sent_list:
        sent_list.remove('')

def q21_cooccur_matrix(filename='raw_sentences.txt', window=2):
    vocab={}#將單字對應到 id
    inv_vocab={}#將 id 對應到單字
    token_size=0
    co_occur={}
    f=open(filename,'r')
    stop=re.compile(r',|.')
    for sentences in f:
        sentences=sentences.strip('\n')
        sentence=re.split(r',|\.',sentences)
        remove_empty(sentence)
        for sent in sentence:
            token=sent.split(' ')
            remove_empty(token)
            for i in range(len(token)):
                for j in range(1,window+1):
                    if i-j>0:
                        if token[i] not in vocab:
                            vocab[token[i]]=token_size
                            inv_vocab[token_size]=token[i]
                            token_size+=1
                        if token[i-j] not in vocab:
                            vocab[token[i-j]]=token_size
                            inv_vocab[token_size]=token[i-j]
                            token_size+=1
                        pair=str(vocab[token[i]])+' '+str(vocab[token[i-j]])
                        if pair not in co_occur:
                            co_occur[pair]=1
                        else:
                            co_occur[pair]+=1
                    if i+j<len(token):
                        if token[i] not in vocab:
                            vocab[token[i]]=token_size
                            inv_vocab[token_size]=token[i]
                            token_size+=1
                        if token[i+j] not in vocab:
                            vocab[token[i+j]]=token_size
                            inv_vocab[token_size]=token[i+j]
                            token_size+=1
                        pair=str(vocab[token[i]])+' '+str(vocab[token[i+j]])
                        if pair not in co_occur:
                            co_occur[pair]=1
                        else:
                            co_occur[pair]+=1
    cooccur_matrix=np.zeros((token_size,token_size),dtype=np.int)
    for key,value in co_occur.items():
        index=key.split(' ')
        cooccur_matrix[int(index[0])][int(index[1])]=value
    
    return vocab, inv_vocab, cooccur_matrix

def q22_word_vectors(cooccur_matrix, dim=10):
    pca=PCA(n_components=dim)
    word_vectors=pca.fittransform(cooccur_matrix)
    return word_vectors


vocab, inv_vocab, cooccur_matrix = q21_cooccur_matrix()
word_vectors = q22_word_vectors(cooccur_matrix)
print(word_vectors)
