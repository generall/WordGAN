#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import string
import numpy as np


# ### Synthetic dataset generation
# 
# Here we are going to create a fake dataset for debugging GAN code.
# 
# This dataset is simple enough to be sure, that NN is able to catch the idea.
# 
# The main point is following: there is an "entity" which is determined by a word and it's context.
# 
# The tricky part is that words are ambigous, and the entity could be identified either with word + close context, or with short + long context.
# 

# In[2]:


def gen_word(word_length):
    word_len = np.random.randint(*word_length)
    return ''.join(random.sample(string.ascii_lowercase, word_len))


# In[3]:


words = [gen_word((4, 5)) for i in range(25)]

context_words = [gen_word((4, 5)) for i in range(10)]

noize_words = [gen_word((4, 5)) for i in range(50)] + [''] * 20

# In[4]:


contexts = {}

for i in range(100):
    context = (
        random.choice(context_words),
        random.choice(context_words),
    )
    contexts[context] = 1

contexts = list(contexts.keys())

# In[33]:


seen_word_context = {}

entities = []
synonyms = []

for context in contexts:
    synonym_words = random.sample(words, 3)
    for word in synonym_words:
        entity = (word, *context)

        if (word, context[0]) in seen_word_context:
            next
        else:
            seen_word_context[(word, context[0])] = 1
        entities.append(entity)

    synonyms.append(synonym_words)

len(entities), len(synonyms)


# In[34]:


def gen_sentence(entity):
    word, short_context, long_context = entity
    short_context = [short_context] + random.sample(noize_words, 1)
    long_context = [long_context] + random.sample(noize_words, 1)
    random.shuffle(short_context)
    random.shuffle(long_context)

    sentence = [long_context[0], short_context[0], word, short_context[1], long_context[1]]

    return ' '.join(sentence).strip()


gen_sentence(('aaa', 'bbb', 'ccc'))

# In[35]:


with open('../data/synthetic/train_data.txt', 'w') as out:
    for i in range(10_000):
        entity = random.choice(entities)
        out.write(gen_sentence(entity))
        out.write('\n')

# In[36]:


import pandas as pd

# In[37]:


pd.DataFrame(entities).to_csv('../data/synthetic/entities.csv', header=None, index=None)

# In[38]:


pd.DataFrame(synonyms).to_csv('../data/synthetic/synonyns.csv', header=None, index=None)

# In[39]:


pd.DataFrame(words).to_csv('../data/synthetic/common.txt', header=None, index=None)

# In[40]:


import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# In[41]:


model = Word2Vec(LineSentence('../data/synthetic/train_data.txt'), size=20, window=5, min_count=0, iter=100)

# In[42]:


model.wv.save_word2vec_format('../data/synthetic/model.txt')

# In[ ]:
