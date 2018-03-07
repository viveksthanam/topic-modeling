
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
import gensim
import numpy as np
import spacy

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim

import os, re, operator, warnings
warnings.filterwarnings('ignore')


# In[9]:


def clean(text):
    return str (''.join([i if ord(i) < 128 else ' ' for i in text]))

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'datasets'])
file = test_data_dir + os.sep + 'bbchealth.txt'
text = open(file).read()


# In[5]:


from spacy.lang.en import English
nlp = spacy.load("en_core_web_sm")


# In[6]:


my_stop_words = ['Jan' , 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True


# In[7]:


doc = nlp(clean(text))


# In[8]:


doc


# In[ ]:


texts, article = [], []
for w in doc:
    if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
        article.append(w.lemma_)
    if w.text == '\n':
        texts.append(article)
        article = []


# In[ ]:


bigram = gensim.models.Phrases(texts)


# In[ ]:


texts = [bigram[line] for line in texts]


# In[ ]:


dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


# In[ ]:


lsimodel = LsiModel(corpus=corpus, num_topics=5, id2word=dictionary)


# In[ ]:


lsimodel.show_topics(num_topics=5)  # Showing only the top 5 topics


# In[ ]:


hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)


# In[ ]:


hdpmodel.show_topics()


# In[ ]:


ldamodel = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary)


# In[ ]:


ldamodel.show_topics()


# In[ ]:


pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

