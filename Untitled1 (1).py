
# coding: utf-8

# In[21]:


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
import numpy as np
import spacy

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim

import os, re, operator, warnings
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now matplotlib inline


# In[2]:


def clean(text):
    return str (''.join([i if ord(i) < 128 else ' ' for i in text]))

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
text = open(lee_train_file).read()


# In[3]:


from spacy.lang.en import English
nlp = spacy.load("en_core_web_sm")


# In[4]:


my_stop_words = [u'say', u'\'s', u'Mr', u'be', u'said', u'says', u'saying', 'the']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True


# In[5]:


doc = nlp(clean(text))


# In[6]:


doc


# In[7]:


texts, article = [], []
for w in doc:
    # if it's not a stop word or punctuation mark, add it to our article!
    if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
        # we add the lematized version of the word
        article.append(w.lemma_)
    # if it's a new line, it means we're onto our next document
    if w.text == '\n':
        texts.append(article)
        article = []


# In[8]:


bigram = gensim.models.Phrases(texts)


# In[9]:


texts = [bigram[line] for line in texts]


# In[10]:


dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


# In[11]:


lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)


# In[12]:


lsimodel.show_topics(num_topics=5)  # Showing only the top 5 topics


# In[13]:


hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)


# In[14]:


hdpmodel.show_topics()


# In[18]:


ldamodel = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary)


# In[19]:


ldamodel.show_topics()


# In[20]:


pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)


# In[ ]:


lsitopics = [[word for word, prob in topic] for topicid, topic in lsimodel.show_topics(formatted=False)]

hdptopics = [[word for word, prob in topic] for topicid, topic in hdpmodel.show_topics(formatted=False)]

ldatopics = [[word for word, prob in topic] for topicid, topic in ldamodel.show_topics(formatted=False)]


# In[ ]:


lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=texts, dictionary=dictionary, window_size=10).get_coherence()

hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=texts, dictionary=dictionary, window_size=10).get_coherence()

lda_coherence = CoherenceModel(topics=ldatopics, texts=texts, dictionary=dictionary, window_size=10).get_coherence()


# In[ ]:


def evaluate_bar_graph(coherences, indices):
    """
    Function to plot bar graph.
    
    coherences: list of coherence values
    indices: Indices to be used to mark bars. Length of this and coherences should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Value')


# In[ ]:


evaluate_bar_graph([lsi_coherence, hdp_coherence, lda_coherence],
                   ['LSI', 'HDP', 'LDA'])

