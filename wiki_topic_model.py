"""
LDA for Wikipedia Data-set
"""

# Import packages
import os
import random
import codecs
import _pickle as cPickle
from gensim.models import LdaModel as Lda
from gensim import corpora
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


# Remove stop words from sentences & lemmatize verbs.
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word, 'v') for word in stop_free.split())
    x = normalized.split()
    y = [s for s in x if len(s) > 2]
    return y


# Define the ath
corpus_path = "articles-corpus/"
article_paths = [os.path.join(corpus_path, p) for p in os.listdir(corpus_path)]

# Read contents of all the articles in a list "doc_complete"
doc_complete = []
for path in article_paths:
    fp = codecs.open(path, 'r', 'utf-8')
    doc_content = fp.read()
    doc_complete.append(doc_content)

# Get all documents
docs = open("docs_wiki.pkl", 'wb')
cPickle.dump(doc_complete, docs)

# Create training set
docs_train = doc_complete[:0.60*len(doc_complete)]

# Cleaning all the documents
stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
doc_clean = [clean(doc) for doc in docs_train]

# Creating the term dictionary of the courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Filter the terms which have occured in less than 100 articles and more than 70% of the articles
dictionary.filter_extremes(no_below=100, no_above=0.7)

# List of some words which has to be removed from dictionary as they are content neutral words
stoplist = set('also use make people know many call include part find become like mean often different \
                usually take wikt come give well get since type list say change see refer actually iii \
                aisne kinds pas ask would way something need things want every str'.split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)

# words,ids = dictionary.filter_n_most_frequent(50)
# print words,"\n\n",ids

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library & Training LDA model on the document term matrix.
lda_model = Lda(doc_term_matrix, num_topics=50, id2word=dictionary, passes=50, iterations=500)
lda_file = open('lda_model_sym_wiki.pkl', 'wb')
cPickle.dump(lda_model, lda_file)
lda_file.close()

# Print all the 50 topics
for topic in lda_model.print_topics(num_topics=50, num_words=10):
    print(topic[0] + 1, " ", topic[1], "\n")
