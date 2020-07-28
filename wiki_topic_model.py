"""
LDA for Wikipedia Data-set
"""

# Import packages
import os
import string
import codecs
import _pickle as cPickle
from gensim.models import LdaModel as Lda
from gensim import corpora
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


# Clean the data-set
def clean_data(doc, stop, lemma):
    """
    :param doc: input document
    :param stop: stop words
    :param lemma: lemmatizer
    :return: y
    """
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word, 'v') for word in stop_free.split())
    x = normalized.split()
    y = [s for s in x if len(s) > 2]

    return y


# Keep nouns and adjectives
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)]
    return ' '.join(nouns_adj)


# Load the data-set
def load_data(path):
    """
    :param path: data path
    :return: doc_clean
    """
    # get the path of the documents
    document_paths = [os.path.join(corpus_path, p) for p in os.listdir(path)]

    # read contents of all the articles in a list "doc_complete"
    print("Loading data...")
    doc_complete = []
    index = 0
    for path in document_paths:
        fp = codecs.open(path, 'r', 'utf-8')
        doc_content = fp.read()
        doc_content = nouns_adj(doc_content)
        doc_complete.append(doc_content)
        print("Processing doc:", index)
        index += 1

    # # Create training set
    # end = int(0.85 * len(doc_complete))
    # docs_train = doc_complete[:end]

    # Cleaning all the documents
    stop = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()
    doc_clean = [clean_data(doc, stop, lemma) for doc in doc_complete]

    # Save all documents
    docs = open("docs_10k_train.pkl", 'wb')
    cPickle.dump(doc_clean, docs)

    print("All documents has been processed.")

    return doc_clean


def create_dictionary(doc_clean):
    """
    :param doc_clean: cleaned documents
    :return: dictionary
    """
    # Creating the term dictionary of the courpus, where every unique term is assigned an index.
    print("Creating dictionary...")
    dictionary = corpora.Dictionary(doc_clean)
    dict = open("dictionary.pkl", "wb")
    cPickle.dump(dictionary, dict)
    dict.close()

    # Filter the terms which have occured in less than 100 articles and more than 70% of the articles
    dictionary.filter_extremes(no_below=100, no_above=0.7)
    print("Dictionary has been created.")

    return dictionary


def create_bow(doc_clean, dictionary):
    """
    :param doc_clean: cleaned documents
    :param dictionary: dictionary
    :return: doc_term_matrix
    """
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    print("Creating bag of words...")
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    print("Bag of words has been created.")

    return doc_term_matrix


# Build and train the LDA model
def train_lda(bow, dictionary, num_topics):
    """
    :param bow: bag of words representation of documents
    :param dictionary: dictionary
    :param num_topics: number of topics
    :return: lda_model
    """
    # Creating the object for LDA model using gensim library & Training LDA model on the document term matrix.
    print("Training LDA model...")
    lda_model = Lda(bow, num_topics=num_topics, id2word=dictionary, passes=20, iterations=100)
    # lda_file = open('lda_model_sym_10k_train.pkl', 'wb')
    # cPickle.dump(lda_model, lda_file)
    # lda_file.close()
    print("LDA model has been trained.")

    return lda_model


# Get topics
def get_topics(lda_model):
    """
    :param lda_model: trained LDA model
    :return: topics
    """
    # Print all the topics
    topics = []
    for idx, topic in lda_model.print_topics(num_topics=10, num_words=10):
        topics.append(topic)
        print("Topic: {} \nWords: {}".format(idx, topic))
        print("\n")

    return topics


# Filter out punctuations
def contains_punctuation(w):
    """
    :param w: target word
    :return: list
    """
    return any(char in string.punctuation for char in w)


# Filter out numeric values
def contains_numeric(w):
    """
    :param w: target word
    :return: list
    """
    return any(char.isdigit() for char in w)


# Document frequency
def D_w(word, corpus):
    """
    :param word: target word
    :param corpus: corpus
    :return: D_w
    """
    D_w = 0
    for i in range(len(corpus)):
        if word in corpus[i]:
            D_w += 1
    return D_w


# Get Topic Coherence(TC)
def topic_coherence(corpus, topics):
    """
    :param corpus: corpus
    :param topics: generated topics
    :return: TC
    """
    # Select top-10 most likely words in each topic
    words = [None] * len(topics)
    for i in range(len(topics)):
        topic = topics[i]
        topic = [w.lower() for w in topic if not contains_punctuation(w)]
        topic = [w for w in topic if not contains_numeric(w)]
        topic = "".join(topic)
        topic = topic.split(sep=' ')
        topic = [w for w in topic if w != '']
        words[i] = topic[0:11]

    # Calculate topic coherence
    D = len(corpus)
    TC = []
    for k in range(len(words)):
        TC_k = 0
        counter = 0
        word_k = words[k]

        for i in range(10):
            w_i = word_k[i]
            tmp = 0

            for j in range(i + 1, 10):
                w_j = word_k[j]
                D_wi = D_w(w_i, corpus)
                D_wj = D_w(w_j, corpus)
                # Joint document frequency
                D_wi_wj = 0
                for i in range(len(corpus)):
                    if (w_i in corpus[i]) and (w_j in corpus[i]):
                        D_wi_wj += 1

                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                tmp += f_wi_wj
                counter += 1

            TC_k += tmp

        TC.append(TC_k)
        TC = np.mean(TC) / counter

        return TC


# Get Topic Diversity(TD)
def topic_diversity(topics):
    """
    :param topics: generated topics
    :return: TD
    """
    # Get all words in generated 100 topics
    topic_words = []
    for i in range(len(topics)):
      topic = topics[i]
      topic = [w.lower() for w in topic if not contains_punctuation(w)]
      topic = [w for w in topic if not contains_numeric(w)]
      topic = "".join(topic)
      topic = topic.split(sep=' ')
      topic = [w for w in topic if w != '']
      topic_words.extend(topic)

    # Get all unique words for topics
    unique_words = []
    for w in topic_words:
      if w not in unique_words:
        unique_words.append(w)

    # Calculate Topic Diversity(TD)
    TD = len(unique_words) / len(topic_words)

    return TD


# Query trained model on new dataset
def query(query_text, dictionary, trained_model, index):
    """
    :param query_text: new corpus to be query
    :param dictionary: dictionary
    :param trained_model: trained LDA model
    :param index: index of document in new corpus
    :return: vector
    """
    query_bow = [dictionary.doc2bow(text) for text in query_text]
    query_doc = query_bow[index]
    vector = trained_model[query_doc]
    return vector


# Main
if __name__ == "__main__":
    # load the data-set
    corpus_path = "C:/Users/Public/LDA_Wiki/10k_training"
    doc_clean = load_data(corpus_path)

    # create dictionary
    dictionary = create_dictionary(doc_clean)

    # create bow representation of documents
    bow = create_bow(doc_clean, dictionary)

    # build and train LDA model
    lda_model = train_lda(bow, dictionary, 20)
    trained_lda = open("trained_lda_10k_20.pkl", 'wb')
    cPickle.dump(lda_model, trained_lda)
    trained_lda.close()

    # get and print out generated topics
    trained_lda = open("trained_lda_10k_20.pkl", 'rb')
    trained_model = cPickle.load(trained_lda)
    topics = get_topics(trained_model)

    # get Topic Coherence(TC)
    topic_coherence = topic_coherence(corpus=doc_clean, topics=topics)
    print("The Topic Coherence(TC) of LDA model is:", topic_coherence)

    # get Topic Diversity(TD)
    topic_diversity = topic_diversity(topics)
    print("The Topic Diversity(TD) of LDA model is: ", topic_diversity)
