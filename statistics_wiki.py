# Import packages
import os
import nltk
import codecs
import _pickle as cPickle

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

"""
Topics
"""
# Art and culture
topic_1 = open("C:/Users/Public/Wiki_Topics/1. Culture and the arts.txt").read().split()

# Geography and places
topic_2 = open("C:/Users/Public/Wiki_Topics/2. Geography and places.txt").read().split()

# Health and fitness
topic_3 = open("C:/Users/Public/Wiki_Topics/3. Health and fitness.txt").read().split()

# History and events
topic_4 = open("C:/Users/Public/Wiki_Topics/4. Human activities.txt").read().split()

# Mathematics and abstractions
topic_5 = open("C:/Users/Public/Wiki_Topics/5. Mathematics and logic.txt").read().split()

# Natural sciences and nature
topic_6 = open("C:/Users/Public/Wiki_Topics/6. Natural and physical sciences.txt").read().split()

# People and self
topic_7 = open("C:/Users/Public/Wiki_Topics/7. People and self.txt").read().split()

# Philosophy and thinking
topic_8 = open("C:/Users/Public/Wiki_Topics/8. Philosophy and thinking.txt").read().split()

# Religion and spirituality
topic_9 = open("C:/Users/Public/Wiki_Topics/9. Religion and belief systems.txt").read().split()

# Social sciences and society
topic_10 = open("C:/Users/Public/Wiki_Topics/10. Society and social sciences.txt").read().split()

# Technology and applied sciences
topic_11 = open("C:/Users/Public/Wiki_Topics/11. Technology and applied sciences.txt").read().split()

"""
Functions
"""


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


# Load the data-set
def load_data(path):
    """
    :param path: data path
    :return: doc_clean
    """
    # get the path of the documents
    document_paths = [os.path.join(corpus_path, p) for p in os.listdir(path)]

    # read contents of all the articles in a list "doc_complete"
    doc_complete = []
    for path in document_paths:
        fp = codecs.open(path, 'r', 'utf-8')
        doc_content = fp.read()
        doc_complete.append(doc_content)

    # Get all documents
    docs = open("docs_wiki.pkl", 'wb')
    cPickle.dump(doc_complete, docs)

    # Create training set and test set
    end = int(0.85 * len(doc_complete))
    docs_train = doc_complete[:end]
    docs_test = doc_complete[end:]

    # Cleaning all the documents
    stop = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()
    doc_train = [clean_data(doc, stop, lemma) for doc in docs_train]
    doc_test = [clean_data(doc, stop, lemma) for doc in docs_test]

    return doc_train, doc_test


# Calculate average number of words per documents
def cal_words(corpus):
    """
    :param corpus: corpus
    :return: avg_words
    """
    num_words = 0
    num_docs = 0
    for i in range(len(corpus)):
        doc = corpus[i]
        num_words += len(doc)
        num_docs += 1
    avg_words = num_words / num_docs
    print("The average number of unique words per document is:", avg_words)


# Calculate number of documents per topics
def cal_topic_docs(corpus, topic_1, topic_2, topic_3, topic_4, topic_5, topic_6, topic_7, topic_8, topic_9, topic_10,
                   topic_11):
    nums_1, nums_2, nums_3, nums_4, nums_5, nums_6, nums_7, nums_8, nums_9, nums_10, nums_11 = [0]*11
    words_1, words_2, words_3, words_4, words_5, words_6, words_7, words_8, words_9, words_10, words_11 = [0]*11

    for i in range(len(corpus)):
        doc = corpus[i]
        if any(word in doc for word in topic_1):
            nums_1 += 1
            words_1 += len(doc)
        if any(word in doc for word in topic_2):
            nums_2 += 1
            words_2 += len(doc)
        if any(word in doc for word in topic_3):
            nums_3 += 1
            words_3 += len(doc)
        if any(word in doc for word in topic_4):
            nums_4 += 1
            words_4 += len(doc)
        if any(word in doc for word in topic_5):
            nums_5 += 1
            words_5 += len(doc)
        if any(word in doc for word in topic_6):
            nums_6 += 1
            words_6 += len(doc)
        if any(word in doc for word in topic_7):
            nums_7 += 1
            words_7 += len(doc)
        if any(word in doc for word in topic_8):
            nums_8 += 1
            words_8+= len(doc)
        if any(word in doc for word in topic_9):
            nums_9 += 1
            words_9 += len(doc)
        if any(word in doc for word in topic_10):
            nums_10 += 1
            words_10 += len(doc)
        if any(word in doc for word in topic_11):
            nums_11 += 1
            words_11 += len(doc)

    print("The number of documents for topic 1 is:", nums_1)
    print("The number of documents for topic 2 is:", nums_2)
    print("The number of documents for topic 3 is:", nums_3)
    print("The number of documents for topic 4 is:", nums_4)
    print("The number of documents for topic 5 is:", nums_5)
    print("The number of documents for topic 6 is:", nums_6)
    print("The number of documents for topic 7 is:", nums_7)
    print("The number of documents for topic 8 is:", nums_8)
    print("The number of documents for topic 9 is:", nums_9)
    print("The number of documents for topic 10 is:", nums_10)
    print("The number of documents for topic 11 is:", nums_11)

    print("The number of words within the topic 1 is:", words_1)
    print("The number of words within the topic 2 is:", words_2)
    print("The number of words within the topic 3 is:", words_3)
    print("The number of words within the topic 4 is:", words_4)
    print("The number of words within the topic 5 is:", words_5)
    print("The number of words within the topic 6 is:", words_6)
    print("The number of words within the topic 7 is:", words_7)
    print("The number of words within the topic 8 is:", words_8)
    print("The number of words within the topic 9 is:", words_9)
    print("The number of words within the topic 10 is:", words_10)
    print("The number of words within the topic 11 is:", words_11)



if __name__ == '__main__':
    # load the data-set
    corpus_path = "articles-corpus/"
    doc_train, doc_test = load_data(corpus_path)

    # Calculate average number of unique words per document
    cal_words(doc_train)
    cal_words(doc_test)

    # Calculate number of documents per topics
    cal_topic_docs(doc_train, topic_1, topic_2, topic_3, topic_4, topic_5, topic_6, topic_7, topic_8, topic_9, topic_10,
                   topic_11)
    cal_topic_docs(doc_test, topic_1, topic_2, topic_3, topic_4, topic_5, topic_6, topic_7, topic_8, topic_9, topic_10,
                   topic_11)
