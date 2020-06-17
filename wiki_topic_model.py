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

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


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

    # Create training set
    end = int(0.85 * len(doc_complete))
    docs_train = doc_complete[:end]

    # Cleaning all the documents
    stop = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()
    doc_clean = [clean_data(doc, stop, lemma) for doc in docs_train]

    return doc_clean


def create_dictionary(doc_clean):
    """
    :param doc_clean: cleaned documents
    :return: dictionary
    """
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

    return dictionary


def create_bow(doc_clean, dictionary):
    """
    :param doc_clean: cleaned documents
    :param dictionary: dictionary
    :return: doc_term_matrix
    """
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

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
    lda_model = Lda(bow, num_topics=num_topics, id2word=dictionary, passes=10, iterations=100)
    lda_file = open('lda_model_sym_wiki.pkl', 'wb')
    cPickle.dump(lda_model, lda_file)
    lda_file.close()

    return lda_model


# Get topics
def get_topics(lda_model):
    """
    :param lda_model: trained LDA model
    :return: topics
    """
    # Print all the topics
    topics = []
    for idx, topic in lda_model.print_topics(num_topics=11, num_words=10):
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
    # Get all words in generated 20 topics
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
    corpus_path = "articles-corpus/"
    doc_clean = load_data(corpus_path)

    # create dictionary
    dictionary = create_dictionary(doc_clean)

    # create bow representation of documents
    bow = create_bow(doc_clean, dictionary)

    # build and train LDA model
    lda_model = train_lda(bow, dictionary, 11)

    # get and print out generated topics
    topics = get_topics(lda_model)

    # get Topic Coherence(TC)
    topic_coherence = topic_coherence(corpus=doc_clean, topics=topics)
    print("The Topic Coherence(TC) of LDA model is:", topic_coherence)

    # get Topic Diversity(TD)
    topic_diversity = topic_diversity(topics)
    print("The Topic Diversity(TD) of LDA model is: ", topic_diversity)

    # query on new document
    new_corpus = [
    ['ubvmsd', 'buffalo', 'neil', 'gandler', 'subject', 'need', 'info', 'bonnevill', 'organ', 'univers', 'buffalo', 'line', 'news', 'softwar', 'vnew', 'nntp', 'post', 'host', 'ubvmsd', 'buffalo', 'littl', 'confus', 'model', 'bonnevill', 'hear', 'ssei', 'tell', 'differ', 'featur', 'perform', 'curious', 'know', 'book', 'valu', 'prefer', 'model', 'book', 'valu', 'usual', 'word', 'demand', 'time', 'year', 'hear', 'spring', 'earli', 'summer', 'best', 'time', 'neil', 'gandler'],
    ['rick', 'miller', 'rick', 'subject', 'face', 'organ', 'line', 'distribut', 'world', 'nntp', 'post', 'host', 'summari', 'ahead', 'swamp', 'familiar', 'format', 'face', 'thingi', 'see', 'folk', 'header', 'mayb', 'view', 'linux', 'display', 'uncompress', 'face', 'manag', 'compil', 'compfac', 'look', 'face', 'anyon', 'news', 'header', 'send', 'face', 'header', 'know', 'probabl', 'littl', 'swamp', 'handl', 'hope', 'rick', 'miller', 'rick', 'ricxjo', 'discus', 'ricxjo', 'muelisto', 'send', 'postcard', 'enposxtigu', 'bildkarton', 'ricevo', 'alion', 'rick', 'miller', 'wood', 'muskego'],
    ['mathew', 'mathew', 'manti', 'subject', 'strong', 'weak', 'atheism', 'organ', 'manti', 'consult', 'cambridg', 'newsread', 'rusnew', 'line', 'acoop', 'macalstr', 'turin', 'turambar', 'depart', 'utter', 'miseri', 'write', 'modifi', 'defin', 'strong', 'atheist', 'assert', 'nonexist', 'assert', 'believ', 'nonexist', 'word', 'mathew'],
    ['bakken', 'arizona', 'dave', 'bakken', 'subject', 'saudi', 'clergi', 'condemn', 'debut', 'human', 'right', 'group', 'keyword', 'intern', 'govern', 'govern', 'civil', 'right', 'social', 'issu', 'polit', 'organ', 'arizona', 'dept', 'tucson', 'line', 'articl', 'benali', 'alcor', 'benali', 'alcor', 'concordia', 'ilyess', 'bdira', 'write', 'look', 'like', 'mind', 'heart', 'blind', 'eye', 'respect', 'today', 'lose', 'minim', 'respect', 'struggl', 'muslim', 'netter', 'give', 'fatwah', 'saudi', 'arabia', 'unit', 'ststes', 'attack', 'iraq', 'attack', 'iraqi', 'drive', 'kuwait', 'countri', 'citizen', 'close', 'blood', 'busi', 'tie', 'saudi', 'citizen', 'think', 'help', 'iraqi', 'swallow', 'saudi', 'arabia', 'eastern', 'oilfield', 'muslim', 'countri', 'help', 'liber', 'kuwait', 'protect', 'saudi', 'arabia', 'mass', 'citizen', 'demonstr', 'favor', 'butcher', 'saddam', 'kill', 'lotsa', 'muslim', 'kill', 'rap', 'loot', 'relat', 'rich', 'muslim', 'thumb', 'nose', 'west', 'defend', 'saudi', 'arabia', 'roll', 'iraqi', 'invas', 'charg', 'saudi', 'arabia', 'fatwah', 'legitim', 'kind', 'clergi', 'islam', 'duti', 'separ', 'religion', 'polit', 'religion', 'mean', 'offici', 'clergi', 'think', 'good', 'idea', 'govern', 'offici', 'religion', 'facto', 'jure', 'human', 'natur', 'like', 'ambiti', 'pious', 'one', 'rise', 'power', 'peopl', 'world', 'countri', 'citizen', 'know', 'leader', 'devout', 'slick', 'oper', 'cairo', 'egypt', 'cairo', 'base', 'arab', 'organ', 'human', 'right', 'aohr', 'thursday', 'welcom', 'establish', 'week', 'committe', 'defens', 'legal', 'right', 'saudi', 'arabia', 'say', 'necessari', 'group', 'oper', 'arab', 'countri', 'sound', 'like', 'guy', 'angel', 'ilyess', 'clarinet', 'post', 'edit', 'stuff', 'follow', 'friday', 'york', 'time', 'report', 'group', 'definit', 'conserv', 'sheikh', 'follow', 'think', 'hous', 'saud', 'rule', 'countri', 'conserv', 'report', 'complain', 'govern', 'conserv', 'assert', 'approx', 'shiit', 'kingdom', 'apost', 'charg', 'saudi', 'islam', 'bring', 'death', 'penalti', 'diplomat', 'sheikh', 'jibrin', 'ilyess', 'call', 'sever', 'punish', 'women', 'drive', 'public', 'protest', 'women', 'drive', 'group', 'say', 'abdelhamoud', 'toweijri', 'say', 'women', 'fire', 'job', 'jail', 'brand', 'prostitut', 'want', 'happen', 'ilyess', 'hear', 'muslim', 'women', 'drive', 'basi', 'ahadith', 'folk', 'like', 'want', 'women', 'fals', 'call', 'prostitut', 'choos', 'hero', 'wise', 'ilyess', 'reflex', 'ralli', 'hat', 'hate', 'women', 'allow', 'work', 'radio', 'immor', 'kingdom', 'hous', 'saud', 'favorit', 'govern', 'earth', 'think', 'restrict', 'religi', 'polit', 'reedom', 'thing', 'think', 'like', 'replac', 'go', 'wors', 'citizen', 'countri', 'think', 'hous', 'saud', 'feel', 'heat', 'late', 'month', 'read', 'step', 'harass', 'muttawain', 'religi', 'polic', 'govern', 'western', 'women', 'fulli', 'veil', 'stupid', 'women', 'send', 'wrong', 'signal', 'moral', 'read', 'crack', 'home', 'base', 'exparti', 'religi', 'gather', 'post', 'reward', 'govern', 'own', 'newspap', 'offer', 'money', 'turn', 'group', 'exparti', 'dare', 'worship', 'home', 'secret', 'place', 'govern', 'grow', 'intoler', 'wind', 'sail', 'conserv', 'opposit', 'unislam', 'thing', 'small', 'tast', 'happen', 'guy', 'overthrow', 'hous', 'saud', 'like', 'tri', 'long', 'rach', 'general', 'west', 'evil', 'zionist', 'rule', 'hate', 'west', 'puppet', 'crowd', 'want', 'ilyess', 'dave', 'bakken', 'presid', 'fine', 'problem', 'know', 'husband', 'jam', 'carvill', 'clinton', 'campaign', 'strategist', 'daddi', 'busi', 'chelsea', 'nurs', 'cspan'],
    ['livesey', 'solntz', 'livesey', 'subject', 'year', 'christian', 'moral', 'organ', 'line', 'distribut', 'world', 'nntp', 'post', 'host', 'solntz', 'articl', 'andrew', 'andrew', 'norman', 'paterson', 'write', 'articl', 'fido', 'livesey', 'solntz', 'livesey', 'write', 'articl', 'andrew', 'andrew', 'norman', 'paterson', 'write', 'claim', 'moral', 'absolut', 'list', 'refer', 'stretch', 'alpha', 'centauri', 'delet', 'think', 'impress', 'refer', 'claim', 'absolut', 'moral', 'claim', 'object', 'assum', 'answer', 'apolog', 'spend', 'solid', 'month', 'argu', 'thing', 'object', 'moral', 'exist']
]
    vector = query(query_text=new_corpus,
                   dictionary=dictionary,
                   trained_model=lda_model,
                   index=1)
    print("Generated vectors are: \n", vector)
