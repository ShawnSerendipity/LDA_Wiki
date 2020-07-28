import os
import pickle
import codecs
import nltk
nltk.download('punkt')


# Load data
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
    docs = open("docs.pkl", 'wb')
    pickle.dump(doc_complete, docs)

    return doc_complete


# Separate docs
def separate_docs(docs):
    dir_path = "separated_new/"

    count = 0
    num_files = 0
    for doc in docs:
        doc = nltk.tokenize.sent_tokenize(doc)
        end = int(len(doc)) - 4
        for i in range(end):
            sub_doc = doc[4 * i:4 * i + 4]
            for j in range(len(sub_doc)):
                if sub_doc[j]:
                    outfile = dir_path + str(num_files + 1) + "_doc.txt"
                    f = codecs.open(outfile, 'w', 'utf-8')
                    f.write(sub_doc[j] + ' ')
                    num_files += 1
        count += 1
        if count > 3:
            break

    print("All documents has been separated.")


# Main
if __name__ == "__main__":
    # load the data-set
    corpus_path = "10k_training/"
    docs = load_data(corpus_path)
    separate_docs(docs)