import os
import pickle
import codecs

# Load data
def load_data(path):
    # get the path of the documents
    document_paths = [os.path.join(corpus_path, p) for p in os.listdir(path)]

    # read contents of all the articles in a list "doc_completre"
    doc_complete = []
    for path in doc_paths:
        fp = codecs.open(path, 'r', 'utf-8')
        doc_content = fp.read()
        doc_complete.append(doc_content)

    # get all documents
    docs = open("docs.pkl", 'wb')
    pickle.dump(doc_complete, docs)

    return doc_complete


# Separate docs
def separate_docs(docs):
    dir_path = "separate_docs/"
    i = 0
    for doc in docs:
        outfile = dir_path + str(i + 8000) + "_doc.txt"
        f = codecs.open(outfile, 'w', 'utf-8')
        j = int(j)
        paragraph = doc[j:j+8000]
        f.write(paragraph)
        i += 1
    print("All documents have been separated.")


# Main
if __name__ == "__main__":
    # load the dataset
    corpus_path = "processed_data/"
    docs = load_data(corpus_path)
    speparate_docs(docs)