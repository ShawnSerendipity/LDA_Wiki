"""
Getting Ground-Truth Topics from Simple Wikipedia dump
"""

# Import package
import xml.etree.ElementTree as ET


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


# Define the path
tree = ET.parse('data/simplewiki-20170201-pages-articles-multistream.xml')
root = tree.getroot()
dir_path = 'articles-corpus//'

# Fetch the topics
topics = []

for i, page in enumerate(root.findall('{http://www.mediawiki.org/xml/export-0.10/}page')):
    for p in page:
        r_tag = "{http://www.mediawiki.org/xml/export-0.10/}revision"
        if p.tag == r_tag:
            for x in p:
                tag = "{http://www.mediawiki.org/xml/export-0.10/}text"
                if x.tag == tag:
                    text = x.text
                    if not text == None:
                        topic = text[text.find("Category:"):text.find("]]")]
                        if topic != "":
                            topic = topic[topic.find(":")+1:]
                            topics.append(topic)

# Get rid of repeated topics
unique_topics = []

for topic in topics:
    if not topic in unique_topics:
        unique_topics.append(topic)
        print(topic)