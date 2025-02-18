"""
Creating Corpus from Simple Wikipedia dump
"""

# Import packages
import xml.etree.ElementTree as ET
import codecs
import re


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


# Define the path
tree = ET.parse('data/simplewiki-20170201-pages-articles-multistream.xml')
root = tree.getroot()
dir_path = 'articles-corpus//'

# Create topic list
topics = ["Art", "culture", "Classical studies ", "Cooking", "Critical theory", "Hobbies", "Literature",
          "Entertainment", "Fiction", "Game", "Poetry", "Sports", "Dance", "Film", "Movie", "Music",
          "Opera ", "Theatre", "Architecture", "Crafts", "Drawing", "Painting", "Photography", "Sculpture",
          "Typography", "Geography", "places", "Health", "fitness", "Exercise", "Health science", "Nutrition",
          "History", "events", "Classical antiquity", "Medieval history (Middle Ages)", "Renaissance", "Mathematics",
          "abstractions", "Arithmetic", "Algebra", "Calculus", "Discrete mathematics", "Geometry", "Trigonometry",
          "Logic", "Statistics", "Natural sciences", "nature", "Animals", "Biochemistry", "Botany", "Ecology",
          "Zoology", "Astronomy", "Chemistry", "Earth sciences", "Physics Fractions", "People", "self", "Biology",
          "Psychology", "Relationships", "Philosophy", "thinking", "Philosophical theories", "Humanism", "Logic",
          "Thinking", "Transhumanism", "Religion", "spirituality", "Social sciences", "society", "Technology",
          "applied sciences"]

# Count words of corpus
word_count = 0
docs_num = 0

# Get each document within the corpus
for i, page in enumerate(root.findall('{http://www.mediawiki.org/xml/export-0.10/}page')):
    for p in page:
        if p.tag == "{http://www.mediawiki.org/xml/export-0.10/}revision":
            for x in p:
                if x.tag == "{http://www.mediawiki.org/xml/export-0.10/}text":
                    article_txt = x.text
                    if article_txt and any(word in article_txt for word in topics):
                        article_txt = article_txt[: article_txt.find("==")]
                        article_txt = re.sub(r"{{.*}}", "", article_txt)
                        article_txt = re.sub(r"\[\[File:.*\]\]", "", article_txt)
                        article_txt = re.sub(r"\[\[Image:.*\]\]", "", article_txt)
                        article_txt = re.sub(r"\n: \'\'.*", "", article_txt)
                        article_txt = re.sub(r"\n!.*", "", article_txt)
                        article_txt = re.sub(r"^:\'\'.*", "", article_txt)
                        article_txt = re.sub(r"&nbsp", "", article_txt)
                        article_txt = re.sub(r"http\S+", "", article_txt)
                        article_txt = re.sub(r"\d+", "", article_txt)
                        article_txt = re.sub(r"\(.*\)", "", article_txt)
                        article_txt = re.sub(r"Category:.*", "", article_txt)
                        article_txt = re.sub(r"\| .*", "", article_txt)
                        article_txt = re.sub(r"\n\|.*", "", article_txt)
                        article_txt = re.sub(r"\n \|.*", "", article_txt)
                        article_txt = re.sub(r".* \|\n", "", article_txt)
                        article_txt = re.sub(r".*\|\n", "", article_txt)
                        article_txt = re.sub(r"{{Infobox.*", "", article_txt)
                        article_txt = re.sub(r"{{infobox.*", "", article_txt)
                        article_txt = re.sub(r"{{taxobox.*", "", article_txt)
                        article_txt = re.sub(r"{{Taxobox.*", "", article_txt)
                        article_txt = re.sub(r"{{ Infobox.*", "", article_txt)
                        article_txt = re.sub(r"{{ infobox.*", "", article_txt)
                        article_txt = re.sub(r"{{ taxobox.*", "", article_txt)
                        article_txt = re.sub(r"{{ Taxobox.*", "", article_txt)
                        article_txt = re.sub(r"\* .*", "", article_txt)
                        article_txt = re.sub(r"<.*>", "", article_txt)
                        article_txt = re.sub(r"\n", "", article_txt)
                        article_txt = re.sub(r"[0-9]","", article_txt)
                        article_txt = re.sub(
                            r"\!|\"|\#|\$|\%|\&|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\?|\@|\[|\\|\]|\^|\_|\`|\{|\||\}|\~",
                            " ", article_txt)
                        article_txt = re.sub(r" +", " ", article_txt)
                        article_txt = article_txt.replace(u'\xa0', u' ')

                        if not article_txt == None and not article_txt == "" and len(article_txt) > 150 and is_ascii(
                                article_txt):
                            outfile = dir_path + str(i + 1) + "_article.txt"
                            f = codecs.open(outfile, "w", "utf-8")
                            f.write(article_txt)
                            f.close()
                            print(article_txt)
                            print('\n=================================================================\n')