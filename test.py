from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


from read_data import read_data

if __name__ == "__main__":
    data = read_data("mentions.csv")
    docs = data['title'] + "\n" + data['description']

    vectorizer = TfidfVectorizer("content", analyzer="word", ngram_range=(1, 2), stop_words="english")
    vectorizer.fit(docs)

    vocab = np.array(vectorizer.get_feature_names())

    for d in np.random.choice(docs, 5):
        dtm = vectorizer.transform([d])
        dtm = dtm.todense()

        highest = (-dtm).argsort(1)
        highest = np.asarray(highest)[0]

        print(d)
        print(vocab[highest[0:10]])
        print("\n=====\n")
