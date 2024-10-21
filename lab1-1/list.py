import os
import json
import jieba
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np


def build_index(file_path='lab1-1/dataset/book_tag_without_stopwords.csv', index_file='lab1-1/dataset/index.json', num_clusters=10000):
    sentences = []
    file_word_map = {}

    # Step 1: Read CSV file and tokenize
    if os.path.exists(file_path):
        original_data = pd.read_csv(file_path)
        for book, booktags in zip(original_data['Book'], original_data['Tags']):
            booktags = booktags.strip("[]")
            booktags_list = [item.strip().strip(",")
                             for item in booktags.split(",")]
            words = booktags_list
            sentences.append(words)
            for word in words:
                if word not in file_word_map:
                    file_word_map[word] = set()
                file_word_map[word].add(book)
    else:
        print(f"File '{file_path}' does not exist.")
        return

    # Step 2: Train word2vec model
    model = Word2Vec(sentences, vector_size=100,
                     window=5, min_count=1, workers=4)

    # Step 3: Cluster word vectors using k-means
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(word_vectors)

    # Step 4: Build inverted index using cluster centers
    index = {}
    for i, word in enumerate(model.wv.index_to_key):
        cluster_id = kmeans.labels_[i]
        if cluster_id not in index:
            index[cluster_id] = set()
        index[cluster_id].update(file_word_map[word])

    # Save the index to a file
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump({str(k): list(v)
                  for k, v in index.items()}, f, ensure_ascii=False)


def search(word, index_file='index.json'):
    with open(index_file, 'r', encoding='utf-8') as f:
        index = json.load(f)

    # Load the word2vec model
    model = Word2Vec.load("word2vec.model")

    # Find the cluster of the word
    if word in model.wv:
        word_vector = model.wv[word].reshape(1, -1)
        cluster_id = KMeans.predict(word_vector)[0]
        return set(index.get(cluster_id, []))
    else:
        return set()


if __name__ == '__main__':
    build_index()
