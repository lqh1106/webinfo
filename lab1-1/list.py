import os
import json
import jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np

def build_index(data_dir='data', index_file='index.json', num_clusters=10):
    sentences = []
    file_word_map = {}
    
    # Step 1: Read files and tokenize
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    words = jieba.lcut(content)
                    sentences.append(words)
                    for word in words:
                        if word not in file_word_map:
                            file_word_map[word] = set()
                        file_word_map[word].add(file)
    else:
        print(f"Directory '{data_dir}' does not exist.")
        return

    # Step 2: Train word2vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
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
        json.dump({k: list(v) for k, v in index.items()}, f, ensure_ascii=False)

def search(word, index_file='index.json'):
    with open(index_file, 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    # Load the word2vec model
    model = Word2Vec.load("word2vec.model")
    
    # Find the cluster of the word
    if word in model.wv:
        word_vector = model.wv[word].reshape(1, -1)
        cluster_id = kmeans.predict(word_vector)[0]
        return set(index.get(cluster_id, []))
    else:
        return set()

if __name__ == '__main__':
    build_index()