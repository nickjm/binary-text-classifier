import os
import gzip
import numpy as np
import project.data.binary_class_dataset_embeddings as emb_dataset
import project.data.binary_class_dataset_bow as bow_dataset
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

NIL_EMBEDDING_ID = 0
PAD_EMBEDDING_ID = 1


def get_embedding_tensor(embedding_path):
    lines = []
    with gzip.open(embedding_path) as file:
        lines = file.readlines()
        file.close()
    embedding_tensor = []
    word_to_indx = {}
    print("Creating embedding tensor...")
    with tqdm(total=len(lines)) as pbar:
        for indx, l in enumerate(lines):
            word, emb = l.split()[0], l.split()[1:]
            vector = [float(x) for x in emb ]
            if indx == 0:
                # Append nil embedding vector
                embedding_tensor.append(np.zeros(len(vector)))
                # Append padding embedding vector
                embedding_tensor.append(np.zeros(len(vector)))
            embedding_tensor.append(vector)
            word_to_indx[word] = indx+2 # for 2 vectors at beginning, nil and pad
            pbar.update()
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word_to_indx


def extract_clamped_text(line, args):
    return " ".join(line.split('\t')[1].split()[:args.max_seq_length])


def get_bow_vectorizer(train_path, dev_path, args):
    with open(train_path) as train:
        with open(dev_path) as dev:
            corpus = []
            print("Creating corpus for BOW generation...")
            lines = train.readlines() + dev.readlines()
            for line in tqdm(lines):
                corpus.append(extract_clamped_text(line, args))
            if args.tfidf:
                vectorizer = TfidfVectorizer()
            else:
                vectorizer = CountVectorizer()
            vectorizer.fit(corpus)
    return vectorizer, corpus


# Depending on args, build dataset
def load_dataset(args):
    print("\nLoading data...")
    train_path = args.data_path + ".train"
    dev_path = args.data_path + ".dev"
    if args.bow:
        vectorizer, corpus = get_bow_vectorizer(train_path, dev_path, args)
        train_data = bow_dataset.BinaryClassTextBOWDataset(train_path, vectorizer, args.max_seq_length)
        dev_data = bow_dataset.BinaryClassTextBOWDataset(dev_path, vectorizer, args.max_seq_length)
        args.vocab_size = len(vectorizer.vocabulary_.keys())
        return train_data, dev_data, None
    else:
        embeddings_path = args.word_embeddings
        embeddings, word_to_indx = get_embedding_tensor(embeddings_path)
        args.embedding_dim = embeddings.shape[1]
        train_data = emb_dataset.BinaryClassTextEmbeddingsDataset(train_path, word_to_indx, nil_id=NIL_EMBEDDING_ID, pad_id=PAD_EMBEDDING_ID)
        dev_data = emb_dataset.BinaryClassTextEmbeddingsDataset(dev_path, word_to_indx, nil_id=NIL_EMBEDDING_ID, pad_id=PAD_EMBEDDING_ID)
        return train_data, dev_data, embeddings
