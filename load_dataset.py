import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import keras as ks

def load_dataset():
    print("Loading dataset...")
    data = pd.read_csv("dataset/lyrics.csv")   
    data = data[data["genre"] != "Not Available"]
    data = data[data["genre"] != "Other"]
    print("Done.")

    print("Extracting relevant data...")
    data["lyrics"] = data["lyrics"].astype(str)
    data["genre"] = data["genre"].astype(str)
    
    print("Removing invalid characters...")
    data["lyrics"] = data["lyrics"].replace(r"\n"," ", regex=True)
    data["lyrics"] = data["lyrics"].replace(r"\s*\[[\w: ]*\]\s*"," ", regex=True)
    data["lyrics"] = data["lyrics"].replace(r"\'s*","", regex=True)
    print("Done.")
    return data


def preprocess_dataset(dataset, seq_length, max_seq_count):
    dataset = dataset.sample(frac=1)
    lyrics = dataset["lyrics"].tolist()
    labels = dataset["genre"].tolist()
    print("Tokenize dataset...")
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(lyrics)
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(lyrics)
    vocab_size = len(t.word_index) + 1
    print("Done.")
    print("Loaded {} documents.".format(len(encoded_docs)))
    label_classes = list(set(labels))
    label_classes_to_index = {label_classes[i]:i for i in range(len(label_classes)) }

    print("Convert docs to sequences of length {}...".format(seq_length))
    sequenced_docs_input = []
    sequenced_docs_label = []
    skip_cnt=0
    for doc_idx in range(len(encoded_docs)):
        doc = encoded_docs[doc_idx]
        label = labels[doc_idx]
        if len(doc) < seq_length:
            skip_cnt+=1
        for i in range(0, len(doc) - seq_length, 5):
            sequenced_docs_input.append(doc[i:i+seq_length])
            sequenced_docs_label.append(label_classes_to_index[label])
            
            if len(sequenced_docs_input) == max_seq_count:
                break
                
        if len(sequenced_docs_input) == max_seq_count:
            break
    
    sequenced_docs_label = np.asarray(sequenced_docs_label)
    sequenced_docs_input = np.asarray(sequenced_docs_input)
    idx=int(0.9*sequenced_docs_label.shape[0])
    
    sequenced_docs_label_train = sequenced_docs_label[:idx]
    sequenced_docs_input_train = sequenced_docs_input[:idx]
    sequenced_docs_label_test = sequenced_docs_label[idx:]
    sequenced_docs_input_test = sequenced_docs_input[idx:]
    
    # Randomize ordering of samples again and split into train/test dataset
    perm_train = np.random.permutation(sequenced_docs_label_train.shape[0])
    perm_test = np.random.permutation(sequenced_docs_label_test.shape[0])
    
    sequenced_docs_label_train = sequenced_docs_label_train[perm_train]
    sequenced_docs_input_train = sequenced_docs_input_train[perm_train]
    sequenced_docs_label_test = sequenced_docs_label_test[perm_test]
    sequenced_docs_input_test = sequenced_docs_input_test[perm_test]
    
    sequenced_docs_label_train = ks.utils.to_categorical(sequenced_docs_label_train, num_classes=len(label_classes))
    sequenced_docs_label_test = ks.utils.to_categorical(sequenced_docs_label_test, num_classes=len(label_classes))
    
    print("Done.")
    print("Generated {} sequences from {} documents.".format(len(sequenced_docs_label), doc_idx+1-skip_cnt))
    print("Skipped {} docs.".format(skip_cnt))

    return (t, sequenced_docs_input_train, sequenced_docs_label_train, sequenced_docs_input_test, sequenced_docs_label_test, label_classes_to_index)


def load_embeddings(embeddings_weights_file="glove.6B.100d.txt"):
    print("Load word embeddings...")
    words = {}
    with open(embeddings_weights_file, encoding="utf8")  as f:
        for line in f:
            vals = line.split(" ")
            w = vals[0]
            coefs = np.asarray(vals[1:], dtype="float32")
            words[w] = coefs

    print("Done.")
    return (words, 100)


def glove_to_matrix(words, tokenizer):
    print("Converting embeddings to weight matrix...")
    vocab_size = len(tokenizer.word_index) + 1
    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, 100))
    idx_to_word_map = {}
    skipped_words=[]
    for word, i in tokenizer.word_index.items():
        embedding_vector = get_vect(words, word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            idx_to_word_map[i] = word
        else:
            skipped_words.append(word)

    print("Skipped {} unknown words.".format(len(skipped_words)))
    print(skipped_words)
    print("Done.")
    return embedding_matrix, idx_to_word_map


def get_vect(words, word):
    if word in words:
        return words[word]
    else:
        return None

def find_closest_word_idx(embedding_matrix, v):
    diff = embedding_matrix - v
    delta = np.sum(diff * diff, axis=1)
    i = np.argmin(delta)
    return i

def idx_vec_to_string(idx_to_word_map, idx_vec):
    return " ".join([idx_to_word_map[idx] if idx in idx_to_word_map else "" for idx in idx_vec])
    
def label_vec_to_embeddings_vec(vec_of_indices, embeddings_matrix):
    res = []
    for idx in vec_of_indices:
        v = embeddings_matrix[idx]
        res.append(v)

    return res
