import load_dataset
import network
import sys
import numpy as np
import keras as ks
from keras.utils import np_utils
import time

lyrics = load_dataset.load_dataset()

(tokenizer, data_input, data_labels, max_input_length, label_classes_to_index) = load_dataset.preprocess_dataset(lyrics)
index_to_label_class = {v: k for k, v in label_classes_to_index.items()}

(embeddings_words, embeddings_vec_size) = load_dataset.load_embeddings()
(embeddings_matrix, idx_to_word_map) = load_dataset.glove_to_matrix(embeddings_words, tokenizer)

vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:" , vocab_size)
print("Inputs Shape:" , data_input.shape)
print("Labels Shape:" , data_labels.shape)
print("Classes are:")
print(label_classes_to_index.keys())

model = network.get_network(max_input_length, data_labels.shape[1], embeddings_matrix, tokenizer, embeddings_vec_size)
model.summary()

name = "100_cells"

callbacks = [
    ks.callbacks.TensorBoard("./logs/{}".format(name), write_graph=True, write_grads=False, write_images=False ),
    ks.callbacks.ModelCheckpoint("./models/{}".format(name + ".dat"),save_best_only=True)
]

model.fit(data_input, data_labels, epochs=30, batch_size=2048, validation_split=0.1, callbacks=callbacks)

# generate characters
for i in range(10):
    pattern_idx = np.random.randint(0, data_input.shape[0]-1)
    pattern = data_input[pattern_idx,:]
    print("Seed:")
    print(load_dataset.idx_vec_to_string(idx_to_word_map, pattern))

    x = np.reshape(pattern, (1, pattern.shape[0]))

    prediction = model.predict(x, verbose=0)
    idx = np.argmax(np.squeeze(prediction))
    print("Predicted class: {}".format(index_to_label_class[idx]))
    print("Actual class: {}".format(index_to_label_class[np.argmax(data_labels[pattern_idx])]))
	
