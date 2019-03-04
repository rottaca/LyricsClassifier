from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.metrics import categorical_accuracy


def get_network(input_length, output_shape, embeddings_matrix, tokenizer, embeddings_vec_size):
    
    print("Building network...")
    vocab_size = len(tokenizer.word_index) + 1

    # define the LSTM model
    model = Sequential()
    model.add(Embedding(vocab_size, embeddings_vec_size, input_length=input_length, trainable=False))
#     model.add(LSTM(10, dropout=0.2, return_sequences=True))
#     model.add(LSTM(10, dropout=0.2, return_sequences=True))
#     model.add(LSTM(10, dropout=0.2, return_sequences=True))
    model.add(LSTM(20, dropout=0.2))
    model.add(Dense(output_shape, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])

    print("Done.")
    return model