from keras.layers import Bidirectional, Conv1D, GlobalMaxPool1D, Dense, GlobalMaxPooling1D, concatenate, \
    GlobalAveragePooling1D, SpatialDropout1D, LSTM
from keras.layers import Embedding


def lstm(max_features, embed_size, embedding_matrix, inp):
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.15))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(6, activation="sigmoid")(x)
    return x


def cnn(max_features, embed_size, embedding_matrix, inp):
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Conv1D(250, kernel_size=3, padding="valid", activation='relu', kernel_initializer='glorot_uniform')(x)
    x = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
    x = Dense(6, activation="sigmoid")(x)
    return x


def lstm_cnn(max_features, embed_size, embedding_matrix, inp):
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.15))(x)
    x = Conv1D(250, kernel_size=3, padding="valid", activation='relu', kernel_initializer='glorot_uniform')(x)
    x = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
    x = Dense(6, activation="sigmoid")(x)
    return x
