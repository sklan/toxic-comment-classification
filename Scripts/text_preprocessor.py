import re

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


def replace_and_remove(train, test, column_name):
    """ Replaces words and removes special character from text in train, test dataset """
    for dataset in [train, test]:
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('\'ll', ' will'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('\'ve', ' have'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('don\'t', ' do not'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('dont', ' do not'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('aren\'t', ' are not'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('won\'t', ' will not'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('wont', ' will not'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('can\'t', ' cannot'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('cant', ' cannot'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('shan\'t', ' shall not'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('shant', ' shall not'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace('\'m', ' am'))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("doesn't", "does not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("doesnt", "does not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("didn't", "did not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("didnt", "did not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("hasn't", "has not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("hasnt", "has not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("haven't", "have not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("havent", "have not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("wouldn't", "would not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("didn't", "did not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("didnt", "did not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("it's", "it is"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("that's", "that is"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("weren't", "were not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace("werent", "were not"))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace(' u ', ' you '))
        dataset[column_name] = dataset[column_name].apply(lambda x: x.replace(' U ', ' you '))
        dataset[column_name] = dataset[column_name].apply(
        lambda x: re.sub('[()\"\t_\n.,:=!@#$%^&*-/[\]?|1234567890—]', ' ', x).strip())


def embedded_glove_matrix(train, test, column_name, glove_path, embed_size, max_features, max_length):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(train[column_name])
    X = pad_sequences(tokenizer.texts_to_sequences(train[column_name]), maxlen=max_length)
    X_ = pad_sequences(tokenizer.texts_to_sequences(test[column_name]), maxlen=max_length)
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    word_index = tokenizer.word_index
    num_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, X, X_


def tf_idf(train, test, column_name, max_df, smooth_idf, max_features, analyzer):
    vectorizer = TfidfVectorizer(max_df=max_df, smooth_idf=smooth_idf, max_features=max_features, analyzer=analyzer)
    X = vectorizer.fit_transform(train[column_name])
    X_ = vectorizer.transform(test[column_name])
    return X, X_
