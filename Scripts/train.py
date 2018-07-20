import argparse

import pandas as pd
from keras.layers import Input
from keras.models import Model

import models
import text_preprocessor as tp

parser = argparse.ArgumentParser()
parser.add_argument("--train-path", type=str, help='Path to training data', default='train.csv')
parser.add_argument("--test-path", type=str, help='Path to testing data', default='test.csv')
parser.add_argument("-m", "--model", type=str, help='Specify model to use', default='lstm-cnn')
parser.add_argument("-g", "--glove-path", type=str, help='Path to glove embeddings', default='glove.840B.300d.txt')
parser.add_argument("--embed-size", type=int, help='Embedding size', default=300)
parser.add_argument("-b","--batch_size", type=int, help='Specify batch size  (default 32)', default=32)
parser.add_argument("--epochs", type=int, help='Specify number of epochs (default 1)', default=1)
parser.add_argument("--max_length", type=int, help='Maximum length of string used in training  (default 200)',
                    default=200)
parser.add_argument("--max_features", type=int, help='Maximum number of features to use in training  (default 5000)',
                    default=5000)
parser.add_argument("--save-weights", type=bool, help='Save trained weights', default=False)
args = parser.parse_args()

if __name__ == '__main__':
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    glove_path = args.glove_path
    model_type = args.model
    max_length = args.max_length
    max_features = args.max_features
    embed_size = args.embed_size
    batch_size = args.batch_size
    epochs = args.epochs
    save_weights = args.save_weights
    tp.replace_and_remove(train, test, 'comment_text')

    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    Y = train[target_columns]

    embedding_matrix, X, X_ = tp.embedded_glove_matrix(train, test, 'comment_text', glove_path, embed_size,
                                                       max_features,
                                                       max_length)
    inp = Input(shape=(max_length,))
    x = None
    if model_type == 'lstm-cnn':
        x = models.lstm_cnn(max_features, embed_size, embedding_matrix, inp)
    elif model_type == 'lstm':
        x = models.lstm(max_features, embed_size, embedding_matrix, inp)
    elif model_type == 'cnn':
        x = models.cnn(max_features, embed_size, embedding_matrix, inp)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, Y, batch_size=batch_size, epochs=epochs)
    if save_weights:
        model.save_weights('checkpoint.csv')

    sub = model.predict(X_)

    temp = test.copy()
    temp[target_columns] = model.predict(X_)
    temp.to_csv('output.csv', index=False)
