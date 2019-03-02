# Toxic Comment Classification

This repository holds the files related to a research project. I experimented with deep learning on the toxic comment classification data set. <br/>
The data set can be found at https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge.


## GloVe: Global Vector Representation 

The GloVe word embedding files used can be found at https://nlp.stanford.edu/projects/glove/.
I specifically used http://nlp.stanford.edu/data/glove.840B.300d.zip. 

## Summary
I tested the following machine learning and deep learning algorithms on the data:
1. Logistic Regression (without embeddings) 
2. Long Short Term Neural Networks (LSTMs)
3. Gated Recurrent Units (GRUs)
4. Convolutional Neural Networks (CNNs)
5. LSTM + CNNs (Best performance)
6. GRU + CNNs 

# Dependencies
You should have Python 3 (preferably 3.6) installed.

### 1. Numpy
https://www.numpy.org/ <br/>
```bash
pip3 install numpy
```
### 2. Pandas
https://pandas.pydata.org <br/>
```bash
pip3 install pandas
```
### 3. Tensorflow
https://tensorflow.org <br/>
```bash
pip3 install tensorflow-gpu
```
### 4. Keras
https://keras.io <br/>
```bash
pip3 install keras
```
### 5. Jupyter Notebook
https://jupyter.org (Optional: If you want to run the ipython notebooks) <br/>
```bash
pip3 install jupyter
```
<br/>

# Running the scripts

After installing dependencies to run the scripts. If the data and the glove files are in the same folder as train.py
simply use:

```bash
python3 train.py
```

### Usage:

```bash
python3 train.py [-h] [--train-path TRAIN_PATH] [--test-path TEST_PATH]
                        [-m MODEL] [-g GLOVE_PATH] [--embed-size EMBED_SIZE]
                        [-b BATCH_SIZE] [--epochs EPOCHS] [--max_length MAX_LENGTH]
                        [--max_features MAX_FEATURES] [--save-weights SAVE_WEIGHTS]
```

```bash
  -h, --help            shows help message
  
  --train-path TRAIN_PATH
                        Path to training data
                        
  --test-path TEST_PATH
                        Path to testing data
                        
  -m MODEL, --model MODEL
                        Specify model to use
                        
  -g GLOVE_PATH, --glove-path GLOVE_PATH
                        Path to glove embeddings
                        
  --embed-size EMBED_SIZE
                        Embedding size
                        
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Specify batch size (default 32)
                        
  --epochs EPOCHS       
                        Specify number of epochs (default 1)
                        
  --max_length MAX_LENGTH
                        Maximum length of string used in training (default 200)
                        
  --max_features MAX_FEATURES
                        Maximum number of features to use in training (default 6000)
                        
  --save-weights SAVE_WEIGHTS
                        Save trained weights

```
