""" train.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This script is to train a text generation model based on the comments in the NYT dataset.
    Concretely, it trains are recurrent neural network (RNN) with a long short-term memory (LSTM) layer that utilizes pre-trained word embeddings from GloVe.
    The model, and all information related to it, is saved in the output directory.

Usage:
    $ python src/train.py -c 5000 -e 75 -d 50 -m 'example_model'
"""

# load packages
import numpy as np
import argparse
from pathlib import Path
import joblib
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, save_model
from utils import *

# attempt to surpress tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def arg_parse():
    '''
    Parse command line arguments.

    It is possible to specify:
    -   Number of comments to be used for training the model.
    -   Number of epochs to train the model.
    -   Dimension of the glove-based embedding layer.
    -   Name of the model to be saved.

    Returns:
    -   args (argparse.Namespace): Parsed arguments.
    '''
    
    # define parser
    parser = argparse.ArgumentParser(description='Train a model to generate text.')

    # add arguments
    parser.add_argument('-c', '--n_comments', default='all', help='Number of comments to be used for training the model.')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('-d', '--embedding_dim', type=int, default=50, choices=[50, 100, 200, 300] ,help='Dimension of the glove-based embedding layer.')
    parser.add_argument('-m', '--model_name', default='example_model', help='Name of the model to be saved.')

    # parse arguments
    args = parser.parse_args()

    # make a print statement based on the arguments to inform the user about their run
    print("\n" + "Initializing training for model based on {} comments from the NYT dataset using {}-dimensional word embeddings. It runs for {} epochs.".format(args.n_comments, args.embedding_dim, args.epochs) + "\n")


    return args

def input_checker(args):
    '''
    Check if input arguments are valid.

    Args:
    -   args (argparse.Namespace): Parsed arguments.
    '''
    
    # check if n_comments is an integer
    if args.n_comments != "all":
        try:
            args.n_comments = int(args.n_comments)
        
        except ValueError:
            print("n_comments must be an integer or 'all'.")
            exit()
    
    # check if embedding_dim is valid
    if args.embedding_dim not in [50, 100, 200, 300]:
        print("embedding_dim must be 50, 100, 200 or 300.")
        exit()


def load_data(inpath):
    '''
    Load data from all csv files in the input directory that have the word "comment" in the filename.
    This allows loading only the files for the comments rather than headlines in the NYT dataset.

    Args:
    -   inpath (pathlib.Path): Path to the input directory.

    Returns:
    -   corpus (list): List of all comments.

    '''
    print("Loading data...")

    # define data path
    data_path = inpath / "news_data"

    # get all csv files in the input directory
    files = [f for f in os.listdir(data_path)]

    # load data from all csv files
    corpus = []

    # loop with progress bar
    for f in tqdm(files):
        if "Comments" in f:

            # load data but fix encoding errors
            comments_df = pd.read_csv(data_path / f, dtype = str)

            # append all comments to a list
            corpus.extend(list(comments_df["commentBody"].values))
        
    return corpus


def sample_comments(corpus, n_comments):
    '''
    Samples n_comments from all_comments.
    It is used to limit the number of comments used for training the model, as it can be too computationally expensive to use all comments.

    Args:
    -   corpus (list): List of all comments.
    -   n_comments (int): Number of comments to be used for training the model (specified in argparse)

    Returns:
    -   corpus (list): List of sampled comments.
    '''

    if n_comments != "all":
        corpus = random.sample(corpus, n_comments)

    return corpus


def get_sequence_of_tokens(tokenizer, corpus):
    '''
    Used to create input sequences for the model from the corpus based on the tokenizer.
    NOTE: Function taken directly from notebook used in class, session 8

    Args:
    -   tokenizer (keras.preprocessing.text.Tokenizer): Tokenizer used to tokenize the corpus.
    -   corpus (list): List of all comments.

    Returns:
    -   input_sequences (list): List of input sequences for the model.

    '''

    print("\n" + "Creating input sequences...")

    input_sequences = []

    # loop over corpus
    for line in tqdm(corpus):
        # convert line to sequence of tokens
        token_list = tokenizer.texts_to_sequences([line])[0]

        # create n-grams
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    return input_sequences
    


def generate_padded_sequences(input_sequences, total_words):
    '''
    Generate padded sequences from the input sequences.
    This is done to ensure that all sequences are of the same length.
    NOTE: Function taken directly from notebook used in class, session 8

    Args:
    -   input_sequences (list): List of input sequences for the model.
    -   total_words (int): Total number of unique words in the corpus.

    Returns:
    -   predictors (np.array): Array of predictors for the model.
    -   label (np.array): Array of labels for the model.
    -   max_sequence_len (int): Length of the longest sequence.
    '''

    print("\n" + "Padding sequences...")

    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])

    # make every sequence the length of the longest one
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))
    
    # split predictors and labels
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

    # convert labels to categorical with one-hot encoding (uses float16 to save memory)
    label = ku.to_categorical(label, 
                            num_classes=total_words,
                            dtype='float16')
    
    return predictors, label, max_sequence_len


def tokenize(corpus):
    '''
    Tokenizes the corpus.
    Utilizes the get_sequence_of_tokens and generate_padded_sequences as support functions

    Args:
    -   corpus (list): List of all comments.

    Returns:
    -   tokenizer (keras.preprocessing.text.Tokenizer): Tokenizer used to tokenize the corpus.
    -   predictors (numpy.ndarray): Array of input sequences for the model.
    -   label (numpy.ndarray): Array of labels for the model.
    -   max_sequence_len (int): Length of the longest sequence.
    -   total_words (int): Total number of words in the corpus.
    '''

    # initialize tokenizer
    tokenizer = Tokenizer()

    # fit tokenizer on corpus
    tokenizer.fit_on_texts(corpus)

    # find total words
    total_words = len(tokenizer.word_index) + 1

    # create input sequences
    inp_sequences = get_sequence_of_tokens(tokenizer, corpus)

    # generate padded sequences
    predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences, total_words)

    return tokenizer, predictors, label, max_sequence_len, total_words


def get_embeddings(inpath, tokenizer, total_words, embedding_dim):
    '''
    Obtains the Glove embeddings for the words in the corpus.
    It is used to create the embedding matrix for the model.

    Args:
    -   inpath (pathlib.Path): Path to the input directory.
    -   tokenizer (keras.preprocessing.text.Tokenizer): Tokenizer used to tokenize the corpus.
    -   total_words (int): Total number of unique words in the corpus.
    -   embedding_dim (int): Dimension of the Glove embeddings (specified in argparse)

    Returns:
    -   embedding_matrix (numpy.ndarray): Array of embeddings for the model.

    '''

    # find filename based on embedding dimension
    glove_name = "glove.6B.{}d.txt".format(embedding_dim)

    # define path to glove embeddings
    glove_path = inpath / "glove_models" / glove_name

    # get embeddings
    embeddings_index = {}

    # loop over embeddings
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("\n" + "Creating embedding matrix...")

    # create embedding matrix
    embedding_matrix = np.zeros((total_words, embedding_dim))

    # loop over words in tokenizer
    for word, i in tqdm(tokenizer.word_index.items()):
        
        # get embedding vector for word
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def create_model(max_sequence_len, total_words, embedding_matrix):
    '''
    Creates the model object and compiles it so that it is ready to be trained.

    Args:
    -   max_sequence_len (int): Length of the longest sequence.
    -   total_words (int): Total number of unique words in the corpus.
    -   embedding_matrix (numpy.ndarray): Array of embeddings for the model.

    Returns:
    -   model (keras.engine.sequential.Sequential): Model object.

    '''
    print("\n" + "Creating and fitting model...")

    # get dimensions from embedding matrix
    embedding_dim = embedding_matrix.shape[1]

    # define input length
    input_len = max_sequence_len - 1

    # create model
    model = Sequential()

    # Add Input Embedding Layer - notice that this is different
    model.add(Embedding(
            total_words,
            embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False,
            input_length=input_len)
    )
    
    # Add LSTM Layer
    model.add(LSTM(30))

    # Add Dropout Layer
    model.add(Dropout(0.1))

    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model


def plot_loss(history):
    '''
    Plots the training loss for the model by epoch.

    Args:
    -   history (keras.callbacks.History): History object from the model.

    Returns:
    -   plt (matplotlib.pyplot): Plot of the training loss by epoch.

    '''

    # plot loss
    plt.plot(history.history['loss'])

    # add labels and legend
    plt.title('Model Loss by Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    
    return plt


def save_objects(model, history, tokenizer, args, outpath):
    '''
    Saves all relevant objects to out directory: The tokenizer, model, model summary and plot for training loss.
    Uses a subdirectory named after the model name.

    Args:
    -   model (keras.engine.sequential.Sequential): Model object.
    -   history (keras.callbacks.History): History object from the model.
    -   tokenizer (keras.preprocessing.text.Tokenizer): Tokenizer used to tokenize the corpus.
    -   args (argparse.Namespace): Arguments passed to the script.
    -   outpath (pathlib.Path): Path to the output directory.

    '''
    print("\n" + "Saving model and tokenizer...")

    # create a folder for the model if it doesn't exist
    if not os.path.exists(outpath / args.model_name):
        os.makedirs(outpath / args.model_name)
    
    # update outpath based on model name
    outpath = outpath / args.model_name
    
    # save tokenizer
    joblib.dump(tokenizer, outpath / "tokenizer.joblib")

    # save model
    save_model(model, outpath / "model.h5")

    # save model summary to text file with n_comments and n_epochs
    with open(outpath / "model_summary.txt", "w") as f:
        
        # save args to text file
        f.write("Number of comments used: {}".format(args.n_comments) + "\n")
        f.write("Number of epochs run: {}".format(args.epochs)+ "\n")

        # save model summary beneath
        model.summary(print_fn=lambda x: f.write(x + "\r \n"))

    # get loss plot
    plt = plot_loss(history)

    # save loss plot
    plt.savefig(outpath / "model_train_loss.png")



def main():
    # define paths
    inpath, outpath = define_paths()

    # parse arguments
    args = arg_parse()

    # check arguments
    input_checker(args)

    # load data
    corpus = load_data(inpath)

    # randomly sample comments if a limit has been set
    corpus = sample_comments(corpus, args.n_comments)

    # clean data with progress bar
    corpus = clean_corpus(corpus)

    # tokenize data
    tokenizer, predictors, label, max_sequence_len, total_words = tokenize(corpus)

    # get embeddings
    embedding_matrix = get_embeddings(inpath, tokenizer, total_words, args.embedding_dim)

    # create model
    model = create_model(max_sequence_len, total_words, embedding_matrix)

    # fit model
    history = model.fit(predictors, label, epochs=args.epochs, verbose=1, batch_size=512)

    # save outputs
    save_objects(model, history, tokenizer, args, outpath)

    # print completion message
    print("\n" + "Run Complete! Model and tokenizer saved to folder named {}.".format(args.model_name) + "\n" + "Check the folder for the tokenizer, model, summary and loss plot." + "\n")
    


# run main
if __name__ == "__main__":
    main()