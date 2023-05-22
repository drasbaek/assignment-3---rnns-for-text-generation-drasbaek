""" generate.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This script is used to generate text based on a trained RNN model, outputted from train.py.
    The text generation outputs are printed to the terminal.

Usage:
    $ python src/generate.py -m 'example_model' -s 'I know this comment is generic but' -n 7
"""

# load packages
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from utils import *

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

def arg_parse():
    '''
    Parse command line arguments.
    It is possible to specify:
    -  Name of model folder to be used for generating text.
    -  Seed text to generate text from.
    -  Number of words to be generated (not including those in input).
    
    Returns:
    -   args (argparse.Namespace): Parsed arguments.
    '''
    
    # define parser
    parser = argparse.ArgumentParser(description='Generate text based on a trained model.')

    # add arguments
    parser.add_argument('-m', '--model', default="example_model", help='Model folder to be used for generating text.')
    parser.add_argument('-s', '--seed_text', default='I know this comment is generic but', help='Seed text to generate text from.')
    parser.add_argument('-n', '--n_next_words', type=int, default=10, help='Number of words to be generated (not including those in input).')

    # parse arguments
    args = parser.parse_args()

    return args


def generate_text(seed_text, next_words, model, tokenizer):
    '''
    Function that generates text based on a trained model, tokenizer and seed text.
    NOTE: Function taken directly from notebook used in class, session 8

    Args:
    -   seed_text (str): Seed text to generate text from.
    -   next_words (int): Number of words to be generated (not including those in input).
    -   model (keras.model): Trained model.
    -   tokenizer (keras.preprocessing.text.Tokenizer): Tokenizer used for training the model.

    Returns:
    -   seed_text (str): Generated text.

    '''
    # find max sequence length
    max_sequence_len = model.layers[0].input_shape[1]

    for _ in range(next_words):
        # tokenize seed text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        # pad token list
        token_list = pad_sequences([token_list], 
                                    maxlen=max_sequence_len, 
                                    padding='pre')
        
        # predict next word
        predicted = np.argmax(model.predict(token_list),
                                            axis=1)
        
        # find word corresponding to predicted index
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        # add word to seed text
        seed_text += " "+output_word

    return "Output Text:" + "\n" + seed_text


def main():
    # define paths
    inpath, outpath = define_paths()

    # parse arguments
    args = arg_parse()

    # load tokenizer
    tokenizer = joblib.load(outpath / args.model / "tokenizer.joblib")

    # load model
    print("Loading model...")
    model = load_model(outpath /  args.model / "model.h5")

    # generate text
    print("Generating text...")
    print(generate_text(args.seed_text, args.n_next_words, model, tokenizer))

if __name__ == "__main__":
    main()




