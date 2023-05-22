# load packages
import numpy as np
import argparse
from pathlib import Path
import joblib
import pandas as pd
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
from tqdm import tqdm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def define_paths():
    '''
    Define paths for input and output data.
    Returns:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   outpath (pathlib.PosixPath): Path to output data.
    '''

    # define paths
    path = Path(__file__)

    # define input dir
    inpath = path.parents[1] / "in"

    # define output dir
    outpath = path.parents[1] / "out"

    return inpath, outpath

def clean_text(txt):
    # change all inputs to strings if they are not
    txt = str(txt)

    # remove punctuation
    txt = "".join(v for v in txt if v not in string.punctuation).lower()

    # remove numbers
    txt = txt.encode("utf8").decode("ascii",'ignore')

    return txt

def clean_corpus(corpus):
    '''
    Clean corpus by removing punctuation and numbers.
    '''
    # clean corpus
    print("\n" + "Cleaning corpus...")
    corpus = [clean_text(x) for x in tqdm(corpus)]

    return corpus
