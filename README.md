# Assignment 3: Language Modelling and Text Generation using RNNs

## Repository Overview
1. [Description](#description)
2. [Data and Methods](#dam)
4. [Repository Tree](#tree)
5. [Setup](#setup)
6. [General Usage](#gusage)
7. [Modified Usage (RECOMMENDED)](#musage)
8. [Discussion](#discussion)


## Description <a name="description"></a>
This repository includes the solution by *Anton Drasbæk Schiønning (202008161)* to assignment 3 in the course "Language Analytics" at Aarhus University. Using TensorFlow, it builds and trains a Recurrent Neural Network (RNN) model, based on a large corpus of comments from the New York Times.

The analysis and exemplary outputs have been tested on a MacBook Pro using macOS Monterey and on [Ucloud](https://cloud.sdu.dk/app), running on Ubuntu v22.10.
</br></br>

## Data and Methods <a name="dam"></a>
### Data
The data that is used comes from the comment section of New York Times Articles in 2017 and 2018. It contains over 2 million comments that can all be utilized in this modelling if desired. For more information, please check the dataset on [Kaggle](https://www.kaggle.com/datasets/aashita/nyt-comments).

### Methods
This analysis uses a Recurrent Neural Network (RNN) to generate text based on the dataset. Concretely, the tensorflow framework is used to construct a sequential model with a single LSTM layer. Each word in the entire training corpus is represented by a GloVe embedding vector. The model training is targeted at predicting the next word in a sequence of words. <br>

To see the model architecture, please refer to `out/example_model/model_summary.txt`.
</br></br>

## Repository Tree <a name="tree"></a>

```
├── README.md                           
├── assign_desc.md              
├── in
│   ├── glove_models       <----- glove_embedding txt files (PLACE HERE)
│   └── news_data          <----- NYT dataset csv files (PlACE HERE)
├── out
│   ├── example_model      <----- exemplary model folder from a training run
│   │   ├── model.h5                  
│   │   ├── model_summary.txt         
│   │   ├── model_train_loss.png      
│   │   └── tokenizer.joblib 
├── requirements.txt
├── run.sh                      
├── setup.sh                    
└── src
    ├── generate.py        <----- script for generating a comment based on a trained model in out
    ├── train.py           <----- script for training a new model
    └── utils.py
```
## Setup <a name="setup"></a>
The analysis can be run by cloning this GitHub repo to your preferred machine. Other than having installed Python3, you must also obtain the New York Times Comments dataset and the Glove word embeddings. <br>

Both of these can be directly downloaded from Kaggle: [The NYT dataset](https://www.kaggle.com/datasets/aashita/nyt-comments) and [the GloVe embeddings](https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation). <br>

These files should be downloaded, unpacked and placed into the `in` folder in this directory. Name the data folder `news_data` and the folder with GloVe embeddings `glove_models`. If in doubt, see *Repository Tree*.
</br></br>

## General Usage <a name="gusage"></a>
If complying with the *setup*, the analysis can then be run by executing the shell script:
```
bash run.sh
```
This achieves the following:
* Creates and activates a virtual environment
* Installs requirements to that environment
* Trains an RNN for text generation using default parameters (`train.py`)
* Generates text using the model with default inputs (`generate.py`)
* Deactivates the virtual environment

**PLEASE NOTE:** Running the full analysis requires vast computuational power, likely only obtainable to most through cloud computing.
Hence, the full analysis has not been tested although it is set up to work properly. For most usecases, please refer to the following section to run a modified, reduced analysis. This modified usage also allows you to specify your own text prompts for the generations.
</br></br>

## Modified Usage (RECOMMENDED) <a name="musage"></a>
### Additional setup
If running a modified analysis, you must run a setup bash script in addition to the general [Setup](#setup) steps mentioned beforehand:
```
bash setup.sh
```
This will create a virtual environment and install requirements to that environment.
</br></br>

### Training a Model
A model, such as the one shown in the `out` folder named `example_model`, can be trained using the `train.py` script. <br>

Several arguments can be specified in order to suit whatever model you want to create:

| Argument              | Default Value | Description                                                             |
|---------------------|---------------|-------------------------------------------------------------------------|
| `--n_comments` `-c`       | 'all'         | Number of comments to train on, specify an integer (e.g., 1000) to not use all comments.       |
| `--epochs` `-e`       | 75           | Number of epochs for training.                                           |
| `--embedding_dim` `-d`    | 50            | Dimensionality of word embeddings (limited to GloVe dimensions: 50, 100, 200, and 300). |
| `--model_name` `-m`      | 'example_model' | Name of the model (used for folder in out).                                                      |
<br>

The example model has been created using these arguments for the training:
```
# create example_model using 5000 comments, run for 75 epochs, using 50 dimenional embeddings
python src/train.py -c 5000 -e 75 -d 50 -m 'example_model'
```
<br>

The outputs for your model can be found in the `out` directory in a folder named after the name you have chosen and contains four objects:
- `model_summary.txt`: Summary of model parameters including what embedding dimension was used, number of epochs, and more.
- `model_train_loss.png`: Image of the training loss development as the model was trained.
- `model.h5`: Fitted model object that may be called to generate text.
- `tokenizer.joblib`: The tokenizer that was used for building the model.
<br>

## Generate Comments
Text is generated by applying either a self-trained model or the `example_model` using the `generate.py` script.

As arguments, it takes what model should be used, the beginning of the text generation and how many words should be generated following this beginning:
| Argument              | Default Value        | Description                                                        |
|---------------------|----------------------|--------------------------------------------------------------------|
| `--model_name` `-m`     | 'example_model'      | Name of the model folder.                                                  |
| `--seed_text` `-s`      | 'I know this comment is generic but' | Seed text to generate based on.                         |
| `--n_next_words` `-n`     | 10                   | Number of next words to generate.           |
<br>

For instance, we may run the model as such:
```
# use example_model to generate next 7 words from "The news article was great, but"
python src/generate.py -m 'example_model' -s 'This article was incredible, I' -n 7
```

Which gives the output:
```
Output Text:
This article was incredible, I am a lot of the way of
```

## Discussion <a name="discussion"></a>
As the example output would indicate, the `example_model` fails to produce comments that mimic a real comment from the dataset. Other tests have shown similar, poor results. <br>

However, due to computational limitations, this `example_model` also has only been trained on a fraction of the data. The training loss curve (`out/example_model/model_train_loss.png`) shows, the model would likely benefit from training for more epochs as the loss has not fully plateued yet. In order to get a better performining model, you should do the following if you have the necessary computational power:
* Train on entire dataset instead of 5000 comments.
* Use a bigger GloVe embedding such as the 200D.
* Potentially train for more epochs than 75 if loss has not plateued completely.

Simply running `run.sh` should achieve creating such a model. However, training `example_model` further or using completely different arguments by following the flexibility specified in *Modified Usage* are also viable options for enhancing performance.





