#!/usr/bin/env bash
# create virtual environment called lang_modelling_env
python3 -m venv lang_modelling_env

# activate virtual environment
source ./lang_modelling_env/bin/activate

# install requirements
python3 -m pip install -r requirements.txt

# run script for training model
python3 src/train.py

# run script for generating text
python3 src/generate.py

# deactivate virtual environment
deactivate lang_modelling_env