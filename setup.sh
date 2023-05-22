#!/usr/bin/env bash
# create virtual environment called lang_modelling_env
python3 -m venv lang_modelling_env

# activate virtual environment
source ./lang_modelling_env/bin/activate

# install requirements
python3 -m pip install -r requirements.txt