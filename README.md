# TF-ranking example

This repository contains an example TFTransform/TFRanking pipeline that : 

- Serializes generated data in the ELWC format and writes the result in TFrecords
- Trains a TFRanking model with the Keras API
- Saves it for later use with a signature function that takes raw example as input for predictions

### Usage
To be able to run the code, you should activate a Python Virtual Environment and install the packages in `requirements.txt`

```bash
vitrualenv --python=python3.5.7 venv
source venv/bin/activate

pip install -r requirements.txt
```

### Running the code
- To serialize data, use:
```bash
rm -rf ./output/*
python transform_pipeline.py
```
It will generate a TFRecords file and the TFTransform metadata in the `./output` directory. Before 

- To train and save the model, use:
```bash
python train_keras_model.py
```