# DyselxiaLearn

Project repository for predicting Dyslexia from 3D Brain MRI scans using 3D Convolutional Autoencoders, and using trained models to identify regions of the brain most important for identifying Dyslexia.


## Environment Setup

1. Please install anaconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. From the project root folder, run the following to setup environment:
`conda env create -f environment.yml`
3. Install the dyslexialearn module, by running the following from the project root:
`pip install -e .`

## Data Setup

data
├── n192_data_for_resid.csv
├── raw_images
└── xTemplate_gm50wm50_mask.nii

- `n192_data_for_resid.csv` contains metadata information about all scans, specifically, it needs to contain the columns for `id`, `site` and `Group`

- Inside the `data/raw_images` folder, 3D 121x145x121 jacobian brain MRI scans are expected with the following naming convention:
`jac_rc1r<<id>>_debiased_deskulled_denoised_xTemplate_subtypes.nii`

- `xTemplate_gm50wm50_mask.nii` is a boolean brain mask of the shape 121x145x121

## Directory Setup

You'll need to create the below directories to run various functions of the code

- models
- out_folder
- out_folder/sensitivities


## Training a new model

The training workflow contains of three broad steps:

1. Train an autoencoder model
`python dyslexialearn/autoencoder_train.py input_configs/ae_train.json`

The trained model gets stored in the models/ directory

2. Infer latent embeddings with trained models
`python dyslexialearn/autoencoder_train.py input_configs/ae_infer.json`

This converts all files in `data/raw_images` to lower dimensional 15x18x15x32 sized 4D images and saves in `out_data/<model_name>`

3. Train classifier experiments
`python dyslexialearn/latent_classifier.py input_configs/classifier.json`

This trains binary classifiers on the latent images, saves new models in `models` and outputs a statistic file for EDA.


## Importance Analysis

To run sensitivity analysis with a trained model, use:
`python dyslexialearn/calculate_sensitivities.py input_configs/sensitivities.json`

The resulting sensitivity files will be saved in `out_data/sensitivities/cases` and `out_data/sensitivities/controls`

## Input Configuration Files

Below is a note on what each of the input json files does:
1. `ae_train.json`

{
	"function": "train",
	"model_name": "model1", # Name of model to save in models/"model_name"
	"batch_size": 24,
	"train_steps": 100,
	"logging_interval": 3
}

2. `ae_infer.json`

{
	"function": "infer",
	"model_name": "model1"  # Model in models/"model_name" will be used for inferring.
}

3. `classifier.json`

{
    "latent_files_dir": "out_data/latent_model1", # Directory to read latent data
    "model_name": "classifier", # Model will be save models/"model_name"
    "num_splits": 2,    # Number of train-test splits to train
    "num_models_per_split": 2, # Number of models to train per split
    "save_all_models": true,  # save models each split and i
    "results_write": "results.json", # Save final results
    "training_options":{
        "train_steps": 1000, 
        "dropout_rate": 0.25,  
        "batch_size": 64    
    }
}

4. `sensitivities.json`

{
	"autoencoder_model": "model1",  # Name of autoencoder model
	"classifier_model": "classifier", # Name of binary classification model
	"num_perturbations_per_image": 1000,
	"noise_shape":  [10, 10, 10], 
	"noise_mean": 0,
	"noise_std": 1
}



