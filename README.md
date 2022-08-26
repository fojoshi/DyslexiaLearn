# DyslexiaLearn

This is the repository for resources used to predict reading disability using defomration-based (Jacobian) images derived from T1-weighted MRI scans using 3D Convolutional Autoencoders, as well as for using trained models to identify regions of the brain that were most important for identifying cases with reading disability. 

This collaborative project between the Multimedia and Bioinformatics lab in Clemson University and the Eckert laboratory at the Medical University of South Carolina (MUSC) was supported by the NICHD (R01 HD 069374).

Email at `foram2494@gmail.com` for questions/suggestions.


## Environment Setup

1. Please install anaconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. From the project root folder, run the following to setup environment:
`conda env create -f environment.yml`
3. Activate your new conda environment
`conda activate DyslexiaLearn`
4. Install the dyslexialearn module, by running the following from the project root:
`pip install -e .`

## Data Setup

The below datafiles are required for the code to function

```
data
├── n192_data_for_resid.csv
├── raw_images
└── brain_mask.nii
```

- `n192_data_for_resid.csv` contains data linked to the imaging data, including participant label (id), research site (site), and reading disability label (Group)

- Inside the `data/raw_images` folder, 3D 121x145x121 Jacobian determinant images art stored with the following naming convention:
`jac_rc1r<<id>>_debiased_deskulled_denoised_xTemplate_subtypes.nii`

- `brain_mask.nii` is a brain mask of the shape 121x145x121

## Directory Setup

You'll need the below directories to run various functions of the code. The `pip install -e .` setup command should create these directories automatically.

- data
- models
- out_folder
- out_folder/sensitivities/cases
- out_folder/sensitivities/controls


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

The output `results.json` file will have the following structure:
```
[	
	# for each split
	{
		"model_idx_1":{
			"train_ids": [ ... Training Image Ids ... ],
			"test_ids": [ ... Test Image Ids ... ],
			"preds: [ ... Predicted Probabilities for all test images ... ],
			"truth": [ ... Ground Truth of all test images ... ],
			"acc": test_accuracy
		}
		"model_idx_2": {
			.....
		}
	}
]
```

### Predicting class probabilities with trained models

The following command will generate a csv file containing predicted class probabilities for each participant in the `data/raw_images` folder

`python dyslexialearn/predictor.py input_configs/predict.json`


## Importance Analysis

To run sensitivity analysis with a trained model, use:
`python dyslexialearn/calculate_sensitivities.py input_configs/sensitivities.json`

The resulting sensitivity files will be saved in `out_data/sensitivities/cases` and `out_data/sensitivities/controls`

## Input Configuration Files

Below is a note on what each of the input json files does:
1. `ae_train.json`

```
{
	"function": "train",
	"model_name": "model1", # Name of model to save in models/"model_name"
	"batch_size": 24,
	"train_steps": 100,
	"logging_interval": 3
}
```

2. `ae_infer.json`

```
{
	"function": "infer",
	"model_name": "model1"  # Model in models/"model_name" will be used for inferring.
}
```
3. `classifier.json`

```
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
```

4. `sensitivities.json`

```
{
	"autoencoder_model": "ae_model",  # Name of autoencoder model
	"classifier_model": "cnn_classifier", # Name of binary classification model
	"num_perturbations_per_image": 1000,
	"noise_shape":  [10, 10, 10], 
	"noise_mean": 0,
	"noise_std": 1
}
```

5. `predict.json`

```
{
	"autoencoder_model": "ae_model",
	"classifier_model": "classifier_split1_init_0"
}
```
