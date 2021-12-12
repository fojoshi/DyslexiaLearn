import numpy as np
from scipy import ndimage
import tensorflow as tf
import nibabel as nib
from sklearn.linear_model import LinearRegression
import multiprocessing
import sys
import copy
import sklearn
import pandas as pd
import sklearn.metrics
import warnings
import json
import concurrent.futures

data = pd.read_csv("data/n192_data_for_resid.csv")
mask = nib.load("data/xTemplate_gm50wm50_mask.nii").get_fdata()

options = json.load(open(sys.argv[1], "r"))

NOISE_SHAPE = options.get('noise_shape', [16,16,16])
MU = options.get('noise_mean', 0)
SIGMA = options.get('noise_std', 1)

autoencoder_model = options["autoencoder_model"]
classifier_model = options["classifier_model"]
model_metadata = json.load(open(f"models/{autoencoder_model}/model_metadata.json", "r"))

tf.reset_default_graph()
latent_graph = tf.Graph()
prediction_graph = tf.Graph()
default_graph = tf.get_default_graph() 
with latent_graph.as_default():
    imported_meta_latent = tf.train.import_meta_graph(f"models/{autoencoder_model}/model.ckpt.meta")
    sess_latent = tf.Session(graph=latent_graph)
    imported_meta_latent.restore(sess_latent, tf.train.latest_checkpoint(f"models/{autoencoder_model}"))

with prediction_graph.as_default():
    imported_meta_pred = tf.train.import_meta_graph(f"models/{classifier_model}/model.ckpt.meta")
    sess_pred = tf.Session(graph=prediction_graph)
    imported_meta_pred.restore(sess_pred, tf.train.latest_checkpoint(f'models/{classifier_model}'))
min_log_data = model_metadata["min_log_data"]
max_log_data = model_metadata["max_log_data"]

def preprocess(image):
    log_data = np.log(image+1e-10)
    data_norm = (log_data - min_log_data)/(max_log_data - min_log_data)
    data_norm = data_norm * mask

    return data_norm

def get_prediction(brain_image):
    latent = sess_latent.run("latent/Relu:0", {"inputs:0": brain_image})
    prediction = sess_pred.run("dense_1/BiasAdd:0", {"inputs:0": latent, 'training:0': False, "keep_prob:0": 0})
    return np.squeeze(1/(1 + np.exp(-prediction)))

def perturb(image, gauss_noise_image, patch):
    # image : [121, 145, 121], float32, original brain image
    # gaussian_noise_image: [15, 15, 15], float32, gaussian image
    # patch: [3, 2]: [[i1, i2], [j1, j2], [k1, k2]]
    
    # patch = cutboxes[cutbox_id]
    i1, i2 = patch[0][0], patch[0][1]
    j1, j2 = patch[1][0], patch[1][1]
    k1, k2 = patch[2][0], patch[2][1]
    perturbed_image = np.copy(image)
    for i in range(i1, i2):
        for j in range(j1, j2):
            for k in range(k1, k2):
                perturbed_image[i][j][k] += gauss_noise_image[i-i1][j-j1][k-k1]
    
    return perturbed_image

def pick_random_box(span):
    while True:
        i = np.random.choice(121)
        j = np.random.choice(145)
        k = np.random.choice(121)
        if mask[i][j][k] != 0:
            break
    return [[i - span[0], i + span[0]],
            [j - span[1], j + span[1]],
            [k - span[2], k + span[2]]]

def getPerturbedDataset(image, iters, fname):
    d ={}
    mu, sigma = MU, SIGMA
    noise_shape = NOISE_SHAPE
    noise_span = [noise_shape[0]//2, noise_shape[1]//2, noise_shape[2]//2]
    actual_pred = get_prediction([image])
    sensitivity_map = np.zeros_like(image)
    count_map = np.ones_like(image)
    minibatch = []
    noise_means = []
    patches = []
    for i in range(iters):
        gauss_noise_image = np.random.normal(mu, sigma, noise_shape)
        patch = pick_random_box(noise_span)
        perturbed_image = perturb(image, gauss_noise_image, patch)
        minibatch.append(perturbed_image)
        noise_means.append(np.mean(np.abs(gauss_noise_image)))
        i1, i2 = patch[0][0], patch[0][1]
        j1, j2 = patch[1][0], patch[1][1]
        k1, k2 = patch[2][0], patch[2][1]
        patches.append([i1, i2, j1, j2, k1, k2])
        if len(minibatch) > 1:
            perturb_preds = get_prediction(minibatch)
            for perturb_pred, patch_ in zip(perturb_preds, patches):
                error = actual_pred - perturb_pred
                i1, i2, j1, j2, k1, k2 = patch_
                sensitivity_map[i1:i2, j1:j2, k1:k2] += error
                count_map[i1:i2, j1:j2, k1:k2] += 1
            minibatch = []
            patches = []

    sensitivity_map = mask * (sensitivity_map/count_map)
    np.save(fname, sensitivity_map)
    print(f"Saving map in {fname}")

X_dys = []
X_non_dys = []
Y_dys = []
Y_non_dys =[]
files_dys = []
files_nondys = []

patient_names = data["id"].values
patient_group = data["Group"].values

for name, group in zip(patient_names, patient_group):
    filename = "data/raw_images/jac_rc1r{}_debiased_deskulled_denoised_xTemplate_subtypes.nii".format(name)
    if group == 1:
        X_dys.append(nib.load(filename).get_fdata())
        Y_dys.append(group)
        files_dys.append(name)
    else:
        X_non_dys.append(nib.load(filename).get_fdata())
        Y_non_dys.append(group)
        files_nondys.append(name)

X_dys = np.array(X_dys)            
Y_dys = np.array(Y_dys)        
X_non_dys = np.array(X_non_dys)        
Y_non_dys = np.array(Y_non_dys)        

Y_dys= np.expand_dims(Y_dys,-1)        
Y_non_dys = np.expand_dims(Y_non_dys, -1)    

sensitivity_map = np.zeros_like(mask)

num_perturbations_per_image = options.get("num_perturbations_per_image", 10000)

num_jobs = 8
num_batch = int(len(X_non_dys)//num_jobs)
with concurrent.futures.ThreadPoolExecutor() as executor: 
    for i in range(num_jobs):
        start = i*num_batch
        end = min((i+1)*num_batch, len(X_non_dys))
        results = [executor.submit(getPerturbedDataset, 
                         preprocess(img), num_perturbations_per_image, 
                         "out_data/sensitivities/controls/sensitivity_map_{}.npy".format(name))
                         for (img, name) in zip(X_non_dys[start:end], files_nondys[start:end])]
        
        for f in concurrent.futures.as_completed(results):
            f.result()

with concurrent.futures.ThreadPoolExecutor() as executor: 
    for i in range(8):
        start = i*num_batch
        end = min((i+1)*num_batch, len(X_non_dys))
        
        results = [executor.submit(getPerturbedDataset, 
                         preprocess(img), num_perturbations_per_image, 
                         "out_data/sensitivities/cases/sensitivity_map_{}.npy".format(name))
                         for (img, name) in zip(X_dys[start:end], files_dys[start:end])]
        
        for f in concurrent.futures.as_completed(results):
            f.result()
