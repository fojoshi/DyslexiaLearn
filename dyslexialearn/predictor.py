import numpy as np
import tensorflow as tf
from dyslexialearn.utils import load_data
import json
import sys
import pandas as pd

options = json.load(open(sys.argv[1], "r"))
out_file_name = options.get("out_filename", "prediction_results.csv")
csv, data, mask = load_data()

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
    image = image * mask
    log_data = np.log(image+1e-10)
    data_norm = (log_data - min_log_data)/(max_log_data - min_log_data)
    data_norm = data_norm * mask
    return data_norm

def get_prediction(brain_image):
    latent = sess_latent.run("latent/Relu:0", {"inputs:0": brain_image})
    prediction = sess_pred.run("dense_1/BiasAdd:0", {"inputs:0": latent, 'training:0': False, "keep_prob:0": 0})
    return np.squeeze(1/(1 + np.exp(-prediction)))

batch_size = 32
num_images = len(data)
iterations = num_images//batch_size
all_preds = []

for i in range(iterations):
    batch = data[i * batch_size: min((i + 1) * batch_size, num_images)]
    predictions = get_prediction(preprocess(batch))
    all_preds.extend(predictions)

csv["preds"] = all_preds

csv.to_csv(out_file_name, index=False)
print(f"Results saved in {out_file_name}")
accuracy = (csv.preds.round() == csv.Group).mean()

print(f"Accuracy : {accuracy}")