import numpy as np
import pandas as pd
import os
import nibabel as nib
import tensorflow as tf
import json
import sys
from tqdm import tqdm

#Getting random mini batches
def getMiniBatch(dataset, batch_size=32):
    random_indices = np.random.choice(len(dataset), [batch_size], replace=False)
    return dataset[random_indices]

def load_data(csv):
    X = []
    for id_name in csv['id']:
        filename = f"data/raw_images/jac_rc1r{id_name}_debiased_deskulled_denoised_xTemplate_subtypes.nii"
        X.append(nib.load(filename).get_fdata())

    return np.array(X)

#Building the graph (You can make a class for this)
tf.reset_default_graph()

#Encoder part begins
X_input = tf.placeholder(tf.float32, shape=[None, 121, 145, 121], name='inputs')
X_input_reshaped = tf.expand_dims(X_input, -1)

conv1 = tf.layers.conv3d(inputs= X_input_reshaped, filters=128, kernel_size=(3,3,3), padding='same', strides = 2, activation=tf.nn.relu)
conv2 = tf.layers.conv3d(inputs=conv1, filters=64, kernel_size=(3,3,3),  strides = 2, padding='same', activation=tf.nn.relu)
conv3 = tf.layers.conv3d(inputs=conv2, filters=32, kernel_size=(2,2,2),  strides = 2, activation=tf.nn.relu,
                        name="latent")
#latent internal representation

#Decoder part begins
deconv1 = tf.layers.conv3d_transpose(inputs=conv3,filters=64, kernel_size=(2,2,2), strides = 2, activation=tf.nn.relu)
deconv2 = tf.layers.conv3d_transpose(inputs=deconv1,filters=32, kernel_size=(2,2,2), strides = 2, activation=tf.nn.relu)
deconv3 = tf.layers.conv3d_transpose(inputs=deconv2, filters=1, kernel_size=(3,3,3), strides = 2, activation=tf.nn.sigmoid,
                                    name='out')

#Loss is the MSD between the voxels of whole brain of each person in the batch
loss= tf.reduce_mean(tf.square(deconv3 - X_input_reshaped))
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()

#192 dataset
csv = pd.read_csv("data/n192_data_for_resid.csv") 
data = load_data(csv)
mask = nib.load("data/xTemplate_gm50wm50_mask.nii").get_fdata()

log_data = np.log(data+1e-10)
masked_data = log_data * mask


def train(options):
    if 'model_name' not in options:
        raise Exception("Please specify model name in input")
    model_name = options["model_name"]

    #Log normalization
    min_log_data = np.min(masked_data)
    max_log_data = np.max(masked_data)
    data_norm = (masked_data - min_log_data)/(max_log_data - min_log_data)
    data_norm = data_norm * mask
    model_metadata = {
        'min_log_data': min_log_data,
        'max_log_data': max_log_data
    }

    #X_train, X_test = train_test_split(data_norm, , 
    #    random_state=42)

    test_size = int(len(data_norm) * options.get("test_size", 0.2))
    
    all_idx = [i for i in range(len(data_norm))]
    np.random.shuffle(all_idx)
    X_train = data_norm[all_idx[:-test_size]]
    X_test = data_norm[all_idx[-test_size:]]

    print("Train Size:", len(X_train))
    print("Test Size:", len(X_test))
    print("Shape of X_train:", np.shape(X_train))
    print("Shape of X_test:", np.shape(X_test))

    sess.run(tf.global_variables_initializer())

    try:
        #Training model
        loss_sum = 0
        iterations=options.get("iterations", 1000)
        logging_interval = options.get("logging_interval", 3)
        batch_size = options.get("batch_size", 32)
    
        i = 0
        print(f"Training Starts. You can Control+C anytime, model gets saved every {logging_interval} iteration(s).")
        while i < iterations:
            for _ in tqdm(range(min(logging_interval, iterations-i))):
                minibatch = getMiniBatch(X_train, batch_size)
                l,_ = sess.run(fetches = [loss,optimizer], feed_dict = {X_input : minibatch})
                loss_sum += l
                i += 1
            
            l_test =  sess.run(fetches = loss, feed_dict = {X_input : X_test})
            print(f"Iterations: {i}/{iterations}: Train Loss {loss_sum/logging_interval}. Test Loss: {l_test}")
            
            print(f"Saving current model weights in models/{model_name}")
            saver.save(sess, f"models/{model_name}/model.ckpt")
            json.dump(model_metadata, open(f"models/{model_name}/model_metadata.json", "w"))
            loss_sum = 0

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping Model Training")

def infer(options):
    model_name = options["model_name"]
    print(f"Using model weights from: 'models/{model_name}'")
    saver.restore(sess, f"models/{model_name}/model.ckpt")
    model_metadata = json.load(open(f"models/{model_name}/model_metadata.json", "r"))
    min_log_data = model_metadata["min_log_data"]
    max_log_data = model_metadata["max_log_data"]

    data_norm = (masked_data - min_log_data)/(max_log_data - min_log_data)
    data_norm = data_norm * mask

    write_dir = f"out_data/latent_{model_name}"
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    #Save the latent representations: These will be the input to our binary classifier
    for name, row in zip(csv.id, data_norm):
        latent = sess.run(tf.squeeze(conv3), feed_dict={X_input: [row]})
        np.save(f"{write_dir}/{name}.npy", latent)
    print(f"All latent files written in : {write_dir}")

if __name__ == "__main__":
    options = json.load(open(sys.argv[1], "r"))
    if options["function"] == "train":
        train(options)
    elif options["function"] == "infer":
        infer(options)