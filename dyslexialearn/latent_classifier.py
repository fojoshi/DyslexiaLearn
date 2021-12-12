import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sys
import json
import pandas as pd

options = json.load(open(sys.argv[1], "r"))
    
def getMiniBatch(dataset, label_dataset, batch_size=16):
    random_indices = np.random.choice(len(dataset), [batch_size], replace=False)
    return dataset[random_indices], label_dataset[random_indices]
    
def getTrainTestSplit(data, dir='out_data/latent', random_state=None):
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for site in data.site.unique():
        this_site = data[data["site"]==site]
        dys_ = this_site[this_site["Group"] == 1]
        non_dys_ = this_site[this_site["Group"] == 0]
        
        if len(dys_) >= 15:
            dys_ratio = 0.2 # 80%train & 20% test
        else:
            dys_ratio = 0.1 # 90%train & 10% test
        if len(non_dys_) >= 15:
            non_dys_ratio = 0.2
        else:
            non_dys_ratio = 0.1

        this_dys_train, this_dys_test = train_test_split(dys_, test_size=dys_ratio, random_state=random_state)
        this_non_dys_train, this_non_dys_test = train_test_split(non_dys_, test_size=non_dys_ratio, random_state=random_state)
        this_train = pd.concat([this_dys_train, this_non_dys_train])
        this_test = pd.concat([this_dys_test, this_non_dys_test])
        train_data = pd.concat([train_data, this_train])
        test_data = pd.concat([test_data, this_test])
    
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for id_name in train_data['id']:
        filename = f"{dir}/{id_name}.npy"
        X_train.append(np.load(filename))
    X_train = np.array(X_train)
    Y_train = train_data["Group"].to_numpy()
    Y_train = np.expand_dims(Y_train, 1)
    for id_name in test_data['id']:
        filename = f"{dir}/{id_name}.npy"
        X_test.append(np.load(filename))
    X_test = np.array(X_test)
    Y_test = test_data["Group"].to_numpy()
    Y_test = np.expand_dims(Y_test, 1)
    return train_data, test_data, X_train, X_test, Y_train, Y_test

def train_model(X_train, Y_train, X_test, Y_test, options, model_name=None):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    loss_sum = 0
    train_steps=options.get("train_steps", 1000)
    testing_interval = 2
    logging_interval = 100
    batch_size = options.get("batch_size", 64)
    dropout_rate = options.get("dropout_rate", 0.25)
    train_acc = 0
    best_acc_yet = 0
    for i in range(train_steps):
        minibatchX, minibatchY = getMiniBatch(X_train, Y_train, batch_size)
        accuracy, l,_ = sess.run(fetches = [acc,loss,optimizer], 
                                 feed_dict = {X_input : minibatchX, label: minibatchY, keep_prob: dropout_rate, istraining:True})
        loss_sum += l
        train_acc += accuracy
        if (i + 1) % testing_interval == 0:
            t_acc, l_test =  sess.run(fetches = [acc, loss], feed_dict = {X_input : X_test, 
                                                                          label: Y_test,
                                                                          keep_prob:0,
                                                                         istraining:False})
            if (i + 1) % logging_interval == 0:
                print(f"Train Steps {i + 1}/{train_steps}: is {loss_sum/logging_interval:.3}. Test Loss: {l_test:.3}, Train Acc:{train_acc/logging_interval:.3}, Test Acc:{t_acc:.3}")
                loss_sum = 0
                train_acc = 0

            if t_acc > best_acc_yet:
                best_acc_yet = t_acc
                saver.save(sess, f"models/{model_name}/model.ckpt")
                print(f"Saving..  Loss: {l_test:.3}, Acc: {t_acc:.3}\n")
    
    saver.restore(sess, f"models/{model_name}/model.ckpt")
    y_pred_best = sess.run(tf.nn.sigmoid(predict), 
                          {X_input: X_test, keep_prob: 0, istraining:False})
    return best_acc_yet, y_pred_best

#Reading the data
data = pd.read_csv("data/n192_data_for_resid.csv") 

#Building the graph
tf.reset_default_graph()

X_input = tf.placeholder(tf.float32, shape=[None, 15, 18, 15, 32], name='inputs')
label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
istraining = tf.placeholder(tf.bool, shape=[], name='training')
init=tf.truncated_normal_initializer(stddev=0.05)
#init=None

conv1 = tf.layers.conv3d(inputs=X_input, filters=24, kernel_size=(3,3,3), strides = 2, 
                         activation=tf.nn.relu,kernel_initializer=init)
conv1 = tf.layers.batch_normalization(conv1, training=istraining)
conv1 = tf.layers.dropout(conv1, rate=keep_prob)
print("conv1:", np.shape(conv1))

flatten = tf.layers.flatten(conv1)
print("flatten",np.shape(flatten))

dense1 = tf.layers.dense(flatten,128,activation=tf.nn.relu,kernel_initializer=init)
dense1 = tf.layers.batch_normalization(dense1, training=istraining)
dense1 = tf.layers.dropout(dense1, rate=keep_prob)

predict = tf.layers.dense(dense1, 1,kernel_initializer=init)
print("predict",np.shape(predict))

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=label, logits=predict, name=None
))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

acc = tf.reduce_mean(tf.cast(tf.equal(label, tf.round(tf.nn.sigmoid(predict))), tf.float32))

num_expts = options.get("num_splits", 10)
latent_files_dir = options.get("latent_files_dir")
num_model_initializations = options.get("num_models_per_split", 1)
results_file = options.get("results_write", "results.csv")
model_name_suffix = options.get("model_name", "results.csv")
all_results = []
for exp in range(num_expts):
    train_data, test_data, X_train, X_test, Y_train, Y_test = getTrainTestSplit(data, dir=latent_files_dir, random_state=exp)
    this_result = {}
    for _ in range(num_model_initializations):
        print(f"Split number: {exp}, Model number: {_}")
        model_name =  f"{model_name_suffix}_split{exp}_init_{_}" if options.get("save_all_models", False) else model_name_suffix
        this_acc, preds = train_model(X_train, Y_train, X_test, Y_test, options=options["training_options"], model_name=model_name)
        this_result[_] = {}
        this_result[_]["train_ids"] = train_data["id"].values.tolist()
        this_result[_]["test_ids"] = test_data["id"].values.tolist()
        this_result[_]["preds"] = preds.tolist()
        this_result[_]["truth"] = Y_test.tolist()
        this_result[_]["acc"] = this_acc.tolist()
        
    all_results.append(this_result)

    with open(results_file, "w") as f:
        json.dump(all_results, f)

    print(f"Updated Results File in: {results_file}")
    