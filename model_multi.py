#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model definition and training for multiple trigger word detection.

This module defines the functions to create, train, save, load and evaluate a
RNN based model for real time detection of a multiple trigger words.

Training is possible either via the main() method or by means of equivalent code,
currently commented, in the bottom of the file, organized in Ipython cells.
The latter way also allowa to reload a pretrained model and resume its training.

If using the main method to train, the following command line arguments has to
be passed:
    * dataset folder
    * model folder, which cannot exist already

Example:
    model_single "datasets/single/1" "models/single/1"

About the dataset: this module assumes that the dataset has been previously
created and saved with the dataset.create_dataset() functions with:
    * 2+ positive classes (+ optional negative class)
    * create_global_feat = True
    * n_samples_per_training_split != None so that the training set was split
        in multiple files.

Attributes:
    MAX_EPOCHS (int): the total number of training epochs if user does not stop
        trining in advance.
    BATCH_SIZE (int): number of samples per training size
    SAVE_EP (int): frequency for model weight saving during training. A value
        of n means that weights are saved once every n epochs.
"""

import dataset as ds
import util_model as um
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed
from tensorflow.keras.layers import LSTM, BatchNormalization, Input, Concatenate
import tensorflow.keras.backend as K
import os
import time
import sys

MAX_EPOCHS = 200
BATCH_SIZE = 32
SAVE_EP = 2


def _composite_loss(y_true, y_pred):
    """Computes loss for multi trigger word detection model.
    
    The target (both true and predicted) is twofold:
        * the first feature goes to one after any trigger word being
            pronounced.
        * the remaining features are one per trigger word class going to one only
            atfer a word of that specific class has been pronounced.
    
    This loss computes sums over samples and timesteps the sum of two terms:
        * The binary cross entropy of the first feature
        * The multiclass cross entropy of the remaining features, but only for
            where y_true is one, otherwise this term goes to 0 --> if no
            trigger word has just finished, it does not matter which class will
            be predicted...
    
    Args:
        y_true (keras.backend.Tensor): target (shape = (#samples, #timesteps, #positive_classes+1))
        y_pred (keras.backend.Tensor): predictions to match against the target,
            same shape of y_true
        
    Returns:
        keras.backend.Tensor: Scalar cost.
    """
    Ytb = y_true[:,:,:1] # first feature is the bynary classification target for the task "any cmd vs no cmd". b stands for binary
    Ytm = y_true[:,:,1:] # all other features are the one hot target for the task "which cmd". m stands for multiple
    Ypb = y_pred[:,:,:1]
    Ypm = y_pred[:,:,1:]

    # COMPUTING BINARY CROSS-ENTROPY
    one = K.ones(K.shape(Ytb))
    Lb = -Ytb*K.log(tf.clip_by_value(Ypb,1e-10,1.0))-(one-Ytb)*K.log(tf.clip_by_value(one-Ypb,1e-10,1.0)) # binary loss
    
    #SETTING BINARY CROSS-ENTROPY ZERO WHERE TARGET IS "DON't CARE"
    thres = tf.fill(K.shape(Lb),-0.001)
    fil = tf.cast(K.greater(Ytb, thres), tf.float32)
    Lb_fil = Lb*fil
    
    # COMPUTING MULTICLASS CROSS-ENTROPY
    parts = []
    for i in range(_n_classes):
        parts.append(-Ytm[:,:,i:i+1]*K.log(tf.clip_by_value(Ypm[:,:,i:i+1],1e-10,1.0)))
    Lm=tf.add_n(parts)
    
    Lmm = Lm*Ytb
    
    Cb = K.sum(Lb_fil)
    Cm = K.sum(Lmm)
    C = Cb+Cm
    
    return C
    #return [C, Cb, Cm, Lb, Lm, Lmm, y_true, y_pred] 


#%% TEST THE LOSS (change the return statement to get the extended results)
'''
_n_classes = 9
with tf.Session() as test:
    yt = np.array([[[0,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,1,0]],  [[0,0,0,0,0,0,0,1,0,0],[1,0,0,0,0,0,1,0,0,0]]])
    yp = np.array([[[0,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,.5,.5]],[[1,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0]]])
    YT= tf.convert_to_tensor(yt,dtype=tf.float32)
    YP= tf.convert_to_tensor(yp,dtype=tf.float32)
    C, Cb, Cm, Lb, Lm, Lmm, y_true, y_pred = _composite_loss(YT,YP)
    test.run(tf.global_variables_initializer())
    c, cb, cm, lb, lm, lmm, _yt, _yp = test.run([C, Cb, Cm, Lb, Lm, Lmm, y_true, y_pred])
    print(c, cb, cm)
    print('lb',lb)
    print('lm',lm)
    print('lmm',lmm)
'''
'''
#EXPECTED OUTPUT
#NOTE: this was the expected values before adding the don't care functionality.
#   If applying to a "dont' care" enabled dataset costs and lb will be slighly
    different.
46.744846 46.0517 0.6931472
lb [[[ 0.] [ 0.]] [[23.02585] [23.02585]]]
lm [[[ 0.] [ 0.6931472]] [[23.02585]  [0.]]]
lmm [[[0.] [ 0.6931472]] [[0.      ]  [0.]]]
'''
#%%

_n_classes=-1

def create_model(model_dir, n_classes=9, n_feat_out=129):
    """
    Creates and compiles a keras LSTM based model for multi trigger word detection.
    
    Also saves the model architecture to json file.
    
    Args:
        model_dir (str): The folder where to save model.json.
        
    Returns:
        keras.models.Model: the model
    """
    global _n_classes
    
    inputs = Input(batch_shape=(None,None,n_feat_out), name='input') #for sake of live prediction it's important to pass batch_input_shape with timesteps dimension set to none
    
    X = LSTM(units=64, return_sequences=True, name='lstm1')(inputs)
    X = Dropout(0.4, name='do1')(X)
    X = BatchNormalization(name='bn1')(X)
    
    X = LSTM(units=128, return_sequences=True, name='lstm2')(X)
    X = Dropout(0.4, name='do2')(X)
    X = BatchNormalization(name='bn2')(X)
    
    X = LSTM(units=256, return_sequences=True, name='lstm3')(X)
    X = Dropout(0.4, name='do3')(X)
    X = BatchNormalization(name='bn3')(X)
    
    Y_bin = TimeDistributed(Dense(1, activation="sigmoid", name='dense1'), name='td1')(X)
    Y_multi = TimeDistributed(Dense(9, activation="softmax", name='dense2'), name='td2')(X)
    
    Y=Concatenate(axis=2, name='concat')([Y_bin,Y_multi])
    
    model = Model(inputs=inputs, outputs=Y)
    
    _n_classes = n_classes # used by _composite_loss()
    model.compile(optimizer='rmsprop', loss=_composite_loss, metrics=None)
    
    with open(os.path.join(model_dir,"model.json"), "w") as f:
        f.write(model.to_json())
    
    return model


def train_more(model, model_dir, train_feeder, X_dev, Y_dev, ep_start, more_ep, save_ep = 2, batch_size=16, h=None):
    """Performs a training round.
    
    Training history is updated. Model weights are periodically saved to file.
    In the end model summary is printed and training history plotted.
    
    Args:
        model (keras.models.Model): The model to train.
        model_dir (str): The folder where to save weights and history.
        ep_start (int): The last epoch trained, 0 if model is brand new.
        more_ep (int): The number of additional epoch to train.
        save_ep (int): The frequency for saving weights and history to file.
        batch_size (int): The number of samples of training batches.
        h (util_model.TrainingHistory): The training history of the model.
    """
    ep = ep_start
    for i in range(0,more_ep,save_ep):
        his = model.fit_generator(train_feeder, epochs=ep+save_ep, validation_data=(X_dev, Y_dev), shuffle=False, initial_epoch=ep, use_multiprocessing=True, workers=1)
        if h is not None:
            h.extend(his)
        ep+=save_ep
        model.save(os.path.join(model_dir,"model_ep_" +str(ep).zfill(3)+".h5"))
        model.save_weights(os.path.join(model_dir,"weights_ep_" +str(ep).zfill(3)+".h5"))
        if h is not None:
            h.save(model_dir)
    model.summary()
    if h is not None:
        h.plot()


#SAVE/LOAD MODEL & WEIGHTS
#model.save('my_model.h5') #model & weights
#keras.models.load_model('my_model.h5') #re-create model, load weights and compile

#SAVE/LOAD MODEL    
#json_string = model.to_json() #only the model
#model = model_from_json(json_string) #re-create model, don't know if compile

#SAVE/LOAD WEIGHTS    
#model.save_weights('my_model_weights.h5')
#model.load_weights('my_model_weights.h5', by_name=True)


def main():
    """Creates and trains the model
    """
    
    if len(sys.argv)!=3:
        raise ValueError("Two paramers must be provided: dataset folder and model folder")
    
    print("Creating model")
    model_dir = sys.argv[2]
    if os.path.isdir(model_dir):
        raise FileExistsError('The target folder exists already. Please remove it manually and retry.')
    os.makedirs(model_dir)
    model = create_model(model_dir)
    #print(model.summary())
    with open(os.path.join(model_dir,"model.json"), "w") as f:
        f.write(model.to_json())
    
    print("Loading dataset")
    dataset_dir = sys.argv[1]
    meta = ds.load_dataset_metadata(dataset_dir)
    n_tr_samples = meta["n_training_samples"]
    n_tr_files = meta["n_training_files"]
    n_sample_per_file = meta["n_samples_per_training_split"]
    train_feeder = ds.BatchFeeder(n_samples=n_tr_samples, n_files=n_tr_files, n_sample_per_file=n_sample_per_file, batch_size=BATCH_SIZE, ds_folder=dataset_dir, base_file_name='train')
    X_dev, Y_dev = ds.load_single_dataset(os.path.join(dataset_dir,"dev.npz"))
    
    print("Training model")
    console_ui = um.TrainingController()
    console_ui.start()
    ep = 0
    h = um.TrainingHistory(model_dir)
    for i in range(MAX_EPOCHS+1):
        h.extend(model.fit_generator(train_feeder, epochs=ep+1, validation_data=(X_dev, Y_dev), shuffle=False, initial_epoch=ep, use_multiprocessing=True, workers=1))
        if ep%SAVE_EP==0 or console_ui.stop:
            model.save(os.path.join(model_dir,"model_ep_" +str(ep).zfill(3)+".h5"))
            model.save_weights(os.path.join(model_dir,"weights_ep_" +str(ep).zfill(3)+".h5"))
            h.save()
        ep+=1
        if console_ui.pauseresume or console_ui.stop:
            model.summary()
            h.plot()
            console_ui.pauseresume = False
            while not console_ui.pauseresume and not console_ui.stop:
                time.sleep(1)
            if console_ui.pauseresume:
                console_ui.pauseresume = False
            else:
                print("Exitining training process")
                break
    if i>=MAX_EPOCHS:
        model.summary()
        h.plot()
        print("Training process completed")


if __name__== "__main__":
    main()

"""
#%% IPYTHON CELL BASED CODE FOLLOWS FOR MODEL TRAINING. THIS IS ALTERNATIVE TO THE MAIN METHOD

#LOAD THE DATASET
dataset_dir = "datasets/command/2"
meta = ds.load_dataset_metadata(dataset_dir)
n_tr_samples = meta["n_training_samples"]
n_tr_files = meta["n_training_files"]
n_sample_per_file = meta["n_samples_per_training_split"]
train_feeder = ds.BatchFeeder(n_samples=n_tr_samples, n_files=n_tr_files, n_sample_per_file=n_sample_per_file, batch_size=32, ds_folder=dataset_dir, base_file_name='train')
X_dev, Y_dev = ds.load_single_dataset(os.path.join(dataset_dir,"dev.npz"))

#%% CREATE THE MODEL
model_dir = 'models/model_command_3'
if os.path.isdir(model_dir):
    print('The target folder exists already. Please remove it manually and retry.')
else:
    os.makedirs(model_dir)
    model = create_model(model_dir)
    h = um.TrainingHistory(model_dir)
    ep = 0
#%% or LOAD A PREVIOUS MODEL TO RESUME TRAINING
model_dir = 'models/model_command_1'
ep = 20 #resume training from the epoch 20
um.load_model(model_dir, ep)
h = um.TrainingHistory(model_dir)
#%% TRAIN THE MODEL
more_ep=4
train_more(model, model_dir, train_feeder, X_dev, Y_dev, ep, more_ep, save_ep = 2, batch_size=32, h=h)
ep+=more_ep
"""


