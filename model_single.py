#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model definition and training for single trigger word detection.

This module defines the functions to create, train, save, load and evaluate a
RNN based model for real time detection of a single trigger word.

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
created and saved with the dataset module functions and that the training set
was saved to a single file.

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed
from tensorflow.keras.layers import LSTM, BatchNormalization
import tensorflow.keras.backend as K
import os
import time
import sys

MAX_EPOCHS = 200
BATCH_SIZE = 32
SAVE_EP = 2


def _custom_loss(y_true, y_pred):
    """Computes loss for single trigger word detection model.
    
    The loss is the sum over samples and timesteps of the binary cross entropy
    between target and prediction.
    
    The only variation is that values of target lower than -0.001 are
    interpreted as a don't care values. Where don't care value are present
    the loss is forced to zero.
    
    Args:
        y_true (keras.backend.Tensor): target (shape = (#samples, #timesteps, 1))
        y_pred (keras.backend.Tensor): predictions to match against the target,
            same shape of y_true
        
    Returns:
        keras.backend.Tensor: Scalar cost.
    """
    # COMPUTING BINARY CROSS-ENTROPY
    one = K.ones(K.shape(y_true))
    loss = -y_true*K.log(tf.clip_by_value(y_pred,1e-10,1.0))-(one-y_true)*K.log(tf.clip_by_value(one-y_pred,1e-10,1.0))
    
    #SETTING TO ZERO WHERE TARGET IS "DON't CARE"
    thres = tf.fill(K.shape(y_true),-0.001)
    fil = tf.cast(K.greater(y_true, thres), tf.float32)
    loss_filtered = loss*fil
    
    #SUMMING OVER TIMESTEP AND SAMPLES
    cost = K.sum(loss_filtered)
    return cost


def create_model(model_dir):
    """
    Creates and compiles a keras LSTM based model for single trigger word detection.
    
    Also saves the model architecture to json file.
    
    Args:
        model_dir (str): The folder where to save model.json.
        
    Returns:
        keras.models.Model: the model
    """
    
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, batch_input_shape=(None,None,129), name='lstm1')) #for sake of live prediction it's important to pass batch_input_shape with timesteps dimension set to none
    model.add(Dropout(0.5, name='do1'))
    model.add(BatchNormalization(name='bn1'))
    
    model.add(LSTM(units=128, return_sequences=True, name='lstm2'))
    model.add(Dropout(0.5, name='do2'))
    model.add(BatchNormalization(name='bn2'))
    
    model.add(LSTM(units=256, return_sequences=True, name='lstm3'))
    model.add(Dropout(0.5, name='do3'))
    model.add(BatchNormalization(name='bn3'))

    model.add(TimeDistributed(Dense(1, activation="sigmoid", name='dense'), name='td'))
    
    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=None)
    model.compile(optimizer='rmsprop', loss=_custom_loss, metrics=None)
    
    with open(os.path.join(model_dir,"model.json"), "w") as f:
        f.write(model.to_json())
    
    return model


def train_more(model, model_dir, X_train, Y_train, X_dev, Y_dev, ep_start, more_ep, save_ep = 2, batch_size=16, h=None):
    """Performs a training round.
    
    Training history is updated. Model weights are periodically saved to file.
    In the end model summary is printed and training history plotted.
    
    Args:
        model (keras.models.Model): The model to train.
        model_dir (str): The folder where to save weights and history.
        X_train (numpy.ndarray): The trainig set samples.
        Y_train (numpy.ndarray): The trainig set targets.
        X_dev (numpy.ndarray): The validation set samples.
        Y_dev (numpy.ndarray): The validation set targets.
        ep_start (int): The last epoch trained, 0 if model is brand new.
        more_ep (int): The number of additional epoch to train.
        save_ep (int): The frequency for saving weights and history to file.
        batch_size (int): The number of samples of training batches.
        h (util_model.TrainingHistory): The training history of the model.
    """
    ep = ep_start
    for i in range(0,more_ep,save_ep):
        his = model.fit(X_train, Y_train, batch_size = batch_size, epochs=ep+save_ep, validation_data=(X_dev, Y_dev), shuffle=False, initial_epoch=ep)
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
    ds_tra, ds_dev, _ = ds.load_dataset(dataset_dir)
    X_train, Y_train = ds_tra
    X_dev, Y_dev = ds_dev
    
    print("Training model")
    console_ui = um.TrainingController()
    console_ui.start()
    ep = 0
    h = um.TrainingHistory(model_dir)
    for i in range(MAX_EPOCHS+1):
        h.extend(model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs=ep+1, validation_data=(X_dev, Y_dev), shuffle=False, initial_epoch=ep))
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
dataset_dir = "datasets/activation/1"
ds_tra, ds_dev, _ = ds.load_dataset(dataset_dir)
X_train, Y_train = ds_tra
X_dev, Y_dev = ds_dev

#%% CREATE THE MODEL
model_dir = 'models/model_trigger_2'
create_model(model_dir)
h = um.TrainingHistory(model_dir)
ep = 0
#%% or LOAD A PREVIOUS MODEL TO RESUME TRAINING
model_dir = 'models/model_trigger_2'
ep = 20 #resume training from the epoch 20
um.load_model(model_dir, ep)
h = um.TrainingHistory(model_dir)

#%% TRAIN THE MODEL
more_ep=4
train_more(model, model_dir, X_train, Y_train, X_dev, Y_dev, ep, more_ep, save_ep = 2, batch_size=16, h=h)
ep+=more_ep
"""

