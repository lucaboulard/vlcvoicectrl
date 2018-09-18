#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utility functions for model definition, training, evaluating and deploying.

This module defines utility functions supporting the lifecycle of keras models.
"""

import numpy as np
import tensorflow.keras.backend as K
import os, os.path
import matplotlib.pyplot as plt
import threading
import json
from tensorflow.keras.models import model_from_json
import re
import h5py


class TrainingController(threading.Thread):
    """Thread providing a console based user interface for model training.
    
    This class provided means for the user to request to stop the training
    session so that the training process can interrupt after completing the 
    current epoch or similar.
    
    The user can issue two kind of requests:
        * pause/resume ('spacebar'): allow to pause and resume the training. This is intended
        to allow the user to review the training console log before proceeding.
        * stop ('q'): to end the trainig as soon as possible, giving back the python
        prompt.
        
    The training process has to reset the flag after handling a pause/resume
    request.
    
    After a receiving a stop request this thread gracefully exits.
    
    Since in python there is no simple crossplatform way to detect keyboard event,
    this impementation use input() function, so that the user is requested to
    press enter after the command key.
    """
    
    def __init__(self):
        """Initializes the BatchFeeder.
        """
        threading.Thread.__init__(self)
        self._lock = threading.RLock()
        with self._lock:
            self._pauseresume = False
            self._stop = False
    
    @property
    def pauseresume(self):
        """ Boolean: whether the user has issued a play/pause request.
        """
        with self._lock:
            return self._pauseresume
        
    @pauseresume.setter
    def pauseresume(self, value):
        with self._lock:
            self._pauseresume = value
            
    @property
    def stop(self):
        """ Boolean: whether the user has issued a stop request.
        """
        with self._lock:
            return self._stop
        
    @stop.setter
    def stop(self, value):
        with self._lock:
            self._stop = value
    
    
    def run(self):
        while(True):
            ch = input()
            if ch=='q':
                self.stop=True
                print('Requested stop')
                break
            elif ch==' ':
                self.pauseresume = True
                print('Requested pause/resume')
            else:
                print("Press a command key followed by enter.")
                print("Available commands:\n\tspacebar --> pause/resume")
                print("\tq --> stop")


class TrainingHistory():
    """Wrapper for loss and validation loss across training epochs.
    
    This class wraps two lists of floats: loss and validation loss, each having
    one element per training epoch. The class allows to save/load to file,
    to extend the lists with the results of successive calls to Model.fit and
    to plot the two curves.
    """
    
    def __init__(self, folder=None):
        """Initializes the TraininigHistory, optionally loading values from file.
        
        If a folder is provided, then the method will try to load data from 
        history.json file in that folder. This is the only way to load a previous
        TrainingHistory instance, as no dedicated load method is provided.
        If folder==None or if there is no history.json in it, this
        instance will be a brand new one with no data in it. The method extend
        allows to add data.
        
        Args:
            folder (str): the folder from which to load and where to save the 
                trining history data.
        """
        
        self._loss = []
        self._val_loss = []
        self._folder = folder
        if folder is not None:
            file_name = os.path.join(folder,"history.json")
            if os.path.isfile(file_name):
                with open(file_name, 'r') as infile:
                    self._loss, self._val_loss = json.load(infile)
    
    
    def save(self, folder=None):
        """Saves the TrainingHistory data to history.json file.
        
        Args:
            folder (str): the folder where to save the trining history data. It
                can be None if a previous folder was already provided either to 
                the constructor, or to a previous call to save().
        """
        
        if folder is not None:
            self._folder=folder
        if self._folder is None:
            raise NotADirectoryError("A directory must be specified either in __init__() or in save()")
        file_name = os.path.join(self._folder,"history.json")
        with open(file_name, 'w') as outfile:
            json.dump([self.loss, self.val_loss], outfile)
        
    def extend(self, his):
        """Extends the history with the outcome of a training round.
        
        Args:
            his (keras.callbacks.History): the object returned by keras.Model.fit()
        """
        self._loss.extend(his.history['loss'])
        self._val_loss.extend(his.history['val_loss'])
        
    
    @property
    def loss(self):
        """ List of float: The training loss across training epochs.
        """
        return self._loss
    
    @property
    def val_loss(self):
        """ List of float: The validation loss across training epochs.
        """
        return self._val_loss
    
    def plot(self):
        """ Plots training and validation loss across epochs.
        """
        plt.rcParams["figure.figsize"] =(12,6)
        plt.plot(self._loss, '-b')
        plt.plot(self._val_loss, '-r')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.grid()
        plt.show()


def load_model_for_live(model_dir, epoch_to_load):
    """Loads a model to be used for live predictions.
    
    The model is converted to stateful and the batch size set to one sample.
    The model should have been previously created with the timestep dimension of
    batch_input_shape (or of batch_shape) to None, so that predictions on a
    variable number of timesteps is supported.
    
    Args:
        model_dir (str): The folder where the model was saved.
        epoch_to_load (int): The epoch whose weights will be loaded in the model.
        
    Returns:
        keras.models.Model: The model, ready for use.
    """
    
    model_file = os.path.join(model_dir,"model.json")
    weight_file = os.path.join(model_dir,"weights_ep_" +str(epoch_to_load).zfill(3)+".h5")
    with open(model_file, 'r') as f:
        json_str = f.read()
        json_str = json_str.replace("\"stateful\": false","\"stateful\": true") # making model stateful
        json_str = re.sub("\"batch_input_shape\": \[[^,]+,","\"batch_input_shape\": [1,", json_str) #making batch of one single example
        model = model_from_json(json_str)
        model.load_weights(weight_file, by_name=True)
    return model


def load_model(model_dir, epoch_to_load):
    """Loads a model to be used for train/test.
    
    This method allow to evaluate and or resume the training of a previously
    created model.
    
    Args:
        model_dir (str): The folder where the model was saved.
        epoch_to_load (int): The epoch whose weights will be loaded in the model.
        
    Returns:
        keras.models.Model: The model, ready for use.
    """
    
    model_file = os.path.join(model_dir,"model.json")
    weight_file = os.path.join(model_dir,"weights_ep_" +str(epoch_to_load).zfill(3)+".h5")
    with open(model_file, 'r') as f:
        json_str = f.read()
        #json_str = json_str.replace("\"stateful\": false","\"stateful\": true") # making model stateful
        #json_str = re.sub("\"batch_input_shape\": \[[0-9]+,","\"batch_input_shape\": [1,", json_str) #making batch of one single example
        model = model_from_json(json_str)
        model.load_weights(weight_file, by_name=True)
    return model
        


def _str_shape(x):
    return 'x'.join(map(str, x.shape))


def load_weights(model, filepath, lookup={}, ignore=[], transform=None, verbose=True):
    """Modified version of keras load_weights that loads as much as it can.
    Useful for transfer learning.

    read the weights of layers stored in file and copy them to a model layer.
    the name of each layer is used to match the file's layers with the model's.
    It is possible to have layers in the model that dont appear in the file.

    The loading stops if a problem is encountered and the weights of the
    file layer that first caused the problem are returned.

    Args:
        model (keras.models.Model): The target.
        filepath (str): Source hdf5 file.
        lookup (dict): (optional) By default, the weights of each layer in the
            file are copied to the
            layer with the same name in the model. Using lookup you can replace
            the file name with a different model layer name, or to a list of
            model layer names, in which case the same weights will be copied
            to all layer models.
        ignore (list): (optional) The list of model layer names to ignore in
        transform (function): (optional) Function that receives the list of
            weights read from a layer in the file and filters them to the
            weights that will be loaded in the target model.
        verbose (bool): Flag. Highly recommended to keep this true and to
            follow the print messages.
    
    Returns:
        weights of the file layer which first caused the load to abort or None
        on successful load.
    """
    
    if verbose:
        print('Loading', filepath, 'to', model.name)
    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            if verbose:
                print(name)
            g = f[name]
            weight_names = [n.decode('utf8') for n in
                            g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in
                                 weight_names]
                if verbose:
                    print('loading', ' '.join(_str_shape(w) for w in weight_values))
                target_names = lookup.get(name, name)
                if isinstance(target_names, str):
                    target_names = [target_names]
                # handle the case were lookup asks to send the same weight to multiple layers
                target_names = [target_name for target_name in target_names if
                                target_name == name or target_name not in layer_names]
                for target_name in target_names:
                    if verbose:
                        print(target_name)
                    try:
                        layer = model.get_layer(name=target_name)
                    except:
                        layer = None
                    if layer:
                        # the same weight_values are copied to each of the target layers
                        symbolic_weights = layer.trainable_weights + layer.non_trainable_weights

                        if transform is not None:
                            transformed_weight_values = transform(weight_values, layer)
                            if transformed_weight_values is not None:
                                if verbose:
                                    print('(%d->%d)'%(len(weight_values),len(transformed_weight_values)))
                                weight_values = transformed_weight_values

                        problem = len(symbolic_weights) != len(weight_values)
                        if problem and verbose:
                            print('(bad #wgts)'),
                        if not problem:
                            weight_value_tuples += zip(symbolic_weights, weight_values)
                    else:
                        problem = True
                    if problem:
                        if verbose:
                            if name in ignore or ignore == '*':
                                print('(skipping)')
                            else:
                                print('ABORT')
                        if not (name in ignore or ignore == '*'):
                            K.batch_set_value(weight_value_tuples)
                            return [np.array(w) for w in weight_values]
                if verbose:
                    print()
            else:
                if verbose:
                    print('skipping this is empty file layer')
        K.batch_set_value(weight_value_tuples)