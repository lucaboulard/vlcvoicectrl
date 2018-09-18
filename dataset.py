# -*- coding: utf-8 -*-
"""Utility functions for dataset generation.

This module defines the functions for the generation of dataset that will be
used to train the ML models used by the VlcVoiceControl software.

Dataset generated with this module are intended to be used to address the
trigger word detection task with either single of or multiple trigger words.

The module assumes that the ML models to be trained are many to many
RNN models with #timestep_in = #timestep_out (T_x = T_y = T).

Furthermore it assumes that, for multiple trigger word detection, the number
of classes is predefined.

Mainly the module provides:
    * extract_backgrounds() --> generates a number of fixed lenght background
        audio sequences, picking at random position from a buch on multimedia
        files on disk. The generated sequences are used as input to the
        create_dataset() function.
    
    * create_dataset() --> creates the dataset and save it to disk.
    
    * load_dataset(), load_single_dataset(), load_dataset_metadata() and
         BatchFeeder() --> loads the previously created datasets.
    
Attributes:
    SAMPLE_LEN_S (int): Length in seconds of the sample audio clips of the dataset.
        Since the x sample is the spectrogram of the audio clip, the length
        number of timesteps T of x is almost proportional to SAMPLE_LEN_S.
        In case SAMPLE_LEN_S is modified, T has to be modified accordingly by
        calling util_audio.spectrogram() on an audio clip and checking the
        number of timesteps in the output (the first dimension).
    MAX_SNIPPETS_PER_SAMPLE (int): The maximum number of trigger word have in 
        each audio clip sample. In case SAMPLE_LEN_S is modified, also 
        MAX_SNIPPETS_PER_SAMPLE can be changed accordingly.
    T (int): Number of sample's timesteps (equal for input and output). It is
        determined by the first dimension of the output of util_audio.spectrogram()
        called on any audio clip of SAMPLE_LEN_S length. So it depends on
        SAMPLE_LEN_S and on the parameters passed to util_audio.spectrogram(),
        in particular nfft and noverlap.
        This is a key value as it determines the extension of required 
        backprop through time, when training the RNN so it should be kept within
        a reasonable range. T is almost proportional to SAMPLE_LEN_S and to
        noverlap parameter. It is almost inversly proportional to nfft parameter.

Todo:
    * Re implement create_dataset() to support the generation of datasets bigger
        than memory.
"""

#import json
import numpy as np
import random
import os
import os.path
from scipy.io import wavfile
import json

import math
import time
import threading
import tensorflow.keras.utils as KU

from util_audio import get_length_s, get_n_audio_streams, spectrogram
from util_audio import normalize_volume, load_audio_files, extract_audio_sequence
from util import print_memory_usage, print_sizeof_vars, list_file_recursive

SAMPLE_LEN_S = 6
MAX_SNIPPETS_PER_SAMPLE = 3
T = 1181

def extract_backgrounds(src_folder, dst_folder, n_seq, seq_len_s):
    """Extract random audio sequences from media files.
    
    Generates a number of fixed lenght background audio sequences, picking at
    random position from a buch on multimedia files on disk. The generated
    sequences are used as input to the create_dataset() function.
    The generated audio clips are saved as wav files with progressive names.
    Supported input multimedia file formats are .avi, .mp3, .mp4, .mkv, *.flac,
    *.wav.
    Files are decoded with ffmpeg, so ffmpeg is a dependency.
    
    For each sequence to be generate a random file is chosen, a random position
    is picked and, in case the file has multiple audio streams, one of them is
    picked at random.
    The audio is re-encoded at 44100Hz and in case it has multiple channels,
    only the first is retained.
    
    Args:
        src_folder (str): The name of the folder where the source media files are located.
        dst_folder (str): The folder where to save the audio sequences. Must not exist yet.
        n_seq (int): The total number of sequences to be extracted.
        seq_len_s (int): The length in seconds of each sequence.
    """
    
    assert os.path.isdir(src_folder)
    assert not os.path.isdir(dst_folder)
    os.makedirs(dst_folder)
    src_files = [f for f in list_file_recursive(src_folder) if f.endswith(".avi") or f.endswith(".mp3") or f.endswith(".mp4") or f.endswith(".mkv") or f.endswith(".flac") or f.endswith(".wav")]
    for i in range(n_seq):
        f = random.choice(src_files)
        f_len = get_length_s(f)
        n_streams = get_n_audio_streams(f)
        of = os.path.join(dst_folder,str(i).zfill(6)+".wav")
        if (f_len is None) or (n_streams is None):
            src_files.remove(f)
            print("Warning: failed to extract from ", f)
            if len(src_files)==0:
                raise OSError("Error: failed to all source files")
            i-=1
            continue
        stream_idx = n_streams if n_streams==0 else random.randrange(n_streams)
        start = random.randrange(int(f_len*0.95-seq_len_s))
        print(i, "of", n_seq, "|", f, "(", start ,"/" , f_len , ") ch", stream_idx)
        extract_audio_sequence(f,of,start,seq_len_s,stream_idx)


def _get_snippet_position(snip, bg, other_positions, max_attempt=50):
    """Tries to randoly pick a position for overlaing a snippet on a audio clip.
    
    The picked position is checked against all other previously reserved slices
    of the background audio clip. In case of ovrlap the position is discarded
    and the function picks another candidate.
    
    Args:
        snip (pydub.AudioSegment): The short audio clip to be overlaid on the background.
        bg (pydub.AudioSegment): The background audio clip.
        other_positions (list of lists two of int): The already reserved segments of background.
        max_attempt (int): The maximum number of attempt to get a position.
    
    Returns:
        list of two int: The start and end position in millisecond for the
            snippet or None if all the max_attempt candidates overlapped.
    """
    for i in range(max_attempt):
        start = random.randrange(SAMPLE_LEN_S*1000-len(snip)-100) #leaving empty ms to the end of sample to have room for setting Y to ones
        end = start + len(snip) -1
        # check for overlap
        overlap = False
        for prev_pos in other_positions:
            a = max(start, prev_pos[0])
            b = min(end, prev_pos[1])
            if a<=b:
                overlap = True
                break
        if not overlap:
            return [start, end]
    #print("KO")
    return None


def _create_audio_sample(backgrounds, snippets, is_neg_class = True, create_global_feat = False, pulse_len_ts = 50, dont_care=None):
    """Creates a single audio clip sample for the dataset and its corresponding target.
    
    Creates a single fixed length audio clip sample and its corresponding target y.
    The x matrix of the sample can be generated by computing the spectrogram of
    the audio clip, but this is not done within this function.
    This is a helper function used by create_dataset()
    
    The audio clip is created overlaing a random number of random trigger word
    snippets over a randomly selected slice of a radomly selected background.
    
    Args:
        backgrounds (list of pydub.AudioSegment): The background audio clips.
        snippets (list of lists of pydub.AudioSegment): The trigger word audio
            clips: one list per each trigger word class. A single word for
            each audio clip.
        is_neg_class (boolean): Same as create_dataset() param.
        create_global_feat (boolean): Same as create_dataset() param.
        pulse_len_ts (int): The number of timesteps to raise target to 1 after a
            trigger word snippet ends.
        dont_care (float): Ranging from 0 to 1. Set to zero or None to disable.
            If greater than zero, in the target and for each inserted snippet an
            interval of timesteps is determined, ending where the snippet end
            and having lenght corresponding to the snippet length (in timestep)
            multiplied by dont_care factor.
            This interval should be 0 valued in the target but its value is
            lowered to -0.0001 so that it can be distingushed from value 0.
            In particular, this can be used by the loss function to interpet
            those timesteps as don't cares having zero loss, whatever the 
            prediced values.
            For example dont_care= 0.3 converts the last 30% of timesteps of 
            each trigger word to don't cares and this, in conjuction with the
            proper custom loss, allows the model learn to detect the trigger word
            in advance, without having to wait it has finished.
            For multiple class, don't care mechanism is applied to each class
            feature and to the optional global feature too.
    
    Returns:
        pydub AudioSegment: The generated audio clip.
        numpy.ndarray: The target y of shape (1, T, ?)
            (the last dimension depends on is_neg_class and create_global_feat)
    """
    
    global counters
    # GETTING A RANDOM SELECTED SLICE OF RANDOM SELECTED BACKGROUND
    bg = random.choice(backgrounds)
    bg_pos = random.randrange(len(bg) -SAMPLE_LEN_S*1000)
    bg = bg[bg_pos:bg_pos+SAMPLE_LEN_S*1000]
    
    # DECIDING HOW MANY SNIPPET TO INSERT
    n_snippets = int(math.ceil(random.triangular(0,MAX_SNIPPETS_PER_SAMPLE,MAX_SNIPPETS_PER_SAMPLE)))
    positions = [] #list of position [start, stop] of snippets already inserted into the sample
    n_classes = len(snippets)
    
    #INTIALIZING THE TARGET
    n_pos_classes = n_classes if is_neg_class is False else n_classes-1
    is_multiclass = n_pos_classes>1
    create_global_feat = create_global_feat if is_multiclass else False
    n_feat = n_pos_classes if not create_global_feat else n_pos_classes+1
    y = np.zeros((1,T,n_feat),np.float32)
    delta = 0 # feature shift in target
    if is_neg_class:
        delta -=1
    if create_global_feat:
        delta +=1
    
    # INSERTING SNIPPETS
    #print(n_snippets)
    for i in range(n_snippets):
        #selecting one class and one snippet form it
        cla = random.randrange(len(snippets))
        #print("\t",i) if cla==0 else print("\t",i, "*")
        counters[cla]+=1
        idx = random.randrange(len(snippets[cla]))
        snip = snippets[cla][idx]
        pos = _get_snippet_position(snip, bg, positions)
        if pos is None: # failed to find room to insert the snippet, the sample is complete
            break
        #overlaying the snippet onto the background
        positions.append(pos)
        bg = bg.overlay(snip, pos[0])
        #setting the target
        snip_start = int(pos[0]*T/(1000*SAMPLE_LEN_S))
        snip_end = int(pos[1]*T/(1000*SAMPLE_LEN_S))
        snip_len = snip_end-snip_start
        if cla==0 and is_neg_class:
            continue
        y[0,snip_end+1:snip_end+pulse_len_ts+1,cla+delta] = 1
        if dont_care is not None:
            dc_len = int(snip_len*dont_care)
            if dc_len>0:
                y[0,snip_end+1-dc_len:snip_end+1,cla+delta] = -0.0001
        #TODO TODO
        if create_global_feat:
            y[0,snip_end+1:snip_end+pulse_len_ts+1,0] = 1
            if dont_care is not None and dc_len>0:
                y[0,snip_end+1-dc_len:snip_end+1,0] = -0.0001
    return normalize_volume(bg) , y


counters=None


def create_dataset(
                   background_dir,
                   class_dirs,
                   class_labels,
                   n_samples,
                   save_dir,
                   is_neg_class = True,
                   create_global_feat = False,
                   n_samples_per_training_split = None,
                   pulse_len_ts = 50,
                   dont_care=0.3
                   ):
    """Create a detaset for trigger word detection task.
    
    Create a dataset, shuffle it, split it and save it to disk.
    The destination folder must not exist yet otherwise the function fails
    (intermediate folders of the provided path are created if needed).
    
    The dataset is created as follows:
        * Background audio clips are loaded from disk.
        * For each class, representative audio clips are loaded from disk.
        * For each sample to generate:
            ** A fixed length background is randomly
                selected from the pool of background audio clips.
            ** A random number of trigger words audio clips are randomly selected
                and overlaid on the background at random positions.
            ** The resulting audio clip is saved to file for for future use
            ** The spectrogram of the audio clip is computed, being the sample x
            ** The corresponding target y is generated with the same number of 
                timesteps of x and as many features as the trigger word classes
                (excepted the optional negative class). y is initialized to 0
                and raised to 1 for 50 timesteps after end of an overlaid 
                trigger word, only in the feature corresponding to the trigger
                word class.
                If create_global_feat is True and if there are 2+ positive
                classes, an additional binary classification feature is created
                in the target (at index 0) being 1 when any of the other
                positive classes is 1. When there is just one positive class
                (and optionally a negative class), create_global_feat flag is
                ignored.
        * All the x and y are stacked in the X, Y dataset matrices.
        * The dataset is then splitted in training, development and test sets
            (70-15-15). Optionally, the training set can be further split in
            fixed size chunks (to save memory when training).
        * Each split is saved to npz file.
        * Class labels and other dataset metadata are saved to metadata.json.
    
    For multi-class classification (2+ non negative classes), the target is one-hot.
    The match of indices between class_labels and the target features may be
    broken: is_neg_class = True removes the first feature shifting all the
    others by -1, while create_global_feat = True adds a feature at index 0
    shifting all the others by +1. If both set they compensates each other...
    The target is 3 dimensional in any case (#samples, #timesteps, #features),
    even for single class when #features=1.
    
    The current implementation generates the whole dataset before splitting and
    saving, so that the dataset has to fit in memory, otherwise the process
    crashes. With the standard settings for timesteps it is possible to generate
    up to 10K samples for each 8GB of phisical memory.
    
    Args:
        background_dir (str): The directory containing background audio clips.
        class_dirs (list of str): The list of directories, one per class, containing
            the trigger words audio samples (negative class, if present, must
            be at 0 index). Each audio file must contain exactely one word.
        class_labels (list of str): The class labels (in the same order of class_dirs).
        n_samples (int): The total number of samples to generate.
        save_dir (str): The dataset destination folder.
        is_neg_class (boolean): If True, the fist class is the "negative class",
            which has no target feature.
        create_global_feat (boolean): 
        n_samples_per_training_split (int): If not None, the training set will
            be split in multiple files with n_samples_per_training_split
            samples each.
        pulse_len_ts (int): The length in timestep of the "one" pulse in 
            sample's target after the end of each trigger word instance.
        dont_care (float): switch and factor for "don't care" functionality.
            Cfr. docs of _create_audio_sample() for details.
    """
    
    global counters
    t0=time.time()
    #CREATING TARGET FOLDER
    print("CREATING TARGET FOLDER")
    if os.path.isdir(save_dir):
        raise FileExistsError('The target folder exists already. Please remove it manually and retry.')
    audio_samples_folder = os.path.join(save_dir,"audio_samples")
    os.makedirs(audio_samples_folder)
    print_memory_usage()
    
    #LOADING RAW AUDIO FILEs
    print("LOADING RAW AUDIO FILEs")
    backgrounds = load_audio_files(background_dir,dBFS_norm=None)
    n_classes = len(class_labels)
    assert len(class_dirs)==n_classes
    counters = np.zeros(n_classes)
    snippets = []
    for i in range(n_classes):
        snippets.append(load_audio_files(class_dirs[i]))
    print("#backgrounds:", len(backgrounds))
    print("#snippets:")
    [print("\t", lab, "-->", len(sni)) for sni,lab in zip(snippets, class_labels)]
    print_memory_usage()
    t1=time.time()
    print("Time:", int(t1-t0), "s")
    input("Press Enter to continue...")
    t1=time.time()
      
    #CREATING SAMPLES
    print("CREATING SAMPLES")
    m=n_samples
    
    def get_sample_shape(backgrounds,snippets,is_neg_class,create_global_feat):
        audio, y = _create_audio_sample(backgrounds,snippets,is_neg_class,create_global_feat)
        audio.set_frame_rate(44100).set_channels(1).export("tmp.wav", format="wav")
        rate, audio44khz = wavfile.read("tmp.wav")
        os.remove("tmp.wav")
        x=spectrogram(audio44khz)
        return x.shape, y.shape
        
    #x_list = []
    #y_list = []
    shax, shay = get_sample_shape(backgrounds,snippets,is_neg_class,create_global_feat)
    #print(shax,shay)
    X = np.empty((m,) + shax,dtype=np.float32)
    Y = np.empty((m,) + shay[1:],dtype=np.float32)
    #print("#X:", X.shape)
    #print("#Y:", Y.shape)
    #print_sizeof_vars(locals())
    #input("Press Enter to continue...")
    
    for i in range(m):
        if i%200==0:
            print("Creating sample ", i, " of ", m)
        #Create audio sample by overlay
        audio, y = _create_audio_sample(backgrounds,snippets,is_neg_class,create_global_feat,pulse_len_ts, dont_care)
        #Compute spectrogram of the audio sample
        filename = os.path.join(audio_samples_folder,str(i).zfill(6)+".wav")
        audio.set_frame_rate(44100).set_channels(1).export(filename, format="wav") # saving audio to file
        rate, audio44khz = wavfile.read(filename) # loading audio so to have 44100Hz freq
        #print(rate)
        assert abs(rate-44100)<1
        x=spectrogram(audio44khz)
        X[i]=x
        Y[i]=y
    del snippets, backgrounds
    #X=np.stack(x_list,axis=0)
    #Y=np.stack(y_list,axis=0)
    #del x_list, y_list
    
    print("#X:", X.shape)
    print("#Y:", Y.shape)
    print("class counters", counters)
    print_memory_usage()
    print_sizeof_vars(locals())
    t2=time.time()
    print("Time:", int(t2-t1), "s")
    input("Press Enter to continue...")
    t2=time.time()
    
    #SPLITTING THE DATASET
    # no use: it's already randomly generated
    print("SPLITTING")
    Xs, Ys, n_tr_samples = _split_dataset(X,Y,n_samples_per_training_split=n_samples_per_training_split)
    #print("Done")
    
    #SAVING THE DATASET
    print("SAVING THE DATASET")
    _save_dataset(Xs, Ys, save_dir)
    with open(os.path.join(save_dir,"metadata.json"), 'w') as metadata_file:
        json.dump({\
                  "class_dirs":class_dirs,
                  "background_dir":background_dir,
                  "class_labels":class_labels,
                  "n_classes":n_classes,
                  "n_samples":n_samples,
                  "is_neg_class": is_neg_class,
                  "create_global_feat": create_global_feat,
                  "n_samples_per_training_split": n_samples_per_training_split,
                  "n_training_samples": n_tr_samples,
                  "n_training_files": 0 if not isinstance(Xs[0],(list,)) else len(Xs[0])
                  }, metadata_file)
    t3=time.time()
    print("Time:", int(t3-t2), "s")


def _split_dataset(X,Y, train_perc = 0.7, dev_perc = 0.15, m_clip=5000, do_shuffle=False, n_samples_per_training_split = None):
    """Splits a dataset into training development and test sets.
    
    Optionally shuffle the dataset before splitting. The splits are sized
    according to the given percentages, but dev and test sets sizes are clipped
    to m_clip samples each.
    Samples are taken in the following order: dev, test, train. This information
    can be useful during error analysis to match the audio samples to the
    samples in each split.
    The training set can be further split in chunks of a predefined size.
    
    Args:
        X (numpy.ndarray): The X matrix.
        Y (numpy.ndarray): The Y matrix.
        train_perc (float): The percentage of data to be destined to training set [0-1].
        dev_perc (float): The percentage of data to be destined to dev set [0-1].
        m_clip (int): The maximum number of samples of dev and test sets.
        do_shuffle (boolean): Wheter to shuffle the dataset before splitting.
        n_samples_per_training_split (int): The number of samples per each
            chunk of training set or None not to split training set in chunks.
    
    Returns:
        list: X splits
        list: Y splits
        int: number of training samples
    """
    
    #SHUFFLING
    m = X.shape[0]
    
    if do_shuffle is True:
        shuffled = list(range(m))
        random.shuffle(shuffled)
        X_shuf = X[shuffled,:]
        Y_shuf = Y[shuffled,:]
    else:
        X_shuf = X
        Y_shuf = Y
        
    #SPLITTING'
    train_perc = 0.7
    dev_perc = 0.15
    
    test_perc = 1.0-train_perc-dev_perc
    dev_cum = int(min(dev_perc*m,10000))
    test_cum = int(dev_cum + min(test_perc*m, m_clip))
    
    X_dev = X_shuf[0:dev_cum]
    Y_dev = Y_shuf[0:dev_cum]
    X_test = X_shuf[dev_cum:test_cum]
    Y_test = Y_shuf[dev_cum:test_cum]
    X_train = X_shuf[test_cum:m]
    Y_train = Y_shuf[test_cum:m]
    
    X_tr_split = []
    Y_tr_split = []
    if n_samples_per_training_split is not None:
        n_splits = int(X_train.shape[0]/n_samples_per_training_split)
        n_splits = n_splits+1 if X_train.shape[0]%n_samples_per_training_split!=0 else n_splits
        for i in range(n_splits-1):
            X_tr_split.append(X_train[i*n_samples_per_training_split:(i+1)*n_samples_per_training_split])
            Y_tr_split.append(Y_train[i*n_samples_per_training_split:(i+1)*n_samples_per_training_split])
        X_tr_split.append(X_train[(n_splits-1)*n_samples_per_training_split:])
        Y_tr_split.append(Y_train[(n_splits-1)*n_samples_per_training_split:])
        return [[X_tr_split, X_dev, X_test],[Y_tr_split, Y_dev, Y_test],X_train.shape[0]]
    
    return [[X_train, X_dev, X_test],[Y_train, Y_dev, Y_test],X_train.shape[0]]


def _save_single_dataset(X, Y, filename):
        try:
                np.savez(filename, X=X, Y=Y)
        except Exception as e:
                print('Unable to save data to', filename, ':', e)


def _save_dataset(Xs, Ys, ds_folder):
    ofile_train = os.path.join(ds_folder,'train.npz')
    ofile_dev = os.path.join(ds_folder,'dev.npz')
    ofile_test = os.path.join(ds_folder,'test.npz')
    ofiles = [ofile_train, ofile_dev, ofile_test]
    for X, Y, f in zip(Xs, Ys, ofiles):
        if isinstance(X, (list,)):
            for i in range(len(X)):
                fi = f[:-4]+str(i).zfill(3)+".npz"
                print("X:", X[i].shape, "Y:", Y[i].shape, "-->", fi)
                _save_single_dataset(X[i],Y[i],fi)
        else:
            print("X:", X.shape, "Y:", Y.shape, "-->", f)
            _save_single_dataset(X,Y,f)


def load_single_dataset(filename):
    """Loads a partial dataset from a single npz file.
    
    Can be used lo load data from either train.npz or dev.npz or test.npz.
    The dataset is a list of X and Y, with:
        #X = (#samples, #timesteps, #feature_in)
        #Y = (#samples, #timesteps, #feature_out)
    Indeed in this project the number of input timesteps is equal to the 
    number of output timesteps.
    
    Args:
        filename (str): The name of the npz file.
    
    Returns:
        list of numpy.ndarray: [X, Y]
    """
    
    try:
        ds = np.load(filename)
        return [ds['X'],ds['Y']]
    except Exception as e:
        print('Unable to load data from', filename, ':', e)


def load_dataset(ds_folder):
    """Loads a dataset from 3 files: train.npz, dev.npz, test.npz.
    
    Suited to load dataset created with create_dataset() with
    n_samples_per_training_split=None.
    This functions loads the 3 splits of the dataset (train, dev, test)
    
    Args:
        ds_folder (str): The folder containing dataset files.

    Returns:
        list of lists of numpy.ndarray: [[X_train, Y_train], [X_dev, Y_dev], [X_test, Y_test]]
    """
    
    if not os.path.isdir(ds_folder):
        raise FileNotFoundError('folder does not exist')
    ifile_train = os.path.join(ds_folder,'train.npz')
    ifile_dev = os.path.join(ds_folder,'dev.npz')
    ifile_test = os.path.join(ds_folder,'test.npz')
    ifiles = [ifile_train, ifile_dev, ifile_test]
    datasets = []
    for f in ifiles:
        datasets.append(load_single_dataset(f))
    return datasets


def load_dataset_metadata(ds_folder):
    """Loads from file the metadata relative to a dataset.
    
    Metadata are loaded from metadata.json file in the dataset folder. They are
    a dictionary including the most relevant parameters passed to
    create_dataset() on dataset creation.
    
    Args:
        ds_folder (str): The folder containing dataset files.

    Returns:
        dict: The dataset metadata
    """
    with open(os.path.join(ds_folder,"metadata.json")) as f:
        data = json.load(f)
        return data


class BatchFeeder(KU.Sequence):
    """Batch generator for dataset spit on multiple files.

    This class is used to serve to a model batch of a large dataset. The
    dataset is usually too large to fir in memory, or can barely fit preventing
    the model to get the memory needed for training.
    Hence, such datasets are split in multiple files when generated. This class
    loads just one file at a time and provide batched of samples seamlessly.
    BatchFeeder is suited to work with training set generated by create_dataset()
    with n_samples_per_training_split parameter set.
    """
    
    
    def __init__(self, n_samples, n_files, n_sample_per_file, batch_size, ds_folder, base_file_name='train', no_target=False):
        """Initializes the BatchFeeder.
        
        Args:
            n_samples (str): The total number of samples in the dataset.
            n_files (str): The number of files the dataset is split in.
            n_sample_per_file (str): The number of samples per file (except the
                last file which can be smaller).
            batch_size (str): The number of samples per batch.
            ds_folder (str): The folder containg the dataset files.
            base_file_name (str): The base name in common to all the dataset
                files, that is the part before the numbering.
            no_target (str): Wheter the returned batch has to include the target
                too (for training), or just the samples (for prediction).
        """
        
        self.lock = threading.Lock()
        with self.lock:
            self.no_target = no_target
            self.batch_size = batch_size
            self.files = [os.path.join(ds_folder,f) for f in sorted(os.listdir(ds_folder)) if f.startswith(base_file_name)]
            #print(self.files)
            assert n_files==len(self.files)
            
            batch_per_file = int(n_sample_per_file/batch_size)
            batch_per_last_file = int((n_samples%n_sample_per_file)/batch_size)
            if batch_per_last_file==0:
                batch_per_last_file = batch_per_file
                
            self.n_batches = int((n_files-1)*batch_per_file+batch_per_last_file)
            
            self.lut = {}
            for i in range(self.n_batches):
                self.lut[i]=(int(i/batch_per_file),i%batch_per_file)
            #print(self.lut)

            self.curr_file_idx = -1
    
    
    def __len__(self):
        return self.n_batches
    
    
    def __getitem__(self, idx):
        with self.lock:
            #print("requested batch ", idx)
            f_idx, b_idx_f = self.lut[idx]
            if f_idx!=self.curr_file_idx:
                #print("loading file ", f_idx)
                self.x, self.y = load_single_dataset(self.files[f_idx])
                self.curr_file_idx=f_idx
            batch_x = self.x[b_idx_f*self.batch_size : (b_idx_f + 1)*self.batch_size]
            if self.no_target:
                return batch_x
            batch_y = self.y[b_idx_f*self.batch_size : (b_idx_f + 1)*self.batch_size]
            return [batch_x, batch_y]
    
    
    def get_target(self):
        """Gets the array of targets of all samples of the dataset (Y).
        
        Since targets are usually smaller than samples, they can usually fit in
        memory. This method loads all the targets from file and return them as
        an array.
    
        Returns:
            numpy.ndarray: The targets Y. #Y=(#samples, #timesteps, #features)
        """
        
        y_li = []
        with self.lock:
            for f in self.files:
                _ , y = load_single_dataset(f)
                last_idx = y.shape[0]-y.shape[0]%self.batch_size
                y_li.append(y[:last_idx])
            Y = np.concatenate(y_li,axis=0)
            return Y
        

