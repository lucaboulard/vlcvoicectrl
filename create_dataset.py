# -*- coding: utf-8 -*-
"""Dataset creation script.

This script is included as an example of how to use the functions of
dataset module to generate your own datasets.
The code is organized into a sequence of independent IPython cells.

* The first cells extracts random background sequences from multimedia files to
    be used as background audio clip for dataset generation
* The second cell generates a dataset for single trigger word detection, being
    "vlc" the trigger word.
* The third cell generates a dataset for multiple trigger word detection,
    specifically some commands you may want to give to vlc player.
    
Each cell includes a call to shutil.rmtree() which removes a folder and all
its content. This line should be commented out, and only enabled if you want 
to overwrite a previous database with a new one. Alternatively, you could manually
erase the folder.
"""

import dataset as ds
import shutil
import os

#%%
#EXTRACT RAW BACKGROUND SEQUENCES FROM MOVIES
dst_folder = "raw-dataset/backgrounds/1"
#shutil.rmtree(dst_folder)
ds.extract_backgrounds("path_to_media_gallery", dst_folder, 100, 30)
#%%
#CREATE DATASET FOR SINGLE TRIGGER WORD DETECTION"
class_dirs = ["raw-dataset/words/not-vlc", "raw-dataset/words/vlc"]
class_labels = ["not-vlc", "vlc"]
dst_folder = "datasets/activation/2"
#shutil.rmtree(dst_folder)
ds.create_dataset("raw-dataset/backgrounds/1", class_dirs, class_labels, 6000, dst_folder)
#%%
#CREATE DATASET FOR COMMAND (MULTI TRIGGER WORD) DETECTION"
base_dir = "raw-dataset/words"
class_labels = ["not-vlc", "back", "close", "fullscreen", "next", "pause", "play", "prev", "volume-down", "volume-up"]
class_dirs = [os.path.join(base_dir,f) for f in class_labels]
dst_folder = "datasets/command/2"
shutil.rmtree(dst_folder)
ds.create_dataset("raw-dataset/backgrounds/1", class_dirs, class_labels, 10000, dst_folder, is_neg_class=True, create_global_feat=True, n_samples_per_training_split = 1024)

