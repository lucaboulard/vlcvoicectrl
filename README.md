# vlcvoicectrl
RNN based voice control for VLC player (using python3, keras and tensorflow)

DEMO VIDEO: https://youtu.be/-tk_nfVBiBM

This is a just-for-fun project I created to practice with LSTM recurrent neural nets. It is about trigger word detection task and inspired by DeepLearning.ai Sequence Model course on Coursera.

Its aim is to create a python application listening to a microphone and allowing the user to send to a separate VLC player instance, a limited number of vocal commands, such as "play", "pause", "close"...

Due to limitation in time, computational power and dataset availability, the result as published in this repository is not going to work out of the box for you. I included the trained weights for the two LSTM models, but they were obtained by training on small datasets created from recordings of my voice only. So basically they works just for me and still my user experience is quite inconsistent and annoying with many false negatives.
Actually this project is just a first step, should be extended and optimized which I doubt I'll have the time to. So I publish it in the hope it cab be useful to somebody else. The code is extensively documented and as long as you have a background in RNN you should be able to make sense of it.

In order to use you need to clone the repo and install the required dependencies. Then you have to create your own dataset. After that you train over the dataset the two LSTM models. Finally you can evaluate your models and load them into the final application that you can use to control VLC.

All this steps are described in the remaining of this document.

## Installation
I created this project under ubuntu, anyway it should be portable to Windows with minor or no changes. I don't know about MacOS portability.

In your devenv you need to have the following dependencies installed:
* Applications:
  * ffmpeg --> to decode multimedia and audio files
  * mediainfo --> to get information about media files such as length and number of audio streams
  * VLC player --> in order to control it...
* Python & c:
  * python 3 --> choose a python version suitable for tensor flow
  * tensor flow --> follow the official installation instruction. The code uses its embedded keras api so no need to install keras as well. I used a prebuilt CPU only version. There is no code for training on GPU: add it if you want.
  * pyaudio --> to record from microphone. It depends on portaudio library. On ubuntu I had to compile both from source in order to have the soundcard recognized. Anyway it's a quite straightforward process; just follow the official instructions.
  * numpy, matplotlib, ... --> the basic stuff included in anaconda. If you don't use anaconda to set up python devenvs, you could consider switching to it.
  * pydub --> to handle audio clips. It leverages ffmpeg.
  * ... --> this is a partial list, only focusing on main dependencies. As you use the code you'll notice that some other modules are missing: just pip install them when you find them.

## Dataset creation
The first stage is to create your own datasets. There are two models to train so two datasets.
Indeed the detection process take place in two stages:
* the detection of "vlc" trigger word
* the detection of one command to send to the player.

So the first dataset (and model) is about detecting a single trigger word ("vlc"), while the second is about multiple trigger words.

### About IPython cells
I wrote this project using Spyder IDE, which lets you run the code in and IPython console and allows you to define cells in your code, just as you do in jupyter notebook. Cells are separated by #%% and can be run independently. I find this a very useful features and used them in some files. In particular:
* create_dataset.py is a very short, pure cell based file. You can use it with spyder, or just copy one cell at a time, modify and run in your own environment.
* model\_single.py and model\_multi.py are about model creation and training. They can be run as standard python scripts and they creates a model and trains it for 200 epochs unless you stop it (refers to scripts documentation). Still they have cells, currently commented out, allowing you to do the same operations in an IPython environment. The two training procedures are equivalent, use the one that better suits you.

### Trigger word audio samples
Datasets are created from audio files to you have to gather audio clips first.
* Using raw-dataset/create-folders.sh create one folder for each trigger word. These are:
  * vlc --> the main trigger word
  * not-vlc --> negative samples
  * command trigger words --> "back", "close", "fullscreen", "next", "pause", "play", "prev", "volume-down", "volume-up".
* Using a sound recorder app and your microphone, record short audio clips of you pronouncing one trigger word at a time.
  * Try to avoid heading and training silences in the audio clips as well as noise (eg. mouse clicks).
  * Record at least 10-20 audio clips for each trigger word, varying voice tone.
  * Put the recordings in the respective folders.
  * For the not-vlc folder record a bunch of random words, still one per audio clips, which are not triggers of this up. Eg. you can use "bread", "milk", "car", "dog" and so on.

### Background audio samples
Triggers are not enough, you need background audio too. Since you want to control the player while playing, the best background audios are similar to the content you usually play with VLC.
Assuming you own a bunch of multimedia files, such as movies, you can extract audio sequences from them.
The first cell in create_dataset.py does just that: extract 100 sequences of 30 seconds from your multimedia file gallery and saves them as separated audio files.
It is a good idea to add to these, some other sequences of silence (actually background noise) recorded with your microphone (no need to make them exactely 30 seconds long).
Put all this file in a single location, such as raw-dataset/backgrounds.

### Create the datasets
The last two cells in create\_dataset.py show you how to create the two datasets from the audio clips you previously gathered. You may want to read the documentation of ds.create_dataset().
Dataset are saved to several files and can grow quite big (10K samples --> 12 GB).

## Model trainining
You have to create two models one for single trigger word detection ("vlc") and the other for multiple trigger words (the various commands to the player). Models are created and trained with model\_single.py and model\_multi.py. Please take a look at these files and their documentation before running them.
You'll notice that model achitecture and weights get saved to file so that model can be reloaded at a later time.

About training itself I found that loss plateau in about 20 epochs, for both models. Depending on your dataset size and computational power you can train even longer.

In order to assess performances of the trained models you can use the two jupyter notebook: nodebook/model\_single\_test.ipynb and nodebook/model\_multi\_test.ipynb

## VLC Voice Control application
Once you are satisfied with your model performances, you can deploy them in the final application, which is contained in vlc_controller.py.

Read the documentation of this file, look at the main() and at the various class, each one implementing one thread of the application.

In the constructor method of Processor2 class you specify path and epoch of models to load.

### Preparing VLC player
VLC player can be controlled via a HTTP API that you have to enable and configure once for all.
* Launch VLC Player
* Tool > Preferences > All > Main interfaces --> Check "Web" checkbox and write “http” in the textfield.
* Tool > Preferences > All > Main interfaces > Lua --> Lua HTTP > Password --> set a password of your choice.
* In vlc_controller.py, write the same password in the constructor method of VlcManager class.
* The file vlc.sh allows you to run both VLC player and vlc_controller.py with a single command. It also allows you to specity host and port of VLC HTTP API. You have to match these values with those in the in the constructor method of VlcManager class.


## Next steps
This project shows an end-to-end two-stage trigger word application. It works, but performs very poorly, actually offering a frustrating user experience. Its usability is limited:
* to the user who recorded the audio clips for the training sets
* to the predefined set of commands defined in the training sets. If a command had to be added, a new dataset would be required and the whole training repeated.
Also, despite the prediction being made every 1/20 of second, the detection arrives quite late.

The latter problem can probably be faced without major changes, just working on prediction post processing in vlc\_controller.py and util\_test.py.

On the contrary, the former issues could only be tackled with a professional size database and with a different model architecture: it seems promising to me the idea of audio embeddings trained on a huge dataset of single spoken words. These audio embeddings could be used to match user words with available commands. Unfortunately the small relevance of a VLC player voice controller is not probably worth this huge effort.

Another promising feature would be to find a way to obtain the output audio signal going to the speaker and subtract it from the signal recorded with the microphone (maybe after applying some delay). This would leave a signal with almost only the spoken trigger words and no background. I expect that a model trained to recognize trigger words with minimal background noise could perform much better than the current one. Unfortunately there is no simple/straightforward/reliable/crossplatform way to get this audio signal in python.


