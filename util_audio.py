# -*- coding: utf-8 -*-
"""Utility functions for audio clip manipulation and filtering.

This module defines functions for audio manipulation that are used throughout
the project.
"""

import subprocess
import numpy as np
import matplotlib.mlab as mlab
import util
from pydub import AudioSegment
import os


def get_length_s(filename):
    """Gets the length in seconds of a media file.
    
    Needs mediainfo installed.
    
    Args:
        filename (str): The name of the media file.

    Returns:
        int: The length in seconds of the media file
    """
    
    res = subprocess.Popen(["mediainfo", filename],stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    s = res.stdout.readlines()
    durations = [str(x) for x in s if 'Duration' in str(x)]
    if durations is None or len(durations)==0:
        return None
    d = durations[0].replace(" ","").replace("\t","").split(":")[1]
    h=0
    if "h" in d:
        h = int(d.split("h")[0])
        d = d.split("h")[1]
    m=0
    if "min" in d:
        m = int(d.split("min")[0])
        d = d.split("min")[1]
    s=0
    if "s" in d:
        s = int(d.split("s")[0])
    return (h*60+m)*60+s


def get_n_audio_streams(filename):
    """Gets the number of audio streams in a media file.
    
    Needs mediainfo installed.
    
    Args:
        filename (str): The name of the media file.

    Returns:
        int: The number of audio streams in the media file.
    """
    
    res = subprocess.Popen(["mediainfo", filename],stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    s = res.stdout.readlines()
    audio_streams = [str(x) for x in s  if 'Audio #' in str(x)]
    if audio_streams is None:
        return None
    return len(audio_streams)


#With default params, for a 6 sec audio, the spectrogram has shape (1181,129):
#1181 timesteps and 129 frequency values.
def spectrogram(audio, nfft=256, fs=8000, noverlap=32):
    """Computes the spectrogram of an audio sample.
    
    Computes the spectrogram of an audio sample using the power spectral
    density. The scale used for frequency axis is dB power (10 * log10).
    In this project spectrograms are used as input x to recurrent neural nets.
    
    Args:
        audio (numpy.ndarray): The audio sample data as 1D array. If 2D, only
            the first channel is processed (ie audio[0]).
        nfft (int): Lenght of FFT window for the computation of a spectrogram
            timestep. Better be a power of 2 for efficiency reasons.
        fs (int): The sampling frequency (samples per time unit).
        noverlap (int): Overlap between adjacent windows.

    Returns:
        numpy.ndarray: The spectrogram. It is 2D: first dimension is time
            (timesteps), the second one is the frequency.
    """
    
    data = audio if audio.ndim==1 else audio[:,0] # working on the first channel only  
    spec, f, t = mlab.specgram(data, nfft, fs, noverlap = noverlap)
    return spec.T.astype(np.float32)


def normalize_volume(audio, dBFS=-15):
    """Normalizes the volume of an audio clip.
    
    Normalizes the amplitude of and audio clip to match a specified value of
    decibels relative to full scale.
    
    Args:
        audio (pydub AudioSegment): The audio clip to be normalized.
        dBFS (int): the desired decibel of ratio between root mean squared
            amplitude of the audio and the max possible amplitude.
    
    Returns:
        pydub.AudioSegment: The normalized audio clip
    """
    
    delta = dBFS - audio.dBFS
    return audio.apply_gain(delta)


#empirically -27 threshold works well for audio clip normalized at -15 with normalize_volume()
def detect_leading_silence(audio, silence_threshold=-27.0, chunk_size=10):
    '''Detects leading silence in an audio clip.
    
    Iterates chunck by chunck from the beginning of the audioclip, 
    until finding the first one with volume above threshold.
    
    Args:
        audio (pydub AudioSegment): The audio clip.
        silence_threshold (int): the volume threshold in dB wrt full scale
            defining the silence.
        chunk_size (int): lenght in ms of audio chuncks tested for silence.
        
    Returns:
        int: the millisecond since the beginning of audio clip where initial
        silence ends
    '''
    
    trim_ms = 0 # ms
    chunk_size = int(max(chunk_size,1))
    while True:
        #print(sound[trim_ms:trim_ms+chunk_size].dBFS)
        if audio[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(audio):
            trim_ms += chunk_size
        else:
            break
    return trim_ms


def trim_audio_sample(audio, silence_threshold=-27.0):
    """Trims an audio clip removing silences at the beginning and at the end.
    
    Args:
        audio (pydub AudioSegment): The audio clip.
        silence_threshold (int): the volume threshold in dB wrt full scale
            defining the silence.
    
    Returns:
        pydub.AudioSegment: trimmed audio clip
    """
    start_trim = detect_leading_silence(audio,silence_threshold)
    end_trim = detect_leading_silence(audio.reverse(),silence_threshold)
    duration = len(audio)
    #print(start_trim, end_trim, duration)
    trimmed_audio = audio[start_trim:duration-end_trim]
    return trimmed_audio


def load_audio_files(folder_name, dBFS_norm=-15, dBFS_min=-28, dBFS_max=-15, clip_start_ms=0, clip_end_ms=0):
    """Load audio all audio files from a folder, recursively.
    
    The entire folder subtree is explored and all audio files are loaded as
    pydub.AudioSegment.
    Actually this functions tries to load all files, so that basically each
    ffmpeg supported audio file type will be correctly loaded.
    As a consequence of trying to load all files, if non audio files are in
    the folder subtree, for each an error will be logged to console. This does
    not compromise the success of the function.
    A number of filters is applied to each audioclip:
        - If frame rate is different from 44100Hz, the audioclip is resampled at
            this frequency.
        - Volume normalization.
        - Optional removal a fixed lenght head and tail.
        - Automated trimming of heading and trailing silence.
    
    Args:
        folder_name (str): Folder to search for audio files.
        dBFS_norm (int): Full scale dB level for volume normalization or None
            not to normalize.
        dBFS_min (int): Ignored if dBFS_norm is not None. Else, if the volume
            level is lower than this threshold, it is amplified to this level.
        dBFS_max (int): Ignored if dBFS_norm is not None. Else, if the volume
            level is higher than this threshold, it is normalized to this level.
        clip_start_ms (int): length in ms of chunck to be removed from audioclips head.
        clip_end_ms (int): length in ms of chunck to be removed from audioclips tail.
    
    Returns:
        list of pydub.AudioSegment: loaded audio clips
    """
    if not os.path.isdir(folder_name):
        raise FileNotFoundError('folder does not exist')
    segments = []
    clip_start_ms = max(int(clip_start_ms),0)
    clip_end_ms = max(int(clip_end_ms),0)
    for f in util.list_file_recursive(folder_name):
        #if f.endswith(".wav") or f.endswith(".mp3") or f.endswith(".flac") or f.endswith(".ogg"):
        try:
            s = AudioSegment.from_file(f)
            s = s.set_frame_rate(44100)
            if dBFS_norm is not None:
                s = normalize_volume(s,dBFS_norm)
            elif dBFS_min is not None and s.dBFS < dBFS_min:
                s = normalize_volume(s,dBFS_min)
            elif dBFS_max is not None and s.dBFS > dBFS_max:
                s = normalize_volume(s,dBFS_max)
            if clip_start_ms==0 and clip_end_ms==0:
                pass
            elif clip_end_ms==0:
                s = s[clip_start_ms:]
            else:
                s = s[clip_start_ms:-clip_end_ms]
            s = trim_audio_sample(s,silence_threshold=-27)
            segments.append(s)
        except Exception as e:
            print("Info: failed to load file: " + f)
            raise e
    return segments


def extract_audio_sequence(infile, outfile, start, length_s, audio_stream_idx=0):
    """Extracts from a media file a fixed length audio clip at specified position.
    
    The extracted audio clip is saved to file.
    The audio is re-encoded at 44100Hz and in case it has multiple channels,
    only the first is retained.
    If multiple audio streams are present, it's possible to select one of them
    by providing the 0-base index.
    
    This fuction depends on ffmpeg which must therfore be installed.
    
    Args:
        infile (str): The name of the input media file.
        outfile (str): The name of the output audio file.
        start (int): The start position in seconds of the audio clip.
        length_s (str): The length in seconds of the audio clip.
        audio_stream_idx (str): The zero based index of the audio stream.
    """
    
    cmd1 = "ffmpeg -y -i"
    cmd2 = " -ss {0} -t {1} -ab 192000 -vn -map 0:a:{2} -af asetrate=44100 -ac 1"
    cmd2_fmt = cmd2.format(str(start), str(length_s), str(audio_stream_idx))
    cmd_all = cmd1.split() + [infile] + cmd2_fmt.split() + [outfile]
    #print(cmd_all)
    process = subprocess.Popen(cmd_all, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    output, error = process.communicate()
    #if output is not None:
    #    print(output)
    #if error is not None:
    #    print(error)