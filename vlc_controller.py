#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application for voice control of Vlc player.

This script is a multitheaded application for live detection of trigger words
aimed at voice controlling Vlc player.

There are several threads involved:
    * input from microphone thread (MicRecorder class)
    * live prediction thread (Processor2 class)
    * sending commands to HTTP interface of Vlc player (VlcManager class)
    * listening to console user input (get_input function)
    * additional theads to reprouce sounds and trigger words are detected
    
The class Processor is a work in progress version of Processor2, currently
unused.

Todo:
    * implement an additional thread to periodically check that Vlc player
        is still alive and exit the application if it has been closed.
    * in current implementation if the user close vlc by voice control, the
        vlcvoicecontrol ui thread keeps the vlcvoicecontrol the process alive
        waiting for user to press enter. This should be fixed.
"""

import threading
import queue
import time
import dataset as ds
import util_test as utt
import numpy as np
import pyaudio
from pydub import AudioSegment
from enum import Enum
from pydub.playback import play
import os
import psutil
import requests
import untangle
import util_model as um

exitFlag = False

class MicRecorder (threading.Thread):
    """Thread that continuously records audio chunks from microphone.
    
    Recorded chuncks are equeued into a synchronized buffer.
    """
    
    def __init__(self, buffer):
        """ Initializes a new MicRecorder instance.
        
        Args:
            buffer (queue.Queue): The buffer for audio chuncks.
        """
        threading.Thread.__init__(self)
        self.buffer = buffer

    def run(self):
        """Thread main method.
        """
        global exitFlag
        print("Starting MicRecorder")
        channels_depth = pyaudio.paInt16
        n_channels = 1
        frame_rate = 44100
        p = pyaudio.PyAudio() #init pyaudio library for audio recording
        assert p.get_default_host_api_info()['deviceCount']>0 #check that pyaudio has found some audio device
        stream = p.open(format=channels_depth,
            channels=n_channels,
            rate=frame_rate,
            input=True,
            frames_per_buffer=frame_rate) #open recording stream
        while True: #thread main loop
            try:
                chunk = stream.read(int(frame_rate/20))#get a chunk of audio data from the mic
            except IOError as e:
                print("MicRecorder: Error - there was an exception in acquiring data from microphone:" , e)
                if exitFlag is True:
                    break
                continue
            try:
                self.buffer.put_nowait(chunk)#put the chunk into the buffer
            except queue.Full as e:
                print("MicRecorder: Warning - the audio buffer is full. Discarding all frames.")
                try:
                    while True:
                        self.buffer.get_nowait()
                except queue.Empty as e:
                    pass
            if exitFlag is True:
                break
        print("Exiting MicRecorder")
        stream.stop_stream()
        stream.close()
        p.terminate()


class Processor(threading.Thread):
    """Work in progress prediction class that only limited to "vlc" trigger detection.
    
    Replaced by Processor2 class.
    """
    
    def __init__(self, buffer):
        """ Initializes a new Processor instance.
        
        Args:
            buffer (queue.Queue): The buffer to get audio chuncks.
        """
        threading.Thread.__init__(self)
        self.buffer = buffer
        self.model_folder = 'models/model_trigger_2'
        self.ep = 20
    
    def run(self):
        """Thread main method.
        """
        global exitFlag
        print("Starting Processor")
        counter = 0
        ts_counter=0
        #ts_factor=0
        framebuf = b''
        nfft = 256 # must match the value used in the spectrogram function
        noverlap = int(nfft/8) # must match the value used in the spectrogram function
        framelen = 2*1 #bytes, must match channels_depth and n_channels in the MicRecorder class
        predbuf = np.empty((1,0,1))
        self.model = utt.load_model_for_live(self.model_folder,self.ep) #model must be loaded in the same thread using it. if loaded in __init__(), then throws exception when used.
        #wf = wave.open("output.wav", 'wb')
        #wf.setnchannels(1)
        #wf.setsampwidth(2)
        #wf.setframerate(44100)
        while True:
            try:
                chunk = self.buffer.get(timeout=1)
            except queue.Empty as e:
                print("Processor: Warning - no data from microphone for a whole second")
                if exitFlag is True:
                    break
                continue
            try:
                framebuf=b"".join([framebuf,chunk])
                #wf.writeframes(chunk)
                n_win=int((len(framebuf)/framelen-nfft)/(nfft-noverlap))+1
                framearr = np.frombuffer(framebuf[:((n_win-1)*(nfft-noverlap)+nfft)*framelen], dtype=np.int16)
                x = np.expand_dims(ds.spectrogram(framearr),axis=0)
                ts_counter += x.shape[1]
                #if int(ts_counter/900)>ts_factor:
                    #ts_factor = int(ts_counter/900)
                    #self.model.reset_states()
                    #print("Resetting")
                #print(x.shape, n_win)
                #assert x.shape[1]==n_win
                framebuf = framebuf[framelen*n_win*(nfft-noverlap):]
                yp = self.model.predict(x, batch_size=1)
                predbuf = np.concatenate((predbuf,yp),axis=1)
                peaks = utt.find_peaks(predbuf[0])
                if len(peaks)>0:
                    print("!!! VLC !!!", counter)
                    counter+=1
                    new_start = min(peaks[-1]+75,predbuf.shape[1])
                    new_start = max(new_start,predbuf.shape[1]-30)
                    predbuf = predbuf[:,new_start:,:]
                    self.model.reset_states()
                else:
                    predbuf = predbuf[:,-30:,:] #should be equal or larger than the window size used in utt.find_peaks(), which is currently 20, so 30 is ok
            except Exception as e:
                print("Processor: Error - there was an exception while processing:" , e)
                exitFlag=True
                #raise e
            if exitFlag is True:
                break
        print("Exiting Processor")
        #wf.close()


class DetectionType(Enum):
    TRIGGER = 0
    COMMAND = 1

class Processor2 (threading.Thread):
    """ Thread that performs live detection of trigger words for Vlc player control.
    
    Detection is real time and continuous. The thread operates on raw audio
    data it gets from a buffer.
    Detection is in two stages:
        * vlc trigger: only "vlc" trigger word is detected. when detected, the 
            detection mode switch to the second stage (and a sound is played)
        * cmd trigger: the various command to vlc player are detected. When a
            command is detected, the command in inserted into a buffer, and a
            sound is played, and the detection mode is switched back to the
            first stage.
            If no command is detected, after some seconds a different sound is
            played and the the detection mode is switched back to the
            first stage.
    """
    
    def __init__(self, buffer, cmd_buffer):
        """ Initializes a new Processor instance.
        
        This method also loads the two RNN models for trigger word detection.
        Loading path are hardcoded and can be changed here to load a different
        models/epochs.
        
        Args:
            buffer (queue.Queue): The buffer to get audio chuncks.
            cmd_buffer (queue.Queue): The buffer to put detected commands.
        """
        threading.Thread.__init__(self)
        self.buffer = buffer
        self.cmd_buffer = cmd_buffer
        self.model_folder_tr = 'models/model_trigger_3' #the model for "vlc" trigger detection
        self.ep_tr = 12 #epoch for weight loading of the first model
        self.model_folder_cmd = 'models/model_command_1' #the model for multiple cmd trigger detection
        self.ep_cmd = 24 #epoch for weight loading of the second model
    
    def run(self):
        """Thread main method.
        """
        global exitFlag
        print("Starting Processor2")
        counter = 0
        ts_counter=0
        #ts_factor=0
        framebuf = b''
        nfft = 256 # must match the value used in the spectrogram function
        noverlap = int(nfft/8) # must match the value used in the spectrogram function
        framelen = 2*1 #bytes, must match channels_depth and n_channels in the MicRecorder class
        predbuf = np.empty((1,0,1))
        self.model_tr = um.load_model_for_live(self.model_folder_tr,self.ep_tr) #model_tr must be loaded in the same thread using it. if loaded in __init__(), then throws exception when used.
        self.model_cmd = um.load_model_for_live(self.model_folder_cmd,self.ep_cmd)
        #wf = wave.open("output.wav", 'wb')
        #wf.setnchannels(1)
        #wf.setsampwidth(2)
        #wf.setframerate(44100)
        T_cmd = int(1181/6*4) #number of timesteps for command detection
        cmd_counter = 0
        detection_type = DetectionType.TRIGGER
        class_labels = ["back", "close", "fullscreen", "next", "pause", "play", "prev", "volume-down", "volume-up"]
        n_class = len(class_labels)
        predbuf_cmd = np.empty((1,0,n_class+1))
        #self.model_tr._make_predict_function()
        while True:
            try:
                chunk = self.buffer.get(timeout=1)
            except queue.Empty as e:
                print("Processor2: Warning - no data from microphone for a whole second")
                if exitFlag is True:
                    break
                continue
            try:
                framebuf=b"".join([framebuf,chunk])
                #wf.writeframes(chunk)
                n_win=int((len(framebuf)/framelen-nfft)/(nfft-noverlap))+1
                framearr = np.frombuffer(framebuf[:((n_win-1)*(nfft-noverlap)+nfft)*framelen], dtype=np.int16)
                x = np.expand_dims(ds.spectrogram(framearr),axis=0)
                ts_counter += x.shape[1]
                #if int(ts_counter/900)>ts_factor:
                #    ts_factor = int(ts_counter/900)
                #    self.model_tr.reset_states()
                #    print("Resetting")
                #print(x.shape, n_win)
                #assert x.shape[1]==n_win
                framebuf = framebuf[framelen*n_win*(nfft-noverlap):]
                if detection_type == DetectionType.TRIGGER:
                    yp = self.model_tr.predict(x, batch_size=1)
                    predbuf = np.concatenate((predbuf,yp),axis=1)
                    peaks = utt.find_peaks(predbuf[0])
                    if len(peaks)>0:
                        threading.Thread(target=play_sound_1).start()
                        print("!!! VLC !!!", counter)
                        counter+=1
                        self.model_tr.reset_states()
                        predbuf = np.empty((1,0,1))
                        detection_type = DetectionType.COMMAND
                        cmd_counter = 0
                    else:
                        predbuf = predbuf[:,-30:,:] #should be equal or larger than the window size used in utt.find_peaks(), which is currently 20, so 30 is ok
                else:
                    yp = self.model_cmd.predict(x, batch_size=1)
                    predbuf_cmd = np.concatenate((predbuf_cmd,yp),axis=1)
                    cmd_counter += yp.shape[1]
                    peak, cla = utt.search_peak_multi(predbuf_cmd[0])
                    if cla is not None:
                        threading.Thread(target=play_sound_1).start()
                        print("!!!", class_labels[cla], "!!!", counter)
                        self.cmd_buffer.put_nowait(class_labels[cla])
                        counter+=1
                        self.model_cmd.reset_states()
                        detection_type = DetectionType.TRIGGER
                        predbuf_cmd = np.empty((1,0,n_class+1))
                    else:
                        predbuf_cmd = predbuf_cmd[:,-30:,:]
                    if cmd_counter>T_cmd:
                        self.model_cmd.reset_states()
                        detection_type = DetectionType.TRIGGER
                        predbuf_cmd = np.empty((1,0,n_class+1))
                        threading.Thread(target=play_sound_2).start()
                        print("Back to trigger mode")
            except Exception as e:
                print("Processor2: Error - there was an exception while processing:" , e)
                exitFlag=True
                raise e
            if exitFlag is True:
                break
        print("Exiting Processor2")
        #wf.close()


class VlcManager(threading.Thread):
    """Thread that read commands from a buffer and send them to HTTP interface of Vlc Player
    """
    
    def __init__(self, cmd_buffer):
        """Initializes a new VlcManager instance.
        
        Hardcoded in this method are the Vlc HTTP interface host, port and
        password. Modify them here if you need to change them.
        Also hardcoded is the dictionary mapping command labels to vlc http command.
        
        Args:
            buffer (queue.Queue): The buffer to get audio chuncks.
            cmd_buffer (queue.Queue): The buffer to put detected commands.
        """
        threading.Thread.__init__(self)
        self.buffer = cmd_buffer
        self.cmd_dict = {
                "back":       "pl_previous",
                "close":      "",
                "fullscreen": "fullscreen",
                "next":       "pl_next",
                "pause":      "pl_pause",
                "play":       "pl_play",
                "prev":       "pl_previous",
                "volume-down":"volume&val={}",
                "volume-up":  "volume&val={}"
                }
        self.base_url = "http://127.0.0.1:8091/requests/status.xml"
        print("Expected Vlc url:",self.base_url)
        self.pwd = "pa55word"
        print("Expected Vlc url:",self.pwd)
    
    
    def _get_volume(self):
        """Gets the current volume level of Vlc player.
        """
        r = requests.get(self.base_url,auth=('',self.pwd))
        xml = untangle.parse(r.text)
        return int(xml.root.volume.cdata)
    
    
    def _close_vlc(self):
        """Kills Vlc process.
        """
        if os.name == 'nt':
            PROCNAME = "vlc.exe"
        else:
            PROCNAME = "vlc"
        for proc in psutil.process_iter():
            if proc.name() == PROCNAME:
                proc.kill()
                gone, alive = psutil.wait_procs([proc], timeout=3)
                return len(alive)==0
        return True
    
    
    def run(self):
        """Thread main method.
        """
        global exitFlag
        print("Starting VlcManager")
        while True:
            try:
                cmd = self.buffer.get(timeout=1)
            except queue.Empty as e:
                if exitFlag is True:
                    break
                continue
            try:
                cmd_uri = self.cmd_dict[cmd]
                if cmd in ("back", "fullscreen", "next", "pause", "play", "prev"):
                    url = self.base_url + "?command=" + cmd_uri
                    r = requests.get(url,auth=('',self.pwd))
                    if r.status_code!=200:
                        print("VlcManager: warning - status code", r.status_code)
                elif cmd in ("volume-down","volume-up"):
                    if cmd=="volume-down":
                        url = url = self.base_url + "?command=" + cmd_uri.format("-200")
                    else:
                        url = url = self.base_url + "?command=" + cmd_uri.format("+200")
                    r = requests.get(url,auth=('',self.pwd))
                    if r.status_code!=200:
                        print("VlcManager: warning - status code", r.status_code)
                else: #close
                    res = self._close_vlc()
                    if res:
                        exitFlag=True
                    else:
                        print("VlcManager: warning - failed to close VLC")
            except Exception as e:
                print("Processor: Error - there was an exception while processing:" , e)
                exitFlag=True
                #raise e
            if exitFlag is True:
                break
        print("Exiting VlcManager")


def get_input():
    """Main function of the thread responsible for console UI of the application.
    
    Basically it just block waiting for user input. As soon as enter is pressed,
    the thread terminates causing the whole application to terminate.
    """
    res = input()
    print("User request to terminate the process:", res)


sound = AudioSegment.from_wav("sound.wav")

def play_sound_1():
    play(sound)

def play_sound_2():
    play(sound.reverse())

def main():
    """Main function.
    
    Starts all the threads and then waits for them to terminate.
    If even just one thread exits, the main thread set the exit flag to true,
    then waits for all other threads to terminate then returns.
    """
    global exitFlag
    print("VlcVoiceControl is running")
    buffer = queue.Queue(20)
    cmd_buffer = queue.Queue()
    recorder = MicRecorder(buffer)
    #processor = Processor(buffer)
    processor = Processor2(buffer,cmd_buffer)
    manager = VlcManager(cmd_buffer)
    exitor = threading.Thread(target=get_input)
    
    processor.start()
    recorder.start()
    exitor.start()
    manager.start()
    
    while True:
        time.sleep(0.5)
        if not (exitor.is_alive() and recorder.is_alive() and processor.is_alive() and manager.is_alive()):
            exitFlag=True
            break
    exitor.join()
    recorder.join()
    processor.join()
    manager.join()
    print("VlcVoiceControl is exiting")
    

if __name__== "__main__":
    main()





