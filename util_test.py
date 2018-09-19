#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for the evaluation of trigger word detection models.

This module contains fuctions to evaluate the performances of models created
with modules model_single and model_multi.

Todo:
    * Implement statistics about peak shift and predicted peak length in
        compute_global_metrics_multi() similar to those in
        compute_global_metrics_single().
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import IPython
import os.path
import math
from scipy.io import wavfile
from util_audio import spectrogram

def evaluate_plot_single(Y, Yp, n_plot=10, ds_folder=None, delta=0, idx_list=None):
    """Plots prediction vs target for randomly selected samples (for the single trigger work detection case).
    
    Each plot represent a sample. Timesteps are on x axis, while y axis range
    from 0 to 1. Two curves are displayed: prediction and ground truth target.
    Sample to plot are randomly selected from the provided sets Y ad Yp.
    
    If called in an IPython console/notebook, it also provides a player for the
    audio clip corresponding to each sample.
    
    The sample indices printed and plotted always refers to the order of samples
    in Y (and Yp), despite delta being different from zero and/or idx_list being
    provided.
    
    Args:
        Y (numpy.ndarray): The set of ground truth target
            #Y = (#samples, #timesteps, #feat_out).
        Yp (numpy.ndarray): The set of predictions
            #Yp = (#samples, #timesteps, #feat_out).
        n_plot (int): The number of samples to plot.
        ds_folder (str): (Optional) The folder containing the dataset.
            If provided and if the method is called from an Ipython console,
            a player is displayed for each sample's audio clip.
        delta (int): The index of the first sample of Y and Yp within the
            whole dataset. For datasets created with "dataset" module this is:
                * 0 if working on the validation set
                * #devset_samples + #testset_samples if working on the training set
        idx_list (list of int): If provided, instead of random picking from all
            samples in Y, the choice is made among this subset of indices. This
            allows to perform error analysis, by providing indices of samples
            with prediction errors.
    """
    
    if idx_list is None:
        idx = random.sample(range(Y.shape[0]), n_plot)
    else:
        idx = random.sample(idx_list, n_plot)
    for i in idx:
        if ds_folder is not None:
            filename = os.path.join(ds_folder, "audio_samples", str(i+delta).zfill(6)+".wav")
            print("Sample:", i, filename)
            IPython.display.display(IPython.display.Audio(filename))
        else:
            print("Sample:", i)
        plt.plot(Y[i], '-b', label='ground truth')
        plt.plot(Yp[i], '-r', label='prediction')
        plt.legend(loc='upper right')
        plt.gca().set_title("Sample " + str(i))
        plt.gca().set_ylim([-.1,1.1])
        plt.show()
        plt.figure()


def evaluate_plot_multi(Y, Yp, n_plot=10, ds_folder=None, delta=0, idx_list=None):
    """Plots prediction vs target for randomly selected samples (for the multiple trigger work detection case).
    
    Each figure represent a sample and has 4 sub plots.
    Timesteps are on x axis, while y axis always range from 0 to 1.
    This is the contents of the sub plots:
        * top-left: prediction and ground truth of the global binary feature:
            "has any trigger word just finished being pronounced?"
        * top-right: prediction and ground truth of the feature corresponding
            the trigger word class having the greatest number of occurrences in
            the sample.
        * bottom-right: prediction and ground truth of the feature corresponding
            the trigger word class having the second greatest number of
            occurrences in the sample.
        * bottom-left: prediction of all remaninig class features.
    
    
    Two curves are displayed: prediction and ground truth target.
    Sample to plot are randomly selected from the provided sets Y ad Yp.
    
    If called in an IPython console/notebook, it also provides a player for the
    audio clip corresponding to each sample.
    
    The sample indices printed and plotted always refers to the order of samples
    in Y (and Yp), despite delta being different from zero and/or idx_list being
    provided.
    
    Args:
        Y (numpy.ndarray): The set of ground truth target
            #Y = (#samples, #timesteps, #feat_out).
        Yp (numpy.ndarray): The set of predictions
            #Yp = (#samples, #timesteps, #feat_out).
        n_plot (int): The number of samples to plot.
        ds_folder (str): (Optional) The folder containing the dataset.
            If provided and if the method is called from an Ipython console,
            a player is displayed for each sample's audio clip.
        delta (int): The index of the first sample of Y and Yp within the
            whole dataset. For datasets created with "dataset" module this is:
                * 0 if working on the validation set
                * #devset_samples + #testset_samples if working on the training set
        idx_list (list of int): If provided, instead of random picking from all
            samples in Y, the choice is made among this subset of indices. This
            allows to perform error analysis, by providing indices of samples
            with prediction errors.
    """
    if idx_list is None:
        idx = random.sample(range(Y.shape[0]), n_plot)
    else:
        idx = random.choices(idx_list,k=n_plot)
    for i in idx:
        if ds_folder is not None:
            filename = os.path.join(ds_folder, "audio_samples", str(i+delta).zfill(6)+".wav")
            print("Sample:", i, filename)
            IPython.display.display(IPython.display.Audio(filename))
        else:
            print("Sample:", i)
        ytb = Y[i,:,:1]
        ypb = Yp[i,:,:1]
        ytm = Y[i,:,1:]
        ypm = Yp[i,:,1:]
        # PLOT BINARY
        ax = plt.subplot(2, 2, 1)
        plt.plot(ytb, '-b', label='binary ground truth')
        plt.plot(ypb, '-r', label='binary prediction')
        plt.legend()
        ax.set_ylim([-.1,1.1])
        
        ytm_avg = ytm.mean(axis=0)
        sorted_idx = np.argsort(ytm_avg)
        
        # PLOT MULTI TOP 1
        ax = plt.subplot(2, 2, 2)
        plt.plot(ytm[:,sorted_idx[-1]], '-b', label='1st top multiple ground truth')
        plt.plot(ypm[:,sorted_idx[-1]], '-r', label='1st top multiple prediction')
        plt.legend()
        ax.set_ylim([-.1,1.1])
        
        # PLOT MULTI TOP 2
        ax = plt.subplot(2, 2, 4)
        plt.plot(ytm[:,sorted_idx[-2]], '-b', label='2nd top multiple ground truth')
        plt.plot(ypm[:,sorted_idx[-2]], '-r', label='2nd top multiple prediction')
        plt.legend()
        ax.set_ylim([-.1,1.1])
        
        # PLOT MULTI OTHERS
        n_classes = ypm.shape[-1]
        if n_classes>2:
            ax = plt.subplot(2, 2, 3)
            for u in range(n_classes-2):
                plt.plot(ypm[:,sorted_idx[u]])
            ax.set_title("multiple predictions of not 1st and 2nd classes") 
            ax.set_ylim([-.1,1.1])
        
        plt.suptitle("Sample " + str(i))
        plt.show()
        plt.figure()


def find_peaks(y, confidence_thres = 0.5, mean_win_size=9, consistency_thres = 0.75, peak_min_len = 15):
    """Finds peaks in a univariate timeseries.
    
    This function applies to single features of target and predictions of single
    samples.
    The first step is binarization of the series: each value is converted to
    either 0 or 1 depending on accuracy threshold.
    Then a 1D mean filter is applied followed by another binary thresholding,
    to deal with short lived spikes in the series.
    The results is the looked for peaks which are returned after the suppression
    of too short ones.
    
    Args:
        y (numpy.ndarray): The univariate time series to analyze.
        confidence_thres (float): The threshold of the first binarization, being
            the minimum level of confidence for a "one" prediction.
        mean_win_size (int): The length in timesteps of the window for the mean
            filtering.
        consistency_thres (float): The threshold for the binarization after mean
            filtering being, the minimum percentage of "one" timesteps around
            the current for it to be declared a "one".
        peak_min_len (int): The threshold in timesteps under which, peaks are
            declared too short and suppressed.
    
    Returns:
        list of (int,int): List of peaks as (first ts, last ts).
    """
    
    peak_ker = np.full((mean_win_size,), 1.0/mean_win_size)
    try:
        tmp = y.flatten()>confidence_thres
        y_filterd = np.convolve(tmp,peak_ker,'same')
    except ValueError as e:
        print(tmp.shape)
        print(peak_ker.shape)
        raise e
    
    #plt.plot(y_filterd, '-r', label="filtered")
    y_bin = y_filterd>consistency_thres
    #plt.plot(y_bin, '-g', label="filtered binary")
    peaks = []
    p_start = p_end = 0
    is_peak = False
    for i in range(y_bin.size):
        if y_bin[i]==1 and not is_peak:
            is_peak = True
            p_start = i
        elif (y_bin[i]==0 or i==y_bin.size-1) and is_peak:
            is_peak = False
            p_end = i
            if p_end-p_start >= peak_min_len:
                peaks.append((p_start, p_end))
    return peaks


def _iou(p1, p2):
    """Computes intersection over union of two intervals.
    Args:
        p1 ((int,int)): First interval as (first ts, last ts)
        p2 ((int,int)): Second interval as (first ts, last ts)
        
    Returns:
        float: intersection over union of the two intervals.
    """
    i_start = max(p1[0],p2[0])
    i_end = min(p1[1],p2[1])
    i_len = max(0,i_end-i_start)
    o_start = min(p1[0],p2[0])
    o_end = max(p1[1],p2[1])
    o_len = o_end-o_start
    return float(i_len)/o_len


def _ioam(p1, p2):
    """Computes intersection of two intervals over the armonic mean of their lengths.
    
    This is a better metric for peak matching than classic intersection over
    union: it still ranges from 0 to 1 but keeps higher values if one interval
    is much shorter than the other, specifically it cannot be lower than
    I/(2*min(L1,L2)).
    
    Args:
        p1 ((int,int)): First interval as (first ts, last ts)
        p2 ((int,int)): Second interval as (first ts, last ts)
        
    Returns:
        float: interval intersection over armonic mean of intervals' lengths.
    """
    i_start = max(p1[0],p2[0])
    i_end = min(p1[1],p2[1])
    i_len = max(0,i_end-i_start)
    l1 = p1[1]-p1[0]
    l2 = p2[1]-p2[0]
    am = 2*l1*l2/(l1+l2)
    return float(i_len)/am

def _delta(p1, p2):
    """Computes the shift between the start of two intervals.
    
    Args:
        p1 ((int,int)): First interval as (first ts, last ts)
        p2 ((int,int)): Second interval as (first ts, last ts)
        
    Returns:
        float: interval intersection over armonic mean of intervals' lengths.
    """
    return p2[0]-p1[0]
    

def _compare_peaks(pt, pp):
    """Compares the peaks extrated from univariate target and univariate prediction.
    
    This function matches the peaks extracted from a single feature of
    target and relative prediction respectively.
    Peaks are consequently classified as:
        * true positives: when there is a match in the two lists
        * false postive: unmatched peaks of the prediction
        * false negative: unmatched peaks of the target
    
    Args:
        pt (list of (int, int)): peaks from a single feature of target
        pp (list of (int, int)): peaks from a single feature of prediction
    
    Returns:
        list of ((int, int),(int, int)): true positives
        list of (int, int): false postives
        list of (int, int): false negatives
    """
    
    TP = [] #true positive peaks
    FP = [] #false positive peaks
    FN = [] #false negative peaks
    it=0
    ip=0
    while it<len(pt) and ip<len(pp):
        if _ioam(pt[it],pp[ip])>0.25:
            TP.append((pt[it],pp[ip]))
            it+=1
            ip+=1
        else:
            if pt[it][1]<pp[ip][1]:
                FN.append(pt[it])
                it+=1
            else:
                FP.append(pp[ip])
                ip+=1
    if it<len(pt):
        for i in range(it,len(pt)):
            FN.append(pt[i])
    elif ip<len(pp):
        for i in range(ip,len(pp)):
            FP.append(pp[i])
    
    return [TP, FP, FN]


def _match_peak(p, pp, return_ioam=False):
    """Checks a list of peaks for matches against a single peak.
    
    
    Given a peak p and a list of peaks pp (computed by find_peaks()) search in pp
    a peak matching with p, base on ioam() metric. If found, the peak in pp is returned.
    Checks the existence in a list of peaks  of a peak near a specified position.
    
    Args:
        p (int,int): The single peak.
        pp (list of (int,int)): The list of peaks.
        return_ioam (boolean): Whether of not to return the matching score.
    
    Returns:
        (int,int): The first matching element in pp or None.
        float: (only if return_ioam=True) The matching score or None.
    """
    
    for p_ in pp:
        score = _ioam(p,p_)
        if(score>0.25):
            return p_ if not return_ioam else [p_ , score]
    return None if not return_ioam else [None, None]


def _compute_metrics(tp, fp, fn):
    """Computes precision, recall and f1-score from count of TP, TN and FN.
    
    Args:
        tp (int): The number of true positives.
        fp (int): The number of false positives.
        fn (int): The number of false negatives.
        
    Returns:
        dict (str -> float): Dictionary including 'precision', 'recall' and 
            'f1_score'.
    """
    res = {}
    res['precision']=tp/(tp+fp)
    res['recall']=tp/(tp+fn)
    res['f1_score']=2*res['precision']*res['recall']/(res['precision']+res['recall'])
    return res


def compute_global_metrics_single(Yt, Yp):
    """Computes precision, recall and f1-score for the single trigger word detection case.
    
    Statistics for metric computation are determined for each sample and summed
    over the samples.
    
    Args:
        Yt (numpy.ndarray): The matrix of targets #Yt = (#samples, #timesteps, 1)
        Yp (numpy.ndarray): The matrix of predictions #Yp = #Yt
        
    Returns:
        dict (str -> float or int): The dictionary of computed metrics including
            the count of tp, fp and including the average shift and root mean
            quared shift (in timesteps) of predicted peaks wrt target peaks
            and including the mean peak length (in timesteps) of tp predicted
            peaks.
        dict (str -> list of int): The dictionary list of samples indices for
            each case:
                * _tp: true positive,
                * _fp: false postive,
                * _fn: false negative
    """
    
    tp = 0
    fp = 0
    fn = 0
    tp_li = []
    fp_li = []
    fn_li = []
    delta_avg = 0.0
    delta_std = 0.0
    peak_len_avg = 0.0
    for i in range(Yt.shape[0]):
        pt = find_peaks(Yt[i])
        pp = find_peaks(Yp[i])
        TP, FP, FN = _compare_peaks(pt, pp)
        tp += len(TP)
        fp += len(FP)
        fn += len(FN)
        if len(TP)>0:
            tp_li.append(i)
        if len(FP)>0:
            fp_li.append(i)
        if len(FN)>0:
            fn_li.append(i)
        for p1, p2 in TP:
            de = _delta(p1,p2)
            delta_avg += de
            delta_std += de*de
            peak_len_avg += p2[1]-p2[0]
    delta_avg /= tp
    delta_std = math.sqrt(delta_std/tp)
    peak_len_avg /= tp
    m = _compute_metrics(tp, fp, fn)
    m['_tp']=tp
    m['_fn']=fn
    m['_fp']=fp
    m['_tp_delta_avg']=delta_avg
    m['_tp_delta_std']=delta_std
    m['_tp_peak_len_avg']=peak_len_avg
    m2={}
    m2['_tp']=tp_li
    m2['_fn']=fn_li
    m2['_fp']=fp_li
    return (m, m2)

def _peak_center(p):
    #get the center of a peak computed by find_peaks
    return int((p[0]+p[1])/2)

def compute_global_metrics_multi(Yt, Yp):
    """Computes precision, recall and f1-score for the multiple trigger word detection case.
    
    Statistics for metric computation are determined for each sample and summed
    over the samples.
    
    Here each true positives requires several conditions:
        * A match between two peaks in target and prediction of the overall
            binary feature
        * The existence of a corresponding prediction peak in the right class
            feature
        * The latter peak being the dominant one among all classes and having
            a significant magnitude
        
    Various kinds of false positive and false negatives exist in this domain:
        * fp due to unmatched prediction peak in the overall binary feature
        * fn due to unmatched target peak in the overall binary feature
        * fp and fn due to a wrong class prediction peak (or a weak peak, or no
            peak) given a correcly matched peak in the overall binary feature.
        in the overall binary feature
    
    Args:
        Yt (numpy.ndarray): The matrix of targets #Yt = (#samples, #timesteps, 1)
        Yp (numpy.ndarray): The matrix of predictions #Yp = #Yt
        
    Returns:
        dict (str -> float or int): The dictionary of computed metrics
        dict (str -> list of int): The dictionary list of samples indices for
            each case (allowing for error analysis):
                * _tp: true positive,
                * _fpb: false postive overall binary,
                * _fnb: false negative overall binary,
                * _fc: false (both positive and negative) class
    """
    
    tp = 0
    fpb = 0
    fnb = 0
    fc = 0
    tp_li = []
    fnb_li = []
    fpb_li = []
    fc_li = []
    for i in range(Yt.shape[0]):
        ytb = Yt[i,:,:1]
        ypb = Yp[i,:,:1]
        ytm = Yt[i,:,1:]
        ypm = Yp[i,:,1:]
        PT = find_peaks(ytb)
        PP = find_peaks(ypb)
        TP, FP, FN = _compare_peaks(PT, PP)
        PMP = [] #peaks of the multiple prediction
        n_classes = ypm.shape[-1]
        for u in range(n_classes):
            PMP.append(find_peaks(ypm[:,u]))
        for pt, pp in TP: #analyze multi-prediction of TP of binary prediction to confirm them
            pt_cen = _peak_center(pt)
            assert np.count_nonzero(ytm[pt_cen])==1
            ct = ytm[pt_cen].argmax() #true class index
            pmp = _match_peak(pp,PMP[ct]) #position of matching peak in PMP[ct]
            if pmp is None: #if no matching peak --> false class
                fc+=1
                fc_li.append(i)
                continue
            p_avg = ypm[pmp[0]:pmp[1]].mean(axis=0) #mean around the peak of prediction for each class
            if p_avg.argmax()==ct and p_avg[ct]>0.5: #check that the maximum mean is scored for the correct class and that the mean itself is over a threshold
                tp+=1
                tp_li.append(i)
            else: #false class
                fc+=1
                fc_li.append(i)
        fpb += len(FP)
        if len(FP)>0:
            fpb_li.append(i)
        fnb += len(FN)
        if len(FN)>0:
            fnb_li.append(i)
    m = _compute_metrics(tp, fpb+fc, fnb+fc)
    m['_tp']=tp
    m['_fbn']=fnb
    m['_fpb']=fpb
    m['_fc']=fc
    m2={}
    m2['_tp']=tp_li
    m2['_fnb']=fnb_li
    m2['_fpb']=fpb_li
    m2['_fc']=fc_li
    return (m, m2)


def search_peak_multi(yp, confidence_thres=0.5):
    """Search peaks in a prediction for the multiple trigger words case.
    
    This function helps to determine whether a trigger words has been detected
    in the prediction of a single sample. It is to be used in the live
    prediction process (not for model evaluation).
    
    Args:
        yp (numpy.nparray): Single sample prediction #yp = (#samples,
            #positive_classes+1)
    
    Returns:
        (int,int): The first peak as (first_ts, last_ts).
        float: The index of the peak class.
    """
    
    yb = yp[:,:1]
    ym = yp[:,1:]
    PB = find_peaks(yb,confidence_thres=confidence_thres)
    if len(PB)==0:
        return None, None
    n_classes = ym.shape[-1]
    pb = PB[0] #focusing on the first binary peak (in time)
    best_score = -1
    best_class = None
    for u in range(n_classes):
        PM = find_peaks(ym[:,u],confidence_thres=confidence_thres)
        pm, ioam = _match_peak(pb,PM,return_ioam=True)
        if pm is None:
            continue
        pm_avg = ym[pm[0]:pm[1]].mean()
        score = ioam*pm_avg
        if score > best_score:
            best_score = score
            best_class = u
    return pb, best_class



def predict_audio_clip_single(model, filename, plot=True, confidence_thres=0.5):
    """Makes prediction on a wav audioclip with a single trigger word model.
    
    Args:
        model (keras.models.Model): Single trigger word prediction model.
        filename (str): The filename (wav format) of the audioclip to predict.
        plot (boolean): Whether to plot the prediction.
        confidence_thres (float): Confidence threshold used to detect peaks (cfr. documentation of find_peaks()).
    
    Returns:
        numpy.ndarray: The prediction (shape = (1, #timesteps, 1)).
        list of (int,int): List of peaks as (first_ts, last_ts).
    """
    
    rate, audio44khz = wavfile.read(filename)
    assert abs(rate-44100)<1
    x=spectrogram(audio44khz)
    #del audio44khz
    x = np.expand_dims(x,axis=0)
    print(x.shape)
    yp = model.predict(x, batch_size=1)
    pp = find_peaks(yp,confidence_thres=confidence_thres)
    print("# peaks", len(pp))
    print("Peaks:")
    [print(p) for p in pp]
    if plot:
        ax = plt.subplot(2, 1, 1)
        plt.plot(yp[0], '-b')
        [plt.axvline(x=p[0], color='r') for p in pp]
        [plt.axvline(x=p[1], color='k') for p in pp]
        #plt.gca().set_title("")
        ax.set_ylim([-.1,1.1])
        ax = plt.subplot(2, 1, 2)
        plt.plot(audio44khz)
        plt.show()
    return yp, pp


def predict_audio_clip_multi(model, filename, plot=True, confidence_thres=0.5):
    """Makes prediction on a wav audioclip with a multiple trigger words model.
    
    Args:
        model (keras.models.Model): Multiple trigger words prediction model.
        filename (str): The filename (wav format) of the audioclip to predict.
        plot (boolean): Whether to plot the prediction.
        confidence_thres (float): Confidence threshold used to detect peaks in global binary feature (cfr. documentation of find_peaks()).
    
    Returns:
        numpy.ndarray: The prediction (shape = (1, #timesteps, 1)).
        list of (int,int): List of peaks as (first_ts, last_ts).
        list of int: Class predicted for each peaks.
    """
    
    rate, audio44khz = wavfile.read(filename)
    assert abs(rate-44100)<1
    x=spectrogram(audio44khz)
    #del audio44khz
    x = np.expand_dims(x,axis=0)
    yp = model.predict(x, batch_size=1)[0]
    
    yb = yp[:,:1]
    ym = yp[:,1:]
    #print(yp.shape, yb.shape, ym.shape)
    PB = find_peaks(yb,confidence_thres=confidence_thres)
    PC=[]
    n_classes = ym.shape[-1]
    for pb in PB:
        best_score = -1
        best_class = None
        for u in range(n_classes):
            PM = find_peaks(ym[:,u])
            pm, ioam = _match_peak(pb,PM,return_ioam=True)
            if pm is None:
                continue
            pm_avg = ym[pm[0]:pm[1]].mean()
            score = ioam*pm_avg
            if score > best_score:
                best_score = score
                best_class = u
        PC.append(best_class)
    print("# peaks", len(PB))
    print("Peaks:")
    [print(p,c) for p,c in zip(PB,PC)]
    if plot:
        # PLOT BINARY
        ax = plt.subplot(3, 1, 1)
        plt.plot(audio44khz)
        ax.set_title("audio clip")
        
        # PLOT BINARY
        ax = plt.subplot(3, 1, 2)
        plt.plot(yb, '-b', label='binary prediction')
        if len(PB)>0:
            plt.plot(ym[:,PC[0]], '-r', label='best class prediction')
        [plt.axvline(x=p[0], color='r') for p in PB]
        [plt.axvline(x=p[1], color='k') for p in PB]
        plt.legend()
        ax.set_ylim([-.1,1.1])
        
        # PLOT MULTI OTHERS
        n_classes = ym.shape[-1]
        if n_classes>1:
            ax = plt.subplot(3, 1, 3)
            for u in range(n_classes-1):
                if len(PB)>0 and u==PC[0]:
                    continue
                plt.plot(ym[:,u])
            ax.set_title("other class predictions") 
            ax.set_ylim([-.1,1.1])
        
        plt.show()
        plt.figure()
    return yp, PB, PC
    
#%% TESTING find_peaks
"""
plt.rcParams["figure.figsize"] =(12,5)
y=np.zeros((1200,))
y[200:287]=1
y[800:850]=1
y[900:950]=1
plt.plot(y, '-b', label="y")
p = find_peaks(y)
plt.scatter(p, np.ones(len(p),), label="peaks") 
plt.legend()
plt.show()
plt.figure()
print(p)
#EXPECTED RES: [215, 815, 915]
"""
#%% TESTING compare_peaks
"""
#pt = find_peaks(yt)
#pp = find_peaks(yp)
pt=[50, 150, 300, 440, 550, 760, 860, 1000, 1200]
pp=[220, 320, 400, 660, 890]
TP, FP, FN = compare_peaks(pt, pp)
print(len(TP), len(FP), len(FN))
print(TP, FP, FN)
#EXPECTED RES: 3 2 6 [(300, 320), (440, 400), (860, 890)] [220, 660] [50, 150, 550, 760, 1000, 1200]
"""
