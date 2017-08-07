# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:09:45 2017

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib import gridspec

import seaborn as sns
import sys
import glob
from collections import OrderedDict
import mne
from mne import Epochs, find_events, create_info
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.time_frequency import AverageTFR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import os

import scipy.stats as stats

matplotlib.rcParams.update({'font.size': 28})
plt.style.use('ggplot')
num_cpu = '8' # Set as a string
os.environ['OMP_NUM_THREADS'] = num_cpu

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Nencki/EEG-IO/') #Replace with the path to the folder with UtilityReadEDF.py script
import UtilityReadEDF as ur
import prepare_data as pr #prepare_data.py should be in the same folder as this script or in the same as UtilityReadEDF.py     

all_freq_scores = {}

def iterate_subjects():

    all_paths = glob.glob('C:\\Users\\Lenovo\\Desktop\\Nencki\\EEG_data\\DMTS_experiment/*')
   
    for path in all_paths:
        exp_path = path +'\\exp_logs\\'
        print(exp_path)
        
        subject = exp_path.split('\\')[-3]
        
        raw,events,event_id = pr.load_prepared(subject)
        ch_names = raw.info['ch_names']
        
        exclude = [e for e in ch_names if 'T' in e]
        raw = raw.drop_channels(exclude)
        
        
        
        all_freq_scores[subject] = CSP_dec(raw, events, event_id, subject)

    return all_freq_scores

def cross_val_test(freq_results):
"""T-Test for cross validation scores."""    
    
    arr_results = []
    for subject, res in freq_results.items():
        arr_results.append(res)
        
    arr_results = np.array(arr_results) 
    p_vals = []
    t_vals = []
    fig, axes = plt.subplots(figsize = (15,10))
    for hz in arr_results.T:
        t,p = stats.ttest_1samp(hz, 0.5)
        p_vals.append(p * arr_results.shape[1])
        t_vals.append(t)
        
        
    f = np.array(freqs[:-1])
    sig = [i for i, val in enumerate(p_vals) if val<0.05]

    axes.bar(left= f , height=arr_results.mean(axis = 0), width=np.diff(freqs)[0],
            align='edge', edgecolor='black', color = 'grey', yerr = arr_results.std(axis = 0), tick_label = [int(round(f)) for f  in freqs][:-1])
    
    axes.bar(left= f[sig], height=arr_results.mean(axis = 0)[sig], width=np.diff(freqs)[0],
            align='edge', edgecolor='black', label = 'Above chance decoding')

    axes.axhline(0.5, color='k', linestyle='--',
                label='chance level')
    axes.legend(prop={'size': 28})
        
    axes.set_xlabel('Frequency (Hz)')
    axes.set_ylabel('Cross-Validation Scores')           

    #fig.suptitle('Single-Trial Frequency Decoding Scores')

    axes.xaxis.set_tick_params(labelsize=18)
    axes.yaxis.set_tick_params(labelsize=18)

    fig.savefig('Decoding.eps')
    fig.savefig('Decoding.svg')
    fig.savefig('Decoding.png', papertype = 'a0')


   

def CSP_dec(raw, events, event_id, subject):  
    """Common spatial pattern method is used for frequency filtering. 
    Linear discriminant analysis is used to label data into conditions based on a frequency component created by CSP."""

    event_id = dict(ISI_time_correct = event_id['ISI_time_correct'], ISI_time_control=event_id['ISI_time_control'])  # motor imagery: hands vs feet   
    # Extract information from the raw file
    sfreq = raw.info['sfreq']
    raw.pick_types(meg=False, eeg=True, stim=False, eog=False)
    
    # Assemble the classifier using scikit-learn pipeline
    clf = make_pipeline(CSP(n_components=4, reg=None, log=True),
                        LinearDiscriminantAnalysis())
    n_splits = 5  # how many folds to use for cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    # Classification & Time-frequency parameters
    tmin, tmax = 1.0, 4.0
    n_cycles = 10.  # how many complete cycles: used to define window size
    min_freq = 2.
    max_freq = 50.
    n_freqs = 24  # how many frequency bins to use
    
    # Assemble list of frequency range tuples
    freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
    freq_ranges = list(zip(freqs[:-1], freqs[1:]))  # make freqs list of tuples
    
    # Infer window spacing from the max freq and number of cycles to avoid gaps
    window_spacing = (n_cycles / np.max(freqs) / 2.)
    centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
    n_windows = len(centered_w_times)
    
    # Instantiate label encoder
    le = LabelEncoder()
            
            # init scores
    freq_scores = np.zeros((n_freqs - 1,))
    
    # Loop through each frequency range of interest
    for freq, (fmin, fmax) in enumerate(freq_ranges):
    
        # Infer window size based on the frequency being used
        w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds
    
        # Apply band-pass filter to isolate the specified frequencies
        raw_filter = raw.copy().filter(fmin, fmax, n_jobs=1, fir_design='firwin')
    
        # Extract epochs from filtered data, padded by window size
        epochs = Epochs(raw_filter, events, event_id, tmin - w_size, tmax + w_size,
                        proj=False, baseline=None, preload=True)
        
        e1 = epochs['ISI_time_correct']
        e2 = epochs['ISI_time_control']
        mne.epochs.equalize_epoch_counts([e1,e2])
        epochs = mne.epochs.concatenate_epochs([e1,e2])
        
        epochs.drop_bad()
        y = le.fit_transform(epochs.events[:, 2])
    
        X = epochs.get_data()
    
        # Save mean scores over folds for each frequency and time window
        freq_scores[freq] = np.mean(cross_val_score(estimator=clf, X=X, y=y,
                                                    scoring='roc_auc', cv=cv,
                                                    n_jobs=8), axis=0)
        
    
    fig, axes = plt.subplots(figsize = (20,10))
    axes.bar(left=freqs[:-1], height=freq_scores, width=np.diff(freqs)[0],
            align='edge', edgecolor='black')
    axes.set_xticks([int(round(f)) for f in freqs])
    axes.set_ylim([0, 1])
    axes.axhline(len(epochs['ISI_time_correct']) / len(epochs), color='k', linestyle='--',
                label='chance level')
    axes.legend()
    axes.set_xlabel('Frequency (Hz)')
    axes.set_ylabel('Decoding Scores')                                       
    fig.suptitle('Frequency Decoding Scores ' + subject)
    
    fig.savefig('CSP_decoding/' + subject)
    
    return freq_scores, freqs
            
        