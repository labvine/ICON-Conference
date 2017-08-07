# -*- coding: utf-8 -*-
"""
@author: Ryszard Cetnarski
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import sys
import glob
from collections import OrderedDict
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from scipy import signal
import scipy.stats as stats
from mne.time_frequency import tfr_morlet
import os
import scipy.io
import matplotlib
from mne.stats import permutation_cluster_test
from mne.stats import permutation_cluster_1samp_test
import matplotlib.ticker as mtick
from scipy.stats import distributions

plt.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 28})
plt.style.use('ggplot')
num_cpu = '8' # Set as a string
os.environ['OMP_NUM_THREADS'] = num_cpu


sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Nencki/EEG-IO/')
import UtilityReadEDF as ur
import prepare_data as pr


def iterate_subjects():
    """Compute PSD and spectrogram for each subject and aggregate."""
    bands = [(0, 8, '0-8'), (8,12, '8-12'), (12,18, '12-18'), (18,30, '18-30')]
    fft_averages = {}
    specgram_avg = {}
    all_paths = glob.glob('C:\\Users\\Lenovo\\Desktop\\Nencki\\EEG_data\\DMTS_experiment/*')
    tfr = {}
    fig, axes = plt.subplots(nrows = 2, ncols =4, figsize = (20,10))
    fig.suptitle('PSD topomap')
    for path in all_paths:
        exp_path = path +'\\exp_logs\\'
        print(exp_path)
        
        subject = exp_path.split('\\')[-3]
        
        
        raw,events,event_id = pr.load_prepared(subject)
        ch_names = raw.info['ch_names']

         
        all_epochs = pr.create_epochs(raw, events, event_id, subject, reject = {'eeg': 0.0009})
        
        specgram_avg, sf, st,  tfr  = epochs_specgram(all_epochs, subject, specgram_avg, tfr)
        fft_averages, f= epochs_fft(all_epochs, subject, fft_averages)

    perm_test(fft_averages, f, ch_names) # Produces PSD plot
    inspect_specgram(specgram_avg) # Produces time-frequency plot

        
    return fft_averages


def perm_test(fft_averages, f, ch_names):
    """Produces PSD comparison figure. 1-D Cluster perm test for the difference in average (per subject) power spectrum amplitude between conditions."""
    times = f[0:30]
    for e_name in ch_names[12:13]:
        print(e_name)
        condition1 = np.array(fft_averages[e_name]['correct'])
        condition2 = np.array(fft_averages[e_name]['control'])
        
        
        difference_arr = condition1-condition2
    
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(difference_arr, n_permutations=1000)
    
    
        #fig, axes = plt.subplots(2, )
        fig = plt.figure(figsize = (20,12))
        
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
        axes = []
        axes.append(plt.subplot(gs[0]))
        axes.append(plt.subplot(gs[1]))
        #fig.suptitle('%s\nCorrect - control'%e_name, fontweight = 'bold', fontsize = '20')
        
        # TOP BOXPLOT
        x_box = np.repeat(times, condition1.shape[0])
        y_box = difference_arr.T.flatten()
        
        df = pd.DataFrame(difference_arr, columns = times,)

        bp = sns.boxplot(df, ax = axes[0],  fliersize = 0, color = 'grey')
        
        sw = sns.swarmplot(x=x_box, y=y_box, color=".25", ax = axes[0], alpha = 0.5)
    
        bp.set(xticklabels = [])
        axes[0].set_ylabel('Log Power difference')
        axes[0].axhline(y= 0, linestyle ='--', alpha = 0.5, color = 'black')

        # BOTTOM CLUSTERS
        for i_c, c in enumerate(clusters):
            c = c[0]
            if cluster_p_values[i_c] <= 0.05:
                h =axes[1].axvspan(times[c.start], times[c.stop - 1],
                                color='#E24A33', alpha=0.3)
                sig_c = c
            else:
                axes[1].axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                            alpha=0.3)
                
        
        for box in bp.artists[sig_c.start:sig_c.stop-1]:
            box.set_facecolor('#E24A33')
        #BOTTOM LINE
        hf = axes[1].plot(times, T_obs, '#348ABD')
        axes[1].legend((h, ), ('cluster p-value < 0.05', ), prop={'size': 28})
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("T-value")
        axes[1].set_xlim(min(times), max(times))
        axes[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

        for ax in axes:
            ax.xaxis.set_tick_params(labelsize=18)
            ax.yaxis.set_tick_params(labelsize=18)
            ax.get_yaxis().set_label_coords(-0.07,0.5)

        fig.savefig('cluster_perms/'+e_name)
        fig.savefig('Cluster_O1.svg')
        fig.savefig('Cluster_O1.eps')
        fig.savefig('Cluster_O1.png', papertype = 'a0')

def inspect_specgram(specgram_avg):
    """Produces time-frequency comparison. 2-D permutation cluster tests are used to compare average (subject) spectrograms between conditions."""
    plt.tick_params(labelsize=18)
    tf_corr = specgram_avg['O1']['correct']
    tf_control = specgram_avg['O1']['control']
    
    wave_corr = tfr['O1']['correct']
    wave_ctrl = tfr['O1']['control']
    
    ee= []
    for corr, ctrl, w_corr, w_ctrl in zip(tf_corr, tf_control, wave_corr, wave_ctrl):

        ee.append((corr - ctrl)[0:30])
        
    ee = np.array(ee)
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(ee, n_permutations=100)
    
    # Create new stats image with only significant clusters
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        print(p_val)
        if p_val <= 0.05:
            print(c)
            T_obs_plot[c] = T_obs[c]
    
    vmax = np.max(np.abs(T_obs))
    vmin = -vmax
    
    fig, axes = plt.subplots(figsize = (20,10))
    
    axes.imshow(T_obs, cmap=plt.cm.gray,
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    
    cbar = axes.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    
    axes.set_ylabel('Frequency (Hz)')
    axes.set_xlabel('Time (ms)')
    #fig.suptitle('Time-Frequency Log Power Difference')
    fig.colorbar(cbar)
    axes.set_xticklabels(np.linspace(0, 5000, 11), fontsize = 18)
    axes.yaxis.set_tick_params(labelsize=18)
    axes.yaxis.set_label_coords(-0.07,0.5)

    fig.savefig('time_frequency.svg')
    fig.savefig('time_frequency.eps')
    fig.savefig('time_frequency.png',papertype='a0')

    

    
        

def epochs_fft(subject_epochs, subject, fft_avg):
    ch_names = subject_epochs[list(subject_epochs.keys())[0]].ch_names
   # color_code = {'correct' : 'blue', 'incorrect' : 'red', 'control': 'green'}
    

    
    for electrode in ch_names:
        if electrode not in fft_avg:
            fft_avg[electrode] = {}
    

        for code, epochs in subject_epochs.items():
            if code not in fft_avg[electrode]:
                fft_avg[electrode][code] = []
            
            electrode_epochs = epochs.copy().pick_channels([electrode]).get_data()
            electrode_epochs = electrode_epochs.reshape(electrode_epochs.shape[0], electrode_epochs.shape[2]).squeeze()
                        
            f, Pxx_den = signal.welch(electrode_epochs, fs = epochs.info['sfreq'], nperseg=512, noverlap = 128)
            
            
            Pxx_mean = np.log(Pxx_den.mean(axis = 0))
            Pxx_mean =  Pxx_mean/np.abs(Pxx_mean.sum())
            fft_avg[electrode][code].append(Pxx_mean[0:30])                   
            
    return fft_avg, f

def epochs_specgram(subject_epochs, subject, specgram_avg, tfr):
    frequencies = np.arange(4, 30, 2)
    
    for electrode in ch_names:
        if electrode not in specgram_avg:
            specgram_avg[electrode] = {}
            tfr[electrode]   = {}

        for code, epochs in subject_epochs.items():
            if code not in specgram_avg[electrode]:
                specgram_avg[electrode][code] = []
                tfr[electrode][code] = []
            electrode_epochs = epochs.copy().pick_channels([electrode]).get_data()
            electrode_epochs = electrode_epochs.reshape(electrode_epochs.shape[0], electrode_epochs.shape[2]).squeeze()
                        
            f, t, Sxx = signal.spectrogram(electrode_epochs, fs = epochs.info['sfreq'], nperseg=512, noverlap = 384)  
            Sxx_mean = np.log(Sxx.mean(axis = 0))
            specgram_avg[electrode][code].append(Sxx_mean)
            
            
            tfr_epochs = tfr_morlet(epochs.copy().pick_channels([electrode]), frequencies, n_cycles=4.,decim = 5, 
                        average=False, return_itc=False, n_jobs=1)
            
            tfr[electrode][code].append(tfr_epochs)

    return specgram_avg, f,t, tfr



def plot_bad_epochs(raw,epochs, events, subject):
    
    tmin = 0.0
    tmax = 5.0
    baseline=(None, 0)
    picks = mne.pick_types(raw.info, meg = False, eeg = True)

    drop_log = epochs.drop_log
    drop_log =  [val for val in drop_log if val != ['IGNORED'] and val != [] ]

    if drop_log != []:
        events_copy = np.copy(events)
        bad_idx = [idx for idx, val in enumerate(epochs.drop_log) if val != ['IGNORED'] and val != [] ]
        
        events_copy[bad_idx, 2] = -1
        
        bad_epochs = mne.Epochs(raw, events_copy, -1, tmin, tmax, picks = picks,
                         baseline = baseline, proj = False, preload=True)
        
        traces = bad_epochs.plot()
        traces.suptitle(subject)
        stats = epochs.plot_drop_log(color = 'black')
        stats.suptitle(subject)
        
    return drop_log 
