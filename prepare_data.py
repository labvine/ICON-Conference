# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:58:11 2017

@author: Lenovo
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
import scipy.io
import matplotlib
import os
num_cpu = '8' # Set as a string
os.environ['OMP_NUM_THREADS'] = num_cpu

sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Nencki/EEG-IO/')
import UtilityReadEDF as ur
plt.style.use('ggplot')


        
def load_prepared(subject):
    """
    1. ICA cleaned eyeblink artifacts
    2. MNE events (with accuracy) parsed from csv logs
    """
    raw_cleaned = mne.io.read_raw_fif('ICA_cleaned/' + subject +'_ica_cleaned.fif', preload=True)
    raw_cleaned.set_eeg_reference(ref_channels=None)
    raw_cleaned = raw_cleaned.apply_proj()
    
    events_cleaned = np.load('mne_events/mne_'+ subject +'.npy')
    events_dict = np.load('mne_events/code_dict_' + subject +'.npy').item()
    return raw_cleaned, events_cleaned,events_dict

def prepare_and_save(path):
    """ Loads digitrack files and parses them for MNE format.
        Bad channels are dropped. 
        Data is cropped to include experiment time only.
        Events are saved to disk.
        NOTE - ICA is not perfomed here. ICA_preprocessing.py removes eyeblink artifacts.
        NOTE - average EEG reference is not applied here.
    """
    
    raw, timestamp = ur.MNE_Read_EDF(path)
    
    #Use the time columns to create MNE events structure
    events_log, event_id = clean_log(path)
    event_sample_indexes = ur.parse_events(events_log, timestamp)
    events = ur.events_for_MNE(event_sample_indexes, event_id)
    
    #Add response correct/incorrect to events
    new_events, new_event_id = expand_events(path, events, event_id)
    #Crop the data to include only the time between start and stop of the experiment - many artifacts outside this interval 
    raw_cropped = raw.copy().crop(tmin = events[0,0]/raw.info['sfreq'], tmax = events[-1,0]/raw.info['sfreq'])
    #Since the raw was cropped to the time of the first event its' new time is now 0. All following events are shifted.
    new_events[:,0] = new_events[:,0] - new_events[0,0]
    
    #Delete bad channels, ears and visually identified channels
    ears = [ch for ch in raw_cropped.ch_names if 'A' in ch]
    raw_cropped = raw_cropped.drop_channels(ears)
    
    subject_bads = {'Adrianna': ['T4'], 'BartekB' : ['Pz'], 'JeremiaszW' : [], 'KonradW' : ['T3'],  'Lucja' : ['T4', 'F8'], 'MaciekG':[], 'MariuszZ' : [], 'OlaS' :['P4'], 'Patrycja' :[]}
    bads = subject_bads[path.split('\\')[-3]]
    if len(bads) != 0:
       raw_cropped = raw_cropped.drop_channels(bads)
    
    #Apply average re-reference
    raw_cropped.save('raw_cropped/' + path.split('\\')[-3] +'_raw_cropped.fif', overwrite = True)
    return raw_cropped, new_events, new_event_id


def clean_log(path):
    """Select from the log only the columns with event times"""
    # Read the experiment log from a csv file
    assert len(glob.glob(path + "/*.csv")) == 1, 'problem with number of .csv files'
    log = pd.read_csv(glob.glob(path + "/*.csv")[0],parse_dates = True, index_col = 0, skiprows = 1, skipfooter = 1, engine='python')
    include = ['time']
    exclude = ['response', 'psychopy' , 'start_time']
    # Select the columns where the timestamp of the event was written
    # event_time_columns = [col_name for col_name in log.columns if any(substring in col_name for substring in include)
    #                                                       and not any(substring in col_name for substring in exclude)]
    
    event_time_columns = [col_name for col_name in log.columns if 'time' in col_name and 'response' not in col_name and 'psychopy' not in col_name and 'start_time' not in col_name]
    
    events_log = log[event_time_columns]
    # Event types have to be encoded with ints starting from 1 for MNE
    event_id = {event_name : idx + 1 for idx, event_name in enumerate(events_log.columns)}    
    return events_log, event_id
    
def expand_events(exp_path, events, event_id):
    """Add accuracy info to event times"""
    #Read the raw log with accuracy info, which was not used when first creating MNE events
    log = pd.read_csv(glob.glob(exp_path + "/*.csv")[0],parse_dates = True, index_col = 0, skiprows = 1, skipfooter = 1, engine='python')
    subject = exp_path.split('\\')[-3]
    #Select the column with accuracy info. Nan's indicate control trial where no answer was given.
    acc_condition = log['accuracy'].fillna('control')
    
    # Create an intermediate dataframe where new codes, including accuracy info will be sotred
    events_df = pd.DataFrame(events, columns = ['sample_nr', 'empty', 'code'])
    events_df['condition'] = np.nan
    
    #Select indices of ITI_time, which always begin a new trial. Number of ITI_time rows == number of trials == number of responses + Nan's(control trials) in log acuracy column
    events_df.loc[events_df['code'] == 2, 'condition'] = acc_condition.as_matrix()
    
    events_df = events_df.fillna(axis = 0, method = 'ffill')
    
    inv_event_dict = {v: k for k, v in event_id.items()}
    
    events_df['e_name'] = events_df['code'].replace(inv_event_dict)
    events_df['combined_codes'] = events_df['e_name'] + '_' + events_df['condition']
    
    # Save the events in the general purpose format
    events_df.to_csv('mne_events/' +'raw_'+subject+'.csv')
    
    new_code_dict = OrderedDict()
    for idx, code in enumerate(sorted(events_df['combined_codes'].unique().astype('str').tolist())):
        new_code_dict[code] = idx
        
    events_df['new_code'] = events_df['combined_codes'].replace(new_code_dict)
    
    new_events = events_df[['sample_nr', 'empty','new_code']].as_matrix()
    
    # Save the events and their codes in mne format
    np.save('mne_events/'+'mne_'+ subject, new_events)
    np.save('mne_events/'+'code_dict_'+ subject, new_code_dict)
    
    return new_events, new_code_dict


####################
#### MNE EPOCHS ####
####################


def create_epochs(raw, events, event_id, subject, reject = {'eeg': 0.0009}  ):
# Note - reject default value reject = {'eeg': 0.0009}, will not remove any epochs - threshold too high.
#Only change the last decimal place of  reject value.
    tmin = 0.0
    tmax = 5.0
    
    baseline=(None, 0)

    picks = mne.pick_types(raw.info, meg = False, eeg = True)
    
    correct_epochs = mne.Epochs(raw, events, event_id['ISI_time_correct'], tmin, tmax, picks = picks,
                     baseline = baseline, reject = reject, proj = False,  preload=True)
    
    incorrect_epochs = mne.Epochs(raw, events, event_id['ISI_time_wrong'], tmin, tmax, picks = picks,
                     baseline = baseline, reject = reject,  proj = False, preload=True)
    
    control_epochs = mne.Epochs(raw, events, event_id['ISI_time_control'], tmin, tmax, picks = picks,
                     baseline = baseline, reject = reject, proj = False, preload=True)
    
    return {'correct': correct_epochs, 'incorrect': incorrect_epochs, 'control' : control_epochs}

def save_epochs(path, raw, events, event_id,_id, subject):
    
    
    tmin = 0.0
    tmax = 5.0
    #baseline=(0.0, 0.5)
    picks = mne.pick_types(raw.info, meg = False, eeg = True)
    try:
        epochs = mne.Epochs(raw, events, event_id[_id], tmin, tmax, picks = picks,
                      proj = False,  preload=True)
        ch_names = epochs.ch_names

        epochs = epochs.get_data()

        np.savetxt(fname = subject +'.txt', X = ch_names, fmt = '%s')    
        scipy.io.savemat('mat_epochs/' +_id + '_' + subject +'.mat', {_id:epochs})
    except KeyError as e:
        print(subject + ' no dont know answers ')
    
    