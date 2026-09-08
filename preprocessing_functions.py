# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 13:11:39 2025

@author: David Teo
"""

import os
from scipy import signal
import scipy.io
import scipy
from scipy.signal import cheby2, lfilter, filtfilt, butter, find_peaks, peak_widths
from scipy.fft import rfft, rfftfreq, fft, fftfreq, ifft
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from striprtf.striprtf import rtf_to_text

##### Get Gains ############
def get_gains(patient, electrode, stim_condition, week):
    def read_rtf_file(file_path):
        """
        Reads an RTF file and extracts its plain text content.

        Args:
            file_path (str): The path to the RTF file.

        Returns:
            str: The plain text content of the RTF file.
        """
        try:
            with open(file_path, 'r', encoding='cp1252') as infile:
                rtf_content = infile.read()
            plain_text = rtf_to_text(rtf_content)
            return plain_text
        except UnicodeDecodeError:
            # If cp1252 fails, try a more general encoding like 'latin-1' or 'utf-8'
            with open(file_path, 'r', encoding='latin-1') as infile:
                rtf_content = infile.read()
            plain_text = rtf_to_text(rtf_content, encoding='latin-1')
            return plain_text
        except FileNotFoundError:
            return f"Error: File not found at {file_path}"
        except Exception as e:
            return f"An error occurred: {e}"
    electrode_labels = {0:'L Psoas Major',
                        1:'L Rectus Femoris',
                        2:'L Vastus Laterali',
                        3:'L Tibialis Anterior',
                        4:'L Gluteus Maximus',
                        5:'L Bicep Femoris',
                        6:'L Gastrocnemius',
                        7:'L Soleus',
                        8:'R Psoas Major',
                        9:'R Rectus Femoris',
                        10:'R Vastus Laterali',
                        11:'R Tibialis Anterior',
                        12:'R Gluteus Maximus',
                        13:'R Bicep Femoris',
                        14:'R Gastrocnemius',
                        15:'R Soleus'}
    # Create dataframe
    muscles = ['R Psoas Major', 'L Psoas Major', 'R Rectus Femoris', 'L Rectus Femoris', 'R Vastus Lateralis', 'L Vastus Lateralis', 'R Tibialis Anterior', 'L Tibialis Anterior', 'R Gluteus Maximus', 'L Gluteus Maximus', 'R Bicep Femoris', 'L Bicep Femoris', 'R Gastrocnemius', 'L Gastrocnemius', 'R Soleus', 'L Soleus']
    gains_off_df = pd.DataFrame(columns=muscles)
    gains_on_df = pd.DataFrame(columns=muscles)
    if patient == 'P1/':
        filepath = 'RS-0101 Post op gain impedance/'
    elif patient == 'P2/':
        filepath = 'RS-0102 Post op gain impedance/'
    elif patient == 'P3/':
        filepath = 'RS-0103_Post Op gain impedance/'
    else:
        print('that is not a patient')
    path = 'C:/Users/David Teo/Desktop/Gain and Impedance Data/' + filepath
    # RS-0101 Post op gain impedance/
    # RS-0102 Post op gain impedance/
    # RS-0103_Post Op gain impedance/
    
    files = os.listdir(path)
    file_dates = []
    month_index = {'Jan' : '01',
                   'Feb' : '02',
                   'Mar' : '03',
                   'Apr' : '04',
                   'May' : '05',
                   'Jun' : '06',
                   'Jul' : '07',
                   'Aug' : '08',
                   'Sep' : '09',
                   'Oct' : '10',
                   'Nov' : '11',
                   'Dec' : '12'}
    
    
    # Load file:
    sorted_by_creation_time = sorted(files, key=lambda item: os.path.getctime(os.path.join(path, item)), reverse=True)
    files = sorted_by_creation_time
    print(sorted_by_creation_time)
    x=0
    for file in files:
        x+=1
        print(x, file)
        file_name = path+file
        text_content = read_rtf_file(file_name)
        
        # Get Date
        date = file.split('- ')[1].split(' session')[0]
        if date[0] == ' ':
            date = date[1:]
        day = date.split(' ')[0]
        if len(day) <2:
            day = '0'+day
        month = month_index[date.split(' ')[1][0:3]]
        year = date.split(' ')[2]
        date = year+month+day
        
        file_dates.append(date)
        
        
        
        # Extract gains
        gains_off = text_content.split('Gains:')[1].split('\nStim off:\n')[-1].split('\nStim on')
        gains_off.pop(-1)
        gains_off = [s.replace('uV','') for s in gains_off]
        gains_on = text_content.split('Gains:')[1].split('\n\x00')[0].split('\nStim on:\n')[1].split('\n')
        gains_on = [s.replace('uV','') for s in gains_on]
        
        #Gains off
        [rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls] = [np.nan for _ in range(16)]
        for g in gains_off:
            if 'all channels' in g.lower():
                [rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls] = [int(g.split(' ')[-1])] * len([rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls])
            if 'all left channels' in g.lower():
                [lpm,lrf,lvl,lta,lgm,lbf,lg,ls] = [int(g.split(' ')[-1])] * len([lpm,lrf,lvl,lta,lgm,lbf,lg,ls])
            if 'all right channels' in g.lower():
                [rpm,rrf,rvl,rta,rgm,rbf,rg,rs] = [int(g.split(' ')[-1])] * len([rpm,rrf,rvl,rta,rgm,rbf,rg,rs])
            if 'psoas' in g.lower():
                if 'bilateral' in g.lower():
                    rpm = int(g.split(': ')[-1])
                    lpm = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rpm = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lpm = int(g.split(': ')[-1])
            if 'rectus' in g.lower():
                if 'bilateral' in g.lower():
                    rrf = int(g.split(': ')[-1])
                    lrf = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rrf = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lrf = int(g.split(': ')[-1])
            if 'vastus' in g.lower():
                if 'bilateral' in g.lower():
                    rvl = int(g.split(': ')[-1])
                    lvl = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rvl = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lvl = int(g.split(': ')[-1])
            if 'tibialis' in g.lower():
                if 'bilateral' in g.lower():
                    rta = int(g.split(': ')[-1])
                    lta = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rta = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lta = int(g.split(': ')[-1])
            if 'gluteus' in g.lower():
                if 'bilateral' in g.lower():
                    rgm = int(g.split(': ')[-1])
                    lgm = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rgm = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lgm = int(g.split(': ')[-1])
            if 'bicep' in g.lower():
                if 'bilateral' in g.lower():
                    rbf = int(g.split(': ')[-1])
                    lbf = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rbf = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lbf = int(g.split(': ')[-1])
            if 'gastroc' in g.lower():
                if 'bilateral' in g.lower():
                    rg = int(g.split(': ')[-1])
                    lg = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rg = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lg = int(g.split(': ')[-1])
            if 'soleus' in g.lower():
                if 'bilateral' in g.lower():
                    rs = int(g.split(': ')[-1])
                    ls = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rs = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    ls = int(g.split(': ')[-1])
            if 'all other channels' in g.lower():
                l = [rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls] # create temp list with current values
                l1 = [int(g.split(': ')[-1])] * len(l)
                #print(l,l1)
                index_retrieve = [i for i,x in enumerate(l) if np.isnan(x)==False]
                #print(index_retrieve)
                for i in index_retrieve:
                    l1[i] = l[i]
                #print(l1)
                [rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls] = l1
                        
        gains_values = [rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls]
        gains_off_df.loc[len(gains_off_df)] = gains_values
        
        # Repeat for stim on
        [rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls] = [np.nan for _ in range(16)]
        for g in gains_on:
            if 'all channels' in g.lower():
                [rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls] = [int(g.split(' ')[-1])] * len([rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls])
            if 'all left channels' in g.lower():
                [lpm,lrf,lvl,lta,lgm,lbf,lg,ls] = [int(g.split(' ')[-1])] * len([lpm,lrf,lvl,lta,lgm,lbf,lg,ls])
            if 'all right channels' in g.lower():
                [rpm,rrf,rvl,rta,rgm,rbf,rg,rs] = [int(g.split(' ')[-1])] * len([rpm,rrf,rvl,rta,rgm,rbf,rg,rs])
            if 'psoas' in g.lower():
                if 'bilateral' in g.lower():
                    rpm = int(g.split(': ')[-1])
                    lpm = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rpm = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lpm = int(g.split(': ')[-1])
            if 'rectus' in g.lower():
                if 'bilateral' in g.lower():
                    rrf = int(g.split(': ')[-1])
                    lrf = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rrf = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lrf = int(g.split(': ')[-1])
            if 'vastus' in g.lower():
                if 'bilateral' in g.lower():
                    rvl = int(g.split(': ')[-1])
                    lvl = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rvl = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lvl = int(g.split(': ')[-1])
            if 'tibialis' in g.lower():
                if 'bilateral' in g.lower():
                    rta = int(g.split(': ')[-1])
                    lta = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rta = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lta = int(g.split(': ')[-1])
            if 'gluteus' in g.lower():
                if 'bilateral' in g.lower():
                    rgm = int(g.split(': ')[-1])
                    lgm = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rgm = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lgm = int(g.split(': ')[-1])
            if 'bicep' in g.lower():
                if 'bilateral' in g.lower():
                    rbf = int(g.split(': ')[-1])
                    lbf = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rbf = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lbf = int(g.split(': ')[-1])
            if 'gastroc' in g.lower():
                if 'bilateral' in g.lower():
                    rg = int(g.split(': ')[-1])
                    lg = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rg = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    lg = int(g.split(': ')[-1])
            if 'soleus' in g.lower():
                if 'bilateral' in g.lower():
                    rs = int(g.split(': ')[-1])
                    ls = int(g.split(': ')[-1])
                if 'right' in g.lower():
                    rs = int(g.split(': ')[-1])
                if 'left' in g.lower():
                    ls = int(g.split(': ')[-1])
            if 'all other channels' in g.lower():
                l = [rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls] # create temp list with current values
                try:
                    l1 = [int(g[-3:])] * len(l)
                    #print(l,l1)
                    index_retrieve = [i for i,x in enumerate(l) if np.isnan(x)==False]
                    #print(index_retrieve)
                    for i in index_retrieve:
                        l1[i] = l[i]
                    #print(l1)
                    [rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls] = l1
                except:
                    continue
        gains_values = [rpm,lpm,rrf,lrf,rvl,lvl,rta,lta,rgm,lgm,rbf,lbf,rg,lg,rs,ls]
        gains_on_df.loc[len(gains_on_df)] = gains_values
    
    
    # Sort according to dates
    gains_on_df.index = file_dates
    gains_off_df.index = file_dates
    
    gains_on_df = gains_on_df.sort_index()
    gains_off_df = gains_off_df.sort_index()
    
    '''# Plot gains across weeks
    weeks = list(range(1,len(file_dates)+1))
    fig_on = gains_on_df.plot(subplots=True, fontsize=5, title = path[-31:-10] + 'Stim On')
    for p in fig_on:
        p.tick_params(axis='x', labelsize=5)
        p.set_xticks(list(range(0,len(file_dates))))
        p.set_xticklabels(weeks)
        p.set_ylim(0,150)
        p.legend(loc='upper right', frameon=False, fontsize = 5, handlelength=0)
    
    fig_off = gains_off_df.plot(subplots=True, fontsize=5, title = path[-31:-10] + 'Stim Off')
    for p in fig_off:
        p.tick_params(axis='x', labelsize=5)
        p.set_xticks(list(range(0,len(file_dates))))
        p.set_xticklabels(weeks)
        p.set_ylim(0,80)
        p.legend(loc='upper right', frameon=False, fontsize = 5, handlelength=0)'''
    
    # Stats
    gains_on_df.mean()
    gains_on_df.std()
    gains_on_df.mean()
    
    #P1
    if patient =='P1/':
        gains_off_df.index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
        gains_on_df.index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
    #P2 week 15 gain file missing
    if patient =='P2/':
        gains_off_df.index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28]
        gains_on_df.index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28]
    if patient =='P3/':
        gains_off_df.index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
        gains_on_df.index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    # Return
    if stim_condition == 0:
        return gains_off_df.at[week, electrode_labels[electrode]]
    elif stim_condition == 1:
        
        return gains_on_df.at[week, electrode_labels[electrode]]
    else:
        print('Specify condition')
##################### NORMAL FILTERS #######################
### FOR P1 #################
def filter_baseline(raw_signal, Fs): #this function is called in the apply_filters function
    raw_signal = np.nan_to_num(raw_signal) # remove nan values in recording
    # Low pass filter
    nyquist_freq = 0.5 * Fs
    normalized_cutoff = 450 / nyquist_freq # change cutoff frequency here
    #b,a = cheby2(10, 40, [20/nyquist_freq, 450/nyquist_freq], btype='bandpass', analog=False, output='ba', fs=Fs)
    b,a = butter(4, [30,450], btype='band', analog=False, fs=10000)
    filtered_signal = filtfilt(b, a, raw_signal)    
    
    # Notch filter @ 50Hz and 17Hz and 40Hz
    Qstim = 10
    Qpowerline = 30
    Nstimoff = list(range(0,500,17))
    Nstimon = list(range(0,500,40))
    Npowerline = [50,100,150,200,250,300,350,400,450]
    '''for notch in Nstimoff:
        b,a = scipy.signal.iirnotch(notch, Qstim, Fs)
        filtered_signal = lfilter(b, a, filtered_signal)
    for notch in Nstimon:
        b,a = scipy.signal.iirnotch(notch, Qstim, Fs)
        filtered_signal = lfilter(b, a, filtered_signal)
    for notch in Npowerline:
        b,a = scipy.signal.iirnotch(notch, Qpowerline, Fs)
        filtered_signal = lfilter(b, a, filtered_signal)'''
    return filtered_signal

def apply_filters(raw_signal, baseline_signal, Fs): #input the raw baseline
    #raw_signal = raw_signal[20000:]
    raw_signal = np.nan_to_num(raw_signal) # remove nan values in recording
    baseline_signal = np.nan_to_num(baseline_signal)
    # Low pass filter
    nyquist_freq = 0.5 * Fs
    normalized_cutoff = 450 / nyquist_freq # change cutoff frequency here
    #b,a = cheby2(10, 40, [20/nyquist_freq, 450/nyquist_freq], btype='bandpass', analog=False, output='ba', fs=Fs)
    b,a = butter(4, [30,450], btype='band', analog=False, fs=10000)
    filtered_signal = filtfilt(b, a, raw_signal)    
    
    # Notch filter @ 50Hz and 17Hz and 40Hz
    Qstim = 30
    Qpowerline = 30
    Nstimoff = list(range(0,500,17))
    Nstimon = list(range(0,500,40))
    Npowerline = [50,100,150,200,250,300,350,400,450]
    for notch in Nstimoff:
        b,a = scipy.signal.iirnotch(notch, Qstim, Fs)
        filtered_signal = lfilter(b, a, filtered_signal)
    for notch in Nstimon:
        b,a = scipy.signal.iirnotch(notch, Qstim, Fs)
        filtered_signal = lfilter(b, a, filtered_signal)
    for notch in Npowerline:
        b,a = scipy.signal.iirnotch(notch, Qpowerline, Fs)
        filtered_signal = lfilter(b, a, filtered_signal)
    
    # Baseline subtraction and rectify
    baseline_signal = filter_baseline(baseline_signal,Fs)
    baseline_mean = np.nanmean(abs(baseline_signal))
    rect_signal = [max(0,x-baseline_mean) for x in abs(filtered_signal)]

    ########## Segmentation using filtered signal ############ Input: filtered_signal ; Output is: widths
    # Lowpass envelope
    nyquist_freq = 0.5 * Fs
    normalized_cutoff = 0.7 / nyquist_freq # change the cutoff frequence here to change smoothness
    b,a = butter(2,0.7,btype='low', analog=False, fs=10000)
    lowpass_envelope = filtfilt(b, a, rect_signal)
    #plt.plot(lowpass_envelope*20)
    # Find peaks and segment
    peaks, properties = find_peaks(lowpass_envelope,  height = 1.2*np.mean(lowpass_envelope)) #,
    widths = peak_widths(lowpass_envelope, peaks, rel_height=0.4)
    
    if len(peaks) == 0: # ensure when no movements detected, file is still processed
        movement_segments, segmented_signal, movements = np.array([0]), np.array([0]), np.array([0])
        return movement_segments, segmented_signal, movements
    else:
        ########## Segment reconstructed signal ############ Input: reconstructed_signal, widths ; Output is: segmented_signal
        # Assign peaks to unrectified signal to extract segments
        segmented_signal = np.full(shape=np.size(filtered_signal), fill_value=np.nan) #np.zeros(np.size(filtered_signal))
        #segmented_signal[:] = np.nan
        movement_segments = []
    
        for i in range(0,len(widths[1])):
            min_movement = max(0, round(widths[2][i])-5000) # min and max functions so we dont exceed recording length -7000
            max_movement = min(len(filtered_signal), round(widths[3][i])+5000) #+7000
            movements = list(range(min_movement, max_movement)) # list containing indexes with segments of interest
            segmented_signal[movements] = filtered_signal[movements] 
            movement_segments = segmented_signal[~np.isnan(segmented_signal)] # only movements extracted

        # Extract ALL movement indices
        movements = [i for i, x in enumerate(segmented_signal) if not np.isnan(x)]
        
        # Normalize to maximum
        #max_amplitude = max(movement_segments)
        #movement_segments = np.array([x/max_amplitude for x in movement_segments])

        return movement_segments, segmented_signal, movements # movement only, whole recording with 0ed dead-time, movement indices

def segmentation(filtered_signal, Fs):
    rect_signal = abs(filtered_signal)
    ########## Segmentation using filtered signal ############ Input: filtered_signal ; Output is: widths
    # Lowpass envelope
    nyquist_freq = 0.5 * Fs
    normalized_cutoff = 0.7 / nyquist_freq # change the cutoff frequence here to change smoothness
    b,a = butter(2,0.7,btype='low', analog=False, fs=10000)
    lowpass_envelope = filtfilt(b, a, rect_signal)
    #plt.plot(lowpass_envelope*20)
    # Find peaks and segment
    peaks, properties = find_peaks(lowpass_envelope,  height = 1.2*np.mean(lowpass_envelope)) #,
    widths = peak_widths(lowpass_envelope, peaks, rel_height=0.4)
    #plt.plot(lowpass_envelope)
    #print(np.mean(lowpass_envelope))
    if len(peaks) == 0: # ensure when no movements detected, file is still processed
        movement_segments, segmented_signal, movements = np.array([0]), np.array([0]), np.array([0])
        print('No peaks detected, returning input signal')
        return rect_signal, movements
    else:
        ########## Segment reconstructed signal ############ Input: reconstructed_signal, widths ; Output is: segmented_signal
        # Assign peaks to unrectified signal to extract segments
        segmented_signal = np.full(shape=np.size(filtered_signal), fill_value=np.nan) #np.zeros(np.size(filtered_signal))
        #segmented_signal[:] = np.nan
        movement_segments = []
    
        for i in range(0,len(widths[1])):
            min_movement = max(0, round(widths[2][i])) # min and max functions so we dont exceed recording length -7000
            max_movement = min(len(filtered_signal), round(widths[3][i])) #+7000
            movements = list(range(min_movement, max_movement)) # list containing indexes with segments of interest
            segmented_signal[movements] = filtered_signal[movements] 
            movement_segments = segmented_signal[~np.isnan(segmented_signal)] # only movements extracted

        # Extract ALL movement indices
        movements = [i for i, x in enumerate(segmented_signal) if not np.isnan(x)]

    return movement_segments, movements

### FOR P3 #################
def apply_filters_P3(raw_signal, baseline_signal, Fs): 
    raw_signal = np.nan_to_num(raw_signal) # remove nan values in recording
    #print('sdf')
    # Low pass filter
    nyquist_freq = 0.5 * Fs
    normalized_cutoff = 450 / nyquist_freq # change cutoff frequency here
    #b,a = cheby2(10, 40, [20/nyquist_freq, 450/nyquist_freq], btype='bandpass', analog=False, output='ba', fs=Fs)
    b,a = butter(4, [30,450], btype='band', analog=False, fs=10000)
    filtered_signal = filtfilt(b, a, raw_signal)    

    # Notch filter @ 50Hz and 17Hz and 40Hz
    Qstim = 10
    Qpowerline = 30
    Nstimoff = list(range(0,500,17))
    #Nstimon = list(range(0,500,40))
    Npowerline = [50,100,150,200,250,300,350,400,450]
    '''for notch in Nstimoff:
        b,a = scipy.signal.iirnotch(notch, Qstim, Fs)
        filtered_signal = lfilter(b, a, filtered_signal)
    #for notch in Nstimon:
    #    b,a = scipy.signal.iirnotch(notch, Qstim, Fs)
    #    filtered_signal = lfilter(b, a, filtered_signal)
    for notch in Npowerline:
        b,a = scipy.signal.iirnotch(notch, Qpowerline, Fs)
        filtered_signal = lfilter(b, a, filtered_signal)
    
    # Baseline subtraction and rectify
    baseline_signal = filter_baseline(baseline_signal,Fs)
    baseline_mean = np.nanmean(abs(baseline_signal))
    rect_signal = np.array([max(0,x-baseline_mean) for x in abs(filtered_signal)])'''
    
    return np.array(filtered_signal)

############ FREQUENCY FILTERS ######################
# Sliding window to get FFT filter from baseline
def create_sliding_window_frequencey_filter(baseline_signal, window_size, fs): # window size is in 1/10000 s
    baseline_signal = np.nan_to_num(baseline_signal)
    n_windows = math.floor(len(baseline_signal)/window_size) #import math package later to round down with matn.floor()
    yf_all_windows = {} # this will contain yf values for all windows
    xf = fftfreq(window_size, 1/fs)
    yf_max = np.zeros(np.size(xf))
    for i in range(0,n_windows):
        signal_window = baseline_signal[i*window_size:(i+1)*window_size]
        yf = abs(fft(signal_window))
        yf_all_windows['Window_' + str(i)] = yf
        yf_max = [max(yf_max[i], yf[i]) for i in range(len(yf_max))]
    
    return yf_max


def sliding_window_frequencey_filter(raw_signal, yf_max, window_size, fs):
    raw_signal = np.nan_to_num(raw_signal)
    n_windows = math.floor(len(raw_signal)/window_size)
    xf = fftfreq(window_size, 1/fs)
    reconstructed_signal = np.empty(0)
    for i in range(0,n_windows):
        signal_window = raw_signal[i*window_size:(i+1)*window_size]
        phase_yf = np.angle(fft(signal_window)) # phase of fft of window
        amplitude_yf = abs(fft(signal_window)) # amplitude of fft of window
        for i in range(0,len(amplitude_yf)):
            amplitude_yf[i] = amplitude_yf[i] - min(amplitude_yf[i], yf_max[i])
        # Reconstruct from phase and amplitude
        reconstructed_spectrum = amplitude_yf * np.exp(1j * phase_yf)
        reconstructed_signal = np.append(reconstructed_signal, ifft(reconstructed_spectrum))
    
    return reconstructed_signal

# Using self as a baseline
def self_baseline_sliding_window_frequencey_filter(raw_signal, deadtime_window, window_size=1000, fs=10000): # uses start and end of recording as dead time
    raw_signal = np.nan_to_num(raw_signal)
    # Use deadtime window to take baseline
    deadtime_window_start = raw_signal[0:deadtime_window] # index from start
    deadtime_window_end = raw_signal[-deadtime_window:] # index from end
    deadtime_window = np.concatenate((deadtime_window_start, deadtime_window_end))
    n_windows = math.floor(len(deadtime_window)/window_size) #import math package later to round down with matn.floor()
    xf = fftfreq(window_size, 1/fs)
    yf_max = np.zeros(np.size(xf))
    for i in range(0,n_windows):
        signal_window = deadtime_window[i*window_size:(i+1)*window_size]
        yf = abs(fft(signal_window))
        yf_max = [max(yf_max[i], yf[i]) for i in range(len(yf_max))]    
    # Perform filtering
    n_windows = math.floor(len(raw_signal)/window_size)
    xf = fftfreq(window_size, 1/fs)
    reconstructed_signal = np.empty(0)
    for i in range(0,n_windows):
        signal_window = raw_signal[i*window_size:(i+1)*window_size]
        phase_yf = np.angle(fft(signal_window)) # phase of fft of window
        amplitude_yf = abs(fft(signal_window)) # amplitude of fft of window
        for i in range(0,len(amplitude_yf)):
            amplitude_yf[i] = amplitude_yf[i] - min(amplitude_yf[i], yf_max[i])
        # Reconstruct from phase and amplitude
        reconstructed_spectrum = amplitude_yf * np.exp(1j * phase_yf)
        reconstructed_signal = np.append(reconstructed_signal, ifft(reconstructed_spectrum))
    
    return reconstructed_signal   

def lowpass(input_signal, freq, Fs):
    rect_signal = abs(input_signal)
    ########## Segmentation using filtered signal ############ Input: filtered_signal ; Output is: widths
    # Lowpass envelope
    nyquist_freq = 0.5 * Fs
    normalized_cutoff = freq / nyquist_freq # change the cutoff frequence here to change smoothness
    b,a = butter(4,normalized_cutoff,btype='low', analog=False)
    lowpass_envelope = filtfilt(b, a, rect_signal)
    return lowpass_envelope