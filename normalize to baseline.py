# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:44:56 2026

@author: David Teo
"""

# Normalize to baseline corrected (use this one)
import os
import scipy.io
import scipy
from scipy.signal import cheby2, lfilter, filtfilt, butter, find_peaks, peak_widths
from scipy.fft import rfft, rfftfreq, fft, fftfreq, ifft
import matplotlib.pyplot as plt
import numpy as np
from preprocessing_functions import create_sliding_window_frequencey_filter, sliding_window_frequencey_filter, self_baseline_sliding_window_frequencey_filter
from preprocessing_functions import apply_filters, filter_baseline, apply_filters_P3, segmentation, get_gains
import math
import traceback
import sys
import antropy as ant
'''
channellabels={'Lt Psoas M','Lt Rectus Femoris','Lt Vastus Laterali','Lt Tibialis Anterior',...
               'Lt Gluteus Maximus','Lt Bicep Femoris','Lt Gastroc','Lt Soleus',...
               'Rt Psoas M','Rt Rectus Femoris','Rt Vastus Laterali','Rt Tibialis Anterior',...
               'Rt Gluteus Maximus','Rt Bicep Femoris','Rt Gastroc','Rt Soleus'}';
'''
def RMS(input_signal):
    rms = np.sqrt(np.nanmean(input_signal**2))
    return rms

def RMS_envelope(input_signal, window_size=500, overlap=250): # using a 0.01s window #window_size=500, overlap=250 window_size=10000, overlap=5000
    for i in range(0,len(input_signal)-window_size,window_size-overlap):
        window = input_signal[i:i+window_size]
        if i ==0:
            rms = [RMS(window)]
        else:
            rms = rms + [RMS(window)]
    return np.array(rms)
exercises_dict = {2:'Left Hip Flexion',
                  3:'Left Hip Extension',
                  4:'Left Knee Flexion',
                  5:'Left Knee Extension',
                  6:'Left Ankle Dorsiflexion',
                  7:'Left Ankle Plantarflexion',
                  8:'Right Hip Flexion',
                  9:'Right Hip Extension',
                  10:'Right Knee Flexion',
                  11:'Right Knee Extension',
                  12:'Right Ankle Dorsiflexion',
                  13:'Right Ankle Plantarflexion',
                  15:'LHF_on',
                  16:'LHE_on',
                  17:'LKF_on',
                  18:'LKE_on',
                  19:'LAD_on',
                  20:'LAP_on',
                  21:'RHF_on',
                  22:'RHE_on',
                  23:'RKF_on',
                  24:'RKE_on',
                  25:'RAD_on',
                  26:'RAP_on'}
electrode_labels = {0:'Left Psoas Major',
                    1:'Left Rectus Femoris',
                    2:'Left Vastus Laterali',
                    3:'Left Tibalis Anterior',
                    4:'Left Gluteus Maximus',
                    5:'Left Bicep Femoris',
                    6:'Left Gastrocnemius',
                    7:'Left Soleus',
                    8:'Right Psoas Major',
                    9:'Right Rectus Femoris',
                    10:'Right Vastus Laterali',
                    11:'Right Tibalis Anterior',
                    12:'Right Gluteus Maximus',
                    13:'Right Bicep Femoris',
                    14:'Right Gastrocnemius',
                    15:'Right Soleus'}
differences = {}
data_off = {'Patient':[],
        'Muscle':[],
        'Exercise':[],
        'Gradient':[],
        'p':[],
        'r^2':[]}
data_on = {'Patient':[],
        'Muscle':[],
        'Exercise':[],
        'Gradient':[],
        'p':[],
        'r^2':[]}
left_test = [(0,2),(1,2),(1,5),(2,5)]
right_test = [(8, 8), (9, 8),(9, 11),(10, 11)]
left_involved =  [(0,2)]#,(1,2),(1,5),(2,5),(3,6),(4,3),(5,3),(5,4),(6,4),(6,7),(7,7)]
right_involved =  [(8, 8), (9, 8),(9, 11),(10, 11),(11, 12),(12, 9),(13, 9),(13, 10),(14, 10),(14, 13),(15, 13)]
#l_all = [(0, 2),(1, 2),(2, 2),(3, 2),(4, 2),(5, 2),(6, 2),(7, 2),(0, 3),(1, 3),(2, 3),(3, 3),(4, 3),(5, 3),(6, 3),(7, 3),(0, 4),(1, 4),(2, 4),(3, 4),(4, 4),(5, 4),(6, 4),(7, 4),(0, 5),(1, 5),(2, 5),(3, 5),(4, 5),(5, 5),(6, 5),(7, 5),(0, 6),(1, 6),(2, 6),(3, 6),(4, 6),(5, 6),(6, 6),(7, 6),(0, 7),(1, 7),(2, 7),(3, 7),(4, 7),(5, 7),(6, 7),(7, 7)]
#r_all = [(8, 8),(9, 8),(10, 8),(11, 8),(12, 8),(13, 8),(14, 8),(15, 8),(8, 9),(9, 9),(10, 9),(11, 9),(12, 9),(13, 9),(14, 9),(15, 9),(8, 10),(9, 10),(10, 10),(11, 10),(12, 10),(13, 10),(14, 10),(15, 10),(8, 11),(9, 11),(10, 11),(11, 11),(12, 11),(13, 11),(14, 11),(15, 11),(8, 12),(9, 12),(10, 12),(11, 12),(12, 12),(13, 12),(14, 12),(15, 12),(8, 13),(9, 13),(10, 13),(11, 13),(12, 13),(13, 13),(14, 13),(15, 13)]
l_ag_ant = [(0, 2),(1, 2),(4, 2),(5, 2),(0, 3),(1, 3),(4, 3),(5, 3),(1, 4),(2, 4),(5, 4),(6, 4),(1, 5),(2, 5),(5, 5),(6, 5),(3, 6),(6, 6),(7, 6),(3, 7),(6, 7),(7, 7)]
r_ag_ant = [(8, 8),(9, 8),(12, 8),(13, 8),(8, 9),(9, 9),(12, 9),(13, 9),(9, 10),(10, 10),(13, 10),(14, 10),(9, 11),(10, 11),(13, 11),(14, 11),(11, 12),(14, 12),(15, 12),(11, 13),(14, 13),(15, 13)]
oi=[l_ag_ant]+[r_ag_ant]
#oi=[[(0,2),(4,3)]] + [[(8,8), (12,9)]]
for pat in ['P2/','P1/', 'P3/']:
    for o in oi:
    #for i in range(1):
        for a in o: # electrode-exercise pairs
            #print('Starting: ' + electrode_labels[a[0]] + ' in ' + exercises_dict[a[1]])
            Fs=10000
            patient = pat
            if patient == 'P1/':
                weeks = list(range(7,27))
            if patient == 'P2/':
                weeks = list(range(9,29))
            if patient == 'P3/':
                weeks = list(range(1,26))
            electrode_1 = [a[0]]
            exercise_numbers_off = [a[1]] # put your exercise of interest
            exercise_numbers_on = [x+13 for x in exercise_numbers_off]
            exercises_to_include = [a[1], a[1]+13] #these are the exercises to search for maximum peak
            if patient == "P3/":
                exercise_numbers_off = [x-1 for x in exercise_numbers_off]
                exercise_numbers_on = [x-1 for x in exercise_numbers_on]
                exercises_to_include = [x-1 for x in exercises_to_include]
                exercises_dict = {1:'Left Hip Flexion',
                                  2:'Left Hip Extension',
                                  3:'Left Knee Flexion',
                                  4:'Left Knee Extension',
                                  5:'Left Ankle Dorsiflexion',
                                  6:'Left Ankle Plantarflexion',
                                  7:'Right Hip Flexion',
                                  8:'Right Hip Extension',
                                  9:'Right Knee Flexion',
                                  10:'Right Knee Extension',
                                  11:'Right Ankle Dorsiflexion',
                                  12:'Right Ankle Plantarflexion',
                                  14:'LHF_on',
                                  15:'LHE_on',
                                  16:'LKF_on',
                                  17:'LKE_on',
                                  18:'LAD_on',
                                  19:'LAP_on',
                                  20:'RHF_on',
                                  21:'RHE_on',
                                  22:'RKF_on',
                                  23:'RKE_on',
                                  24:'RAD_on',
                                  25:'RAP_on'}
            save_directory = "C:/Users/David Teo/Desktop/after fyp cleanup/norm to baseline/"
            raw_signals_off = {}
            raw_signals_on = {}
            filtered_signals_off = {}
            filtered_signals_on = {}
            RMS_off_dictionary = {}
            RMS_on_dictionary = {}
            RMS_off_mean_dictionary = {}
            RMS_on_mean_dictionary = {}
    
            
            ######################################### Pre-Processing ##################################
            
            ################ FOR P1 #################
            
            if patient == "P1/":
                # Get raw signals in dictionary
                for week in weeks:
                    week = str(week)
                    path = "C:/Users/David Teo/Desktop/Extracted Mat Files/"
                    path = path + patient + "W" + week + '/'
                    try:
                        files = os.listdir(path)
                        baseline_off_signal = scipy.io.loadmat(path+files[0])['data'][electrode_1[0]]
                        baseline_average = np.sqrt(np.nanmean(RMS_envelope(baseline_off_signal)**2))
                        baseline_off_psoas_signal = scipy.io.loadmat(path+files[0])['data'][0] # for segmentation
                        try: # cos week 2 has no stim on exercises
                            baseline_on_file = [file for file in files if file[0:2]=='14']
                            baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data'][electrode_1[0]]
                        except:
                            pass
                        # Stim off block
                        ### ensures exercise of interest loaded first so items are index out correctly later so ratio is calculated properly later
                        exe = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_off[0]]
                        exe1 = [file for file in files if round(float(file.split('.')[0])) in exercise_numbers_off[1:len(exercise_numbers_off)]]
                        files = exe + exe1
                        for file in files:
                            loadpath = path + file
                            #print('starting electrode ', electrode_1[0])
                            # Extract segments by psoas major
                            if electrode_1[0] <8:
                                psoas_signal = scipy.io.loadmat(loadpath)['data'][0][20000:] # remove first 2 seconds cos of artifacts
                            else:
                                psoas_signal = scipy.io.loadmat(loadpath)['data'][8][20000:] # remove first 2 seconds cos of artifacts
                            #print('sdfds')
                            psoas_signal = np.nan_to_num(psoas_signal)
                            _,_,movement_indices = apply_filters(psoas_signal, baseline_off_psoas_signal, Fs)
                            # Real Signal processing
                            raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]][20000:]
                            raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                            filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                            segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                            segmented_signal[movement_indices] = filtered_signal[movement_indices]
                            segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                            # Add filtered signal to dictionary
                            filtered_signals_off['W' + week + '_Exe' + file.split('.')[0]] = np.real(segmented_signal)
                            
                            
                            sd_baseline = 1000
                            mean_baseline = 0
                            for i in range(0,len(filtered_signal)-30000,20000): #3s windows with 1s overlap
                                window = filtered_signal[i:i+30000]
                                mean = np.mean(abs(window))
                                sd = np.std(window)
                                if sd < sd_baseline:
                                    sd_baseline = sd
                                    mean_baseline = mean
                            
                            normalized_signal = segmented_signal/mean_baseline
                            power = 10*math.log10(np.sqrt(np.nanmean(normalized_signal**2)))
                            #print(mean_baseline, 10*math.log10(np.sqrt(np.nanmean(envelope**2))))
                            
                            # Calculate RMS and add to dictionary
                            RMS_off_dictionary['W' + week + '_Exe' + file.split('.')[0]] = power
                            print('off done',patient, week, ' Exe', file.split('.')[0], ' Elec', electrode_1[0])
                            #print(loadpath)  
                        
                        # Stim on block
                        files = os.listdir(path)
                        ### ensures exercise of interest loaded first so items are index out correctly later so ratio is calculated properly later
                        exe = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_on[0]]
                        exe1 = [file for file in files if round(float(file.split('.')[0])) in exercise_numbers_on[1:len(exercise_numbers_off)]]
                        files = exe + exe1
                        #print(files)
                        for file in files:
                            loadpath = path + file
                            #print('starting electrode ', electrode_1[0])
                            # Extract segments by psoas major
                            if electrode_1[0] <8:
                                psoas_signal = scipy.io.loadmat(loadpath)['data'][0][20000:] # remove first 2 seconds cos of artifacts
                            else:
                                psoas_signal = scipy.io.loadmat(loadpath)['data'][8][20000:] # remove first 2 seconds cos of artifacts
                            psoas_signal = np.nan_to_num(psoas_signal)
                            _,_,movement_indices = apply_filters(psoas_signal, baseline_off_psoas_signal, Fs)
                            # Real Signal processing
                            raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]][20000:]
                            raw_signal = np.nan_to_num(raw_signal) # Remove nan values
            
                            filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                            segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                            segmented_signal[movement_indices] = filtered_signal[movement_indices]
                            segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                            # Add filtered signal to dictionary
                            filtered_signals_on['W' + week + '_Exe' + file.split('.')[0]] = np.real(segmented_signal)
                            
                            
                            sd_baseline = 1000
                            mean_baseline = 0
                            for i in range(0,len(filtered_signal)-30000,20000): #3s windows with 1s overlap
                                window = filtered_signal[i:i+30000]
                                mean = np.mean(abs(window))
                                sd = np.std(window)
                                if sd < sd_baseline:
                                    sd_baseline = sd
                                    mean_baseline = mean
                            
                            normalized_signal = segmented_signal/mean_baseline
                            power = 10*math.log10(np.sqrt(np.nanmean(normalized_signal**2)))
                            
                            # Calculate RMS and add to dictionary
                            RMS_on_dictionary['W' + week + '_Exe' + file.split('.')[0]] = power
                            print('on done',patient, week, ' Exe', file.split('.')[0], ' Elec', electrode_1[0])
                            #print(loadpath)  
            
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
            
            if patient =='P2/':
                print('lol')
                # Get raw signals in dictionary
                for week in weeks:
                    if week <21:
                        print(week)
                        week = str(week)
                        path = "C:/Users/David Teo/Desktop/Extracted Mat Files/"
                        path = path + patient + "W" + week + '/'
                        try:
                            files = os.listdir(path)
                            baseline_off_signal = scipy.io.loadmat(path+files[0])['data'][electrode_1[0]]
                            baseline_average = np.sqrt(np.nanmean(RMS_envelope(baseline_off_signal)**2))
                            baseline_off_psoas_signal = scipy.io.loadmat(path+files[0])['data'][0] # for segmentation
                            try: # cos week 2 has no stim on exercises
                                baseline_on_file = [file for file in files if file[0:2]=='14']
                                baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data'][electrode_1[0]]
                            except:
                                pass
                            '''### Find maximum activity in session
                            session_maximum = 0
                            for file in files:
                                if round(float(file.split('.')[0])) in exercises_to_include:#[2,3,4,5,6,7,15,16,17,18,19,20]:
                                    #print(file)
                                    loadpath = path + file
                                    # extract movement segments
                                    psoas_signal = scipy.io.loadmat(loadpath)['data'][0][20000:] # remove first 2 seconds cos of artifacts
                                    psoas_signal = np.nan_to_num(psoas_signal)
                                    _,_,movement_indices = apply_filters(psoas_signal, baseline_off_psoas_signal, Fs)
                                    #print('psoas')
                                    #get muscle of interest signal, filter, and segment
                                    next_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]][20000:]
                                    next_signal = np.nan_to_num(next_signal) # Remove nan values
                                    next_signal = apply_filters_P3(next_signal,baseline_off_signal,Fs)
                                    next_signal = np.nan_to_num(next_signal) # Remove nan values
                                    #print('loaded')
                                    segmented_signal = np.full(shape=np.size(next_signal), fill_value=np.nan)
                                    segmented_signal[movement_indices] = next_signal[movement_indices]
                                    segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                    envelope = RMS_envelope(segmented_signal)
                                    #print(max(envelope))
                                    #add to session and find max
                                    session_maximum = max(session_maximum, max(envelope))
                            #print(session_maximum)'''
                            # Stim off block
                            ### ensures exercise of interest loaded first so items are index out correctly later so ratio is calculated properly later
                            exe = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_off[0]]
                            exe1 = [file for file in files if round(float(file.split('.')[0])) in exercise_numbers_off[1:len(exercise_numbers_off)]]
                            files = exe + exe1
                            for file in files:
                                loadpath = path + file
                                #print('starting electrode ', electrode_1[0])
                                # Extract segments by psoas major
                                if electrode_1[0] <8:
                                    psoas_signal = scipy.io.loadmat(loadpath)['data'][0][20000:] # remove first 2 seconds cos of artifacts
                                else:
                                    psoas_signal = scipy.io.loadmat(loadpath)['data'][8][20000:] # remove first 2 seconds cos of artifacts
                                psoas_signal = np.nan_to_num(psoas_signal)
                                _,_,movement_indices = apply_filters(psoas_signal, baseline_off_psoas_signal, Fs)
                                # Real Signal processing
                                raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]][20000:]
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                # Add filtered signal to dictionary
                                filtered_signals_off['W' + week + '_Exe' + file.split('.')[0]] = np.real(segmented_signal)
                                
                                sd_baseline = 1000
                                mean_baseline = 0
                                for i in range(0,len(filtered_signal)-30000,20000): #3s windows with 1s overlap
                                    window = filtered_signal[i:i+30000]
                                    mean = np.mean(abs(window))
                                    sd = np.std(window)
                                    if sd < sd_baseline:
                                        sd_baseline = sd
                                        mean_baseline = mean
                                
                                normalized_signal = segmented_signal/mean_baseline
                                power = 10*math.log10(np.sqrt(np.nanmean(normalized_signal**2)))
                                
                                # Calculate RMS and add to dictionary
                                RMS_off_dictionary['W' + week + '_Exe' + file.split('.')[0]] = power
        
                                print('off done',patient, week, ' Exe', file.split('.')[0], ' Elec', electrode_1[0])
                            
                            # Stim on block
                            files = os.listdir(path)
                            ### ensures exercise of interest loaded first so items are index out correctly later so ratio is calculated properly later
                            exe = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_on[0]]
                            exe1 = [file for file in files if round(float(file.split('.')[0])) in exercise_numbers_on[1:len(exercise_numbers_off)]]
                            files = exe + exe1
                            #print(files)
                            for file in files:
                                print(files)
                                loadpath = path + file
                                #print('starting electrode ', electrode_1[0])
                                # Extract segments by psoas major
                                if electrode_1[0] <8:
                                    psoas_signal = scipy.io.loadmat(loadpath)['data'][0][20000:] # remove first 2 seconds cos of artifacts
                                else:
                                    psoas_signal = scipy.io.loadmat(loadpath)['data'][8][20000:] # remove first 2 seconds cos of artifacts
                                psoas_signal = np.nan_to_num(psoas_signal)
                                _,_,movement_indices = apply_filters(psoas_signal, baseline_off_psoas_signal, Fs)
                                # Real Signal processing
                                raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]][20000:]
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                # Add filtered signal to dictionary
                                filtered_signals_on['W' + week + '_Exe' + file.split('.')[0]] = np.real(segmented_signal)
                                
                                sd_baseline = 1000
                                mean_baseline = 0
                                for i in range(0,len(filtered_signal)-30000,20000): #3s windows with 1s overlap
                                    window = filtered_signal[i:i+30000]
                                    mean = np.mean(abs(window))
                                    sd = np.std(window)
                                    if sd < sd_baseline:
                                        sd_baseline = sd
                                        mean_baseline = mean
                                
                                normalized_signal = segmented_signal/mean_baseline
                                power = 10*math.log10(np.sqrt(np.nanmean(normalized_signal**2)))
                                
                                
                                
                                # Calculate RMS and add to dictionary
                                RMS_on_dictionary['W' + week + '_Exe' + file.split('.')[0]] = power
                                
                                print('on done',patient, week, ' Exe', file.split('.')[0], ' Elec', electrode_1[0])
                
                        except Exception as e:
                            print(f"An unexpected error occurred: {e}")
                    
                    else:
                        week = str(week)
                        path = "C:/Users/David Teo/Desktop/Extracted Mat Files/"
                        path = path + patient + "W" + week + '/'
                        print(week)
                        try:
                            files = os.listdir(path)
                            baseline_off_signal = scipy.io.loadmat(path+files[0])['data'][electrode_1[0]]
                            baseline_off_psoas_signal = scipy.io.loadmat(path+files[0])['data'][0]
                            baseline_on_file = [file for file in files if file[0:2]=='13']
                            baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data'][electrode_1[0]]
                            baseline_off_rms = np.sqrt(np.nanmean(RMS_envelope(baseline_off_signal)**2))
                            baseline_on_rms = np.sqrt(np.nanmean(RMS_envelope(baseline_on_signal)**2))
                            session_maximum_off = 0
                            session_maximum_on = 0
                            '''for file in files:
                                if round(float(file.split('.')[0])) in exercises_to_include:
                                    if round(float(file.split('.')[0])) < 14:
                                        #print(file)
                                        loadpath = path + file
                                        #get muscle of interest signal, filter, and segment
                                        next_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]]
                                        next_signal = np.nan_to_num(next_signal) # Remove nan values
                                        next_signal = apply_filters_P3(next_signal,baseline_off_signal,Fs)
                                        next_signal = RMS_envelope(next_signal)
                                        #add to session and find max
                                        session_maximum_off = max(session_maximum_off, max(next_signal))
                                        print(session_maximum_off)
                                    else:
                                        print(file)
                                        loadpath = path + file
                                        #get muscle of interest signal, filter, and segment
                                        next_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]]
                                        next_signal = np.nan_to_num(next_signal) # Remove nan values
                                        next_signal = apply_filters_P3(next_signal,baseline_off_signal,Fs)
                                        next_signal = RMS_envelope(next_signal)
                                        #add to session and find max
                                        session_maximum_on = max(session_maximum_on, max(next_signal))
                                        print(session_maximum_on)
                            session_maximum = max(session_maximum_off,session_maximum_on)'''
                            # Stim off block
                            exe_off = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_off[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                            files = exe_off #+ exe1
                            #print(files)
                            
                            for i in range(0,len(files),3):
                                loadpath = path + files[i]
                                loadpath1 = path + files[i+1]
                                loadpath2 = path + files[i+2]
                                
                                #psoas signal for segmentation
                                if electrode_1[0] <8:
                                    psoas_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][0], scipy.io.loadmat(loadpath1)['data'][0], scipy.io.loadmat(loadpath2)['data'][0]))
                                else:
                                    psoas_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][8], scipy.io.loadmat(loadpath1)['data'][8], scipy.io.loadmat(loadpath2)['data'][8]))
                                
                                psoas_signal = np.nan_to_num(psoas_signal)
                                _,_,movement_indices = apply_filters(psoas_signal, baseline_off_psoas_signal, Fs)
                                # Real Signal processing
                                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[0]], scipy.io.loadmat(loadpath1)['data'][electrode_1[0]], scipy.io.loadmat(loadpath2)['data'][electrode_1[0]]))
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                segmented_signal = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                envelope = RMS_envelope(segmented_signal)
                                #segmented_signal = segmented_signal/max(segmented_signal)#/session_maximum#_off
                                plt.plot(envelope)
                                filtered_signals_off['W' + week + '_Exe' + files[i].split('.')[0]] = envelope
                                
                                sd_baseline = 1000
                                mean_baseline = 0
                                for j in range(0,len(filtered_signal)-30000,20000): #3s windows with 1s overlap
                                    window = filtered_signal[j:j+30000]
                                    mean = np.mean(abs(window))
                                    sd = np.std(window)
                                    if sd < sd_baseline:
                                        sd_baseline = sd
                                        mean_baseline = mean
                                
                                normalized_signal = segmented_signal/mean_baseline
                                power = 10*math.log10(np.sqrt(np.nanmean(normalized_signal**2)))
                                
                                # Calculate RMS and add to dictionary
                                RMS_off_dictionary['W' + week + '_Exe' + files[i].split('.')[0]] = power
                                print('off done',patient, week, ' Exe', files[i].split('.')[0], ' Elec', electrode_1[0])
                                
                            # Stim on block
                            files = os.listdir(path)
                            exe_on = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_on[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                            files = exe_on
                            #print(files)
                            for i in range(0,len(files),3):
                                loadpath = path + files[i]
                                loadpath1 = path + files[i+1]
                                loadpath2 = path + files[i+2]
                                #psoas signal for segmentation
                                if electrode_1[0] <8:
                                    psoas_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][0], scipy.io.loadmat(loadpath1)['data'][0], scipy.io.loadmat(loadpath2)['data'][0]))
                                else:
                                    psoas_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][8], scipy.io.loadmat(loadpath1)['data'][8], scipy.io.loadmat(loadpath2)['data'][8]))
                                psoas_signal = np.nan_to_num(psoas_signal)
                                _,_,movement_indices = apply_filters(psoas_signal, baseline_off_psoas_signal, Fs)
                                # Real Signal processing
                                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[0]], scipy.io.loadmat(loadpath1)['data'][electrode_1[0]], scipy.io.loadmat(loadpath2)['data'][electrode_1[0]]))
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                segmented_signal = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                envelope = RMS_envelope(segmented_signal)
                                #segmented_signal = segmented_signal/max(segmented_signal)#/session_maximum#_off
                                plt.plot(envelope)
                                plt.title(week)
                                plt.pause(0.01)
                                filtered_signals_on['W' + week + '_Exe' + files[i].split('.')[0]] = envelope
                                #print(loadpath)
                                # Find baseline
                                
                                sd_baseline = 1000
                                mean_baseline = 0
                                for j in range(0,len(filtered_signal)-30000,20000): #3s windows with 1s overlap
                                    window = filtered_signal[j:j+30000]
                                    mean = np.mean(abs(window))
                                    sd = np.std(window)
                                    if sd < sd_baseline:
                                        sd_baseline = sd
                                        mean_baseline = mean
                                
                                normalized_signal = segmented_signal/mean_baseline
                                power = 10*math.log10(np.sqrt(np.nanmean(normalized_signal**2)))
                                
                                # Calculate RMS and add to dictionary
                                RMS_on_dictionary['W' + week + '_Exe' + files[i].split('.')[0]] = power
                                print('on done',patient, week, ' Exe', files[i].split('.')[0], ' Elec', electrode_1[0])
                        except Exception as e:
                            print(f"An unexpected error occurred: {e}")
                            traceback.print_exc()
                            continue            
                
                
            if patient == "P3/":
                temp=[]
                for week in weeks:
                    week = str(week)
                    path = "C:/Users/David Teo/Desktop/Extracted Mat Files/"
                    path = path + patient + "W" + week + '/'
                    print(week)
                    try:
                        files = os.listdir(path)
                        baseline_off_signal = scipy.io.loadmat(path+files[0])['data'][electrode_1[0]]
                        baseline_off_psoas_signal = scipy.io.loadmat(path+files[0])['data'][0]
                        baseline_on_file = [file for file in files if file[0:2]=='13']
                        baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data'][electrode_1[0]]
                        baseline_off_rms = np.sqrt(np.nanmean(RMS_envelope(baseline_off_signal)**2))
                        baseline_on_rms = np.sqrt(np.nanmean(RMS_envelope(baseline_on_signal)**2))
                        session_maximum_off = 0
                        session_maximum_on = 0
                        '''for file in files:
                            if round(float(file.split('.')[0])) in exercises_to_include:
                                if round(float(file.split('.')[0])) < 14:
                                    #print(file)
                                    loadpath = path + file
                                    #get muscle of interest signal, filter, and segment
                                    next_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]]
                                    next_signal = np.nan_to_num(next_signal) # Remove nan values
                                    next_signal = apply_filters_P3(next_signal,baseline_off_signal,Fs)
                                    next_signal = RMS_envelope(next_signal)
                                    #add to session and find max
                                    session_maximum_off = max(session_maximum_off, max(next_signal))
                                    print(session_maximum_off)
                                else:
                                    print(file)
                                    loadpath = path + file
                                    #get muscle of interest signal, filter, and segment
                                    next_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]]
                                    next_signal = np.nan_to_num(next_signal) # Remove nan values
                                    next_signal = apply_filters_P3(next_signal,baseline_off_signal,Fs)
                                    next_signal = RMS_envelope(next_signal)
                                    #add to session and find max
                                    session_maximum_on = max(session_maximum_on, max(next_signal))
                                    print(session_maximum_on)
                        session_maximum = max(session_maximum_off,session_maximum_on)'''
                        # Stim off block
                        exe_off = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_off[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                        files = exe_off #+ exe1
                        #print(files)
                        temp.append(int(week))
                        for i in range(0,len(files),3):
                            loadpath = path + files[i]
                            loadpath1 = path + files[i+1]
                            loadpath2 = path + files[i+2]
                            
                            #psoas signal for segmentation
                            if electrode_1[0] <8:
                                psoas_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][0], scipy.io.loadmat(loadpath1)['data'][0], scipy.io.loadmat(loadpath2)['data'][0]))
                            else:
                                psoas_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][8], scipy.io.loadmat(loadpath1)['data'][8], scipy.io.loadmat(loadpath2)['data'][8]))
                            psoas_signal = np.nan_to_num(psoas_signal)
                            _,_,movement_indices = apply_filters(psoas_signal, baseline_off_psoas_signal, Fs)
                            # Real Signal processing
                            raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[0]], scipy.io.loadmat(loadpath1)['data'][electrode_1[0]], scipy.io.loadmat(loadpath2)['data'][electrode_1[0]]))
                            raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                            filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                            segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                            segmented_signal[movement_indices] = filtered_signal[movement_indices]
                            segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                            segmented_signal = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                            envelope = RMS_envelope(segmented_signal)
                            #segmented_signal = segmented_signal/max(segmented_signal)#/session_maximum#_off
                            plt.plot(envelope)
                            filtered_signals_off['W' + week + '_Exe' + files[i].split('.')[0]] = envelope
                            
                            sd_baseline = 1000
                            mean_baseline = 0
                            for j in range(0,len(filtered_signal)-30000,20000): #3s windows with 1s overlap
                                window = filtered_signal[j:j+30000]
                                mean = np.mean(abs(window))
                                sd = np.std(window)
                                if sd < sd_baseline:
                                    sd_baseline = sd
                                    mean_baseline = mean
                            
                            normalized_signal = segmented_signal/mean_baseline
                            power = 10*math.log10(np.sqrt(np.nanmean(normalized_signal**2)))
                            
                            
                            
                            # Calculate RMS and add to dictionary
                            RMS_off_dictionary['W' + week + '_Exe' + files[i].split('.')[0]] = power
                            print('off done',patient, week, ' Exe', files[i].split('.')[0], ' Elec', electrode_1[0])
                        # Stim on block
                        files = os.listdir(path)
                        exe_on = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_on[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                        files = exe_on
                        #print(files)
                        for i in range(0,len(files),3):
                            loadpath = path + files[i]
                            loadpath1 = path + files[i+1]
                            loadpath2 = path + files[i+2]
                            #psoas signal for segmentation
                            if electrode_1[0] <8:
                                psoas_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][0], scipy.io.loadmat(loadpath1)['data'][0], scipy.io.loadmat(loadpath2)['data'][0]))
                            else:
                                psoas_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][8], scipy.io.loadmat(loadpath1)['data'][8], scipy.io.loadmat(loadpath2)['data'][8]))
                            psoas_signal = np.nan_to_num(psoas_signal)
                            _,_,movement_indices = apply_filters(psoas_signal, baseline_off_psoas_signal, Fs)
                            # Real Signal processing
                            raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[0]], scipy.io.loadmat(loadpath1)['data'][electrode_1[0]], scipy.io.loadmat(loadpath2)['data'][electrode_1[0]]))
                            raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                            filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                            segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                            segmented_signal[movement_indices] = filtered_signal[movement_indices]
                            segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                            segmented_signal = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                            envelope = RMS_envelope(segmented_signal)
                            #segmented_signal = segmented_signal/max(segmented_signal)#/session_maximum#_off
                            plt.plot(envelope)
                            plt.title(week)
                            plt.pause(0.01)
                            filtered_signals_on['W' + week + '_Exe' + files[i].split('.')[0]] = envelope
                            
                            sd_baseline = 1000
                            mean_baseline = 0
                            for j in range(0,len(filtered_signal)-30000,20000): #3s windows with 1s overlap
                                window = filtered_signal[j:j+30000]
                                mean = np.mean(abs(window))
                                sd = np.std(window)
                                if sd < sd_baseline:
                                    sd_baseline = sd
                                    mean_baseline = mean
                            
                            normalized_signal = segmented_signal/mean_baseline
                            power = 10*math.log10(np.sqrt(np.nanmean(normalized_signal**2)))
                            
                            
                            
                            # Calculate RMS and add to dictionary
                            RMS_on_dictionary['W' + week + '_Exe' + files[i].split('.')[0]] = power
                            print('on done',patient, week, ' Exe', files[i].split('.')[0], ' Elec', electrode_1[0])
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
                        traceback.print_exc()
                        continue            
            ######################################### Analysis ##################################
            ### Stim on analysis  #########
            off_weeks = [int(x.split('_')[0].split('W')[1]) for x in list(RMS_off_dictionary.keys())]
            on_weeks = [int(x.split('_')[0].split('W')[1]) for x in list(RMS_on_dictionary.keys())]
            #plt.plot(off_weeks,list(RMS_off_dictionary.values()), marker = 'o', label = 'stim off')
            #plt.plot(on_weeks,list(RMS_on_dictionary.values()), marker = 'x', label = 'stim on')
            plt.title(patient[0:2]+' '+electrode_labels[electrode_1[0]] + ' ' + exercises_dict[exercise_numbers_off[0]] + ' Normalized /n Stimulation On')
            plt.xlabel('Weeks')
            plt.xticks(range(1,max(on_weeks)+1), fontsize=7)
            plt.ylabel('Normalized RMS')
            #plt.ylim(0,1)
            
            
            
            # Create data
            from sklearn.linear_model import LinearRegression
            x = np.array(on_weeks).reshape(-1, 1) # P1: [7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24]
            y = np.array(list(RMS_on_dictionary.values()))
            # 1. Create a scatter plot
            plt.scatter(x, y, color='black')
            # 2. Create and fit the linear regression model
            model = LinearRegression().fit(x, y)
            y_pred = model.predict(x) # Predict y values using the model
            # Access the coefficients and intercept
            intercept = model.intercept_
            coefficients = model.coef_
            #P value
            #correlation_coefficient, p_value = scipy.stats.pearsonr(np.array(on_weeks), y)
            slope, intercept, correlation_coefficient, p_value, std_err = scipy.stats.linregress(np.array(on_weeks), y)
            # Print the results
            print("Intercept:", intercept)
            print("Coefficients:", coefficients)
            # 3. Plot the regression line
            r2_test = model.score(x,y)
            plt.plot(x, y_pred, color='red', label = 'Regression line: y='+str(np.round(coefficients[0], decimals=2))+'x + ' + str(np.round(intercept, decimals=2))+ ', p = ' + str(np.round(p_value, decimals=3)) + ', r^2 = ' + str(np.round(r2_test, decimals=3))) # R: 'Regression line: y=0.024x + 3.14' L: 'Regression line: y=0.056x + 2.92'        
            plt.legend(loc='upper right')
            print(f"R^2: {r2_test:.4f}")
            
            save_directory = "C:/Users/David Teo/Desktop/after fyp cleanup/norm to baseline/"#redo real left and right/on norm to baseline/"
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(save_directory + patient[0:2]+electrode_labels[electrode_1[0]] + ' ' + exercises_dict[exercise_numbers_off[0]] + ' Normalized Stimulation On'+".svg")
            plt.pause(0.01)
            
            
            #save as table
            data_on['Patient'].append(patient[0:2])
            data_on['Exercise'].append(exercises_dict[exercise_numbers_off[0]])
            data_on['Muscle'].append(electrode_labels[electrode_1[0]])
            data_on['Gradient'].append(coefficients[0])
            data_on['p'].append(p_value)
            data_on['r^2'].append(r2_test)
            
            
            ### Stim off analysis  #########
            off_weeks = [int(x.split('_')[0].split('W')[1]) for x in list(RMS_off_dictionary.keys())]
            on_weeks = [int(x.split('_')[0].split('W')[1]) for x in list(RMS_on_dictionary.keys())]
            #plt.plot(off_weeks,list(RMS_off_dictionary.values()), marker = 'o', label = 'stim off')
            #plt.plot(on_weeks,list(RMS_on_dictionary.values()), marker = 'x', label = 'stim on')
            plt.title(patient[0:2]+' '+electrode_labels[electrode_1[0]] + ' ' + exercises_dict[exercise_numbers_off[0]] + ' Normalized /n Stimulation Off')
            plt.xlabel('Weeks')
            plt.xticks(range(1,max(on_weeks)+1), fontsize=7)
            plt.ylabel('Normalized RMS')
            #plt.ylim(0,1)
            
            
            
            # Create data
            from sklearn.linear_model import LinearRegression
            x = np.array(off_weeks).reshape(-1, 1) # P1: [7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24]
            y = np.array(list(RMS_off_dictionary.values()))
            # 1. Create a scatter plot
            plt.scatter(x, y, color='black')
            # 2. Create and fit the linear regression model
            model = LinearRegression().fit(x, y)
            y_pred = model.predict(x) # Predict y values using the model
            # Access the coefficients and intercept
            intercept = model.intercept_
            coefficients = model.coef_
            #P value
            #correlation_coefficient, p_value = scipy.stats.pearsonr(np.array(off_weeks), y)
            slope, intercept, correlation_coefficient, p_value, std_err = scipy.stats.linregress(np.array(off_weeks), y)
            # Print the results
            print("Intercept:", intercept)
            print("Coefficients:", coefficients)
            # 3. Plot the regression line
            r2_test = model.score(x,y)
            plt.plot(x, y_pred, color='red', label = 'Regression line: y='+str(np.round(coefficients[0], decimals=2))+'x + ' + str(np.round(intercept, decimals=2))+ ', p = ' + str(np.round(p_value, decimals=3)) + ', r^2 = ' + str(np.round(r2_test, decimals=3))) # R: 'Regression line: y=0.024x + 3.14' L: 'Regression line: y=0.056x + 2.92'        
            plt.legend(loc='upper right')
            print(f"R^2: {r2_test:.4f}")
            
            save_directory = "C:/Users/David Teo/Desktop/after fyp cleanup/norm to baseline/"#redo real left and right/off norm to baseline/"
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(save_directory + patient[0:2]+electrode_labels[electrode_1[0]] + ' ' + exercises_dict[exercise_numbers_off[0]] + ' Normalized Stimulation Off'+".svg")
            plt.pause(0.01)
            
            
            #save as table
            data_off['Patient'].append(patient[0:2])
            data_off['Exercise'].append(exercises_dict[exercise_numbers_off[0]])
            data_off['Muscle'].append(electrode_labels[electrode_1[0]])
            data_off['Gradient'].append(coefficients[0])
            data_off['p'].append(p_value)
            data_off['r^2'].append(r2_test)
            
import pandas as pd
df = pd.DataFrame(data_on)
df.to_csv(save_directory + 'P12RMS across weeks stim on_norm to baseline.csv', index=False)
df = pd.DataFrame(data_off)
df.to_csv(save_directory + 'P12RMS across weeks stim off_norm to baseline.csv', index=False)