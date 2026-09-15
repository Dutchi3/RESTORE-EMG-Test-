# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:58:36 2026

@author: David Teo
"""

import os
import scipy.io
import scipy
import scipy.signal as signal
from scipy.signal import cheby2, lfilter, filtfilt, butter, find_peaks, peak_widths
from scipy.fft import rfft, rfftfreq, fft, fftfreq, ifft
import matplotlib.pyplot as plt
import numpy as np
from preprocessing_functions import create_sliding_window_frequencey_filter, sliding_window_frequencey_filter, self_baseline_sliding_window_frequencey_filter
from preprocessing_functions import apply_filters, filter_baseline, apply_filters_P3, segmentation, get_gains
import math
import traceback
import sys
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

coh_dict = {'Patient':[],
            'Frequency Band':[],
            'Exercise':[],
            'Muscles':[],
            'Difference':[],
            'P value':[]            
            }
custom = [(0,1,2)]
left_involved = [(0,1,2),(4,5,3),(5,6,4),(1,2,5),(6,7,7)] #e1,e1,exercise
# finding 06: the last right-side entry was (13,14,13) - Bicep Femoris + Gastrocnemius,
# a hamstring paired with a calf muscle, for ankle plantarflexion.  The left side
# correctly mirrors as (6,7,7) = Gastrocnemius + Soleus, and the study's own
# "Muscles for exercises" document specifies electrodes 14,15 for that exercise.
right_involved =  [(8,9,8),(12,13,9),(13,14,10),(9,10,11),(14,15,13)]
#l_all = [(0, 2),(1, 2),(2, 2),(3, 2),(4, 2),(5, 2),(6, 2),(7, 2),(0, 3),(1, 3),(2, 3),(3, 3),(4, 3),(5, 3),(6, 3),(7, 3),(0, 4),(1, 4),(2, 4),(3, 4),(4, 4),(5, 4),(6, 4),(7, 4),(0, 5),(1, 5),(2, 5),(3, 5),(4, 5),(5, 5),(6, 5),(7, 5),(0, 6),(1, 6),(2, 6),(3, 6),(4, 6),(5, 6),(6, 6),(7, 6),(0, 7),(1, 7),(2, 7),(3, 7),(4, 7),(5, 7),(6, 7),(7, 7)]
#r_all = [(8, 8),(9, 8),(10, 8),(11, 8),(12, 8),(13, 8),(14, 8),(15, 8),(8, 9),(9, 9),(10, 9),(11, 9),(12, 9),(13, 9),(14, 9),(15, 9),(8, 10),(9, 10),(10, 10),(11, 10),(12, 10),(13, 10),(14, 10),(15, 10),(8, 11),(9, 11),(10, 11),(11, 11),(12, 11),(13, 11),(14, 11),(15, 11),(8, 12),(9, 12),(10, 12),(11, 12),(12, 12),(13, 12),(14, 12),(15, 12),(8, 13),(9, 13),(10, 13),(11, 13),(12, 13),(13, 13),(14, 13),(15, 13)]
oi=[left_involved]+[right_involved]
for pat in ["P3/", "P2/", "P1/"]:
    differences = {}
    for c in [(32,48)]:#[(32,40),(32,60),(52,96),(104,148)]:#[(32,50)]:#
        for o in oi: # this if to iterate left than right (if i wanna iterate overnight)
            for a in o: # electrode-exercise pairs
                #print('Starting: ' + electrode_labels[a[0]] + '-' + electrode_labels[a[1]] + ' in ' + exercises_dict[a[2]])
                Fs=10000
                patient = pat#"P1/"
                if patient == 'P1/':
                    weeks = list(range(7,28))
                if patient == 'P2/':
                    weeks = list(range(9,28))
                if patient == 'P3/':
                    weeks = list(range(1,26))
                electrode_1 = [a[0], a[1]]
                exercise_numbers_off = [a[2]] # put your exercise of interest
                exercise_numbers_on = [x+13 for x in exercise_numbers_off]
                exercises_to_include = [a[2], a[2]+13] #these are the exercises to search for maximum peak
                if patient == "P3/":
                    exercise_numbers_off = [x-1 for x in exercise_numbers_off]
                    exercise_numbers_on = [x-1 for x in exercise_numbers_on]
                    exercises_to_include = [x-1 for x in exercises_to_include]
                    exercises_dict = {1:'Left Hip Flexion',
                                      2:'Left Hip Extension',
                                      3:'Left Knee Flexion',
                                      4:'Left Knee Extension',
                                      5:'Left Ankle Dorsiflexion',
                                      6:'Left Ankle PLantarflexion',
                                      7:'Right Hip Flexion',
                                      8:'Right Hip Extension',
                                      9:'Right Knee Flexion',
                                      10:'Right Knee Extension',
                                      11:'Right Ankle Dorsiflexion',
                                      12:'Right Ankle PLantarflexion',
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
                save_directory = "C:/Users/David Teo/Desktop/test images/"
                raw_signals_off = {}
                raw_signals_on = {}
                filtered_signals_off = {}
                filtered_signals_on = {}
                RMS_off_dictionary = {}
                RMS_on_dictionary = {}
                RMS_off_mean_dictionary = {}
                RMS_on_mean_dictionary = {}
                total_coherence_off = {}
                total_coherence_on = {}
                #Coherence parameters
                noverlap = 1250
                nperseg = 2500
                coherence_start, coherence_end = c[0],c[1]
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
                                    session_maximum = max(session_maximum, max(envelope))'''
                            #print(session_maximum)
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
                                # Calulate for electrode 1
                                raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]][20000:]
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                signal_1 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                plt.pause(0.01)
                                
                                #print(loadpath)  
                                
                                # Calculate for electrode 2
                                raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[1]][20000:]
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                signal_2 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                
                                # Calculate coherence and add to dictionary
                                f1, Cxy1 = signal.coherence(signal_1, signal_2, fs=Fs,nperseg = nperseg, noverlap=noverlap) # ususally use nperseg = 10000; nperseg = 2500 sets freq bins to 2.5Hz
                                #coherence_on[str(week)]= [f1, Cxy1]
                                confidence_limit = 1-(0.05)**(1/((len(signal_1)-noverlap)//(nperseg-noverlap)-1))
                                print(week, ' Stim on CL: ', confidence_limit)
                                freq_band = sum(x for x in Cxy1[int(coherence_start//(Fs/nperseg)):int((coherence_end+1)//(Fs/nperseg))] if x>=confidence_limit) #15:21 for nperseg=10000; if nperseg=2500, 3:6 for 12-20Hz; 5:11 for 20-40Hz
                                print('Frequency band: ', f1[int(coherence_start//(Fs/nperseg))], '-', f1[int((coherence_end+1)//(Fs/nperseg))])
                                total_coherence_off[week] = freq_band
                                
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
                                # Calulate for electrode 1
                                raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]][20000:]
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                signal_1 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                plt.pause(0.01)
                                
                                #print(loadpath)  
                                
                                # Calculate for electrode 2
                                raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[1]][20000:]
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                signal_2 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                
                                # Calculate coherence and add to dictionary
                                f1, Cxy1 = signal.coherence(signal_1, signal_2, fs=Fs,nperseg = nperseg, noverlap=noverlap) # ususally use nperseg = 10000; nperseg = 2500 sets freq bins to 2.5Hz
                                #coherence_on[str(week)]= [f1, Cxy1]
                                confidence_limit = 1-(0.05)**(1/((len(signal_1)-noverlap)//(nperseg-noverlap)-1))
                                print(week, ' Stim on CL: ', confidence_limit)
                                freq_band = sum(x for x in Cxy1[int(coherence_start//(Fs/nperseg)):int((coherence_end+1)//(Fs/nperseg))] if x>=confidence_limit) #15:21 for nperseg=10000; if nperseg=2500, 3:6 for 12-20Hz; 5:11 for 20-40Hz
                                print('Frequency band: ', f1[int(coherence_start//(Fs/nperseg))], '-', f1[int((coherence_end+1)//(Fs/nperseg))])
                                total_coherence_on[week] = freq_band
                
                        except Exception as e:
                            print(f"An unexpected error occurred: {e}")
                
                
                if patient =='P2/':
                    # Get raw signals in dictionary
                    for week in weeks:
                        if week <21:
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
                                    #print('sdfds')
                                    psoas_signal = np.nan_to_num(psoas_signal)
                                    _,_,movement_indices = apply_filters(psoas_signal, baseline_off_psoas_signal, Fs)
                                    # Real Signal processing
                                    # Calulate for electrode 1
                                    raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]][20000:]
                                    raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                    filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                    segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                    segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                    segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                    signal_1 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                    plt.pause(0.01)
                                    
                                    #print(loadpath)  
                                    
                                    # Calculate for electrode 2
                                    raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[1]][20000:]
                                    raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                    filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                    segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                    segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                    segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                    signal_2 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                    
                                    # Calculate coherence and add to dictionary
                                    f1, Cxy1 = signal.coherence(signal_1, signal_2, fs=Fs,nperseg = nperseg, noverlap=noverlap) # ususally use nperseg = 10000; nperseg = 2500 sets freq bins to 2.5Hz
                                    #coherence_on[str(week)]= [f1, Cxy1]
                                    confidence_limit = 1-(0.05)**(1/((len(signal_1)-noverlap)//(nperseg-noverlap)-1))
                                    print(week, ' Stim on CL: ', confidence_limit)
                                    freq_band = sum(x for x in Cxy1[int(coherence_start//(Fs/nperseg)):int((coherence_end+1)//(Fs/nperseg))] if x>=confidence_limit) #15:21 for nperseg=10000; if nperseg=2500, 3:6 for 12-20Hz; 5:11 for 20-40Hz
                                    print('Frequency band: ', f1[int(coherence_start//(Fs/nperseg))], '-', f1[int((coherence_end+1)//(Fs/nperseg))])
                                    total_coherence_off[week] = freq_band
                                    
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
                                    # Calulate for electrode 1
                                    raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[0]][20000:]
                                    raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                    filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                    segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                    segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                    segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                    signal_1 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                    plt.pause(0.01)
                                    
                                    #print(loadpath)  
                                    
                                    # Calculate for electrode 2
                                    raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[1]][20000:]
                                    raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                    filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                    segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                    segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                    segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                    signal_2 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                    
                                    # Calculate coherence and add to dictionary
                                    f1, Cxy1 = signal.coherence(signal_1, signal_2, fs=Fs,nperseg = nperseg, noverlap=noverlap) # ususally use nperseg = 10000; nperseg = 2500 sets freq bins to 2.5Hz
                                    #coherence_on[str(week)]= [f1, Cxy1]
                                    confidence_limit = 1-(0.05)**(1/((len(signal_1)-noverlap)//(nperseg-noverlap)-1))
                                    print(week, ' Stim on CL: ', confidence_limit)
                                    freq_band = sum(x for x in Cxy1[int(coherence_start//(Fs/nperseg)):int((coherence_end+1)//(Fs/nperseg))] if x>=confidence_limit) #15:21 for nperseg=10000; if nperseg=2500, 3:6 for 12-20Hz; 5:11 for 20-40Hz
                                    print('Frequency band: ', f1[int(coherence_start//(Fs/nperseg))], '-', f1[int((coherence_end+1)//(Fs/nperseg))])
                                    total_coherence_on[week] = freq_band
                    
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
                                    # signal 1
                                    raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[0]], scipy.io.loadmat(loadpath1)['data'][electrode_1[0]], scipy.io.loadmat(loadpath2)['data'][electrode_1[0]]))
                                    raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                    filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                    segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                    segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                    segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                    signal_1 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                    
                                    # signal 2
                                    raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[1]], scipy.io.loadmat(loadpath1)['data'][electrode_1[1]], scipy.io.loadmat(loadpath2)['data'][electrode_1[1]]))
                                    raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                    filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                    segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                    segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                    segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                    signal_2 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                    # Calculate coherence and add to dictionary
                                    f1, Cxy1 = signal.coherence(signal_1, signal_2, fs=Fs,nperseg = nperseg, noverlap=noverlap) # ususally use nperseg = 10000; nperseg = 2500 sets freq bins to 2.5Hz
                                    #coherence_on[str(week)]= [f1, Cxy1]
                                    confidence_limit = 1-(0.05)**(1/((len(signal_1)-noverlap)//(nperseg-noverlap)-1))
                                    print(week, ' Stim on CL: ', confidence_limit)
                                    freq_band = sum(x for x in Cxy1[int(coherence_start//(Fs/nperseg)):int((coherence_end+1)//(Fs/nperseg))] if x>=confidence_limit) #15:21 for nperseg=10000; if nperseg=2500, 3:6 for 12-20Hz; 5:11 for 20-40Hz
                                    print('Frequency band: ', f1[int(coherence_start//(Fs/nperseg))], '-', f1[int((coherence_end+1)//(Fs/nperseg))])
                                    total_coherence_off[week] = freq_band
                                    
                                    
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
                                    # signal 1
                                    raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[0]], scipy.io.loadmat(loadpath1)['data'][electrode_1[0]], scipy.io.loadmat(loadpath2)['data'][electrode_1[0]]))
                                    raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                    filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                    segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                    segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                    segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                    signal_1 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                    
                                    # signal 2
                                    raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[1]], scipy.io.loadmat(loadpath1)['data'][electrode_1[1]], scipy.io.loadmat(loadpath2)['data'][electrode_1[1]]))
                                    raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                    filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                    segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                    segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                    segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                    signal_2 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                    # Calculate coherence and add to dictionary
                                    f1, Cxy1 = signal.coherence(signal_1, signal_2, fs=Fs,nperseg = nperseg, noverlap=noverlap) # ususally use nperseg = 10000; nperseg = 2500 sets freq bins to 2.5Hz
                                    #coherence_on[str(week)]= [f1, Cxy1]
                                    confidence_limit = 1-(0.05)**(1/((len(signal_1)-noverlap)//(nperseg-noverlap)-1))
                                    print(week, ' Stim on CL: ', confidence_limit)
                                    freq_band = sum(x for x in Cxy1[int(coherence_start//(Fs/nperseg)):int((coherence_end+1)//(Fs/nperseg))] if x>=confidence_limit) #15:21 for nperseg=10000; if nperseg=2500, 3:6 for 12-20Hz; 5:11 for 20-40Hz
                                    print('Frequency band: ', f1[int(coherence_start//(Fs/nperseg))], '-', f1[int((coherence_end+1)//(Fs/nperseg))])
                                    total_coherence_on[week] = freq_band
                                    
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
                                # signal 1
                                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[0]], scipy.io.loadmat(loadpath1)['data'][electrode_1[0]], scipy.io.loadmat(loadpath2)['data'][electrode_1[0]]))
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                signal_1 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                
                                # signal 2
                                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[1]], scipy.io.loadmat(loadpath1)['data'][electrode_1[1]], scipy.io.loadmat(loadpath2)['data'][electrode_1[1]]))
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                signal_2 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                #print(loadpath)
                                # Calculate coherence and add to dictionary
                                f1, Cxy1 = signal.coherence(signal_1, signal_2, fs=Fs,nperseg = nperseg, noverlap=noverlap) # ususally use nperseg = 10000; nperseg = 2500 sets freq bins to 2.5Hz
                                #coherence_on[str(week)]= [f1, Cxy1]
                                confidence_limit = 1-(0.05)**(1/((len(signal_1)-noverlap)//(nperseg-noverlap)-1))
                                print(week, ' Stim on CL: ', confidence_limit)
                                freq_band = sum(x for x in Cxy1[int(coherence_start//(Fs/nperseg)):int((coherence_end+1)//(Fs/nperseg))] if x>=confidence_limit) #15:21 for nperseg=10000; if nperseg=2500, 3:6 for 12-20Hz; 5:11 for 20-40Hz
                                print('Frequency band: ', f1[int(coherence_start//(Fs/nperseg))], '-', f1[int((coherence_end+1)//(Fs/nperseg))])
                                total_coherence_off[week] = freq_band
                                
                                
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
                                # signal 1
                                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[0]], scipy.io.loadmat(loadpath1)['data'][electrode_1[0]], scipy.io.loadmat(loadpath2)['data'][electrode_1[0]]))
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                signal_1 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                
                                # signal 2
                                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[1]], scipy.io.loadmat(loadpath1)['data'][electrode_1[1]], scipy.io.loadmat(loadpath2)['data'][electrode_1[1]]))
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                filtered_signal = apply_filters_P3(raw_signal,baseline_off_signal,Fs)
                                segmented_signal = np.full(shape=np.size(raw_signal), fill_value=np.nan)
                                segmented_signal[movement_indices] = filtered_signal[movement_indices]
                                segmented_signal = segmented_signal[~np.isnan(segmented_signal)] # remove dead time
                                signal_2 = np.abs(segmented_signal)#/session_maximum#/baseline_average#/session_maximum
                                #print(loadpath)
                                # Calculate coherence and add to dictionary
                                f1, Cxy1 = signal.coherence(signal_1, signal_2, fs=Fs,nperseg = nperseg, noverlap=noverlap) # ususally use nperseg = 10000; nperseg = 2500 sets freq bins to 2.5Hz
                                #coherence_on[str(week)]= [f1, Cxy1]
                                confidence_limit = 1-(0.05)**(1/((len(signal_1)-noverlap)//(nperseg-noverlap)-1))
                                print(week, ' Stim on CL: ', confidence_limit)
                                freq_band = sum(x for x in Cxy1[int(coherence_start//(Fs/nperseg)):int((coherence_end+1)//(Fs/nperseg))] if x>=confidence_limit) #15:21 for nperseg=10000; if nperseg=2500, 3:6 for 12-20Hz; 5:11 for 20-40Hz
                                print('Frequency band: ', f1[int(coherence_start//(Fs/nperseg))], '-', f1[int((coherence_end+1)//(Fs/nperseg))])
                                total_coherence_on[week] = freq_band
                                
                        except Exception as e:
                            print(f"An unexpected error occurred: {e}")
                            traceback.print_exc()
                            continue            
                ######################################### Analysis ##################################
                ### Stim off analysis redo #########
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
                off_weeks = [int(x) for x in list(total_coherence_off.keys())]
                on_weeks = [int(x) for x in list(total_coherence_on.keys())]
                plt.plot(off_weeks,list(total_coherence_off.values()), marker = 'o', label = 'stim off')
                plt.plot(on_weeks,list(total_coherence_on.values()), marker = 'x', label = 'stim on')
                plt.title(patient[0:2]+' '+exercises_dict[a[2]]+' Coherence between \n'+electrode_labels[a[0]] + '-' + electrode_labels[a[1]] + ' at '+str(f1[int(coherence_start//(Fs/nperseg))])+ '-'+ str(f1[int((coherence_end+1)//(Fs/nperseg))])+'Hz')
                
                plt.xlabel('Weeks')
                plt.xticks(list(range(min(off_weeks[0],on_weeks[0]),max(off_weeks[-1],on_weeks[-1])+1)))
                plt.ylabel('Coherence')
                
                from scipy.stats import ttest_rel, wilcoxon
                x=list(total_coherence_on.keys())
                y=list(total_coherence_off.keys())
                x1 = [i.split('_')[0] for i in x]
                y1 = [i.split('_')[0] for i in y]
                z = []
                for i in range(0,len(y)):
                    if y[i].split('_')[0] in x1:
                        z.append(y[i])
                y=z
                z = []
                for i in range(0,len(x)):
                    if x[i].split('_')[0] in y1:
                        z.append(x[i])
                x=z
                x=[total_coherence_on[i] for i in x]
                y=[total_coherence_off[i] for i in y]
                #t_statistic, p_value = ttest_rel(x, y, alternative='greater')
                w = wilcoxon(x, y, alternative='greater')
                print('p_value = ', w.pvalue)
                
                
                difference = [x[i]-y[i] for i,_ in enumerate(x)]
                differences[a] = difference
                
                plt.pause(0.1)
                
                
                '''#plot
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
                plt.pause(0.001)
                plt.plot([int(x) for x in list(total_coherence_off.keys())], list(total_coherence_off.values()), label='Stim off')
                plt.plot([int(x) for x in list(total_coherence_on.keys())], list(total_coherence_on.values()), label='Stim on')
                plt.xticks(list(range(1,1+max([int(x) for x in list(total_coherence_on.keys())]))), fontsize = 7)
                #plt.axhline(y=0, color='r', linestyle='--')
                #plt.ylim(0,4)
                plt.title(patient[0:2]+' '+exercises_dict[a[2]]+'\n Coherence between '+electrode_labels[a[0]] + '-' + electrode_labels[a[1]] + '\n at '+str(f1[int(coherence_start//(Fs/nperseg))])+ '-'+ str(f1[int((coherence_end+1)//(Fs/nperseg))])+'Hz')
                plt.tick_params(axis='x', labelsize=8)
                plt.tight_layout()
                plt.legend()
                save_directory = "C:/Users/David Teo/Desktop/rms images/"
                plt.savefig(save_directory + patient[0:2]+' '+exercises_dict[a[2]]+' Coherence between '+electrode_labels[a[0]] + '-' + electrode_labels[a[1]] + ' at '+str(f1[int(coherence_start//(Fs/nperseg))])+ '-'+ str(f1[int((coherence_end+1)//(Fs/nperseg))])+'Hz'+".png")
                '''
    #### PLOTTY PLOTTY #### Barplots of stim on-stim off compared to 0
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
    muscle_exercise_pairs = list(differences.keys())
    exercises = [i[2] for i in muscle_exercise_pairs]
    muscles = []
    diff = []
    medians = []
    q1_q3 = []
    signif = []
    for i in muscle_exercise_pairs:
        
        muscles.append(electrode_labels[i[0]]+'-'+electrode_labels[i[1]]+'\n'+exercises_dict[i[2]])
        diff.append(differences[i])
        medians.append(np.median(differences[i]))
        q1_q3.append([np.percentile(differences[i], 25), np.percentile(differences[i], 75)])
        p = wilcoxon(differences[i])[1]
        '''if p > 0.05:
            p ='n.s.'
        elif p < 0.05 and p>0.01:
            p = '*'
        elif p<0.01 and p>0.001:
            p = '**'
        elif p<0.001:
            p = '***'
        else:
            p = 'gobbeldygook'''
        signif.append(p)
        
        # Append to dictionary to save as csv later
        coh_dict['Patient'].append(pat)
        coh_dict['Frequency Band'].append([coherence_start,coherence_end])
        coh_dict['Exercise'].append(exercises_dict[i[2]])
        coh_dict['Muscles'].append(electrode_labels[i[0]]+'-'+electrode_labels[i[1]])
        coh_dict['Difference'].append(diff)
        coh_dict['P value'].append(p)
    
        # conduct wilcoxn signed rank test
        
    #plot
    plt.pause(0.001)
    plt.boxplot(diff,tick_labels=muscles)
    plt.axhline(y=0, color='r', linestyle='--')
    for j in range(1,len(medians)+1):
        plt.text(j, max(diff[j-1])+0.1, signif[j-1], ha='center', va='bottom')
    plt.ylim(-4,4)
    plt.title(patient[0:2]+' '+exercises_dict[a[2]]+' Coherence at '+str(f1[int(coherence_start//(Fs/nperseg))])+ '-'+ str(f1[int((coherence_end+1)//(Fs/nperseg))])+'Hz')
    plt.tick_params(axis='x', labelsize=6.5, labelrotation = 80)
    save_directory = "C:/Users/David Teo/Desktop/fyp reals/"
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(save_directory + patient[0:2]+' '+exercises_dict[a[2]]+' Coherence between '+electrode_labels[a[0]] + '-' + electrode_labels[a[1]] + ' at '+str(f1[int(coherence_start//(Fs/nperseg))])+ '-'+ str(f1[int((coherence_end+1)//(Fs/nperseg))])+'Hz'+".svg")

import pandas as pd
df = pd.DataFrame(coh_dict)
df.to_csv(save_directory + 'Coherencepval.csv', index=False)
