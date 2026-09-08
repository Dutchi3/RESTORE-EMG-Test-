# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:53:13 2026

@author: David Teo
"""


import os
import scipy.io
import scipy
from scipy.signal import cheby2, lfilter, filtfilt, butter, find_peaks, peak_widths
from scipy.fft import rfft, rfftfreq, fft, fftfreq, ifft
import matplotlib.pyplot as plt
import numpy as np
from preprocessing_functions import apply_filters, filter_baseline, apply_filters_P3, segmentation
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





exercises_dict = {2:'LHF_off',
                  3:'LHE_off',
                  4:'LKF_off',
                  5:'LKE_off',
                  6:'LAD_off',
                  7:'LAP_off',
                  8:'RHF_off',
                  9:'RHE_off',
                  10:'RKF_off',
                  11:'RKE_off',
                  12:'RAD_off',
                  13:'RAP_off',
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
electrode_labels = {0:'L Psoas Major',
                    1:'L Rectus Femoris',
                    2:'L Vastus Laterali',
                    3:'L Tibalis Anterior',
                    4:'L Gluteus Maximus',
                    5:'L Bicep Femoris',
                    6:'L Gastrocnemius',
                    7:'L Soleus',
                    8:'R Psoas Major',
                    9:'R Rectus Femoris',
                    10:'R Vastus Laterali',
                    11:'R Tibalis Anterior',
                    12:'R Gluteus Maximus',
                    13:'R Bicep Femoris',
                    14:'R Gastrocnemius',
                    15:'R Soleus'}
for pat in ['P1/','P3/','P2/']:
    for d in [8,9,10,11]: #exercises offf
        Fs=10000
        patient = pat
        weeks = list(range(1,29))
        if d<8:
            electrode_1 = [0,1,2,3,4,5,6,7]
        else:
            electrode_1 = [8,9,10,11,12,13,14,15]
        exercise_numbers_off = [d] # put your exercise of interest first, rest are reference exercises
        exercise_numbers_on = [x+13 for x in exercise_numbers_off]
        if patient == "P3/":
            exercise_numbers_off = [x-1 for x in exercise_numbers_off]
            exercise_numbers_on = [x-1 for x in exercise_numbers_on]
            exercises_dict = {1:'LHF_off',
                              2:'LHE_off',
                              3:'LKF_off',
                              4:'LKE_off',
                              5:'LAD_off',
                              6:'LAP_off',
                              7:'RHF_off',
                              8:'RHE_off',
                              9:'RKF_off',
                              10:'RKE_off',
                              11:'RAD_off',
                              12:'RAP_off',
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
        save_directory = "C:/Users/David Teo/Desktop/right side real/"
        raw_signals_off = {}
        raw_signals_on = {}
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
                    baseline_off_signal = scipy.io.loadmat(path+files[0])['data'][electrode_1]
                    try: # cos week 2 has no stim on exercises
                        baseline_on_file = [file for file in files if file[0:2]=='14']
                        baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data'][electrode_1]
                    except:
                        pass
                    # Stim off block
                    ### ensures exercise of interest loaded first so items are index out correctly later so ratio is calculated properly later
                    exe = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_off[0]]
                    exe1 = [file for file in files if round(float(file.split('.')[0])) in exercise_numbers_off[1:len(exercise_numbers_off)]]
                    files = exe + exe1
                    for file in files:
                        loadpath = path + file
                        plt.figure(figsize=(15,15))
                        plt.title('P1_off' + exercises_dict[round(float(file.split('.')[0]))] + ' W'+week)
                        for a in range(0,len(electrode_1)):
                            print('starting electrode ', electrode_1[a])
                            plt.subplot(len(electrode_1),1,a+1)
                            raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[a]]
                            raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                            time=[x/Fs for x in range(0,len(raw_signal))]
                            raw_signals_off['W' + week + '_Exe' + file.split('.')[0]] = raw_signal
                            #plt.plot(time,raw_signal, label='raw')
                            filtered_signal,view_signal,indices = apply_filters(raw_signal,baseline_off_signal,Fs)
                            #plt.plot(time,view_signal, label='filtered')
                            
                            #plt.xticks(np.arange(min(time), max(time), 1.0), fontsize = 10)
                            
         
                            envelope = RMS_envelope(raw_signal)
                            time=list(range(0,250*len(envelope),250))
                            time=[x/Fs for x in time]
                            segmented_signal = np.full(shape=np.size(envelope), fill_value=np.nan)
                            indices = [int(np.floor(x/250)) for x in indices]
                            indices = np.unique(indices)
                            segmented_signal[indices] = envelope[indices]
                            
                            plt.plot(time,envelope)
                            plt.plot(time,segmented_signal)
                            plt.ylabel(str(electrode_1[a]))
                            plt.tight_layout()
                            # Calculate RMS and add to dictionary
                            #RMS_off_dictionary['W' + week + '_Exe' + file.split('.')[0]] = np.sqrt(np.nanmean(filtered_signal**2))
                            print(loadpath)  
                     
                    plt.xlabel('Time')
                    plt.savefig(save_directory + "P1_" + exercises_dict[round(float(file.split('.')[0]))] +"W"+week+".png")
                    
                    # Stim on block
                    files = os.listdir(path)
                    ### ensures exercise of interest loaded first so items are index out correctly later so ratio is calculated properly later
                    exe = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_on[0]]
                    exe1 = [file for file in files if round(float(file.split('.')[0])) in exercise_numbers_on[1:len(exercise_numbers_off)]]
                    files = exe + exe1
                    print(files)
                    for file in files:
                        loadpath = path + file
                        plt.clf()
                        plt.figure(figsize=(15,15))
                        plt.title('P1_on' + exercises_dict[round(float(file.split('.')[0]))] + ' W'+week)
                        for a in range(0,len(electrode_1)):
                            print('starting electrode ', electrode_1[a])
                            plt.subplot(len(electrode_1),1,a+1)
                            raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[a]]
                            raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                            time=[x/Fs for x in range(0,len(raw_signal))]
                            raw_signals_off['W' + week + '_Exe' + file.split('.')[0]] = raw_signal
                            #plt.plot(time,raw_signal, label='raw')
                            filtered_signal,view_signal,_ = apply_filters(raw_signal,baseline_off_signal,Fs)
                            #plt.plot(time,view_signal, label='filtered')
                            #plt.ylabel(str(electrode_1[a]))
                            #plt.xticks(np.arange(min(time), max(time), 1.0), fontsize = 10)
                            #plt.tight_layout()
                            
                            envelope = RMS_envelope(raw_signal)
                            time=list(range(0,250*len(envelope),250))
                            time=[x/Fs for x in time]
                            segmented_signal = np.full(shape=np.size(envelope), fill_value=np.nan)
                            indices = [int(np.floor(x/250)) for x in indices]
                            indices = np.unique(indices)
                            segmented_signal[indices] = envelope[indices]
                            
                            plt.plot(time,envelope)
                            plt.plot(time,segmented_signal)
                            plt.ylabel(str(electrode_1[a]))
                            plt.tight_layout()
                            
                            # Calculate RMS and add to dictionary
                            #RMS_off_dictionary['W' + week + '_Exe' + file.split('.')[0]] = np.sqrt(np.nanmean(filtered_signal**2))
                            print(loadpath)  
                        plt.xlabel('Time')
                        plt.savefig(save_directory + "P1_" + exercises_dict[round(float(file.split('.')[0]))] +"W"+week+".png")
                        
                        plt.clf()
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
                        baseline_off_signal = scipy.io.loadmat(path+files[0])['data'][electrode_1]
                        try: # cos week 2 has no stim on exercises
                            baseline_on_file = [file for file in files if file[0:2]=='14']
                            baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data'][electrode_1]
                        except:
                            pass
                        # Stim off block
                        ### ensures exercise of interest loaded first so items are index out correctly later so ratio is calculated properly later
                        exe = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_off[0]]
                        exe1 = [file for file in files if round(float(file.split('.')[0])) in exercise_numbers_off[1:len(exercise_numbers_off)]]
                        files = exe + exe1
                        for file in files:
                            loadpath = path + file
                            plt.figure(figsize=(15,15))
                            plt.title('P2_off' + exercises_dict[round(float(file.split('.')[0]))] + ' W'+week)
                            for a in range(0,len(electrode_1)):
                                print('starting electrode ', electrode_1[a])
                                plt.subplot(len(electrode_1),1,a+1)
                                raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[a]]
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                time=[x/Fs for x in range(0,len(raw_signal))]
                                raw_signals_off['W' + week + '_Exe' + file.split('.')[0]] = raw_signal
                                #plt.plot(time,raw_signal, label='raw')
                                filtered_signal,view_signal,indices = apply_filters(raw_signal,baseline_off_signal,Fs)
                                #plt.plot(time,view_signal, label='filtered')
                                
                                #plt.xticks(np.arange(min(time), max(time), 1.0), fontsize = 10)
                                
             
                                envelope = RMS_envelope(raw_signal)
                                time=list(range(0,250*len(envelope),250))
                                time=[x/Fs for x in time]
                                segmented_signal = np.full(shape=np.size(envelope), fill_value=np.nan)
                                indices = [int(np.floor(x/250)) for x in indices]
                                indices = np.unique(indices)
                                segmented_signal[indices] = envelope[indices]
                                
                                plt.plot(time,envelope)
                                plt.plot(time,segmented_signal)
                                plt.ylabel(str(electrode_1[a]))
                                plt.tight_layout()
                                # Calculate RMS and add to dictionary
                                #RMS_off_dictionary['W' + week + '_Exe' + file.split('.')[0]] = np.sqrt(np.nanmean(filtered_signal**2))
                                print(loadpath)  
                         
                        plt.xlabel('Time')
                        plt.savefig(save_directory + "P2_" + exercises_dict[round(float(file.split('.')[0]))] +"W"+week+".png")
                        
                        # Stim on block
                        files = os.listdir(path)
                        ### ensures exercise of interest loaded first so items are index out correctly later so ratio is calculated properly later
                        exe = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_on[0]]
                        exe1 = [file for file in files if round(float(file.split('.')[0])) in exercise_numbers_on[1:len(exercise_numbers_off)]]
                        files = exe + exe1
                        print(files)
                        for file in files:
                            loadpath = path + file
                            plt.clf()
                            plt.figure(figsize=(15,15))
                            plt.title('P2_on' + exercises_dict[round(float(file.split('.')[0]))] + ' W'+week)
                            for a in range(0,len(electrode_1)):
                                print('starting electrode ', electrode_1[a])
                                plt.subplot(len(electrode_1),1,a+1)
                                raw_signal = scipy.io.loadmat(loadpath)['data'][electrode_1[a]]
                                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                                time=[x/Fs for x in range(0,len(raw_signal))]
                                raw_signals_off['W' + week + '_Exe' + file.split('.')[0]] = raw_signal
                                #plt.plot(time,raw_signal, label='raw')
                                filtered_signal,view_signal,_ = apply_filters(raw_signal,baseline_off_signal,Fs)
                                #plt.plot(time,view_signal, label='filtered')
                                #plt.ylabel(str(electrode_1[a]))
                                #plt.xticks(np.arange(min(time), max(time), 1.0), fontsize = 10)
                                #plt.tight_layout()
                                
                                envelope = RMS_envelope(raw_signal)
                                time=list(range(0,250*len(envelope),250))
                                time=[x/Fs for x in time]
                                segmented_signal = np.full(shape=np.size(envelope), fill_value=np.nan)
                                indices = [int(np.floor(x/250)) for x in indices]
                                indices = np.unique(indices)
                                segmented_signal[indices] = envelope[indices]
                                
                                plt.plot(time,envelope)
                                plt.plot(time,segmented_signal)
                                plt.ylabel(str(electrode_1[a]))
                                plt.tight_layout()
                                
                                # Calculate RMS and add to dictionary
                                #RMS_off_dictionary['W' + week + '_Exe' + file.split('.')[0]] = np.sqrt(np.nanmean(filtered_signal**2))
                                print(loadpath)  
                            plt.xlabel('Time')
                            plt.savefig(save_directory + "P2_" + exercises_dict[round(float(file.split('.')[0]))] +"W"+week+".png")
                            
                            plt.clf()
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
                
                else:
                    week = str(week)
                    path = "C:/Users/David Teo/Desktop/Extracted Mat Files/"
                    path = path + patient + "W" + week + '/'
                    print(week)
                    try:
                        files = os.listdir(path)
                        baseline_off_signal = scipy.io.loadmat(path+files[0])['data'][electrode_1]
                        baseline_on_file = [file for file in files if file[0:2]=='13']
                        baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data'][electrode_1]
                        # Stim off block
                        exe_off = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_off[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                        files = exe_off #+ exe1
                        #print(files)
                       
                        print('stim off start')
                        for i in range(0,len(files),3):
                            loadpath = path + files[i]
                            loadpath1 = path + files[i+1]
                            loadpath2 = path + files[i+2]
                            plt.clf()
                            plt.figure(figsize=(15,15))
                            plt.title('P3_off' + exercises_dict[round(float(files[i].split('.')[0]))] + ' W'+week)
                            for a in range(0,len(electrode_1)):
                                print('stim off starting electrode ', electrode_1[a])
                                plt.subplot(len(electrode_1),1,a+1)
                                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[a]], scipy.io.loadmat(loadpath1)['data'][electrode_1[a]], scipy.io.loadmat(loadpath2)['data'][electrode_1[a]]))
                                raw_signal = np.nan_to_num(raw_signal)#[electrode_1[0]] # Remove nan values
                                time=[x/Fs for x in range(0,len(raw_signal))]
                                raw_signals_off['W' + week + '_Exe' + files[i].split('.')[0]] = raw_signal
                                #plt.plot(time,raw_signal, label='raw')
                                filtered_signal,view_signal,_ = apply_filters(raw_signal,baseline_off_signal,Fs)
                                #plt.plot(time,view_signal, label='filtered')
                                #plt.ylabel(str(electrode_1[a]))
                                #plt.xticks(np.arange(min(time), max(time), 1.0), fontsize = 10)
                                #plt.tight_layout()
                                
                                
                                envelope = RMS_envelope(raw_signal)
                                time=list(range(0,250*len(envelope),250))
                                time=[x/Fs for x in time]
                                segmented_signal = np.full(shape=np.size(envelope), fill_value=np.nan)
                                indices = [int(np.floor(x/250)) for x in indices]
                                indices = np.unique(indices)
                                segmented_signal[indices] = envelope[indices]
                                
                                plt.plot(time,envelope)
                                plt.plot(time,segmented_signal)
                                plt.ylabel(str(electrode_1[a]))
                                plt.tight_layout()
                                
                            plt.xlabel('Time')
                            plt.savefig(save_directory + "P2_" + exercises_dict[round(float(files[i].split('.')[0]))] +"W"+week+".png")
                                
                          
                        # Stim on block
                        files = os.listdir(path)
                        exe_on = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_on[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                        files = exe_on
                        print('stim on start', files)
                        #print(files)
                        for i in range(0,len(files),3):
                            loadpath = path + files[i]
                            loadpath1 = path + files[i+1]
                            loadpath2 = path + files[i+2]
                            plt.clf()
                            plt.figure(figsize=(15,15))
                            print('dfd')
                            plt.title('P3_on' + exercises_dict[round(float(files[i].split('.')[0]))] + ' W'+week)
                            for a in range(0,len(electrode_1)):
                                print('stim on starting electrode ', electrode_1[a])
                                plt.subplot(len(electrode_1),1,a+1)
                                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[a]], scipy.io.loadmat(loadpath1)['data'][electrode_1[a]], scipy.io.loadmat(loadpath2)['data'][electrode_1[a]]))
                                raw_signal = np.nan_to_num(raw_signal)#[electrode_1[0]] # Remove nan values
                                time=[x/Fs for x in range(0,len(raw_signal))]
                                raw_signals_on['W' + week + '_Exe' + files[i].split('.')[0]] = raw_signal
                                #plt.plot(time,raw_signal, label='raw')
                                filtered_signal,view_signal,_ = apply_filters(raw_signal,baseline_off_signal,Fs)
                                #plt.plot(time,view_signal, label='filtered')
                                #plt.ylabel(str(electrode_1[a]))
                                #plt.xticks(np.arange(min(time), max(time), 1.0), fontsize = 10)
                                #plt.tight_layout()
                                
                                envelope = RMS_envelope(raw_signal)
                                time=list(range(0,250*len(envelope),250))
                                time=[x/Fs for x in time]
                                segmented_signal = np.full(shape=np.size(envelope), fill_value=np.nan)
                                indices = [int(np.floor(x/250)) for x in indices]
                                indices = np.unique(indices)
                                segmented_signal[indices] = envelope[indices]
                                
                                plt.plot(time,envelope)
                                plt.plot(time,segmented_signal)
                                plt.ylabel(str(electrode_1[a]))
                                plt.tight_layout()
                            plt.xlabel('Time')
                            plt.savefig(save_directory + "P2_" + exercises_dict[round(float(files[i].split('.')[0]))] +"W"+week+".png")
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
                        traceback.print_exc()
                        continue            
         
        if patient =='P3/':
            temp=[]
            for week in weeks:
                week = str(week)
                path = "C:/Users/David Teo/Desktop/Extracted Mat Files/"
                path = path + patient + "W" + week + '/'
                print(week)
                try:
                    files = os.listdir(path)
                    baseline_off_signal = scipy.io.loadmat(path+files[0])['data'][electrode_1]
                    baseline_on_file = [file for file in files if file[0:2]=='13']
                    baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data'][electrode_1]
                    # Stim off block
                    exe_off = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_off[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                    files = exe_off #+ exe1
                    #print(files)
                    temp.append(int(week))
                    print('stim off start')
                    for i in range(0,len(files),3):
                        loadpath = path + files[i]
                        loadpath1 = path + files[i+1]
                        loadpath2 = path + files[i+2]
                        plt.clf()
                        plt.figure(figsize=(15,15))
                        plt.title('P3_off' + exercises_dict[round(float(files[i].split('.')[0]))] + ' W'+week)
                        for a in range(0,len(electrode_1)):
                            print('stim off starting electrode ', electrode_1[a])
                            plt.subplot(len(electrode_1),1,a+1)
                            raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[a]], scipy.io.loadmat(loadpath1)['data'][electrode_1[a]], scipy.io.loadmat(loadpath2)['data'][electrode_1[a]]))
                            raw_signal = np.nan_to_num(raw_signal)#[electrode_1[0]] # Remove nan values
                            time=[x/Fs for x in range(0,len(raw_signal))]
                            raw_signals_off['W' + week + '_Exe' + files[i].split('.')[0]] = raw_signal
                            #plt.plot(time,raw_signal, label='raw')
                            filtered_signal,view_signal,_ = apply_filters(raw_signal,baseline_off_signal,Fs)
                            #plt.plot(time,view_signal, label='filtered')
                            #plt.ylabel(str(electrode_1[a]))
                            #plt.xticks(np.arange(min(time), max(time), 1.0), fontsize = 10)
                            #plt.tight_layout()
                            
                            
                            envelope = RMS_envelope(raw_signal)
                            time=list(range(0,250*len(envelope),250))
                            time=[x/Fs for x in time]
                            segmented_signal = np.full(shape=np.size(envelope), fill_value=np.nan)
                            indices = [int(np.floor(x/250)) for x in indices]
                            indices = np.unique(indices)
                            segmented_signal[indices] = envelope[indices]
                            
                            plt.plot(time,envelope)
                            plt.plot(time,segmented_signal)
                            plt.ylabel(str(electrode_1[a]))
                            plt.tight_layout()
                            
                        plt.xlabel('Time')
                        plt.savefig(save_directory + "P3_" + exercises_dict[round(float(files[i].split('.')[0]))] +"W"+week+".png")
                            
                      
                    # Stim on block
                    files = os.listdir(path)
                    exe_on = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers_on[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                    files = exe_on
                    print('stim on start', files)
                    #print(files)
                    for i in range(0,len(files),3):
                        loadpath = path + files[i]
                        loadpath1 = path + files[i+1]
                        loadpath2 = path + files[i+2]
                        plt.clf()
                        plt.figure(figsize=(15,15))
                        print('dfd')
                        plt.title('P3_on' + exercises_dict[round(float(files[i].split('.')[0]))] + ' W'+week)
                        for a in range(0,len(electrode_1)):
                            print('stim on starting electrode ', electrode_1[a])
                            plt.subplot(len(electrode_1),1,a+1)
                            raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrode_1[a]], scipy.io.loadmat(loadpath1)['data'][electrode_1[a]], scipy.io.loadmat(loadpath2)['data'][electrode_1[a]]))
                            raw_signal = np.nan_to_num(raw_signal)#[electrode_1[0]] # Remove nan values
                            time=[x/Fs for x in range(0,len(raw_signal))]
                            raw_signals_on['W' + week + '_Exe' + files[i].split('.')[0]] = raw_signal
                            #plt.plot(time,raw_signal, label='raw')
                            filtered_signal,view_signal,_ = apply_filters(raw_signal,baseline_off_signal,Fs)
                            #plt.plot(time,view_signal, label='filtered')
                            #plt.ylabel(str(electrode_1[a]))
                            #plt.xticks(np.arange(min(time), max(time), 1.0), fontsize = 10)
                            #plt.tight_layout()
                            
                            envelope = RMS_envelope(raw_signal)
                            time=list(range(0,250*len(envelope),250))
                            time=[x/Fs for x in time]
                            segmented_signal = np.full(shape=np.size(envelope), fill_value=np.nan)
                            indices = [int(np.floor(x/250)) for x in indices]
                            indices = np.unique(indices)
                            segmented_signal[indices] = envelope[indices]
                            
                            plt.plot(time,envelope)
                            plt.plot(time,segmented_signal)
                            plt.ylabel(str(electrode_1[a]))
                            plt.tight_layout()
                        plt.xlabel('Time')
                        plt.savefig(save_directory + "P3_" + exercises_dict[round(float(files[i].split('.')[0]))] +"W"+week+".png")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    traceback.print_exc()
                    continue            