# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 12:25:09 2025

@author: David Teo
"""

import os
import scipy.io
import scipy
from scipy.signal import cheby2, lfilter, filtfilt, butter, find_peaks, peak_widths
from scipy.fft import rfft, rfftfreq, fft, fftfreq, ifft
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from preprocessing_functions import create_sliding_window_frequencey_filter, sliding_window_frequencey_filter, self_baseline_sliding_window_frequencey_filter
from preprocessing_functions import apply_filters, filter_baseline, apply_filters_P3, segmentation, lowpass
import math
from sklearn.decomposition import NMF
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
def z_norm(input_signal):
    mean = np.mean(input_signal)
    sd = np.std(input_signal)
    z_signal = (input_signal-mean)/sd
    return z_signal

def RMS(input_signal):
    rms = np.sqrt(np.nanmean(input_signal**2))
    return rms

def RMS_envelope(input_signal, window_size=1250, overlap=625): #(input_signal, window_size=500, overlap=250):
    for i in range(0,len(input_signal)-window_size,window_size-overlap):
        window = input_signal[i:i+window_size]
        if i ==0:
            rms = [RMS(window)]
        else:
            rms = rms + [RMS(window)]
    return np.array(rms)
data = {'Patient':[],
        'Exercise':[],
        'Synergies':[],
        'Gradient':[],
        'p':[],
        'r^2':[]}
#plt.plot(RMS_envelope(raw_signal[0]))    
for pat in ['P3/']:
    for m in [[2,3,4,5,6,7]] + [[8,9,10,11,12,13]]:
        for e in m:#[8,9,10,11,12,13]: #[2,3,4,5,6,7]
        
            Fs=10000
            patient = pat
            if patient == 'P1/':
                weeks = list(range(7,28))
            if patient == 'P2/':
                weeks = list(range(9,28))
            if patient == 'P3/':
                weeks = list(range(1,26))
            
            if e<8:#m == [2]:
                electrodes = [0,1,2,3,4,5,6,7]#[8,9,10,11,12,13,14,15]#
            if e>=8:#m ==[8]:
                electrodes = [8,9,10,11,12,13,14,15]#
            exercise_number_off = [e]
            exercise_number_on = [x+13 for x in exercise_number_off]
            
            if patient == 'P3/':
                exercise_number_off = [x-1 for x in exercise_number_off]
                exercise_number_on = [x-1 for x in exercise_number_on]
                exercises_dict = {1:'LHF',
                                  2:'LHE',
                                  3:'LKF',
                                  4:'LKE',
                                  5:'LAD',
                                  6:'LAP',
                                  7:'RHF',
                                  8:'RHE',
                                  9:'RKF',
                                  10:'RKE',
                                  11:'RAD',
                                  12:'RAP',
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
            exercise = exercises_dict[exercise_number_off[0]]
            print(exercise)
            save_directory = "C:/Users/David Teo/Desktop/after fyp cleanup/synergies/"
            filtered_signals_off = {}
            lowpassed_signals_off = {}
            RMS_envelopes_off = {}
            filtered_signals_on = {}
            lowpassed_signals_on = {}
            RMS_envelopes_on = {}
            
            # for testing file concatenation
            #exercise_numbers = list(range(1,14))
            #files = ['1. stim off-baseline_Extracted_GainAdjusted.mat', '10. no stim- Rt knee flexion_Extracted_GainAdjusted.mat', '11. no stim- Rt knee extension_Extracted_GainAdjusted.mat', '12. no stim-rt ankle dorsiflex_Extracted_GainAdjusted.mat', '13. no stim - rt ankle plantarflex_Extracted_GainAdjusted.mat', '14. stim on-baseline_Extracted_GainAdjusted.mat', '15. stim on- lt hip flex_Extracted_GainAdjusted.mat', '16. stim on-lt hip extension_Extracted_GainAdjusted.mat', '17. stim on -lt knee flex_Extracted_GainAdjusted.mat', '18. stim on-lt knee extension_Extracted_GainAdjusted.mat', '19. stim on-lt ankle dorsiflex_Extracted_GainAdjusted.mat', '2. Lt Hip flex_Extracted_GainAdjusted.mat', '20. stim on - lt ankle plantarflex_Extracted_GainAdjusted.mat', '21. stim on - rt hip flex_Extracted_GainAdjusted.mat', '22. stim on- rt hip extension_Extracted_GainAdjusted.mat', '23. stim on-rt knee flex_Extracted_GainAdjusted.mat', '24. stim on-rt knee extension_Extracted_GainAdjusted.mat', '25. stim on - rt ankle dorsiflex_Extracted_GainAdjusted.mat', '26. stim on- rt ankle plantarflex_Extracted_GainAdjusted.mat', '3. no stim-Lt hip extension_Extracted_GainAdjusted.mat', '4. no stim-Lt knee flexion_Extracted_GainAdjusted.mat', '5. no stim-Lt knee extension_Extracted_GainAdjusted.mat', '6. no stim-lt ankle dorsiflex_Extracted_GainAdjusted.mat', '7. no stim -lt ankle plantarflex_Extracted_GainAdjusted.mat', '8. no stim-rt hip flexion_Extracted_GainAdjusted.mat', '9. no stim-rt hip extension_Extracted_GainAdjusted.mat']
            #files_segmented = [file.split('.')[0] + '.1' + file.split('.')[1] for file in files] + [file.split('.')[0] + '.2' + file.split('.')[1] for file in files]  + [file.split('.')[0] + '.3' + file.split('.')[1] for file in files]
            ################ FOR P1 #################
            temp=[]
            if patient == "P1/":
                # Get raw signals in dictionary
                for week in weeks:
                    week = str(week)
                    path = "C:/Users/David Teo/Desktop/Extracted Mat Files/"
                    path = path + patient + "W" + week + '/'
                            
                    try:
                        files = os.listdir(path)
                        temp.append(week)
                        # Create filters
                        baseline_off_signal = scipy.io.loadmat(path+files[0])['data']
                        #baseline_off_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_off_signal]
                        try:
                            baseline_on_file = [file for file in files if file[0:2]=='14']
                            baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data']
                            #baseline_on_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_on_signal]
                        except:
                            pass
                        
                        # Stim off
                        # This orders the files
                        exercises = []
                        for e in exercise_number_off:
                            for file in files:
                                if float(file[0:2]) == e:
                                    exercises.append(file)
                                    continue
                        files = exercises
                        print(files)
                        order = len(files)
                        for file in files:
                            # load in first file
                            loadpath = path + file
                            raw_signal = scipy.io.loadmat(loadpath)['data'][electrodes]
                            raw_signal = np.nan_to_num(raw_signal)
                            y=raw_signal[0]
                            # Get movement indices by psoas
                            _,segments,movement_indices = apply_filters(raw_signal[0], baseline_off_signal, Fs)
                            print('sdf')
                            raw_signal = [x[movement_indices] for x in raw_signal]
                            filtered_signal = np.array([apply_filters_P3(x, baseline_off_signal, Fs) for x in raw_signal])
                            #filtered_signal = np.array([abs(np.real(sliding_window_frequencey_filter(x, baseline_off_yf_max[i], 10000, Fs))) for i,x in enumerate(raw_signal)])
                            maximas = [max(x) for x in filtered_signal]
                            max_normalized_signal = np.array([x/maximas[i] for i,x in enumerate(filtered_signal)])
                            #z_norm_signal = z_norm(filtered_signal)
                            #RMS envelope
                            RMS_envelopes_off[week] = np.array([RMS_envelope(x) for x in max_normalized_signal])
                            
                            #lowpass
                            lowpassed_signals_off[week] = abs(lowpass(filtered_signal, 6, Fs))
                            print(loadpath)
                            
                            #plt.plot(y)
                            #plt.plot(segments)
                            #plt.title(patient[0:2]+'_W'+week+'_off_filtered')
                            #plt.savefig(save_directory+patient[0:2]+'_W'+week+'_off_filtered')
                            #plt.clf()
                        
                        # Stim on
                        # This orders the files
                        files = os.listdir(path)
                        exercises = []
                        for e in exercise_number_on:
                            for file in files:
                                if float(file[0:2]) == e:
                                    exercises.append(file)
                                    continue
                        files = exercises
                        print(files)
                        order = len(files)
                        for file in files:
                            # load in first file
                            loadpath = path + file
                            raw_signal = scipy.io.loadmat(loadpath)['data'][electrodes]
                            raw_signal = np.nan_to_num(raw_signal)
                            y=raw_signal[0]
                            # Get movement indices by psoas
                            _,segments,movement_indices = apply_filters(raw_signal[0],baseline_on_signal, Fs)
                            raw_signal = [x[movement_indices] for x in raw_signal]
                            filtered_signal = np.array([apply_filters_P3(x, baseline_off_signal, Fs) for x in raw_signal])
                            #filtered_signal = np.array([abs(np.real(sliding_window_frequencey_filter(x, baseline_on_yf_max[i], 10000, Fs))) for i,x in enumerate(raw_signal)])
                            maximas = [max(x) for x in filtered_signal]
                            max_normalized_signal = np.array([x/maximas[i] for i,x in enumerate(filtered_signal)])
                            #z_norm_signal = z_norm(filtered_signal)
                            #RMS envelope
                            RMS_envelopes_on[week] = np.array([RMS_envelope(x) for x in max_normalized_signal])
                            #lowpass
                            lowpassed_signals_on[week] = abs(lowpass(filtered_signal, 6, Fs))
                            print(loadpath)
                            
                            #plt.plot(y)
                            #plt.plot(segments)
                            #plt.title(patient[0:2]+'_W'+week+'_off_filtered')
                            #plt.savefig(save_directory+patient[0:2]+'_W'+week+'_off_filtered')
                            #plt.clf()
                        
                            
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
            
            ################ FOR P2 #################
            if patient == "P2/":
                # Get raw signals in dictionary
                for week in weeks:
                    if week < 21:
                        week = str(week)
                        path = "C:/Users/David Teo/Desktop/Extracted Mat Files/"
                        path = path + patient + "W" + week + '/'
                                
                        try:
                            files = os.listdir(path)
                            temp.append(week)
                            # Create filters
                            baseline_off_signal = scipy.io.loadmat(path+files[0])['data']
                            #baseline_off_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_off_signal]
                            try:
                                baseline_on_file = [file for file in files if file[0:2]=='14']
                                baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data']
                                #baseline_on_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_on_signal]
                            except:
                                pass
                            
                            # Stim off
                            # This orders the files
                            exercises = []
                            for e in exercise_number_off:
                                for file in files:
                                    if float(file[0:2]) == e:
                                        exercises.append(file)
                                        continue
                            files = exercises
                            print(files)
                            order = len(files)
                            for file in files:
                                # load in first file
                                loadpath = path + file
                                raw_signal = scipy.io.loadmat(loadpath)['data'][electrodes]
                                raw_signal = np.nan_to_num(raw_signal)
                                x=raw_signal[0]
                                # Get movement indices
                                _,segments,movement_indices = apply_filters(raw_signal[0], baseline_off_signal, Fs)
                                raw_signal = [x[movement_indices] for x in raw_signal]
                                filtered_signal = np.array([apply_filters_P3(x, baseline_off_signal, Fs) for x in raw_signal])
                                #filtered_signal = np.array([abs(np.real(sliding_window_frequencey_filter(x, baseline_off_yf_max[i], 10000, Fs))) for i,x in enumerate(raw_signal)])
                                maximas = [max(x) for x in filtered_signal]
                                max_normalized_signal = np.array([x/maximas[i] for i,x in enumerate(filtered_signal)])
                                #z_norm_signal = z_norm(filtered_signal)
                                #RMS envelope
                                RMS_envelopes_off[week] = np.array([RMS_envelope(x) for x in max_normalized_signal])
                                #lowpass
                                lowpassed_signals_off[week] = abs(lowpass(filtered_signal, 6, Fs))
                                print(loadpath)
                                
                                #plt.plot(x)
                                #plt.plot(segments)
                                #plt.title(patient[0:2]+'_W'+week+'_off_filtered')
                                #plt.savefig(save_directory+patient[0:2]+'_W'+week+'_off_filtered')
                                #plt.clf()
                            
                            # Stim on
                            # This orders the files
                            files = os.listdir(path)
                            exercises = []
                            for e in exercise_number_on:
                                for file in files:
                                    if float(file[0:2]) == e:
                                        exercises.append(file)
                                        continue
                            files = exercises
                            print(files)
                            order = len(files)
                            for file in files:
                                # load in first file
                                loadpath = path + file
                                raw_signal = scipy.io.loadmat(loadpath)['data'][electrodes]
                                raw_signal = np.nan_to_num(raw_signal)
                                y=raw_signal[0]
                                # Get movement indices
                                _,segments,movement_indices = apply_filters(raw_signal[0],baseline_on_signal, Fs)
                                raw_signal = [x[movement_indices] for x in raw_signal]
                                filtered_signal = np.array([apply_filters_P3(x, baseline_off_signal, Fs) for x in raw_signal])
                                #filtered_signal = np.array([abs(np.real(sliding_window_frequencey_filter(x, baseline_on_yf_max[i], 10000, Fs))) for i,x in enumerate(raw_signal)])
                                maximas = [max(x) for x in filtered_signal]
                                max_normalized_signal = np.array([x/maximas[i] for i,x in enumerate(filtered_signal)])
                                #z_norm_signal = z_norm(filtered_signal)
                                #RMS envelope
                                RMS_envelopes_on[week] = np.array([RMS_envelope(x) for x in max_normalized_signal])
                                #lowpass
                                lowpassed_signals_on[week] = abs(lowpass(filtered_signal, 6, Fs))
                                print(loadpath)
                                
                                #plt.plot(x)
                                #plt.plot(segments)
                                #plt.title(patient[0:2]+'_W'+week+'_off_filtered')
                                #plt.savefig(save_directory+patient[0:2]+'_W'+week+'_off_filtered')
                                #plt.clf()                    
                        except Exception as e:
                            print(f"An unexpected error occurred: {e}")
                    
                    else:
                        week = str(week)
                        path = "C:/Users/David Teo/Desktop/Extracted Mat Files/"
                        path = path + patient + "W" + week + '/'
                        print(week)
                        try:
                            files = os.listdir(path)
                            baseline_off_signal = scipy.io.loadmat(path+files[0])['data']
                            baseline_off_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_off_signal]
                            baseline_on_file = [file for file in files if file[0:2]=='13']
                            baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data']
                            baseline_on_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_on_signal]
                            # Stim off block
                            exe_off = [file for file in files if round(float(file.split('.')[0])) == exercise_number_off[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                            files = exe_off #+ exe1
                            print(files)
                            temp.append(int(week))
                            for i in range(0,len(files),3):
                                loadpath = path + files[i]
                                loadpath1 = path + files[i+1]
                                loadpath2 = path + files[i+2]
                                # load signals
                                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrodes], scipy.io.loadmat(loadpath1)['data'][electrodes], scipy.io.loadmat(loadpath2)['data'][electrodes]), axis = 1)
                                raw_signal = np.nan_to_num(raw_signal)
                                y=raw_signal[0]
                                # Get movement indices
                                _,segments,movement_indices = apply_filters(raw_signal[0],baseline_on_signal, Fs)
                                raw_signal = [x[movement_indices] for x in raw_signal]
                                filtered_signal = np.array([apply_filters_P3(x, baseline_off_signal,Fs) for x in raw_signal])
                                #filtered_signal = np.array([abs(np.real(sliding_window_frequencey_filter(x, baseline_on_yf_max[i], 10000, Fs))) for i,x in enumerate(raw_signal)])
                                maximas = [max(x) for x in filtered_signal]
                                max_normalized_signal = np.array([x/maximas[i] for i,x in enumerate(filtered_signal)])
                                #z_norm_signal = z_norm(filtered_signal)
                                #RMS envelope
                                RMS_envelopes_off[week] = np.array([RMS_envelope(x) for x in max_normalized_signal])
                                #lowpass
                                lowpassed_signals_off[week] = abs(lowpass(filtered_signal, 6, Fs))
                                print(loadpath)
                        
                            # Stim on block
                            files = os.listdir(path)
                            exe_on = [file for file in files if round(float(file.split('.')[0])) == exercise_number_on[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                            files = exe_on
                            print(files)
                            for i in range(0,len(files),3):
                                loadpath = path + files[i]
                                loadpath1 = path + files[i+1]
                                loadpath2 = path + files[i+2]
                                # For 1st electrode
                                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrodes], scipy.io.loadmat(loadpath1)['data'][electrodes], scipy.io.loadmat(loadpath2)['data'][electrodes]), axis = 1)
                                raw_signal = np.nan_to_num(raw_signal)
                                y=raw_signal[0]
                                # Get movement indices
                                _,segments,movement_indices = apply_filters(raw_signal[0],baseline_on_signal, Fs)
                                raw_signal = [x[movement_indices] for x in raw_signal]
                                filtered_signal = np.array([apply_filters_P3(x,baseline_off_signal, Fs) for x in raw_signal])
                                #filtered_signal = np.array([abs(np.real(sliding_window_frequencey_filter(x, baseline_on_yf_max[i], 10000, Fs))) for i,x in enumerate(raw_signal)])
                                maximas = [max(x) for x in filtered_signal]
                                max_normalized_signal = np.array([x/maximas[i] for i,x in enumerate(filtered_signal)])
                                #z_norm_signal = z_norm(filtered_signal)
                                #RMS envelope
                                RMS_envelopes_on[week] = np.array([RMS_envelope(x) for x in max_normalized_signal])
                                #lowpass
                                lowpassed_signals_on[week] = abs(lowpass(filtered_signal, 6, Fs))
                                print(loadpath)
                        except:
                            continue
            
            
            # Patient 3 Block #################################
            if patient == "P3/":
                temp=[]
                for week in weeks:
                    week = str(week)
                    path = "C:/Users/David Teo/Desktop/Extracted Mat Files/"
                    path = path + patient + "W" + week + '/'
                    print(week)
                    try:
                        files = os.listdir(path)
                        baseline_off_signal = scipy.io.loadmat(path+files[0])['data']
                        #baseline_off_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_off_signal]
                        baseline_on_file = [file for file in files if file[0:2]=='13']
                        baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data']
                        #baseline_on_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_on_signal]
                        # Stim off block
                        exe_off = [file for file in files if round(float(file.split('.')[0])) == exercise_number_off[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                        files = exe_off #+ exe1
                        print(files)
                        temp.append(int(week))
                        for i in range(0,len(files),3):
                            loadpath = path + files[i]
                            loadpath1 = path + files[i+1]
                            loadpath2 = path + files[i+2]
                            # load signals
                            raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrodes], scipy.io.loadmat(loadpath1)['data'][electrodes], scipy.io.loadmat(loadpath2)['data'][electrodes]), axis = 1)
                            raw_signal = np.nan_to_num(raw_signal)
                            y=raw_signal[0]
                            # Get movement indices
                            _,segments,movement_indices = apply_filters(raw_signal[0],baseline_on_signal, Fs)
                            raw_signal = [x[movement_indices] for x in raw_signal]
                            filtered_signal = np.array([apply_filters_P3(x, baseline_on_signal, Fs) for x in raw_signal])
                            #filtered_signal = np.array([abs(np.real(sliding_window_frequencey_filter(x, baseline_on_yf_max[i], 10000, Fs))) for i,x in enumerate(raw_signal)])
                            maximas = [max(x) for x in filtered_signal]
                            max_normalized_signal = np.array([x/maximas[i] for i,x in enumerate(filtered_signal)])
                            #z_norm_signal = z_norm(filtered_signal)
                            #RMS envelope
                            RMS_envelopes_off[week] = np.array([RMS_envelope(x) for x in max_normalized_signal])
                            #lowpass
                            lowpassed_signals_off[week] = abs(lowpass(filtered_signal, 6, Fs))
                            print(loadpath)
                    
                        # Stim on block
                        files = os.listdir(path)
                        exe_on = [file for file in files if round(float(file.split('.')[0])) == exercise_number_on[0] and float(file.split('.')[1][0])<4] # take exercises minus controlled release
                        files = exe_on
                        print(files)
                        for i in range(0,len(files),3):
                            loadpath = path + files[i]
                            loadpath1 = path + files[i+1]
                            loadpath2 = path + files[i+2]
                            # For 1st electrode
                            raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'][electrodes], scipy.io.loadmat(loadpath1)['data'][electrodes], scipy.io.loadmat(loadpath2)['data'][electrodes]), axis = 1)
                            raw_signal = np.nan_to_num(raw_signal)
                            y=raw_signal[0]
                            # Get movement indices
                            _,segments,movement_indices = apply_filters(raw_signal[0],baseline_on_signal, Fs)
                            raw_signal = [x[movement_indices] for x in raw_signal]
                            filtered_signal = np.array([apply_filters_P3(x, baseline_on_signal, Fs) for x in raw_signal])
                            #filtered_signal = np.array([abs(np.real(sliding_window_frequencey_filter(x, baseline_on_yf_max[i], 10000, Fs))) for i,x in enumerate(raw_signal)])
                            maximas = [max(x) for x in filtered_signal]
                            max_normalized_signal = np.array([x/maximas[i] for i,x in enumerate(filtered_signal)])
                            #z_norm_signal = z_norm(filtered_signal)
                            #RMS envelope
                            RMS_envelopes_on[week] = np.array([RMS_envelope(x) for x in max_normalized_signal])
                            #lowpass
                            lowpassed_signals_on[week] = abs(lowpass(filtered_signal, 6, Fs))
                            print(loadpath)
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
                        #traceback.print_exc()
                        continue  
                
            weeks=temp
            ################## Analysis #######################
            print('Pre-Processing Finished')
            def calculate_vaf(X_original, W_matrix, C_matrix):
                X_reconstructed = np.dot(W_matrix, C_matrix)
                # Calculate the sum of squares of residuals (variance not accountd for)
                ss_res = np.sum((X_original - X_reconstructed)**2)
                # Calculate the total sum of squares of the original matrix
                ss_total = np.sum(X_original**2)
                # Calculate VAF (proportion of variance accounted for)
                vaf = (1 - (ss_res / ss_total)) * 100
                return np.round(vaf, decimals=1)
            
            vaf_syn1_off = []
            vaf_syn1_on = []
            optimal_n_off = []
            optimal_n_on = []
            weeks = [int(x) for x in weeks]
            for i in weeks:
                ###########STIM OFF#############
                print('start week' + str(i) + 'off')
                # NNMF
                n_synergies = [1,2,3,4,5,6,7,8]  # Choose the desired number of synergies
                vaf_per_synergy = []
                vaf = 0 #initialize
                try:
                    '''for n in n_synergies:
                        model = NMF(n_components=n, init='random', random_state=1, solver='cd') # Other initializations like 'sparse' can be better
                        X =RMS_envelopes_off[str(i)] # Original matrix
                        W = model.fit_transform(X) # W will be (n_muscles x n_synergies) # amplitudes
                        C = model.components_     # C will be (n_synergies x n_time_points) # over time
                        temp=vaf
                        vaf = calculate_vaf(X,W,C)
                        if n==1:
                            vaf_syn1_off.append(vaf)
                        vaf_per_synergy.append(vaf-temp)
                        if vaf>95:
                            optimal_n = n
                            optimal_n_off.append(optimal_n)
                            break
                        optimal_n = n
                        if optimal_n == max(n_synergies):
                            optimal_n_off.append(optimal_n)            
                        
                    # Plot W (muscle weights)
                    plt.figure(figsize=(10, 6))
                    for j in range(optimal_n):
                        plt.subplot(2, optimal_n, j + 1)
                        plt.bar(list(range(1,len(electrodes)+1)),W[:, j])
                        plt.title(f'Synergy {j+1} Activation Pattern '+'var='+str(vaf_per_synergy[j]), fontsize = 9-0.5*optimal_n)
                        plt.xlabel('Muscles')
                        plt.xticks(list(range(1,len(electrodes)+1)), fontsize = 7)
                        plt.ylabel('Activation (Normalized)', fontsize=7.5)
            
                
                    # Plot C (time series)
                    #plt.figure(figsize=(10, 4))
                    for j in range(optimal_n):
                        plt.subplot(2, optimal_n, j + 1+optimal_n)
                        plt.plot(C[j, :])
                        plt.title(f'Synergy {j+1} Activation Pattern', fontsize = 9-0.5*optimal_n)
                        plt.xlabel('Time Points')
                        plt.ylabel('Activation (Normalized)', fontsize=7.5)
                        plt.grid(True)
                    
            
                    plt.suptitle(patient + 'W' + str(int(i)) + '_STIM OFF_Optimal synergies=' + str(optimal_n) + '_VAF='+str(vaf))
                    plt.tight_layout()
                    plt.savefig(save_directory + patient[0:2] + '_W' + str(int(i)) + '_STIM OFF_Optimal synergies=' + str(optimal_n) + '.png')
                    plt.clf()'''
                        
                    ###########STIM ON#############
                    print('start week' + str(i) + 'on')
                    # NNMF
                    n_synergies = [1,2,3,4,5,6,7,8]  # Choose the desired number of synergies
                    vaf_per_synergy = []
                    vaf = 0 #initialize
                    try:
                        for n in n_synergies:
                            model = NMF(n_components=n, init='random', random_state=1, solver='cd') # Other initializations like 'sparse' can be better
                            X =RMS_envelopes_on[str(i)] # Original matrix or lowpassed_signals_on or RMS_envelopes_on
                            W = model.fit_transform(X) # W will be (n_muscles x n_synergies) # amplitudes
                            C = model.components_     # C will be (n_synergies x n_time_points) # over time
                            temp=vaf
                            vaf = calculate_vaf(X,W,C)
                            if n==1:
                                vaf_syn1_on.append(vaf)
                            vaf_per_synergy.append(vaf-temp)
                            
                            if vaf>95:
                                optimal_n = n
                                optimal_n_on.append(optimal_n)
                                break
                            optimal_n = n
                            if optimal_n == max(n_synergies):
                                optimal_n_on.append(optimal_n)
                        
                        # Plot W (muscle weights)
                        plt.figure(figsize=(10, 6))
                        for j in range(optimal_n):
                            plt.subplot(2, optimal_n, j + 1)
                            plt.bar(list(range(1,len(electrodes)+1)),W[:, j])
                            plt.title(f'Synergy {j+1} Activation Pattern', fontsize = 9-0.5*optimal_n) #(f'Synergy {j+1} Activation Pattern '+'var='+str(np.round(vaf_per_synergy[j],decimals=3)), fontsize = 9-0.5*optimal_n)
                            plt.xlabel('Muscles')
                            plt.xticks(list(range(1,9)),['PM', 'RF', 'VL', 'TA', 'GM', 'BF', 'MG', 'S'], fontsize = 7) #(list(range(1,len(electrodes)+1)), fontsize = 7)
                            plt.ylabel('Relative Activation', fontsize=7.5)
            
                    
                        # Plot C (time series)
                        #plt.figure(figsize=(10, 4))
                        for j in range(optimal_n):
                            plt.subplot(2, optimal_n, j + 1+optimal_n)
                            plt.plot(C[j, :])
                            plt.title(f'Synergy {j+1} Activation Pattern', fontsize = 9-0.5*optimal_n)
                            plt.xlabel('Time Points')
                            plt.ylabel('Activation (Normalized)', fontsize=7.5)
                            plt.grid(True)
                    except Exception as e:
                        vaf_syn1_on.append(np.nan)
                        print('sdf')
                        optimal_n_on.append(np.nan)
                        print('abc')
                        print(f"An unexpected error occurred: {e}")
                        continue
            
                    plt.suptitle(patient[0:2] + ' W' + str(int(i)) + ' Stim On '+ exercise) #(patient[0:2] + ' W' + str(int(i)) + ' Stim On '+ exercise+' Optimal synergies=' + str(optimal_n) + ', VAF='+str(vaf))
                    plt.tight_layout()
                    plt.rcParams['svg.fonttype'] = 'none'
                    plt.savefig(save_directory + patient[0:2] + '_W' + str(int(i)) + '_STIM ON_Optimal synergies=' + str(optimal_n) + '_'+exercise +'.svg')
                    plt.savefig(save_directory + patient[0:2] + '_W' + str(int(i)) + '_STIM ON_Optimal synergies=' + str(optimal_n) + '_'+exercise +'.png')
                    plt.pause(0.01)
                except:
                    continue
            
                
            
            
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1,2,1)
            plt.plot(vaf_syn1_off, label = 'off', marker = 'x')
            plt.plot(vaf_syn1_on, label = 'on', marker = 'x')
            plt.xticks(range(0,len(weeks)),weeks, fontsize=8.5)
            plt.title('Var of Synergy 1')
            plt.xlabel('Weeks')
            plt.ylabel('Variance accounted for')
            plt.legend()
            
            
            df = {'Optimal n': optimal_n_off + optimal_n_on,
                  'Categories':['off']*len(optimal_n_off) + ['on']*len(optimal_n_on)}
            plt.subplot(1,2,2)
            sns.barplot(data=df, x='Categories', y='Optimal n', hue='Categories', errorbar='sd')
            plt.title('Number of synergies to account for >95% of variance')
            plt.legend()
            plt.suptitle(patient[0:2] + ' ' + exercise)
            
            plt.pause(0.02)
            
            plt.plot(optimal_n_off, label = 'stim off', marker = 'o', color = 'blue')
            plt.plot(optimal_n_on, label = 'stim on', marker = 'x', color = 'orange')
            plt.xticks(range(0,len(weeks)),weeks)
            plt.legend()
            plt.ylabel('Number of synergies')
            plt.xlabel('Weeks')
            plt.yticks([0,1,2,3,4,5])
            plt.title(patient[0:2]+' N synergies to account for >95% of variance in ' + exercise, fontsize=9)
            plt.pause(0.01)
            # Plot regression line
            from sklearn.linear_model import LinearRegression
            
            # Sample data (sklearn expects 2D array for X)
            x = np.array([int(x) for x in list(RMS_envelopes_on.keys())]).reshape(-1, 1) # P1: [7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24]
            y = np.array(np.array(optimal_n_on)[list(~np.isnan(optimal_n_on))])
            # 1. Create a scatter plot
            plt.scatter(x, y, color='black')
            # 2. Create and fit the linear regression model
            model = LinearRegression().fit(x, y)
            y_pred = model.predict(x) # Predict y values using the model
            # Access the coefficients and intercept
            intercept = model.intercept_
            coefficients = model.coef_
            #P value
            correlation_coefficient, p_value = scipy.stats.pearsonr(np.array([int(x) for x in list(RMS_envelopes_on.keys())]), y)
            #R2 value
            r2_test = model.score(x,y)
            # Print the results
            print("Intercept:", intercept)
            print("Coefficients:", coefficients)
            # 3. Plot the regression line
            plt.plot(x, y_pred, color='red', label = 'Regression line: y='+str(np.round(coefficients[0], decimals=2))+'x + ' + str(np.round(intercept, decimals=2)) + ', p=' + str(np.round(p_value, decimals=3)) + ', R^2=' + str(np.round(r2_test, decimals=3)) ) # R: 'Regression line: y=0.024x + 3.14' L: 'Regression line: y=0.056x + 2.92'
            plt.xticks([int(x) for x in list(RMS_envelopes_on.keys())])
            plt.yticks([0,1,2,3,4,5])
            plt.xlabel('Weeks')
            plt.ylabel('Number of synegies')
            plt.title(patient[0:2]+' Number of synergies required for VAF>95% in the ' + exercises_dict[exercise_number_off[0]])
            plt.legend(loc='lower right')
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(save_directory + patient[0:2]+' Number of synergies required for VAF 95% in the ' + exercises_dict[exercise_number_off[0]] + '.svg')
            plt.savefig(save_directory + patient[0:2]+' Number of synergies required for VAF 95% in the ' + exercises_dict[exercise_number_off[0]] + '.png')
            
            print(f"R^2: {r2_test:.4f}")
            plt.pause(0.01)
            #save as table
            data['Patient'].append(patient[0:2])
            data['Exercise'].append(exercises_dict[exercise_number_off[0]])
            data['Synergies'].append([min(y),max(y)])
            data['Gradient'].append(coefficients[0])
            data['p'].append(p_value)
            data['r^2'].append(r2_test)
import pandas as pd
df = pd.DataFrame(data)
df.to_csv(save_directory + 'P3Synergies_data.csv', index=False)

################## OLD STUFFF ########################################
''' 
################ FOR P2 #################
if patient == "P2/":
    # Get raw signals in dictionary
    for week in weeks:
        if week < 21:
            week = str(week)
            path = "C:/Users/David Teo/Desktop/Gain Adjusted Mat Files/"
            path = path + patient + "W" + week + '/'
            try:
                files = os.listdir(path)
                baseline_off_signal = scipy.io.loadmat(path+files[0])['data']
                baseline_off_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_off_signal]
                try:
                    baseline_on_file = [file for file in files if file[0:2]=='14']
                    baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data']
                    baseline_on_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_on_signal]
                except:
                    pass
                
                # define no stim exercises to append to later
                concatenated_signal = baseline_off_signal
                # This orders the files
                exercises = []
                for e in exercise_numbers:
                    for file in files:
                        if float(file[0:2]) == e:
                            exercises.append(file)
                            continue
                files = exercises
                for file in files:
                    # load in first file
                    loadpath = path + file
                    raw_signal = scipy.io.loadmat(loadpath)['data']
                    raw_signal = np.nan_to_num(raw_signal)
                    # concatennate
                    concatenated_signal = np.concatenate((concatenated_signal, raw_signal), axis=1)
                    print(loadpath)  
                
            except:
                continue
        
        else:
            week = str(week)
            path = "C:/Users/David Teo/Desktop/Gain Adjusted Mat Files/"
            path = path + patient + "W" + week + '/'
            try:
                files = os.listdir(path)
                baseline_off_signal = scipy.io.loadmat(path+files[0])['data']
                baseline_off_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_off_signal]
                baseline_on_file = [file for file in files if file[0:2]=='13']
                baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data']
                baseline_on_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_on_signal]
                
                # define no stim exercises to append to later
                concatenated_signal = baseline_off_signal
                # This orders the files
                exercises = []
                for e in exercise_numbers:
                    for file in files:
                        if float(file[0:2]) == e:
                            exercises.append(file)
                            continue
                files = exercises
                for file in files:
                    # load in first file
                    loadpath = path + file
                    raw_signal = scipy.io.loadmat(loadpath)['data']
                    raw_signal = np.nan_to_num(raw_signal)
                    # concatennate
                    concatenated_signal = np.concatenate((concatenated_signal, raw_signal), axis=1)
                    print(loadpath)  
            
            except:
                continue
            
            
# Patient 3 Block
if patient == "P3/":
    temp=[]
    for week in weeks:
        week = str(week)
        path = "C:/Users/David Teo/Desktop/Gain Adjusted Mat Files/"
        path = path + patient + "W" + week + '/'
        print(week)
        try:
            files = os.listdir(path)
            baseline_off_signal = scipy.io.loadmat(path+files[0])['data']
            baseline_off_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_off_signal]
            baseline_on_file = [file for file in files if file[0:2]=='13']
            baseline_on_signal = scipy.io.loadmat(path+baseline_on_file[0])['data']
            baseline_on_yf_max = [create_sliding_window_frequencey_filter(x, 10000, Fs) for x in baseline_on_signal]
            # Stim off block
            exe_off = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers[0] and float(file.split('.')[1][0])<3] # take exercises minus controlled release
            files = exe_off #+ exe1
            print(files)
            temp.append(int(week))
            for i in range(0,len(files),2):
                loadpath = path + files[i]
                loadpath1 = path + files[i+1]
                # load signals
                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'], scipy.io.loadmat(loadpath1)['data']), axis = 1)
                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                # filter and rectify
                filtered_signal = [abs(np.real(sliding_window_frequencey_filter(x, baseline_off_yf_max[i], 10000, Fs))) for i,x in enumerate(raw_signal)]
                filtered_signals_off[week] = filtered_signal
                filtered_signal_envelope = [abs(lowpass(x, 6, Fs)) for x in filtered_signal]
                filtered_signals_envelopes_off[week] = filtered_signal_envelope
                filtered_signal_envelope_normalized = [x/max(x) for x in filtered_signal_envelope]
                filtered_signals_envelopes_normalized_off[week] = filtered_signal_envelope_normalized
                print(loadpath)
        
            # Stim on block
            files = os.listdir(path)
            exe_on = [file for file in files if round(float(file.split('.')[0])) == exercise_numbers[1] and float(file.split('.')[1][0])<3] # take exercises minus controlled release
            files = exe_on
            for i in range(0,len(files),2):
                loadpath = path + files[i]
                loadpath1 = path + files[i+1]
                # For 1st electrode
                raw_signal = np.concatenate((scipy.io.loadmat(loadpath)['data'], scipy.io.loadmat(loadpath1)['data']), axis = 1)
                raw_signal = np.nan_to_num(raw_signal) # Remove nan values
                # filter and rectigy
                filtered_signal = [abs(np.real(sliding_window_frequencey_filter(x, baseline_off_yf_max[i], 10000, Fs))) for i,x in enumerate(raw_signal)]
                filtered_signals_on[week] = filtered_signal
                filtered_signal_envelope = [abs(lowpass(x, 6, Fs)) for x in filtered_signal]
                filtered_signals_envelopes_on[week] = filtered_signal_envelope
                filtered_signal_envelope_normalized = [np.nan_to_num(x/max(x)) for x in filtered_signal_envelope]
                filtered_signals_envelopes_normalized_on[week] = filtered_signal_envelope_normalized
                print(loadpath)
        except:
            continue
    weeks=temp

'''