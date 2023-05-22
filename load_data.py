import numpy as np
import pandas as pd
import os
import argparse
import mne
from scipy.signal import butter, sosfilt

 # Bandpass Butterworth filter
def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Directory that contains the downloaded CHB-MIT', default="/scratch2/msc22h15/EEG-AD-main/data/physionet.org/files/chbmit/1.0.0/")
    parser.add_argument('--data_save', type=str, help='Directory to save processed data', default="/scratch2/msc22h15/CHB-MIT-Preprocessed_4s/")
    parser.add_argument('--segment_length', type=int, default=4, help="length of the EEG segment to be created")
    parser.add_argument('--filter', action="store_true", default=False, help="Butterworth bandpass filter")
    args = parser.parse_args()

    print("\tdata_dir: ", args.data_dir)
    print("\tdata_save: ", args.data_save)
    print("\tsegment_length: ", args.segment_length)

    patients = os.listdir(args.data_dir)
    patients.sort()
    patients = [p for p in patients if p.startswith('chb') and '24' not in p] # for now, patient 24 is not considere because it was added later

    # Read the file which lists the seizure files, it is named RECORDS-WITH-SEIZURES
    seizure_files = pd.read_csv(os.path.join(args.data_dir, 'RECORDS-WITH-SEIZURES'), header=None)
    # Seizure files is a list of all the files that have seizures
    # Strip chbXX/ from the beginning of the file name
    seizure_files = [f[6:] for f in seizure_files[0]]
    # Next for each patient only read the summary file which has the name 'chbXX-summary.txt'
    summary_files = [os.path.join(args.data_dir, p, p + '-summary.txt') for p in patients]
    # Each summary file represents one patient

    # ----------- 1ST STEP: CREATE SEIZURES DICTIONARY -----------
    # Make a dictionary that has the patient name as key, the value is a dictionary with the seizure file as key and that value has the start and end times of the seizures.
    patient_seizure_dict = {}
    for i, patient in enumerate(patients):
        seizure_dict = {}
        # Only look at seizure files that correspond to the patient
        summary = pd.read_csv(summary_files[i], sep='\t', header=None)
        # Remove all capitalization from the summary file
        summary = summary[0].str.lower()
        patient_seizure_file = [f for f in seizure_files if f.startswith(patient)]
        
        for seizure_file in patient_seizure_file:
            times = []
            # Find the line that corresponds to the seizure file, it has the string seizure_file in the line
            line = summary[summary.str.contains(seizure_file)].index[0]
            # Find the place where the number of seizures is written, it is 2 lines below the line where the seizure file is
            num_seizures = summary[line + 3]
            # Read the number of seizures, it folows "number of seizures in file: X"
            num_seizures = int(num_seizures.split(': ')[1])
            # Next we want to read the seizure file start and end time.
            # Start time is right below the File name, and end time is below that.
            start_time = summary[line + 1]
            start_time = start_time.split(': ')[1]
            end_time = summary[line + 2]
            end_time = end_time.split(': ')[1]
            # Next we read the start and end seconds of the seizures.
            # The start and end times are in the format "xxxx seconds" we then convert that to samples (256 samples per second)
            # We know there are num_seizures seizures in this file so we loop over them
            for j in range(num_seizures):
                start_seiz = summary[line + 4 + 2*j]
                start_seiz = start_seiz.split(': ')[1]
                # Strip spaces from start_seiz
                start_seiz = start_seiz.strip()
                start_seiz = int(start_seiz.split(' ')[0])
                start_seiz = start_seiz * 256
                end_seiz = summary[line + 4 + (2*j+1)]
                end_seiz = end_seiz.split(': ')[1]
                end_seiz = end_seiz.strip()
                end_seiz = int(end_seiz.split(' ')[0])
                end_seiz = end_seiz * 256
                # We now have the start and end times of the seizures in samples, we add them to the times list
                times.append([start_seiz, end_seiz])
            # We now have the start and end times of the seizures for this patient, we add them to the seizure_dict
            seizure_dict[seizure_file] = times
        # We now add the seizure_dict to the patient_seizure_dict
        patient_seizure_dict[patient] = seizure_dict

    # ----------- 2ND STEP: CREATE EEG SEGMENTS FOR EACH PATIENT -----------
    # 20 channels
    TPC_channels = ['FP1-F7','F7-T3','T3-T5','T5-O1','FP2-F8','F8-T4','T4-T6','T6-O2','T3-C3','C3-CZ',\
                    'CZ-C4','C4-T4','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2']

    # 4 channels 
    # TPC_channels = ['F7-T3', 'T3-T5', 'F8-T4', 'T4-T6'] # 4 channels used by Thorir

    #Rename such that T3 is T7 and T4 is T8, T5 is P7 and T6 is P8
    TPC_channels_renamed = [c.replace('T3','T7') for c in TPC_channels]
    TPC_channels_renamed = [c.replace('T4','T8') for c in TPC_channels_renamed]
    TPC_channels_renamed = [c.replace('T5','P7') for c in TPC_channels_renamed]
    TPC_channels_renamed = [c.replace('T6','P8') for c in TPC_channels_renamed]

    for patient in patient_seizure_dict:
        # If data_save + patient does not exist, create it
        if not os.path.exists(args.data_save + patient):
            os.makedirs(args.data_save + patient)
        print("Processing patient: ", patient)
        # Loop over all the seizure files for this patient
        counter = 0
        for seizure_file in patient_seizure_dict[patient]:
            # Read the data

            # data = mne.io.read_raw_edf(args.data_dir+patient+'/'+seizure_file,include=TPC_channels_renamed,verbose = 'CRITICAL',preload=True) # old mne version
            data = mne.io.read_raw_edf(args.data_dir+patient+'/'+seizure_file,verbose = 'CRITICAL',preload=True) # new mne version
            #If T8-P8-1 is in the channels, remove it and rename T8-P8-0 to T8-P8
            if 'T8-P8-1' in data.ch_names:
                data.drop_channels(['T8-P8-1'])
                data.rename_channels({'T8-P8-0':'T8-P8'})
            # Add the channels that are in TPC_channels_renamed but not in data.ch_names as a dummy channel
            dummy_channels = [c for c in TPC_channels_renamed if c not in data.ch_names]
            for c in dummy_channels:
                data.add_channels([mne.io.RawArray(np.zeros((1, data.n_times)), mne.create_info([c], data.info['sfreq']),verbose = 'WARNING')])
            # Reorder the channels such that they are in the same order as TPC_channels_renamed
            data.reorder_channels(TPC_channels_renamed)
            raw_data = data.get_data()* 1000000
            # print("Before filtering: len={}, max={}, min={}, mean={}".format(raw_data.shape, np.max(raw_data), np.min(raw_data), np.mean(raw_data)))
            # Filter signal
            if args.filter:
                lowcut = 0.5  # Hz
                highcut = 50  # Hz        
                fs = 256  # Sampling frequency
                raw_data = butter_bandpass_filter(raw_data, lowcut, highcut, fs, order=5)
            # print("After filtering: len={}, max={}, min={}, mean={}".format(raw_data.shape, np.max(raw_data), np.min(raw_data), np.mean(raw_data)))
            # Reshape the raw_data such that it is (n_windows,n_channels, window_size) 
            n_channels = len(TPC_channels_renamed)
            window_size = 256 * args.segment_length # 256
            n_windows = int(raw_data.shape[1]/window_size)
            raw_data = raw_data[:,0:n_windows*window_size]
            raw_data = raw_data.reshape((n_channels,window_size,n_windows), order='F')
            # Swap the axes such that it is (n_windows, n_channels, window_size)
            raw_data = np.swapaxes(raw_data,0,2)
            raw_data = np.swapaxes(raw_data,1,2)
            labels = np.zeros(raw_data.shape[0]*raw_data.shape[2])
            for seizure in patient_seizure_dict[patient][seizure_file]:
                # Make the labels
                labels[seizure[0]:seizure[1]] = 1
            # reshape the labels such that the if the majority of the window is seizure, the label is 1
            new_labels = np.zeros(int(labels.shape[0]/window_size))
            for i in range(new_labels.shape[0]):
                if(np.sum(labels[i*window_size:(i+1)*window_size])>window_size/2):
                    new_labels[i] = 1
            labels = new_labels
            np.save(args.data_save+patient+'/d'+str(counter)+'.npy',raw_data)
            np.save(args.data_save+patient+'/l'+str(counter)+'.npy',labels)
            counter += 1