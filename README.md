This is the implementation code for part of the Master's Thesis on "Unsupervised Machine-Learning for Epilepsy detection on EEG data". 
The repository contains the code for the BENDR adaptation for seizure detection on CHB-MIT.

## Download dataset
> ```wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/```

## Run dataloader
`load_data.py` process the CHB-MIT dataset (downloaded at the step before), considering only edf files that contain a seizure, and splitting files into data and labels.
The parameters to chose are: 
- data_dir: path to the `physionet.org/files/chbmit/1.0.0/` folder (where the dataset has been saved in the step before). Default = `/scratch2/msc22h15/EEG-AD-main/data/CHB/files/chbmit/1.0.0/`
- data_save: path to saving directory. Default = `/scratch2/msc22h15/CHB-MIT-Preprocessed_4s/`
- segment_length: length of the EEG segments. Default = 4
- filter: use Butterworth Bandpass filter during loading. Default=False

Run this file with: 
> ```python load_data.py --data_dir /path/to/chbmit/physionet.org/files/chbmit/1.0.0/ --data_save /path/to/save ```