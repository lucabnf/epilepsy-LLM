This is the implementation code for part of the Master's Thesis on "Unsupervised Machine-Learning for Epilepsy detection on EEG data". 
The repository contains the code for the BENDR adaptation for seizure detection on CHB-MIT.

## Download dataset
```
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
```

## Run dataloader
`load_data.py` process the CHB-MIT dataset (downloaded at the step before), considering only edf files that contain a seizure, and splitting files into data and labels.
The parameters to chose are: 
- data_dir: path to the `physionet.org/files/chbmit/1.0.0/` folder (where the dataset has been saved in the step before). Default = `/scratch2/msc22h15/EEG-AD-main/data/CHB/files/chbmit/1.0.0/`
- data_save: path to saving directory. Default = `/scratch2/msc22h15/CHB-MIT-Preprocessed_4s/`
- segment_length: length of the EEG segments. Default = 4
- filter: use Butterworth Bandpass filter during loading. Default=False

Run this file with: 
```
python load_data.py --data_dir /path/to/chbmit/physionet.org/files/chbmit/1.0.0/ --data_save /path/to/save
```

## Train model
`main.py` is the main file to running both training and testing.
The parameters to chose are: 
- batch_size: standard is 256;
- pretrain_epochs: number of epochs for which we want to pre-train our model on the remaining patients of CHB-MIT. This parameter makes sense only when parameter "pretraining"==True
- finetune_epochs: number of epochs for which we want to fine-tune our model on single patient
- save_every: checkpoint step
- learning_rate
- device: gpu cuda device to use during training (parallelel computing is used when possible)
- data_folder: where is the dataset saved
- pretraining: boolean, pretrain or not on the remaining patients of CHB-MIT

With this file, it is possible to test several configurations. For example: 
- try different models: BENDR, MAEEG, BENDR Linear, CNN
- try different pre-training: pre-training only on TUEG, pre-training on TUEG and CHB-MIT
- try different model sizes of BENDR

## Other info
- `utils.py`: this file contains some utility funcitons (normalization techniques, oversampling, pruning, rolling smoothing windows)
- `model.py` and `layers.py`: files to define the models used. In `layers.py` it is possible to modify and tweak the size of the model (changing number of convolutional blocks used in the first stage and transformers heads and layers in the second stage).
- `utils_folder`: contains other data utils and train utils functions and classes.