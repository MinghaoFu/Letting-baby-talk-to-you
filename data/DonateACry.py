import os
import torch
import torch.nn as nn
import re
import librosa
import pickle
import numpy as np
import random
import glob

from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from utils import center_data, flatten_list

class SegmentedAudio(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = data['audio'].shape[0]
        self.n_frames = data['audio'].shape[1]
        self.n_mfcc_coeffs = data['audio'].shape[2]
        self.n_flatten_feats = self.n_frames * self.n_mfcc_coeffs
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data['audio'][idx], self.data['label'][idx], self.data['id'][idx]
    
    def shape(self):
        return self.data['audio'].shape


class DonateACryDataset:
    def __init__(self, data_dir, labels, val_rate):
        self.data_dir = data_dir
        self.labels = labels
        self.cls2ind = {cls: i for i, cls in enumerate(self.labels)}
        train_ids, val_ids = self.generate_train_val_ids(self.baby_ids, self.n_baby, val_rate, split_type, load_path, save_path)
    
        
        all_data = {'audio': [], 'label': [], 'id': []}
        all_data['audio'] = torch.cat([train_data['audio'], val_data['audio']], axis=0)
        all_data['label'] = torch.cat([train_data['label'], val_data['label']], axis=0)
        all_data['id'] = torch.cat([train_data['id'], val_data['id']], axis=0)

        self.all_data = all_data
        self.train_data = train_data
        self.val_data = val_data
        
        self.train_dataset = SegmentedAudio(train_data)
        self.val_dataset = SegmentedAudio(val_data)
        self.all_dataset = SegmentedAudio(all_data)
            
        print('Training samples: {}, validation samples: {}, total samples: {}'.format(len(self.train_dataset), \
            len(self.val_dataset), len(self.all_dataset)))

    def preprocess(self):
        '''
            Wav to mel, reference https://github.com/seungwonpark/melgan/blob/master/preprocess.py
        '''
        pass
    
    def load_data(self):
        mels = []
        labels = []
        print('--- Loading mel-spectrum files.')
        for melpath in tqdm.tqdm(glob.glob(os.path.join(self.data_dir, '**'), recursive=True)):
            ind = self.cls2ind[melpath.split(os.path.sep)[-2]]
            age = melpath[]
            gender = melpath[]
            
            mel = torch.load(melpath)
            mel = mel.cuda()
            mels.append(mels)
            labels.append(ind)
        mels = torch.stack(mels)
        labels = torch.stack(labels)

        return mels, labels
            
        for i, dir in enumerate(one_second_audio_dirs):
            for file in os.listdir(dir):
                if file.endswith(".mel"):
                    
                    y, sr = librosa.load(os.path.join(dir, file))
                    if sr != 22050:
                        print(sr, file)
                    
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc_coeffs).T
                    #n_frames = mfcc.shape[0]
                    pitch = librosa.yin(y, fmin=75, fmax=600)
                    #intensity = librosa.feature.rms(y=y).flatten()
                    feature = np.concatenate((mfcc, pitch), axis=0)
                    
                    _id = int(file[2:4])
                    if _id in train_ids:
                        train_data['audio'].append(feature)
                        train_data['label'].append(i)
                        train_data['id'].append(_id)
                    elif _id in val_ids:
                        val_data['audio'].append(feature)
                        val_data['label'].append(i)
                        val_data['id'].append(_id)
                    else:
                        print('Data {}/{} is not included.'.format(dir, file))
                    
                    data_idx += 1
            
        for data in [train_data, val_data]:
            data['audio'] = torch.from_numpy(center_data(np.stack(data['audio']))).float()
            data['label'] = torch.tensor(data['label'])
            data['id'] = torch.tensor(data['id'])
            
        return train_data, val_data

    def load_n_seconds_audio(self, seg_len, train_ids, val_ids, n_mfcc_coeffs, prefix='Full_'):
        data_idx = 0 
        train_data = {'audio': [], 'label': [], 'id': []}
        val_data = {'audio': [], 'label': [], 'id': []}
        full_audio_dirs = [self.data_dir + prefix + label for label in self.labels]
        print('--- Loading and segmenting {} second audios'.format(seg_len))
        for i, dir in enumerate(full_audio_dirs):
            for file in os.listdir(dir):
                if file.endswith(".wav"):
                    y, sr = librosa.load(os.path.join(dir, file))
                    seg_samples = int(seg_len * sr)
                    total_segs = len(y) // seg_samples
                    _id = int(re.findall(r'\d+', file)[0])
                    for seg_index in range(total_segs):
                        start = seg_index * seg_samples
                        end = (seg_index + 1) * seg_samples
                        segment = y[start:end]
                        
                        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc_coeffs).T

                        #n_frames = mfcc.shape[0]
                        pitch = librosa.yin(segment, fmin=75, fmax=600)
                        #intensity = librosa.feature.rms(y=y).flatten()
                        feature = np.concatenate((mfcc, np.expand_dims(pitch, axis=1)), axis=1)
                        if _id in train_ids:
                            train_data['audio'].append(feature)
                            train_data['label'].append(i)
                            train_data['id'].append(_id)
                        elif _id in val_ids:
                            val_data['audio'].append(feature)
                            val_data['label'].append(i)
                            val_data['id'].append(_id)
                        else:
                            print('Data {}/{} is not included.'.format(dir, file))

        for data in [train_data, val_data]:
            data['audio'] = torch.from_numpy(center_data(np.stack(data['audio']))).float()
            data['label'] = torch.tensor(data['label'])
            data['id'] = torch.tensor(data['id'])
        
        return train_data, val_data
