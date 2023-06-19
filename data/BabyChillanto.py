import os
import torch
import torch.nn as nn
import re
import librosa
import pickle
import numpy as np
import random

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


class BabyChillantoDataset:
    def __init__(self, data_dir, labels, seg_len, n_mfcc_coeffs, val_rate, split_type, shift, save_path=None, load_path=None):
        self.data_dir = data_dir
        self.labels = labels
        self.seg_len = seg_len
        self.shift = shift
        self.n_mfcc_coeffs = n_mfcc_coeffs
        self.id2audios, self.n_baby, self.baby_ids = self.get_dataset_ids()
        train_ids, val_ids = self.generate_train_val_ids(self.baby_ids, self.n_baby, val_rate, split_type, load_path, save_path)
        
        if self.seg_len == 1:
            train_data, val_data = self.load_one_second_audio(train_ids, val_ids, n_mfcc_coeffs)
        else:
            if shift != seg_len:
                train_data, val_data = self.load_n_seconds_audio_shift(self.seg_len, self.shift, train_ids, val_ids, n_mfcc_coeffs)
            else:
                train_data, val_data = self.load_n_seconds_audio(self.seg_len, train_ids, val_ids, n_mfcc_coeffs)
        
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
        
    def get_dataset_ids(self, prefix='Full_'):
        full_audio_dirs = [self.data_dir + prefix + label for label in self.labels]
        
        id2audios = {}
        n_baby = 0
        baby_ids = []

        for idx, dir in enumerate(full_audio_dirs):
            class_ids = []
            for file in os.listdir(dir):
                if file.lower().endswith(".wav"):
                    # record individual ids
                    _id = int(re.findall(r'\d+', file)[0])
                    
                    if _id in id2audios:
                        id2audios[_id].append(file)
                    else:
                        id2audios[_id] = [file]
                        n_baby += 1
                        class_ids.append(_id)

            baby_ids.append(class_ids)
            
        print("--- Total baby number: {}, class distribution: {}".format(n_baby, [f"{self.labels[i]}: {len(cls)}" \
                for i, cls in enumerate(baby_ids)]))
        
        return id2audios, n_baby, baby_ids
        
    
    def generate_train_val_ids(self, individual_ids, n_baby, val_rate, split_type, save_path, load_path):
    
        if load_path is not None:
            with open(load_path, "rb") as file:
                data = pickle.load(file)
            train_ids = data['train']
            val_ids = data['val']
            assert len(val_ids + train_ids) == len(flatten_list(individual_ids))
            
        else:
            if split_type == 'class':
                train_ids = []
                val_ids = []
                for cls in individual_ids:
                    random.shuffle(cls)
                    n_val = int(len(cls) * val_rate)
                    val_ids.extend(cls[:n_val])
                    train_ids.extend(cls[n_val:])
                    
            elif split_type == 'baby':
                
                individual_ids = flatten_list(individual_ids)
                random.shuffle(individual_ids)
                n_val_individual = int(val_rate * n_baby)
                n_train_individuals = n_baby - n_val_individual
                train_ids = individual_ids[:n_train_individuals]
                val_ids = individual_ids[n_train_individuals:]
            
            else:
                raise ValueError('Unknow splitting type {}'.format(split_type))
            
            if save_path is not None:
                data = {'train': train_ids, 'val': val_ids}
                with open(save_path, "wb") as file:
                    pickle.dump(data, file)
        
        return train_ids, val_ids


    def load_one_second_audio(self, train_ids, val_ids, n_mfcc_coeffs, prefix='1s_'):
        data_idx = 0 
        train_data = {'audio': [], 'label': [], 'id': []}
        val_data = {'audio': [], 'label': [], 'id': []}
        one_second_audio_dirs = [self.data_dir + prefix + label for label in self.labels]
        print('--- Loading 1 second audios')
        for i, dir in enumerate(one_second_audio_dirs):
            for file in os.listdir(dir):
                if file.endswith(".wav"):
                    if data_idx == 1921: # dirty data
                        #print(dir + '/' + file)
                        data_idx += 1
                        continue
                    
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
    
    def load_n_seconds_audio_shift(self, seg_len, shift, train_ids, val_ids, n_mfcc_coeffs, prefix='Full_'):
        train_data = {'audio': [], 'label': [], 'id': []}
        val_data = {'audio': [], 'label': [], 'id': []}
        full_audio_dirs = [self.data_dir + prefix + label for label in self.labels]
        print('--- Loading and segmenting {} second audios'.format(seg_len))
        for i, dir in enumerate(full_audio_dirs):
            for file in os.listdir(dir):
                if file.endswith(".wav"):
                    y, sr = librosa.load(os.path.join(dir, file))
                    seg_samples = int(seg_len * sr)
                    shift_samples = int(shift * sr)
                    total_segs = (len(y) - seg_samples) // shift_samples 
                    _id = int(re.findall(r'\d+', file)[0])
                    for seg_index in range(total_segs):
                        start = seg_index * shift_samples
                        end = start + seg_samples
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

