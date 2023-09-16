import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import soundfile as sf
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
        if len(data['audio'].shape) == 3:
            self.n_flatten_feats = data['audio'].shape[1] * data['audio'].shape[2]
        elif len(data['audio'].shape) == 2:
            self.n_flatten_feats = data['audio'].shape[-1]
        else:
            raise ValueError('Feature shape {} is not available.'.format(data['audio'].shape))
            
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data['audio'][idx], self.data['label'][idx], self.data['id'][idx], self.data['idx'][idx]
    
    def shape(self):
        return self.data['audio'].shape


class Mix:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.labels = args.labels
        self.seg_len = args.seg_len
        self.shift = args.shift
        self.n_mfcc_coeffs = args.n_mfcc_coeffs
        self.remove_ids = args.remove_ids
        self.seg_save_dir = args.seg_save_dir
        self.center_data = args.center_data
        if self.seg_save_dir is not None:
            if os.path.exists(self.seg_save_dir):
                raise ValueError('Directory of segmented audios exists!')
            else:
                for type in ['train', 'val', 'test']:
                    os.makedirs(os.path.join(self.seg_save_dir, type))
        self.audio_dirs = [os.path.join(self.data_dir, label) for label in self.labels]
        self.cls_num = [0] * len(self.labels)
        
        self.id2audios, self.n_baby, self.baby_ids = self.get_dataset_ids()
        args.train_ids, args.val_ids, args.test_ids = self.generate_train_val_test_ids(self.baby_ids, self.n_baby, args.split_type, args.load_path, args.save_path)
        print('--- Number of baby, train: {}, valid: {}, tets: {}'.format(len(args.train_ids), len(args.val_ids), len(args.test_ids)))
        if self.seg_len == 1:
            train_data, val_data = self.load_one_second_audio(args.train_ids, args.val_ids, args.n_mfcc_coeffs)
        elif self.seg_len == 0:
            train_data, val_data = self.load_full_audio(args.train_ids, args.val_ids, args.n_mfcc_coeffs)
        else:
            train_data, val_data, test_data = self.load_n_seconds_audio_shift(self.seg_len, self.shift, args.train_ids, args.val_ids, args.test_ids, args.n_mfcc_coeffs)
                
        if torch.cuda.is_available():
            for data in [train_data, val_data, test_data]:
                for k in data.keys():
                    data[k] = data[k].cuda()
        
        all_data = {}
        for k in train_data.keys():
            all_data[k] = torch.cat([train_data[k], val_data[k], test_data[k]], axis=0)

        self.all_data = all_data
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        self.train_dataset = SegmentedAudio(train_data)
        self.val_dataset = SegmentedAudio(val_data)
        self.test_dataset = SegmentedAudio(test_data)
        self.all_dataset = SegmentedAudio(all_data)
            
        print('--- Training samples: {}, validation samples: {}, validation samples: {}, total samples: {}'.format(len(self.train_dataset), \
            len(self.val_dataset), len(self.test_dataset), len(self.all_dataset)))
        print('--- Class number: {}'.format({label: num for label, num in zip(args.labels, self.cls_num)}))
        
    def get_dataset_ids(self):
        id2audios = {}
        n_baby = 0
        baby_ids = []

        for idx, dir in enumerate(self.audio_dirs):
            class_ids = []
            for file in os.listdir(dir):
                if file.lower().endswith(".wav"):
                    # record individual ids
                    infos = file.split('-')
                    id = int(infos[0])
                    if id not in self.remove_ids:
                        if id in id2audios:
                            id2audios[id].append(file)
                        else:
                            id2audios[id] = [file]
                            n_baby += 1
                            class_ids.append(id)

            if idx == 1:
                n = len(class_ids) // 5
                baby_ids.append(class_ids[:n])
            else:
                baby_ids.append(class_ids)
            
        print("--- Total baby number: {}, class baby num: {}".format(n_baby, [f"{self.labels[i]}: {len(cls)}" \
                for i, cls in enumerate(baby_ids)]))
        
        return id2audios, n_baby, baby_ids
        
    def generate_train_val_test_ids(self, individual_ids, n_baby, split_type, save_path, load_path):
        if load_path is not None:
            with open(load_path, "rb") as file:
                data = pickle.load(file)
            train_ids = data['train']
            val_ids = data['val']
            test_ids = data['test']
            assert len(val_ids + train_ids + test_ids) == len(flatten_list(individual_ids))
        else:
            if split_type == 'class':
                train_ids = []
                val_ids = []
                test_ids = []
                for cls in individual_ids:
                    random.shuffle(cls)
                    n_val = max(1, round(len(cls) * 0.2))
                    n_test = max(1, round(len(cls) * 0.2))
                    n_train = len(cls) - n_val - n_test
                    train_ids.extend(cls[:n_train])
                    val_ids.extend(cls[n_train:n_train + n_val])
                    test_ids.extend(cls[-n_test:])
                    
            elif split_type == 'baby':
                
                individual_ids = flatten_list(individual_ids)
                random.shuffle(individual_ids)
                n_train_individual = int(0.8 * n_baby)
                n_val_individuals = int(0.1 * n_baby)
                n_test_individuals = n_baby - n_train_individuals - n_val_individuals
                train_ids = individual_ids[:n_train_individuals]
                val_ids = individual_ids[n_train_individuals:n_train_individuals + n_val_individuals]
                test_ids = individual_ids[-n_test_individuals:]
            else:
                raise ValueError('Unknow splitting type {}'.format(split_type))
            
            if save_path is not None:
                data = {'train': train_ids, 'val': val_ids}
                with open(save_path, "wb") as file:
                    pickle.dump(data, file)
        
        return train_ids, val_ids, test_ids

    def load_n_seconds_audio_shift(self, seg_len, shift, train_ids, val_ids, test_ids, n_mfcc_coeffs, prefix='Full_'):
        train_data = {'audio': [], 'label': [], 'id': [], 'idx': []}
        val_data = {'audio': [], 'label': [], 'id': [], 'idx': []}
        test_data = {'audio': [], 'label': [], 'id': [], 'idx': []}
        print('--- Loading and segmenting {} second audios with shift {}'.format(seg_len, shift))
        for i, dir in enumerate(self.audio_dirs):
            for file in os.listdir(dir):
                if file.endswith(".wav"):
                    
                    infos = file[:-4].split('-')
                    id = int(infos[0])
                    idx = int(infos[1])
                    gender = infos[2]
                    age = infos[3]
                    
                    y, sr = librosa.load(os.path.join(dir, file))
                    # Add white noise
                    # if id in train_ids:
                    #     noise_factor = 0.0001
                    #     white_noise = np.random.randn(len(y)) * noise_factor
                    #     y = y + white_noise
                    
                    seg_samples = int(seg_len * sr)
                    shift_samples = int(shift * sr)
                    total_segs = max((len(y) - seg_samples) // shift_samples + 1, 1)
                    for seg_index in range(total_segs):
                        start = seg_index * shift_samples
                        end = start + seg_samples
                        if end - start > len(y):
                            segment = np.pad(y, (0, end - start - len(y)), mode='constant')
                        else:
                            segment = y[start:end]
                        
                        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc_coeffs).T
                        
                        #n_frames = mfcc.shape[0]
                        pitch = librosa.yin(segment, fmin=75, fmax=600)
                        #intensity = librosa.feature.rms(y=y).flatten()
                        feature = np.concatenate((mfcc, np.expand_dims(pitch, axis=1)), axis=1)
                        
                        if id in train_ids or id in self.remove_ids:
                            type = 'train'
                            train_data['audio'].append(feature)
                            train_data['label'].append(i)
                            train_data['id'].append(id)
                            train_data['idx'].append(idx)
                        elif id in val_ids:
                            type = 'val'
                            val_data['audio'].append(feature)
                            val_data['label'].append(i)
                            val_data['id'].append(id)
                            val_data['idx'].append(idx)
                        elif id in test_ids:
                            type = 'test'
                            test_data['audio'].append(feature)
                            test_data['label'].append(i)
                            test_data['id'].append(id)
                            test_data['idx'].append(idx)

                        self.cls_num[i] += 1
                        
                        if self.seg_save_dir is not None:
                            seg_save_path = os.path.join(self.seg_save_dir, type, '{}-{}-{}-{}-{}.wav'.format(id, idx, seg_index, gender, age))
                            #librosa.output.write_wav(seg_save_path, segment, sr) libsora.output has been removed in 0.8.0
                            sf.write(seg_save_path, segment, sr)

        for data in [train_data, val_data, test_data]:
            np_audios = np.stack(data['audio'])
            if self.center_data:
                np_audios = center_data(np_audios)
            data['audio'] = torch.from_numpy(np_audios).float() 
            data['label'] = torch.tensor(data['label'])
            data['id'] = torch.tensor(data['id'])
            data['idx'] = torch.tensor(data['idx'])
        
        return train_data, val_data, test_data

