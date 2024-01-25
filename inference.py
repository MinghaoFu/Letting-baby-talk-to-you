import librosa
import numpy as np
import argparse
import torch

import noisereduce as nr

from models import resnet
from utils import center_data


parser = argparse.ArgumentParser(description='Descriptions')

parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/resnet_2_Mix/best_model.pth', help='Path to checkpoint file')  
parser.add_argument('--audio_path', type=str, default='/home/minghao.fu/workspace/Letting-baby-talk-to-you/audio.wav', help='')
parser.add_argument('--n_mfcc_coeffs', type=int, default=20, help='Number of MFCC coefficients')
parser.add_argument('--seg_len', type=int, default=2, help='Segment length of audio')
parser.add_argument('--shift', type=float, default=1, help='Shift length of each segment')
parser.add_argument('--dropout', type=float, default=None, help='Dropout')
parser.add_argument('--labels', type=list, default=['pain', 'hungry', 'asphyxia', 'deaf'], help='Labels')

args = parser.parse_args()

y, sr = librosa.load(args.audio_path)
y = nr.reduce_noise(y=y, sr=sr)
seg_samples = int(args.seg_len * sr)
shift_samples = int(args.shift * sr)
total_segs = max((len(y) - seg_samples) // shift_samples + 1, 1)
for seg_index in range(total_segs):
    start = seg_index * shift_samples
    end = start + seg_samples
    if end - start > len(y):
        segment = np.pad(y, (0, end - start - len(y)), mode='constant')
    else:
        segment = y[start:end]

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=args.n_mfcc_coeffs).T
    pitch = librosa.yin(segment, fmin=75, fmax=600)
    feature = np.concatenate((mfcc, np.expand_dims(pitch, axis=1)), axis=1)
    feature_tensor = torch.from_numpy(feature).unsqueeze(0).float()

    model = resnet.resnet(args)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device)) 
    model.eval()
    output = model(feature_tensor)
    probs = torch.nn.functional.softmax(output)
    predicted = torch.argmax(probs)
    print(predicted, probs)
    
    