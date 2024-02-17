import librosa
import numpy as np
import argparse
import torch
import librosa
import numpy as np
import noisereduce as nr
import glob
import os

from model import resnet
from collections import Counter

def most_frequent(list_of_elements):
    """
    Returns the most frequent element in the given list.
    
    Parameters:
    - list_of_elements: A array-like list containing the elements to analyze.
    
    Returns:
    - The most frequent element in the list.
    """
    counter = Counter(list_of_elements)
    most_common = counter.most_common(1)[0][0]
    return most_common

def extract_most_energetic_segments(y, sr, segment_length=2.0, n_segments=1):
    """
    Extracts the top-n {segment_length}-seconds segments with the highest energy from an audio file.

    Parameters:
    - y: Audio time series.
    - sr: Sampling rate of y.
    - segment_length: Length of the segment to extract in seconds.
    - n_segments: Number of energetic segments to return.

    Returns:
    - List of tuples containing start and end times and the audio data of the most energetic segments.
    """
    segment_samples = int(segment_length * sr)
    segments_energy = []

    # Calculate energy for each segment and store with start sample index
    for start_sample in range(0, len(y) - segment_samples, segment_samples // 2):  # 50% overlap
        segment = y[start_sample:start_sample + segment_samples]
        energy = np.sum(segment**2)
        segments_energy.append((start_sample, energy))

    # Sort segments by energy in descending order and select top n_segments
    top_segments = sorted(segments_energy, key=lambda x: x[1], reverse=True)[:n_segments]

    # Extract the audio data for the top n_segments
    energetic_segments = []
    for start_sample, _ in top_segments:
        start_time = start_sample / sr
        end_time = (start_sample + segment_samples) / sr
        segment_data = y[start_sample:start_sample + segment_samples]
        energetic_segments.append((start_time, end_time, segment_data))

    return energetic_segments

def useful_information_proportion(y, sr, silence_threshold=0.01, frame_length=2048, hop_length=512):
    """
    Calculate the proportion of 'useful information' in an audio file based on a simple silence detection algorithm.

    Parameters:
    - audio_path: Path to the audio file.
    - silence_threshold: The threshold value below which sound is considered silence.
    - frame_length: The number of samples per analysis frame.
    - hop_length: The number of samples to shift the analysis window for each frame.

    Returns:
    - Proportion of the audio file that is above the silence threshold.
    """

    energy = np.array([sum(abs(y[i:i+frame_length]**2))
                       for i in range(0, len(y), hop_length)])

    normalized_energy = energy / np.max(energy)

    non_silent_frames = np.sum(normalized_energy > silence_threshold)

    total_frames = len(energy)
    proportion_non_silent = non_silent_frames / total_frames

    return proportion_non_silent

def inference(audio_path, model, checkpoint_path, seg_len=2, n_classes=4, n_segments=3, n_mfcc_coeffs=20):
    """
    Infers the label of an audio file using a trained model (resnet).
    Parameters:
    - audio_path: Path to the audio file.
    - model: torch.nn.Module object. The trained model to use for inference.
    - args: argparse.Namespace object. The command-line arguments of the script.
    
    Returns:
    - start_time, end_time: Start and end times of the most energetic segment in the audio file.
    - energetic_segment: The audio data of the most energetic segment.
    """
    y, sr = librosa.load(audio_path)
    y = nr.reduce_noise(y=y, sr=sr)
    
    score = useful_information_proportion(y, sr)
    segs = extract_most_energetic_segments(y, sr, seg_len, n_segments)
    model = resnet(n_classes)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    feature_tensors = []
    for start_time, end_time, seg in segs:
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc_coeffs).T
        pitch = librosa.yin(seg, fmin=75, fmax=600)
        feature = np.concatenate((mfcc, np.expand_dims(pitch, axis=1)), axis=1)
        feature_tensor = torch.from_numpy(feature).unsqueeze(0).float().to(device)
        feature_tensors.append(feature_tensor)
        
    batched_tensor = torch.cat(feature_tensors, dim=0)
    
    output = model(batched_tensor)
    print(f'Model Output: \n {output}')
    probs = torch.nn.functional.softmax(output, dim=-1)
    print(f'Probability: \n {probs}')
    predictions = torch.argmax(probs, dim=-1)
        
    predicted_label = most_frequent(predictions.cpu().numpy())
    return predicted_label, score


    
if __name__ == "__main__":
    N_CLASSES = 4
    AUDIO_PATH = './files/audio/'
    CHECKPOINT_PATH = './files/checkpoint/resnet_1.pth'
    SEG_LEN = 2.0
    N_SEGMENTS = 3
    N_MFCC_COEFFS = 20
    
    paths = glob.glob(os.path.join(AUDIO_PATH, '*.wav'))
    
    for path in paths:
        model = resnet(N_CLASSES)
        predicted_label, score = inference(path, model, CHECKPOINT_PATH, seg_len=SEG_LEN, n_segments=N_SEGMENTS, n_mfcc_coeffs=N_MFCC_COEFFS)
        print(f'Most Frequent Predicted Label: {predicted_label}, Useful Score {score}')