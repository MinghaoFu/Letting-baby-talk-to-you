from flask import Flask, request, jsonify
import torch
import librosa
import numpy as np
import random

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def read_audio_to_tensor(sample_method='auto_detect', 
                         seg_len=2, 
                         n_mfcc_coeffs=20, 
                         device=None, # if GPUs
                         ):
    if sample_method not in ['random', 'first', 'auto_detect']:
        raise ValueError("Sample method must be 'random', 'first', or 'auto_detect'")
    
    if 'file' not in request.files:
        return jsonify(error='No file part')

    file = request.files['file']

    if file.filename == '':
        return jsonify(error='No selected file')

    # Assume the file received ends with .wav
    if file and file.filename.endswith('.wav'):
        if sample_method == 'first':
            seg, sr = librosa.load(file, duration=seg_len)
            # Add white noise
            # if id in train_ids:
            #     noise_factor = 0.0001
            #     white_noise = np.random.randn(len(y)) * noise_factor
            #     y = y + white_noise
        else:
            y, sr = librosa.load(file)
            seg_len_size = seg_len * sr
            if seg_len_size > len(y):
                seg = np.pad(y, (0, seg_len_size - len(y)), mode='constant')
            segments = [y[i : i + seg_len_size] for i in range(0, len(y), seg_len_size)] # shift window runs too slow
            if sample_method == 'random':
                seg = random.choice(segments)
            elif sample_method == 'auto_detect':
                energies = [np.sum(segment**2) for segment in segments]
                seg = segments[np.argmax(energies)]

        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc_coeffs).T

        #n_frames = mfcc.shape[0]
        pitch = librosa.yin(seg, fmin=75, fmax=600)
        #intensity = librosa.feature.rms(y=y).flatten()
        feature = np.concatenate((mfcc, np.expand_dims(pitch, axis=1)), axis=1)
        
        feature_tensor = torch.tensor(feature).unsqueeze(0) # As model input, shape: (1, n_frames, n_mfcc_coeffs + 1)
        if device is not None:
            feature_tensor = feature_tensor.to(device)

        # You can now work with the tensor as needed
        # ...
        # ...
        # ...

        return jsonify(success=True)

    return jsonify(error='Invalid file type')

if __name__ == '__main__':
    app.run(debug=True)
