# Using pretrained model and cluster to do find data mistake
import torch
import torch.nn as nn

def find_mistakes(audio, label, id):
    vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
    vocoder.eval()
    if len(audio.shape) != 3:
        raise ValueError('--- Mel shape {} should be (n_samples, frames, n_mfcc_coeffs)'.format(audio.shape))
    
    if torch.cuda.is_available():
        vocoder = vocoder.cuda()
        audio = audio.cuda()
        label = label.cuda()
        id = id.cuda
        
    print(vocoder.mel_channel)
    with torch.no_grad():
        print(audio.shape)
        audio = vocoder.inference(audio)
        print(audio.shape)
        print(audio[0])
        print(audio)


if __name__ == "__main__":
    pass