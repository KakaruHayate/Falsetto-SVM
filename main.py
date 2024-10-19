import os
import argparse
import torch

from SVM import SVM

import numpy as np
import librosa

from logger import utils
from solver import convert_tensor_values
import yaml


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="path to the model file",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    return parser.parse_args(args=args, namespace=namespace)


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


def load_model(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    model = SVM(input_dim=args.model.input_size, output_dim=args.model.output_size)
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, args


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    model, args = load_model(cmd.model_path, device='cpu')
    
    audio, sample_rate = librosa.load(cmd.input, sr=args.data.sampling_rate)
    
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    
    audio_t = librosa.effects.preemphasis(audio, args.data.pre_emph)
    mfcc = librosa.feature.mfcc(y=audio_t, sr=args.data.sampling_rate, n_mfcc=args.data.mfcc_order, n_fft=args.data.fft_size, hop_length=args.data.hop_length)
    mfcc = torch.from_numpy(mfcc).float().unsqueeze(0)
    
    with torch.no_grad():
        pred_label = model(mfcc)
    pred_label = convert_tensor_values(pred_label).cpu().numpy()
    
    if np.mean(pred_label) > 0.5:
        print("falsetto")
    else:
        print("chest")
