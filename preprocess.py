import os
import numpy as np
import librosa
import argparse
import shutil
from logger import utils
from tqdm import tqdm
from logger.utils import traverse_dir
import concurrent.futures
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


def preprocess(
        path_srcdir, 
        path_mfccdir,
        path_labeldir,
        mfcc_order, 
        fft_size,
        sampling_rate,
        hop_length, 
        pre_emph
        ):
        
    # list files
    filelist =  traverse_dir(
        path_srcdir,
        extension='wav',
        is_pure=True,
        is_sort=True,
        is_ext=True)


    # run
    
    def process(file):
        ext = file.split('.')[-1]
        binfile = file[:-(len(ext)+1)]+'.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_mfccfile = os.path.join(path_mfccdir, binfile)
        path_labelfile = os.path.join(path_labeldir, binfile)
        
        # load audio
        audio_t, _ = librosa.load(path_srcfile, sr=sampling_rate)
        
        # extract MFCC
        audio_t = librosa.effects.preemphasis(audio_t, pre_emph) # PreEmphasis
        mfcc = librosa.feature.mfcc(y=audio_t, sr=sampling_rate, n_mfcc=mfcc_order, n_fft=fft_size, hop_length=hop_length)
        # Normalize MFCC
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # mfcc = scaler.fit_transform(mfcc.T).T
        # extract label
        mfcc_frames = mfcc.shape[1]
        if "chest" in file[:-(len(ext)+1)]:
            labels = -np.ones(mfcc_frames)
        else:
            labels = np.ones(mfcc_frames)
        
        os.makedirs(os.path.dirname(path_mfccfile), exist_ok=True)
        np.save(path_mfccfile, mfcc)
        os.makedirs(os.path.dirname(path_labelfile), exist_ok=True)
        np.save(path_labelfile, labels)

    print('Preprocess the audio clips in :', path_srcdir)
    
    # single process
    for file in tqdm(filelist, total=len(filelist)):
        process(file)
    
    # multi-process (have bugs)
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process, filelist), total=len(filelist)))
    '''
                
if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    mfcc_order = args.data.mfcc_order
    fft_size = args.data.fft_size
    sampling_rate = args.data.sampling_rate
    hop_length = args.data.hop_length
    train_path = args.data.train_path
    valid_path = args.data.valid_path
    pre_emph = args.data.pre_emph
    label_path = args.data.label_path
    
    # run
    for path in [train_path, valid_path]:
        path_srcdir  = os.path.join(path, 'audio')
        path_mfccdir  = os.path.join(path, 'mfcc')
        path_labeldir  = os.path.join(path, 'label')
        preprocess(
            path_srcdir, 
            path_mfccdir,
            path_labeldir,
            mfcc_order, 
            fft_size, 
            sampling_rate,
            hop_length, 
            pre_emph
            )
    
