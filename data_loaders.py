import os
import random
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset

def traverse_dir(
        root_dir,
        extension,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args):
    data_train = AudioDataset(
        args.data.train_path, args.data.train_frames)
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.train.num_workers,
        persistent_workers=(args.train.num_workers > 0),
        pin_memory=True
    )
    data_valid = AudioDataset(
        args.data.valid_path, args.data.train_frames)
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(self, path_root, data_lengths):
        super().__init__()
        self.path_root = path_root
        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extension='wav',
            is_pure=True,
            is_sort=True,
            is_ext=False
        )
        self.data_buffer = {}
        self.max_mfcc_length = data_lengths
        self.max_label_length = data_lengths
        print('Load all the data from :', path_root)
        for name in tqdm(self.paths, total=len(self.paths)):
            path_mfcc = os.path.join(self.path_root, 'mfcc', name) + '.npy'
            mfcc = np.load(path_mfcc)
            mfcc = torch.from_numpy(mfcc).float()
            
            path_label = os.path.join(self.path_root, 'label', name) + '.npy'
            label = np.load(path_label)
            label = torch.from_numpy(label).float()
            
            self.data_buffer[name] = {
                'mfcc': mfcc,
                'label': label,
            }


    def __getitem__(self, file_idx):
        name = self.paths[file_idx]
        data_buffer = self.data_buffer[name]

        # get item
        return self.get_data(name, data_buffer)

    def get_data(self, name, data_buffer):
        # load mfcc
        mfcc = data_buffer.get('mfcc')
        # pad mfcc to max_mfcc_length using the last frame if needed
        pad_size = self.max_mfcc_length - mfcc.size(1)
        if pad_size > 0:
            # Repeat the last frame instead of padding with zeros
            pad = mfcc[:, -1:].repeat(1, pad_size)
            mfcc = torch.cat((mfcc, pad), dim=1)
        elif pad_size < 0:
            mfcc = mfcc[:, :self.max_mfcc_length]

        # load label
        label = data_buffer.get('label')
        # pad label to max_label_length using the last value if needed
        pad_size = self.max_label_length - label.size(0)
        if pad_size > 0:
            # Repeat the last value instead of padding with -1
            pad = label[-1:].repeat(pad_size)
            label = torch.cat((label, pad), dim=0)
        elif pad_size < 0:
            label = label[:self.max_label_length]

        return dict(mfcc=mfcc, label=label, name=name)

    def __len__(self):
        # 返回批次数量
        return len(self.data_buffer)
