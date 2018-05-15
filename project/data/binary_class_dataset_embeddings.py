import torch
import torch.utils.data as data
import gzip
from tqdm import tqdm
import numpy as np


class BinaryClassTextEmbeddingsDataset(data.Dataset):

    def __init__(self, path, word_to_indx, max_length=50, data_limit=-1, nil_id=0, pad_id=0):
        self.path = path
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.nil_id = nil_id
        self.pad_id = pad_id
        with open(path, 'r') as f:
            lines = f.readlines()[:data_limit]
            print("Transforming corpus using embedding ids...")
            for line in tqdm(lines):
                sample = self.processLine(line)
                self.dataset.append(sample)
            f.close()

    ## Convert one line from beer dataset to {Text, Tensor, Labels}
    def processLine(self, line):
        label = torch.FloatTensor([float(line.split('\t')[0])])
        text = line.split('\t')[1].split()[:self.max_length]
        x = get_indices_tensor(text, self.word_to_indx, self.max_length, self.nil_id, self.pad_id)
        sample = {'x':x, 'y':label}
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample


def get_indices_tensor(text_arr, word_to_indx, max_length, nil_id, pad_id):
    text_indx = [word_to_indx[x] if x in word_to_indx else nil_id for x in text_arr][:max_length]
    if len(text_indx) < max_length:
        text_indx.extend([pad_id for _ in range(max_length - len(text_indx))])
    x = torch.LongTensor(text_indx)
    return x
