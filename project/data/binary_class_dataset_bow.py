import torch
import torch.utils.data as data
import gzip
from tqdm import tqdm
import numpy as np


class BinaryClassTextBOWDataset(data.Dataset):

    def __init__(self, path, vectorizer, max_length=50, data_limit=-1):
        self.dataset = []
        self.max_length = max_length
        self.vectorizer = vectorizer
        with open(path, 'r') as f:
            lines = f.readlines()[:data_limit]
            print("Transforming corpus into BOW representation")
            for line in tqdm(lines):
                sample = self.processLine(line)
                self.dataset.append(sample)
            f.close()

    ## Convert one line from beer dataset to {Text, Tensor, Labels}
    def processLine(self, line):
        label = torch.FloatTensor([int(line.split('\t')[0])])
        text = extract_clamped_text(line, self.max_length)
        x = self.vectorizer.transform([text])
        arr = torch.FloatTensor(x.todense())
        x = arr[0]
        sample = {'x':x, 'y':label}
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample


def extract_clamped_text(line, max_length):
    return " ".join(line.split('\t')[1].split()[:max_length])
