import deepchem
import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset

def get_data(path,split=True,test_size=0.2):
    data = pd.read_csv(path)
    smiles = data['smiles']
    targets = data['HIV_active']
    smiles = np.array(smiles)
    targets = np.array(targets)

    if split:
        X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(smiles,targets,test_size=test_size)
        return X_train,X_test,y_train,y_test
    else:
        X = smiles
        y = targets
        return X,y

def tokenize_smiles(smiles_features,pad_len=410,vocab_path='data/smiles_vocab.txt'):
    from tensorflow.keras.utils import to_categorical
    # Returns array of sequences with one-hot encoded smiles
    from deepchem.feat.smiles_tokenizer import SmilesTokenizer
    tokenizer = SmilesTokenizer(vocab_path)
    new_smiles = []
    for smiles in smiles_features:
        new = np.array(tokenizer.add_padding_tokens(tokenizer.encode(smiles),length=pad_len))
        new_smiles.append(new)
    new_smiles = np.array(new_smiles)
    new_smiles = new_smiles.reshape((-1,pad_len))
    new_smiles = to_categorical(new_smiles)
    return new_smiles

class SmilesDataset(Dataset):
    def __init__(self,smiles_features,targets,pad_len,vocab_path='data/smiles_vocab.txt'):
        self.smiles_features = smiles_features
        self.pad_len = pad_len
        self.vocab_path = vocab_path
        self.targets = targets

    def __getitem__(self,idx):
        feature = self.tokenize_smiles(self.smiles_features[idx],self.pad_len,self.vocab_path)
        target = self.targets[idx]
        return feature,target

    def __len__(self):
        return len(self.smiles_features)

    def tokenize_smiles(self,smiles_features,pad_len=410,vocab_path='data/smiles_vocab.txt'):
        # Returns array of sequences with one-hot encoded smiles
        from tensorflow.keras.utils import to_categorical
        from deepchem.feat.smiles_tokenizer import SmilesTokenizer
        tokenizer = SmilesTokenizer(vocab_path)
        new_smiles = []
        if isinstance(smiles_features,list):
            pass
        else:
            smiles_features = [smiles_features]
        for smiles in smiles_features:
            new = np.array(tokenizer.add_padding_tokens(tokenizer.encode(smiles),length=pad_len))
            new_smiles.append(new)
        new_smiles = np.array(new_smiles).reshape((1,pad_len))
        # vocab_file = pd.read_csv(vocab_path,header=None)
        # num_classes = len(vocab_file)
        # new_smiles = to_categorical(new_smiles,num_classes=num_classes)
        # new_smiles  = torch.from_numpy(new_smiles)
        # new_smiles = new_smiles.argmax(axis=1)
        # print('New smiles shape')
        # print(new_smiles.shape)
        new_smiles = torch.from_numpy(new_smiles)
        return new_smiles

#def load_dataset(smiles_features,pad_len=410,vocab_path='data/smiles.txt'):

def get_random_data_sampler_weights(y):
    weights = []
    for val in y:
        if y == 0:
            weights.append(0.3)
        elif y == 1:   # There are way more 0s than 1s, so we want to oversample the 1s
            weights.append(0.7)
    return weights
