import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelOne(nn.Module):
    def __init__(self,seq_length,vec_length):
        super(ModelOne, self).__init__()
        self.embedding = nn.Embedding(591,100) #100 is embedding dim
        self.lstm = nn.LSTM(100,10) # 10 is LSTM dim
        self.linear = nn.Linear(4100,1)

    def forward(self,seq):
        embeddings = self.embedding(seq)
        lstm_out,_ = self.lstm(embeddings)
        x = self.linear(lstm_out.view(len(seq), -1))
        x = torch.sigmoid(x)
        return x

class ModelTwo(nn.Module): # Does not currently work
    def __init__(self,seq_length,vec_length):
        super(ModelTwo, self).__init__()
        self.embedding = nn.Embedding(591,100) #100 is embedding dim
        self.lstm = nn.LSTM(100,20,batch_first=True) # 10 is LSTM dim
        self.linear = nn.Linear(20,1)

    def forward(self,seq):
        embeddings = self.embedding(seq)
        lstm_out,(ht,hc) = self.lstm(embeddings)
        x = self.linear(lstm_out)
        print(x.shape)
        x = torch.sigmoid(x)
        return x
