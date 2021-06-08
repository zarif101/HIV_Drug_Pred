import torch
from torch.utils.data import DataLoader
from util import get_data,SmilesDataset
from models import ModelOne
import time
import warnings

warnings.filterwarnings("ignore")

X_train,X_test,y_train,y_test = get_data('data/HIV.csv',split=True,test_size=0.19)
train_dataset = SmilesDataset(X_train,y_train,410)
test_dataset = SmilesDataset(X_test,y_test,410)

train_loader = DataLoader(train_dataset,batch_size=10,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

NUM_EPOCHS = 10
model = ModelOne(1,1)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters())

def train_step():
    total_loss_this_step = 0
    for data in train_loader:
        features,labels = next(iter(train_loader))
        #print(features.shape)
        features = features.reshape((10,410))
        labels = labels.reshape((10,1)).type(torch.FloatTensor)
        optimizer.zero_grad()

        out = model(features)
        loss = criterion(out,labels)
        loss.backward()
        optimizer.step()
        total_loss_this_step+=loss
    return total_loss_this_step/len(train_loader.dataset) #Returning average loss for this epoch

# Main issue right now: DeepChem has a stupid and useless warning it keeps tossing, which SEVERELY clogs up the console during training
# I'm going to try to suppress it, but if that doesn't work, I'll probably write trainings history/info to a file while looking for a better solution

if __name__ == '__main__':
    e = 0
    f = open('model_training_history.txt','a')
    for epoch in range(NUM_EPOCHS):
        e+=1
        train_loss = train_step()
        print('Average loss: '+str(train_loss))
        f.write('Epoch '+str(e)+'...Train Loss: '+str(train_loss))
    f.close()
        #time.sleep(10)
