from models import ModelOne
from util import *
import torch

def main():
    X_train,X_test,y_train,y_test = get_data('data/HIV.csv',split=True,test_size=0.19)

    train_dataset = SmilesDataset(X_train,y_train,410)
    test_dataset = SmilesDataset(X_test,y_test,410)
    feat,target = train_dataset.__getitem__(1)
    model = ModelOne(1,1) # these 1s are temporary bc right now they aren't doing anything in ModelOne
    out = model(feat)
    print(out)

if __name__ == '__main__':
    main()
