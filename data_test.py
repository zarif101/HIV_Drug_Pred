from torch.utils.data import DataLoader
from util import get_data,SmilesDataset

def main():
    X_train,X_test,y_train,y_test = get_data('data/HIV.csv',split=True,test_size=0.19)

    train_dataset = SmilesDataset(X_train,y_train,410)
    test_dataset = SmilesDataset(X_test,y_test,410)
    a,b = train_dataset.__getitem__(1)
    c,d = test_dataset.__getitem__(1)
    assert a.shape == (1, 410, 591)
    assert b == 0 or b == 1
    assert c.shape == (1, 410, 591)
    assert d == 0 or d == 1

    train_loader = DataLoader(train_dataset,batch_size=10,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

    print('All tests passed!')
if __name__ == '__main__':
    main()
