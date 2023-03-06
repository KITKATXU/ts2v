import math

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import scipy.io as scio

def scalar(coarse):
    Max = float(max(coarse))
    Min = float(min(coarse))
    sin_list = [(i - Min)/((Max-Min)*2) for i in coarse]
    tan_list = [min(i/math.sqrt(1-i*i),56) for i in sin_list]
    return tan_list


class ToyDataset(Dataset):
    def __init__(self,error_type):
        super(ToyDataset, self).__init__()

        dataFile_l = '.\loads_train.mat'
        data_l = scio.loadmat(dataFile_l)


        dataFile = '.\moves_train.mat'
        data_m = scio.loadmat(dataFile)
        DATA=[]
        TYPE=[]
        for num in {1, error_type}:
            for i in range(len(data_m['moves_' + str(num)])):
                # i= i[:, np.newaxis]  # 从列的维度扩维
                for j in range(200):
                    if data_m['moves_' + str(num)][i][j] == 0:
                        tp_list = list(data_m['moves_' + str(num)][i])
                        data_m['moves_' + str(num)][i] = tp_list[j:] + tp_list[:j]
                        data_m['moves_' + str(num)][i] = scalar(data_m['moves_' + str(num)][i])
                        tp_list_l = list(data_l['loads_' + str(num)][i])
                        data_l['loads_' + str(num)][i] = tp_list_l[j:] + tp_list_l[:j]
                        data_l['loads_' + str(num)][i] = scalar(data_l['loads_' + str(num)][i])
                        break
                # DATA.append([list(t).append(num) for t in zip(data_m['moves_' + str(num)][i], data_l['loads_' + str(num)][i])])
                tmp=[]
                tmp=tmp +[a for a in data_m['moves_' + str(num)][i]]
                tmp=tmp +[a for a in data_l['loads_' + str(num)][i]]
                TYPE.append(int((num-1)>0))
                DATA.append(tmp)
        Ccolumns =[]
        Ccolumns = Ccolumns+['moves'+str(i) for i in range(1,201)]
        Ccolumns = Ccolumns+['loads'+str(i) for i in range(1,201)]
        # Ccolumns.append('type')
        # print((DATA))
        df = pd.DataFrame(DATA, columns=Ccolumns)
        print(len(np.array(TYPE)))


        self.x = df.to_numpy()
        # print(self.x)
        self.y = np.array(TYPE)
        # print(len(self.x))
        # print(len(self.y))
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return np.array(self.x[idx]), self.y[idx]

class TestDataset(Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()

        dataFile_l = '.\loads_test.mat'
        data_l = scio.loadmat(dataFile_l)


        dataFile = '.\moves_test.mat'
        data_m = scio.loadmat(dataFile)
        DATA=[]
        TYPE=[]
        for num in range(1,14):
            for i in range(len(data_m['moves_' + str(num)])):
                # i= i[:, np.newaxis]  # 从列的维度扩维
                for j in range(200):
                    if data_m['moves_' + str(num)][i][j] == 0:
                        tp_list = list(data_m['moves_' + str(num)][i])
                        data_m['moves_' + str(num)][i] = tp_list[j:] + tp_list[:j]
                        data_m['moves_' + str(num)][i] = scalar(data_m['moves_' + str(num)][i])
                        tp_list_l = list(data_l['loads_' + str(num)][i])
                        data_l['loads_' + str(num)][i] = tp_list_l[j:] + tp_list_l[:j]
                        data_l['loads_' + str(num)][i] = scalar(data_l['loads_' + str(num)][i])
                        break
                # DATA.append([list(t).append(num) for t in zip(data_m['moves_' + str(num)][i], data_l['loads_' + str(num)][i])])
                tmp=[]
                tmp=tmp +[a for a in data_m['moves_' + str(num)][i]]
                tmp=tmp +[a for a in data_l['loads_' + str(num)][i]]
                TYPE.append(int((num-1)>0))
                DATA.append(tmp)
        Ccolumns =[]
        Ccolumns = Ccolumns+['moves'+str(i) for i in range(1,201)]
        Ccolumns = Ccolumns+['loads'+str(i) for i in range(1,201)]
        # Ccolumns.append('type')
        # print((DATA))
        df = pd.DataFrame(DATA, columns=Ccolumns)
        print(len(np.array(TYPE)))


        self.x = df.to_numpy()
        # print(self.x)
        self.y = np.array(TYPE)
        # print(len(self.x))
        # print((self.y))
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return np.array(self.x[idx]), self.y[idx]

if __name__ == "__main__":
    dataset = ToyDataset()
    # print(dataset[2200])