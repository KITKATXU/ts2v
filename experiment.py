from Data import ToyDataset, TestDataset
from periodic_activations import SineActivation, CosineActivation
import torch
from torch.utils.data import DataLoader
from Pipeline import AbstractPipelineClass
from torch import nn
from Model import Model
import numpy as np

class ToyPipeline(AbstractPipelineClass):
    def __init__(self, model,error_type,epoch):
        self.model = model
        self.error_type = error_type
        self.epoch = epoch
    
    def train(self):
        loss_fn = nn.CrossEntropyLoss()

        dataset = ToyDataset(self.error_type)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-1)

        num_epochs = self.epoch

        for ep in range(num_epochs):
            for x, y in dataloader:

                optimizer.zero_grad()

                y_pred = self.model(x.to(torch.float32))

                loss = loss_fn(y_pred, y.long())


                loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=5, norm_type=2)


                optimizer.step()

                
                # print("epoch: {}, loss:{}".format(ep, loss.item()))
    
    def preprocess(self, x):
        return x
    
    def decorate_output(self, x):
        return x

if __name__ == "__main__":
    pipe2 = ToyPipeline(Model("sin", 72),2,1000)
    pipe2.train()
    pipe3 = ToyPipeline(Model("sin", 72), 3, 1000)
    pipe3.train()
    pipe4 = ToyPipeline(Model("sin", 72), 4, 800)
    pipe4.train()
    pipe5 = ToyPipeline(Model("sin", 72), 5, 1000)
    pipe5.train()
    pipe6 = ToyPipeline(Model("sin", 72), 6, 650)
    pipe6.train()
    pipe7 = ToyPipeline(Model("sin", 72), 7, 1000)
    pipe7.train()
    pipe8 = ToyPipeline(Model("sin", 72), 8, 1000)
    pipe8.train()
    pipe9 = ToyPipeline(Model("sin", 72), 9, 800)
    pipe9.train()
    pipe10 = ToyPipeline(Model("sin", 72), 10, 1000)
    pipe10.train()
    pipe11 = ToyPipeline(Model("sin", 72), 11, 800)
    pipe11.train()
    pipe12 = ToyPipeline(Model("sin", 72), 12, 1200)
    pipe12.train()
    pipe13 = ToyPipeline(Model("sin", 72), 13, 300)
    pipe13.train()

    print("Testing Begining ... ")  # 模型测试
    total = 0
    correct = 0

    testset = TestDataset()
    testloader = DataLoader(testset, batch_size=512, shuffle=True)

    for i, data_tuple in enumerate(testloader, 0):
        data, labels = data_tuple
        print(data)
        print(labels)
        old_preds = [0]*len(labels)
        for pi in {pipe2,pipe3,pipe4,pipe5,pipe6,\
                   pipe7,pipe8,pipe9,pipe10,pipe11,pipe12,pipe13}:
            output = pi.predict(data)
            _, preds_tensor = torch.max(output, 1)
            tmp_old = []
            for tmp_i in range(len(preds_tensor)):
                tmp_old.append(int((preds_tensor[tmp_i]+old_preds[tmp_i])>0))
            old_preds = torch.tensor(tmp_old)
            # print(old_preds)
            # total += labels.size(0)
            # correct += np.squeeze((old_preds == labels).sum().numpy())
            # print("Accuracy : {} %".format(correct / total))
        preds_tensor = old_preds

        print(preds_tensor)

        total += labels.size(0)
        correct += np.squeeze((preds_tensor == labels).sum().numpy())
    print("Accuracy : {} %".format(correct / total))




    #pipe = ToyPipeline(Model("cos", 12))
    #pipe.train()
