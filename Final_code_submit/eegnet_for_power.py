#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score,mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats.stats import pearsonr
torch.manual_seed(10)
from torchviz import make_dot
from scipy import io
import pickle


# In[8]:


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 30), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 18 timepoints.
        self.fc1 = nn.Linear(4*2*1, 1)


    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
#         print(x.shape)
        x = x.view(-1, 4*2*1)
        x = F.sigmoid(self.fc1(x))
        return x



net = EEGNet().cuda(0)
# print(net.forward(Variable(torch.Tensor(np.random.rand(1, 1, 120, 30)).cuda(0))))
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr= 0.0001)


# In[23]:


def evaluate(model, X, Y, params = ["acc"]):
    results = []
    batch_size = 32

    predicted = []

    for i in range(int(len(X)/batch_size)):
        s = i*batch_size
        e = i*batch_size+batch_size

        inputs = Variable(torch.from_numpy(X[s:e]).cuda(0))
        pred = model(inputs)

        predicted.append(pred.data.cpu().numpy())


    inputs = Variable(torch.from_numpy(X).cuda(0))
    predicted = model(inputs)

    predicted = predicted.data.cpu().numpy()

    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "loss":
            results.append(mean_squared_error(Y,predicted))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2*precision*recall/ (precision+recall))

    a = Y.reshape(Y.shape[0])
    b = predicted.reshape(predicted.shape[0])

    results.append(pearsonr(a,b)[0])
    return results


# In[10]:

substring = {'1','2','4','5','6','9','11','12','13','14','22'}
# substring = {'1'}
addr = '/home/vinay/Documents/tfs/codes/Final_code_submit/EEG_TS/session_'
res_addr = '/home/vinay/Documents/tfs/codes/Final_code_submit/EEG_TS/session_'
for ind in substring:
    print('File',ind,'begin')

    address = addr + ind + '/power_data_session_'+ind +'.mat'
    data = io.loadmat(address)
    X = data['x']
    y = data['y']

    X = np.expand_dims(X,axis=1).astype('float32')


    # In[11]:


    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, random_state=None)




    # In[24]:


    batch_size = 32
    test_losses_over_kfold=[]
    mapes=[]
    correlations_over_kfold = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        me = np.mean(X_train)
        sd = np.std(X_train)
        X_train = (X_train - me)/sd
        X_test = (X_test - me)/sd
        train_loss_over_epoch = []
        test_loss_over_epoch= []
        for epoch in range(25):  # loop over the dataset multiple times
            # print("\nEpoch ", epoch)

            running_loss = 0.0
            for i in range(int(len(X_train)/batch_size)-1):
                s = i*batch_size
                e = i*batch_size+batch_size

                inputs = torch.from_numpy(X_train[s:e])
                labels = torch.FloatTensor(np.array([y_train[s:e]]).T*1.0)

                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
    #             print(inputs)
                outputs = net(inputs)
    #             print(outputs)
                loss = criterion(outputs, labels)
                loss.backward()


                optimizer.step()

                running_loss += loss.item()
            make_dot(outputs)

            # Validation accuracy
            params = ["loss"]
    #         print(params)
    #         print("Training Loss ", running_loss)
    #         print("Train - ", evaluate(net, X_train, y_train, params))
    #         print("Test - ", evaluate(net, X_test, y_test, params))
            train_loss_over_epoch.append(evaluate(net, X_train, y_train, params)[0])
            test_loss_over_epoch.append(evaluate(net, X_train, y_train, params)[0])
        loss  , correlation = evaluate(net, X_test, y_test, params)
        test_losses_over_kfold.append(loss)
        correlations_over_kfold.append(correlation)


    # In[26]:


    mean_test_loss = np.mean(test_losses_over_kfold)
    std_test_loss = np.std(test_losses_over_kfold)
    mean_correlation = np.mean(correlations_over_kfold)
    std_correlation = np.std(correlations_over_kfold)





    Results = {'train_loss_over_epoch':train_loss_over_epoch,'test_loss_over_epoch':test_loss_over_epoch,'mean_test_loss': mean_test_loss, 'std_test_loss': std_test_loss,'mean_correlation':mean_correlation,'std_correlation':std_correlation}
    res_address = res_addr + ind + '/original_results.pkl'
    output_file = open(res_address, 'wb')
    pickle.dump(Results, output_file)
    output_file.close()
    print('File',ind,'done')
