import CMAPSSDataset
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

window_size = 32
datasets = CMAPSSDataset.CMAPSSDataset(fd_number='1', batch_size=32, sequence_length=window_size)
train_data = datasets.get_train_data()
train_feature_slice = datasets.get_feature_slice(train_data)
train_label_slice = datasets.get_label_slice(train_data)
test_data = datasets.get_test_data()
test_feature_slice, test_label_slice = datasets.get_last_data_slice(test_data)

timesteps = train_feature_slice.shape[1]
input_dim = train_feature_slice.shape[2]

X_train = torch.tensor(train_feature_slice).float()
y_train = torch.tensor(train_label_slice).float()
X_test = torch.tensor(test_feature_slice).float()
y_test = torch.tensor(test_label_slice).float()

ds = torch.utils.data.TensorDataset(X_train,y_train)
dataloader_train = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

@variational_estimator
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,out_channels=36,kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=72, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Linear(504,timesteps*input_dim) #300待求
        self.lstm_1 = BayesianLSTM(input_dim, 50, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.layer1(x)
        x = self.layer2(x) #8,7,72
        x = x.permute(0,2,1)
        x = x.reshape(x.size(0),-1) #8,504
        x = self.fc(x)
        x = x.view(-1,timesteps,input_dim)

        x_, _ = self.lstm_1(x)

        # gathering only the latent end-of-sequence for the linear layer
        x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        return x_
net = NN()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

#train
iteration = 0
for epoch in range(3):#test
    for i, (datapoints, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()

        loss = net.sample_elbo(inputs=datapoints,
                               labels=labels,
                               criterion=criterion,
                               sample_nbr=3,
                               complexity_cost_weight=1 / X_train.shape[0])
        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 250 == 0:
            preds_test = net(X_test)[:, 0].unsqueeze(1)
            loss_test = criterion(preds_test, y_test)
            print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss_test))


def pred_future(test,sample_nbr):
    preds_test = []
    for i in range(len(test)):
        input = test[i].unsqueeze(0)
        pred = [net(input).cpu().item() for i in range(sample_nbr)]
        preds_test.append(pred)
    return preds_test

def get_confidence_intervals(preds_test, ci_multiplier):
    preds_test = torch.tensor(preds_test)

    pred_mean = preds_test.mean(1)
    pred_std = preds_test.std(1).detach().cpu().numpy()

    pred_std = torch.tensor((pred_std))

    upper_bound = (pred_mean + (pred_std * ci_multiplier))
    lower_bound = (pred_mean - (pred_std * ci_multiplier))
    return pred_mean,upper_bound,lower_bound

preds_test=pred_future(X_test,5)
pred_mean,upper_bound,lower_bound=get_confidence_intervals(preds_test,2)
y_true = y_test.reshape(-1)
y=np.arange(len(y_true))

#评估
explained_variance=sklearn.metrics.explained_variance_score(y_true,pred_mean)
mse = sklearn.metrics.mean_squared_error(y_true,pred_mean)
mae = sklearn.metrics.mean_absolute_error(y_true,pred_mean)
medae=sklearn.metrics.median_absolute_error(y_true,pred_mean)
R2 = sklearn.metrics.r2_score(y_true,pred_mean)
print('explained variance:',explained_variance)  #保留n 位小数：round( ,n)
print('MSE:',mse)
print('MAE:',mae)
print('MedAE:',medae)
print('R2:',R2)

# plt.plot(y,
#          y_test.reshape(-1),
#          color='black',
#          label="Real")
#
# plt.plot(y,
#          pred_mean,
#          label="Prediction",
#          color="red")
#
# plt.fill_between(x=y,
#                  y1=upper_bound,
#                  y2=lower_bound,
#                  facecolor='green',
#                  label="Confidence interval",
#                  alpha=0.5)
#
# plt.legend()
# plt.show()