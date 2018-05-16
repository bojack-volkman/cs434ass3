import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns

cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

torch.manual_seed(42)
if cuda:
    torch.cuda.manual_seed(42)

batch_size = 32

kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}

#loads training data from CIFAR10 db into training object
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

#loads testing data from CIFAR10 db into validation object
validation_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)

#directly from source provided by instructor
for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

#commented out plotting stuff since it doesn't run on server	
'''
pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray")
    plt.title('Class: '+str(y_train[i]))
'''

#nn class from PyTorch
#code for this section can be found here: 
#https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 100)
        #self.fc2 = nn.Linear(120, 84) removed this layer
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.sigmoid(self.conv1(x)))#change to F.relu for question 2
        x = self.pool(F.sigmoid(self.conv2(x)))#change to F.relu for question 2
        x = x.view(-1, 16 * 5 * 5)
        x = F.sigmoid(self.fc1(x)) #change to relu for question 2
        #x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x

model = Net()
if cuda:
    model.cuda()

#change learning rate (lr), momentum, or add parameter weight_decay for "experiments"       
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

print(model)

#directly from source provided by instructor
def train(epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

#directly from source provided by instructor
def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        val_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))


epochs = 10

#directly from source provided by instructor
lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)


#plt.figure(figsize=(5,3))
#plt.plot(np.arange(1,epochs+1), lossv)
#plt.title('validation loss')

#plt.figure(figsize=(5,3))
#plt.plot(np.arange(1,epochs+1), accv)
#plt.title('validation accuracy');	
