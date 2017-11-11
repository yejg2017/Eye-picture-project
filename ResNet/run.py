import torch 
import torch.nn as nn
#import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import data_loader
import optparse
import os
import ResNet
#from utils import progress_bar

paser=optparse.OptionParser(usage='Specify paramenters for ResNet model')
paser.add_option('-b',dest='batch_size',type='int',default=16)

paser.add_option('--gw',dest='IMG_W',type='int',default=208)

paser.add_option('--gh',dest='IMG_H',type='int',default=208)

paser.add_option('--lr',dest='learning_rate',type='float',default=0.001)

paser.add_option('-d',dest='data_path',type='str')
paser.add_option('-M',dest='MAX_STEP',type='int',default=50000)
paser.add_option('--md',dest='method',type='str',default='train')

#paser.add_option('-o',dest='output',type='str')
paser.add_option('-s',dest='model_path',type='str')
paser.add_option('-n',dest='n_classes',type='int',default=2)
paser.add_option('-w',dest='weight_decay',type='float',default=1e-8)

# Parameters
(options,arg)=paser.parse_args()
batch_size=options.batch_size
image_size=[options.IMG_H,options.IMG_W]
learning_rate=options.learning_rate
image_path=options.data_path
model_path=options.model_path
#outfile=options.output
num_classes=options.n_classes
num_epochs=options.MAX_STEP
method=options.method
weight_decay=options.weight_decay


def accuracy(outputs,labels):
    total=labels.size(0)
    _, predicted = torch.max(outputs.data, 1)    
    correct=(predicted == labels).sum()
    return correct/total



transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.Scale(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_val = transforms.Compose([
    transforms.Scale(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])



train_loader=data_loader.get_loader(image_path,batch_size,transform=transform_train,method='train')
test_loader=data_loader.get_loader(image_path,batch_size,transform=transform_val,method='val')
print(len(train_loader))
print(len(test_loader))
net=ResNet.resnet50()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=weight_decay)

model_style=model_path+'checkpoint_lr_%f_bs_%d_img_%d/'%(learning_rate,batch_size,image_size[0])+'cnn.pkl'

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        
        images,labels=Variable(images),Variable(labels)
        optimizer.zero_grad()
        outputs=net(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        train_loss+=loss.data[0]
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=predicted.eq(labels.data).sum()
        
        
#        if (i+1) % 20 == 0:
#            
#            print ("TRIAN: Epoch [%d/%d], Iter [%d/%d] Loss: %.4f ------ Acc: %.4f%%" %(epoch+1,num_epochs, i+1,len(train_loader)//batch_size,\
#                     loss.data[0],correct/total*100.))

#        progress_bar(i, len(train_loader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(i+1), 100.*correct/total, correct, total))
        if (i+1)%50==0:
            print('Train Epoch:%d---Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch,train_loss/(i+1), 100.*correct/total, correct, total))





def test(epoch):
    #global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for i, (images,labels) in enumerate(test_loader):
        images,labels=Variable(images),Variable(labels)
        outputs=net(images)
        loss=criterion(outputs,labels)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total +=labels.size(0)
        correct += predicted.eq(labels.data).sum()
        
#        print ("VAL *** Epoch [%d/%d], Iter [%d/%d] Loss: %.4f ------ Acc: %.4f%% ***" %(epoch+1,num_epochs, i+1,len(test_loader)//batch_size,\
#                     loss.data[0],correct/total*100.))
#        progress_bar(i, len(test_loader), 'Val ** Loss: %.3f | Acc: %.3f%% (%d/%d) **'% (test_loss/(i+1), 100.*correct/total, correct, total))
        if (i+1)%50==0:
           print('Val Epoch:%d---Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch,test_loss/(i+1), 100.*correct/total, correct, total))

for epoch in range(num_epochs):
    
    if (epoch+1) % 10000 == 0:
        learning_rate/= 3
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    train(epoch)
    test(epoch)


torch.save(cnn.state_dict(),model_style)
