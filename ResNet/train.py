import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import data_loader
import optparse
import os
import ResNet


paser=optparse.OptionParser(usage='Specify paramenters for ResNet model')
paser.add_option('-b',dest='batch_size',type='int',default=64)

paser.add_option('--gw',dest='IMG_W',type='int',default=224)

paser.add_option('--gh',dest='IMG_H',type='int',default=224)

paser.add_option('--lr',dest='learning_rate',type='float',default=0.01)

paser.add_option('-d',dest='data_path',type='str')
paser.add_option('-M',dest='MAX_STEP',type='int',default=50000)
paser.add_option('--md',dest='method',type='str',default='train')

#paser.add_option('-o',dest='output',type='str')
#paser.add_option('-s',dest='model_path',type='str')
paser.add_option('-n',dest='n_classes',type='int',default=2)

# Parameters
(options,arg)=paser.parse_args()
batch_size=options.batch_size
image_size=[options.IMG_H,options.IMG_W]
learning_rate=options.learning_rate
image_path=options.data_path
#model_path=options.model_path
#outfile=options.output
num_classes=options.n_classes
num_epochs=options.MAX_STEP
method=options.method



def accuracy(outputs,labels):
    total=labels.size(0)
    _, predicted = torch.max(outputs.data, 1)    
    correct=(predicted == labels).sum()
    return correct/total



transform = transforms.Compose([
                    transforms.Scale(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_loader=data_loader.get_loader(image_path,batch_size,transform=transform,method='train')
test_loader=data_loader.get_loader(image_path,batch_size,transform=transform,method='val')
net=ResNet.resnet50()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)




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
        
        
        if (i+1) % 50 == 0:
            
            print ("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f ------ Acc: %.4f%%" %(epoch+1,num_epochs, i+1,len(train_loader)//batch_size,\
                     loss.data[0],correct/total*100.))

      # Decaying Learning Rate
#    if (epoch+1) % 10000 == 0:
#        learning_rate/= 3
#        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)





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

    print ("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f ------ Acc: %.4f%%" %(epoch+1,num_epochs, i+1,len(test_loader)//batch_size,\
                     loss.data[0],correct/total*100.))


for epoch in range(num_epochs):
    train(epoch)
    if (epoch+1) % 10000 == 0:
        learning_rate/= 3
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    test(epoch)
