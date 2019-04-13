import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils import data
import torchvision
import torchvision.models as models
import os,csv,math
import numpy as np
import visdom
from PIL import Image
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
vis = visdom.Visdom(env=u'traffic_sign_classifier',use_incoming_socket=False)

BATCH_SIZE = 64
LR = 0.01
WEIGHT_DECAY = 0.00005
MOMENTUM = 0.9
NUM_WORKER = 4
PATIENCE = 4
EPOCH = 20

train_img_path = './GTSRB/Final_Training/Images'
test_img_path = './GTSRB/Final_Test/Images'
model_path = './model'

class GTSRB(data.Dataset):
    def __init__(self, root, is_train=True, transform=None):
        self.root = root
        self.transform = transform
        self.is_train = is_train  # training set or test set
        
        self.images = [] # images
        self.labels = [] # corresponding labels

        if self.is_train == True:
            # loop over all 42 classes
            for c in range(0,43):
                prefix = self.root + '/' + format(c, '05d') + '/' # subdirectory for class
                gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
                gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
                next(gtReader) # skip header
                # loop over all images in current annotations file
                for row in gtReader:
                    self.images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
                    self.labels.append(row[7]) # the 8th column is the label
            gtFile.close()
        else:
            prefix = self.root + '/' # subdirectory for class
            gtFile = open(prefix + 'GT-final_test.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader) # skip header
                # loop over all images in current annotations file
            for row in gtReader:
                self.images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
                self.labels.append(row[7]) # the 8th column is the label
            gtFile.close()

    def __getitem__(self, index):

        img, target = self.images[index], self.labels[index]
        img = Image.fromarray(np.uint8(img))
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        if self.transform is not None:
            img = self.transform(img)
#         print(img.shape)
        target = torch.tensor(int(target))
#         print(target)
        return img, target

    def __len__(self):
        return len(self.labels)
    
class VGG_GTSRB(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.conv = torch.nn.Sequential(*list(self.features.children())) 
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential( #分类器结构
            #fc6
            torch.nn.Linear(512*7*7, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
 
            #fc7
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
 
            #fc8
            torch.nn.Linear(4096, 43))
        
        self._initialize_weights()
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
  
    def forward(self,X):
        X = self.conv(X)
        X = self.avgpool(X)
        X = X.view(X.size(0),-1)
        X = self.classifier(X)
        return X
    
train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=512),  # Let smaller edge match
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor()
        ])
test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor()
        ])   
    
    
train_dataset = GTSRB(train_img_path,is_train=True,transform=train_transforms)
test_dataset = GTSRB(test_img_path,is_train=False,transform=test_transforms)
train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=NUM_WORKER, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=NUM_WORKER, pin_memory=True)

    

net = net = torch.nn.DataParallel(VGG_GTSRB()).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
solver = torch.optim.SGD(
            net.parameters(), lr=LR,
            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(solver,patience=PATIENCE)



best_score = 0
best_epoch = 0
best_acc = 0

#TEST
def test(data_loader):
    net.train(False)
    num_correct = 0
    num_total = 0
    for X, y in data_loader:
        # Data.
        X = torch.autograd.Variable(X.cuda())
        y = torch.autograd.Variable(y.cuda(non_blocking = True))
        # Prediction.
        score = net(X)
        _, prediction = torch.max(score.data, 1)
        num_total += y.size(0)
        num_correct += torch.sum(prediction == y.data)
    net.train(True)  # Set the model to training phase
    num_correct = torch.tensor(num_correct).float().cuda()
    num_total = torch.tensor(num_total).float().cuda()
    return 100 * num_correct / num_total



# TRAIN
ii = 0
for t in range(EPOCH):
    epoch_loss = []
    num_correct = 0
    num_total = 0
    for X, y in train_loader:
        X = torch.autograd.Variable(X.cuda())
        y = torch.autograd.Variable(y.cuda(non_blocking = True))
        solver.zero_grad()
        score = net(X)
        loss = criterion(score, y)
        epoch_loss.append(loss.item())
        _, prediction = torch.max(score.data, 1)
        num_total += y.size(0)
        num_correct += torch.sum(prediction == y.data)
        loss.backward()
        solver.step()
       
        ii += 1
        x = torch.Tensor([ii])
        y = torch.Tensor([loss.item()])
        vis.line(X=x, Y=y, win='loss', update='append' if ii>0 else None)
        
    num_correct = torch.tensor(num_correct).float().cuda()
    num_total = torch.tensor(num_total).float().cuda()

    train_acc = 100 * num_correct / num_total
    test_acc = test(test_loader)
    scheduler.step(test_acc)
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = t + 1
        print('*', end='')
        # Save model onto disk.
        torch.save(net.state_dict(),os.path.join(model_path,'tsc_epoch_%d.pth' % (t + 1)))
        print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
        
print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))



    
