'''
Recurrent implementation of ResNet to train on Cifar100

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] Q. Liao, T. Poggio. Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex. In CBBM, 2016.
'''

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from IPython import embed


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class RecNetBlock_Affine(nn.Module):

    def __init__(self, in_channels, out_channels, affineType, stride=1, downsample=None, parent=None):
        super(RecNetBlock_Affine, self).__init__()
        self.parent=parent
        if parent is None:
            self.convs=nn.ModuleList([conv3x3(in_channels, out_channels, stride), conv3x3(out_channels, out_channels, stride=1)])
            self.stride=stride
            self.downsample=downsample
        else:
            self.convs = parent.convs
            self.stride= parent.stride
            self.downsample=parent.downsample
        self.affineType=affineType
        self.batchNorms=nn.ModuleList([nn.BatchNorm2d(out_channels, affine=False) for _ in range(2)])
        self.relu=nn.ReLU()
        self.affine_size = out_channels
        if affineType=='linear':
            self.affineA = nn.Sequential(nn.Linear(2*out_channels, out_channels//2), nn.Linear(out_channels//2, 2*out_channels))
            self.affineB=nn.Sequential(nn.Linear(2*out_channels, out_channels//2), nn.Linear(out_channels//2, 2*out_channels))
        elif affineType=='gru':
            self.affineA = nn.GRU(2*out_channels,2*out_channels,1)
            self.affineB = nn.GRU(2*out_channels,2*out_channels,1)
            
    def forward(self, x):
        (x, bn1Affines, bn2Affines)=x
        bn1_WB = bn1Affines.split(self.affine_size)
        bn2_WB = bn2Affines.split(self.affine_size)
        residual=x
        if self.downsample is not None:
            residual=self.relu(self.downsample(x))
        #print('Residual shape: ',residual.shape)
        out=self.batchNorms[0](self.convs[0](x))
        out=self.relu(out*bn1_WB[0].view(1,-1,1,1)+bn1_WB[1].view(1,-1,1,1))
        #print("output shape 1: ",out.shape)
        out=self.batchNorms[1](self.convs[1](out))
        out=self.relu(out*bn2_WB[0].view(1,-1,1,1)+bn2_WB[1].view(1,-1,1,1))
        #print("output shape 2: ",out.shape)
        out+=residual
        if self.affineType=='gru':
            (new_bn1Affines, _) =self.affineA(bn1Affines.view(1,1,-1))
            (new_bn2Affines, _) = self.affineB(bn2Affines.view(1,1,-1))
            new_bn1Affines = new_bn1Affines.squeeze()
            new_bn2Affines = new_bn2Affines.squeeze()
        else:
            new_bn1Affines = self.affineA(bn1Affines)
            new_bn2Affines = self.affineB(bn2Affines)
        return (out, new_bn1Affines, new_bn2Affines)
    
class RecNet_Affine(nn.Module):
    def __init__(self, block, layers, affineType, num_classes=100):
        super(RecNet_Affine, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.sizes = [16,32,64]
        self.alphas = nn.ParameterList([nn.Parameter(torch.FloatTensor(2*sz)) for sz in self.sizes])
        self.betas = nn.ParameterList([nn.Parameter(torch.FloatTensor(2*sz)) for sz in self.sizes])
        self.affineType=affineType
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        for alpha in self.alphas:
            alpha.data.fill_(0)
        for beta in self.betas:
            beta.data.fill_(0)
    def AffineIterator(self):
        for (name, param) in self.named_parameters():
            if 'linear' in name or 'alphas' in name or 'betas' in name:
                yield param
    def AllButAffineIterator(self):
        for(name, param) in self.named_parameters():
            if 'linear' not in name and 'alphas' not in name and 'betas' not in name:
                yield param
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers.append(block(self.inplanes, planes, self.affineType, stride, downsample))
        self.inplanes = planes
        parent=block(self.inplanes, planes, self.affineType, downsample=None)
        layers.append(parent)
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.affineType, parent=parent))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        (x, l1_alphas, l1_betas) = self.layer1((x, self.alphas[0], self.betas[0]))
        (x, l2_alphas, l2_betas) = self.layer2((x, self.alphas[1], self.betas[1]))
        (x, l3_alphas, l3_betas) = self.layer3((x, self.alphas[2], self.betas[2]))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    
class RecNetAffineModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride, affineType=None, gruLoss=False):
        super(RecNetAffineModule, self).__init__()
        #self.in_channels = in_channels
        #self.out_channels = out_channels
        self.conv = conv3x3(in_channels, out_channels, stride)
        self.affineType = affineType
        if affineType is not None:
            self.gru = nn.GRU(2*in_channels, 2*out_channels)
        else:
            self.gru = None
        self.relu = nn.ReLU()
        self.affineSize = out_channels
        self.gruLoss=gruLoss
    def forward(self, x): 
        (x, affines, loss)=x
        out = self.conv(x)
        if self.affineType is None:
            return (self.relu(out), affines, loss)
        mean = torch.sum(torch.sum(x, 3), 2).mean(0)
        std = torch.sum(torch.sum(x, 3), 2).std(0)
        print('std for each channel: ', std)
        (new_affines, _) = self.gru(affines.view(1,1,-1), torch.cat((mean, std)).view(1,1,-1))
        print(new_affines)
        (new_mean, new_std) = new_affines.squeeze().split(self.affineSize)
        
        out = out*new_std.view(1,-1,1,1) + new_mean.view(1,-1,1,1)
        out = self.relu(out)
        if self.gruLoss==True:
            mean_var = Variable(new_mean)
            std_var = Variable(new_std)
            criterion = nn.L1Loss()
            #print(mean_var)
            new_loss = criterion(mean_var, torch.FloatTensor(self.affineSize).fill_(0))+criterion(std_var, torch.FloatTensor(self.affineSize).fill_(1))
            return (out, new_affines.squeeze(), loss+new_loss)
        else:
            return (out, new_affines.squeeze(), loss)
        

        
        
class RecNetBlockAffineModular(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, parent=None, affineType=None, getLoss=False):
        super(RecNetBlockAffineModular, self).__init__()
        self.parent=parent
        if parent is None:
            self.module1=RecNetAffineModule(in_channels, out_channels, stride, affineType, getLoss)
            self.module2=RecNetAffineModule(out_channels, out_channels, 1, affineType, getLoss)
            self.stride=stride
            self.downsample=downsample
        else:
            self.module1=parent.module1
            self.module2=parent.module2
            self.stride=parent.stride
            self.downsample=parent.downsample
        self.getLoss=getLoss
        self.relu=nn.ReLU()
    def forward(self, x):
        (x, (alphas, betas), loss)=x
        if self.downsample is not None:
            residual=self.relu(self.downsample(x))
        else:
            residual = x
        (out, new_alphas, new_loss)=self.module1((x,alphas,loss))
        (out, new_betas, new_loss)=self.module2((out, betas, new_loss))
        out+=residual
        return(out, (new_alphas, new_betas), new_loss)
       
        
class RecNetAffineModular(nn.Module):
    def __init__(self, layers, num_classes=100, gruLoss=False):
        super(RecNetAffineModular, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.sizes = [16,32,64]
        self.alphas = nn.ParameterList([nn.Parameter(torch.FloatTensor(2*sz)) for sz in self.sizes])
        self.betas = nn.ParameterList([nn.Parameter(torch.FloatTensor(2*sz)) for sz in self.sizes])
        self.relu = nn.ReLU(inplace=True)
        self.gruLoss=gruLoss
        self.layer1 = self._make_layer(16, layers[0])
        self.layer2 = self._make_layer(32, layers[1], stride=2)
        self.layer3 = self._make_layer(64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, num_classes)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        for alpha in self.alphas:
            alpha.data.fill_(0)
        for beta in self.betas:
            beta.data.fill_(0)
    def AffineIterator(self):
        for (name, param) in self.named_parameters():
            if 'linear' in name or 'alphas' or 'betas' in name:
                yield param
    def AllButAffineIterator(self):
        for(name, param) in self.named_parameters():
            if 'linear' not in name and 'alphas' or 'betas' not in name:
                yield param
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        layers = []
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False)
            layers.append(RecNetBlockAffineModular(self.inplanes, planes, stride, downsample, affineType=None, getLoss=self.gruLoss))
        else:
            layers.append(RecNetBlockAffineModular(self.inplanes, planes, stride, downsample, affineType='gru', getLoss=self.gruLoss))
        self.inplanes = planes
        parent=RecNetBlockAffineModular(self.inplanes, planes, downsample=None, affineType='gru', getLoss=self.gruLoss)
        layers.append(parent)
        for i in range(2, blocks):
            layers.append(RecNetBlockAffineModular(self.inplanes, planes, parent=parent, getLoss=self.gruLoss))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        loss = torch.zeros(1, requires_grad=True)
        (x, (l1_alphas,l1_betas), loss) = self.layer1((x, (self.alphas[0], self.betas[0]), loss))
        (x, (l2_alphas,l2_betas), loss) = self.layer2((x, (self.alphas[1],self.betas[1]), loss))
        (x, (l3_alphas,l3_betas), loss) = self.layer3((x, (self.alphas[2],self.betas[2]), loss))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return (x, loss)
    
    
    

class RecNetAffineNewInput(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(RecNetAffineNewInput, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.sizes = [16,32,64]
        self.alphas = nn.ParameterList([nn.Parameter(torch.FloatTensor(2*sz)) for sz in self.sizes])
        self.betas = nn.ParameterList([nn.Parameter(torch.FloatTensor(2*sz)) for sz in self.sizes])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        for alpha in self.alphas:
            alpha.data.fill_(0)
        for beta in self.betas:
            beta.data.fill_(0)
    def AffineIterator(self):
        for (name, param) in self.named_parameters():
            if 'linear' in name or 'alphas' in name or 'betas' in name:
                yield param
    def AllButAffineIterator(self):
        for(name, param) in self.named_parameters():
            if 'linear' not in name and 'alphas' not in name and 'betas' not in name:
                yield param
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
            )
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        parent=block(self.inplanes, planes, downsample=None)
        layers.append(parent)
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, parent=parent))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        (x, l1_alphas, l1_betas) = self.layer1((x, self.alphas[0], self.betas[0]))
        (x, l2_alphas, l2_betas) = self.layer2((x, self.alphas[1], self.betas[1]))
        (x, l3_alphas, l3_betas) = self.layer3((x, self.alphas[2], self.betas[2]))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    


    
class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print("output shape 1: ",out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out=self.relu(out)
        #print("output shape 2: ",out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
        #print("residual shape: ",residual.shape)

        out += residual
        return out

class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def recnet_affine(**kwargs):
    model = RecNet_Affine(RecNetBlock_Affine, [9,9,9], 'linear', **kwargs)
    return model

def recnet_affine_gru(**kwargs):
    model = RecNet_Affine(RecNetBlock_Affine, [9,9,9], 'gru', **kwargs)
    return model

def recnet_affine_new(**kwargs):
    model = RecNetAffineNewInput(RecNetBlockAffineNewInputs, [9,9,9], **kwargs)
    return model
def recnet_affine_modular(**kwargs):
    model = RecNetAffineModular([9,9,9], **kwargs)
    return model
if __name__ == '__main__':
    net = preact_resnet110_cifar()
    y = net(torch.autograd.Variable(torch.randn(1, 3, 32, 32)))
    print(net)
    print(y.size())

