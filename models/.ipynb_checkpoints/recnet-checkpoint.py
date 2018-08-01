'''
Recurrent implementation of ResNet to train on Cifar100

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] Q. Liao, T. Poggio. Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex. In CBBM, 2016.
'''

import torch
import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class RecNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, parent=None):
        super(RecNetBlock, self).__init__()
        self.parent=parent
        if parent is None:
            self.convs=nn.ModuleList([conv3x3(in_channels, out_channels, stride), conv3x3(out_channels, out_channels, stride=1)])
            self.stride=stride
            self.downsample=downsample
        else:
            self.convs = parent.convs
            self.stride= parent.stride
            self.downsample=parent.downsample
        self.batchNorms=nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(2)])
        self.relu=nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual=x
        if self.downsample is not None:
            residual=self.downsample(x)
        #print('Residual shape: ',residual.shape)
        out=self.relu(self.batchNorms[0](self.convs[0](x)))
        print("output shape 1: ",out.shape)
        out=self.batchNorms[1](self.convs[1](out))
        #print("output shape 2: ",out.shape)
        out+=residual
        return self.relu(out)
    
class RecNetBlock_postrelu(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, parent=None):
        super(RecNetBlock_postrelu, self).__init__()
        self.parent=parent
        if parent is None:
            self.convs=nn.ModuleList([conv3x3(in_channels, out_channels, stride), conv3x3(out_channels, out_channels, stride=1)])
            self.stride=stride
            self.downsample=downsample
        else:
            self.convs = parent.convs
            self.stride= parent.stride
            self.downsample=parent.downsample
        self.batchNorms=nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(2)])
        self.relu=nn.ReLU()
        
    def forward(self, x):
        residual=x
        if self.downsample is not None:
            residual=self.relu(self.downsample(x))
        #print('Residual shape: ',residual.shape)
        out=self.relu(self.batchNorms[0](self.convs[0](x)))
        #print("output shape 1: ",out.shape)
        out=self.relu(self.batchNorms[1](self.convs[1](out)))
        #print("output shape 2: ",out.shape)
        out+=residual
        return out
    
    
class RecNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(RecNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
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
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
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
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    

class RecNetBlock_postrelu_affine(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, parent=None):
        super(RecNetBlock_postrelu_affine, self).__init__()
        self.parent=parent
        if parent is None:
            self.convs=nn.ModuleList([conv3x3(in_channels, out_channels, stride), conv3x3(out_channels, out_channels, stride=1)])
            self.stride=stride
            self.downsample=downsample
        else:
            self.convs = parent.convs
            self.stride= parent.stride
            self.downsample=parent.downsample
        self.batchNorms=nn.ModuleList([nn.BatchNorm2d(out_channels, affine=False) for _ in range(2)])
        self.relu=nn.ReLU()
        self.linearA=nn.Sequential(nn.Linear(2*out_channels, out_channels//2), nn.Linear(out_channels//2, 2*out_channels))
        self.linearB=nn.Sequential(nn.Linear(2*out_channels, out_channels//2),nn.Linear(out_channels//2, 2*out_channels))
        self.affine_size=out_channels
    def forward(self, x):
        (x, alphas, betas)=x
        split_alphas = alphas.split(self.affine_size)
        split_betas = betas.split(self.affine_size)
        residual=x[0]
        if self.downsample is not None:
            residual=self.relu(self.downsample(x))
        #print('Residual shape: ',residual.shape)
        out=self.batchNorms[0](self.convs[0](x))
        out=self.relu((out*split_alphas[0].view(1,-1,1,1))+split_alphas[1].view(1,-1,1,1))
        #print("output shape 1: ",out.shape)
        out=self.batchNorms[1](self.convs[1](out))
        out=self.relu((out*split_betas[0].view(1,-1,1,1))+split_betas[1].view(1,-1,1,1))
        #print("output shape 2: ",out.shape)
        out+=residual
        new_alphas = self.linearA(alphas)
        new_betas = self.linearB(betas)
        return (out, new_alphas, new_betas)
    

class RecNet_Affine(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(RecNet_Affine, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
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
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
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
        x = self.bn1(x)
        x = self.relu(x)
        
        (x, l1_alphas, l1_betas) = self.layer1((x, self.alphas[0], self.betas[0]))
        (x, l2_alphas, l2_betas) = self.layer2((x, self.alphas[1], self.betas[1]))
        (x, l3_alphas, l3_betas) = self.layer3((x, self.alphas[2], self.betas[2]))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    

 

class RecNet_FourLayers(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(RecNet_FourLayers, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            layers = []
            if stride != 1 or self.inplanes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes)
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
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
   
    
    
class RecNetLayerNormBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, parent=None):
        super(RecNetLayerNormBlock, self).__init__()
        self.parent=parent
        if parent is None:
            self.convs=nn.ModuleList([conv3x3(in_channels, out_channels, stride), conv3x3(out_channels, out_channels, stride=1)])
            self.stride=stride
            self.downsample=downsample
        else:
            self.convs = parent.convs
            self.stride= parent.stride
            self.downsample=parent.downsample
        self.relu=nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual=x
        if self.downsample is not None:
            residual=self.downsample(x)
        out=self.relu(self.convs[0](x))
        out=self.convs[1](out)
        out+=residual
        layerNorm=nn.LayerNorm(out.shape, eps=10, elementwise_affine=False).cuda()
        return self.relu(layerNorm(out))
    
    
class RecNet_LayerNorm(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(RecNet_LayerNorm, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
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
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
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
    
    
    
    
    
    
class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class BasicBlock_postrelu(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_postrelu, self).__init__()
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
    
class BasicBlock_postelu(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_postelu, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.elu = nn.ELU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        #print("output shape 1: ",out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out=self.elu(out)
        #print("output shape 2: ",out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
        #print("residual shape: ",residual.shape)

        out = out + residual
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


class ResNet_Cifar_ELU(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        super(ResNet_Cifar_ELU, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.elu = nn.ELU()
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
        x = self.elu(x)

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

def resnet_postrelu(**kwargs):
    model = ResNet_Cifar(BasicBlock_postrelu, [9, 9, 9], **kwargs)
    return model

def resnet_postelu(**kwargs):
    model = ResNet_Cifar_ELU(BasicBlock_postelu, [9, 9, 9], **kwargs)
    return model

def recnet(**kwargs):
    model = RecNet(RecNetBlock, [9,9,9], **kwargs)
    return model

def recnet4(**kwargs):
    model = RecNet_FourLayers(ResNetBlock_Recurrent, [9,9,9,9], **kwargs)
    return model

def recnet_postrelu(**kwargs):
    model = RecNet(RecNetBlock_postrelu, [9,9,9], **kwargs)
    return model

def recnet_affine(**kwargs):
    model = RecNet_Affine(RecNetBlock_postrelu_affine, [9,9,9], **kwargs)
    return model

def recnet_layernorm(**kwargs):
    model = RecNet_LayerNorm(RecNetLayerNormBlock, [9,9,9], **kwargs)
    return model

if __name__ == '__main__':
    net = preact_resnet110_cifar()
    y = net(torch.autograd.Variable(torch.randn(1, 3, 32, 32)))
    print(net)
    print(y.size())

