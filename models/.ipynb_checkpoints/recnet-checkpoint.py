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

"""
Modular Affine RecNet component definitions
"""

class marModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride, affine_type=None, gru_loss=False):
        super(marModule, self).__init__()
        self.conv = self.conv3x3(in_channels, out_channels, stride)
        self.norm = nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False)
        self.affine_type = affine_type
        if affine_type is not None:
            self.gru = nn.GRU(2*in_channels, 2*out_channels, 1)
        else:
            self.gru = None
        self.relu = nn.ReLU()
        self.affine_size = out_channels
        self.gru_loss=gru_loss
        
    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
    def forward(self, x): 
        (x, affines, loss)=x
        out = self.conv(x)
        if self.affine_type is not None:
            intermediate = x.transpose(0, 1).contiguous().view(x.shape[1], -1)
            mean = intermediate.mean(-1)
            std = intermediate.std(-1, unbiased=False)
            (new_affines, _) = self.gru(affines.view(1,1,-1), torch.cat((mean, std)).view(1,1,-1))
            (new_mean, new_std) = new_affines.squeeze().split(self.affine_size)
            out = self.norm(out)
            out = out*new_std.view(1,-1,1,1) + new_mean.view(1,-1,1,1)
            out = self.relu(out)
            if self.gru_loss is True:
                new_loss = torch.abs(new_mean).mean()+torch.abs(new_std-1).mean()
                return (out, new_affines.squeeze(), loss+new_loss)
            else:
                return (out, new_affines.squeeze(), loss)
        else:
            return (self.relu(self.norm(out)), affines, loss)

        
class marBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, affine_type=None, get_gru_loss=False):
        super(marBlock, self).__init__()        
        self.module1=marModule(in_channels, out_channels, stride, affine_type, get_gru_loss)
        self.module2=marModule(out_channels, out_channels, 1, affine_type, get_gru_loss)
        self.stride=stride
        self.downsample=downsample
        self.get_gru_loss=get_gru_loss
        self.relu=nn.ReLU()
        
    def forward(self, x):
        (x, (alphas, betas), loss)=x
        if self.downsample is not None:
            residual=self.relu(self.downsample(x))
        else:
            residual = x
        (out, new_alphas, new_loss)=self.module1((x,alphas,loss))
        (out, new_betas, new_loss)=self.module2((out, betas, new_loss))
        out = out + residual
        return(out, (new_alphas, new_betas), new_loss)
       
    def stop_gru_loss(self):
        self.module1.gru_loss=False
        self.module2.gru_loss=False

class marLayer(nn.Module):
    def __init__(self, inplanes, planes, blocks, gru_loss, stride=1):
        super(marLayer, self).__init__()
        self.downsample = None
        self.upsampling_block = None
        self.timesteps = blocks
        self.inplanes = inplanes
        self.gru_loss = gru_loss
        self.stride = stride
        
        if self.stride is not 1 or self.inplanes is not planes:
            self.downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=self.stride, bias=False), 
                                       nn.BatchNorm2d(planes, affine=False, track_running_stats=False))
            self.upsampling_block = marBlock(self.inplanes, planes, self.stride, downsample=self.downsample, affine_type=None, get_gru_loss=self.gru_loss)
            self.timesteps = blocks - 1
        
        self.inplanes = planes
        self.layer_block=marBlock(self.inplanes, planes, downsample=None, affine_type='gru', get_gru_loss=self.gru_loss)
    
    def forward(self, x):
        self.timestep_alphas = nn.ParameterList([])
        self.timestep_betas = nn.ParameterList([])
        out = x
        if self.upsampling_block is not None:
            out = self.upsampling_block(x)
        for _ in range(self.timesteps):
            out = self.layer_block(out)
            (_, (alphas, betas), _) = out
            self.timestep_alphas.append(nn.Parameter(alphas))
            self.timestep_betas.append(nn.Parameter(betas))
        return out
        
    def kill_pretraining(self):
        if self.upsampling_block is not None:
            self.upsampling_block.stop_gru_loss()
        self.layer_block.stop_gru_loss()

class ModularAffineRecnet(nn.Module):
    def __init__(self, layers, num_classes=100, get_gru_loss=False):
        super(ModularAffineRecnet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(16, affine=False, track_running_stats=False)
        self.sizes = [16,32,64]
        self.alphas = nn.ParameterList([nn.Parameter(torch.FloatTensor(2*sz)) for sz in self.sizes])
        self.betas = nn.ParameterList([nn.Parameter(torch.FloatTensor(2*sz)) for sz in self.sizes])
        self.relu = nn.ReLU(inplace=True)
        self.get_gru_loss=get_gru_loss
        self.layer1 = marLayer(self.inplanes, self.sizes[0], layers[0], self.get_gru_loss)
        self.layer2 = marLayer(self.sizes[0], self.sizes[1], layers[1], self.get_gru_loss, stride=2)
        self.layer3 = marLayer(self.sizes[1], self.sizes[2], layers[2], self.get_gru_loss, stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, num_classes)
        self.dtype = torch.cuda.FloatTensor
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
        for i in range(len(self.sizes)):
            (alpha_mean, alpha_std)=self.alphas[i].split(self.sizes[i])
            (beta_mean, beta_std)=self.betas[i].split(self.sizes[i])
            alpha_mean.fill_(0)
            alpha_std.fill_(1)
            beta_mean.fill_(0)
            beta_std.fill_(1)
            self.alphas[i] = nn.Parameter(torch.cat((alpha_mean, alpha_std)))
            self.betas[i] = nn.Parameter(torch.cat((beta_mean, beta_std)))
        
        
    def turn_off_pretraining(self):
        self.layer1.kill_pretraining()
        self.layer2.kill_pretraining()
        self.layer3.kill_pretraining()
    
    def get_affine_parameters(self):
        for (name, param) in self.named_parameters():
            if 'linear' in name or 'alphas' or 'betas' in name:
                yield param
                
    def get_all_but_affine_params(self):
        for(name, param) in self.named_parameters():
            if 'linear' not in name and 'alphas' or 'betas' not in name:
                yield param
            
            
    def forward(self, x):
        gru_loss = torch.zeros(1).type(self.dtype)
        x = self.relu(self.norm1(self.conv1(x)))
        (x, _, loss) = self.layer1((x, (self.alphas[0], self.betas[0]), gru_loss))
        (x, _, loss) = self.layer2((x, (self.alphas[1],self.betas[1]), loss))
        (x, _, loss) = self.layer3((x, (self.alphas[2],self.betas[2]), loss))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return (x, loss)
    


    
"""
Old Networks and definitions needed to run them
"""


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class AffineRecnetBlock(nn.Module):
    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def __init__(self, in_channels, out_channels, affine_type, stride=1, downsample=None, parent=None):
        super(AffineRecnetBlock, self).__init__()
        self.parent=parent
        if parent is None:
            self.convs=nn.ModuleList([self.conv3x3(in_channels, out_channels, stride), self.conv3x3(out_channels, out_channels, stride=1)])
            self.stride=stride
            self.downsample=downsample
        else:
            self.convs = parent.convs
            self.stride= parent.stride
            self.downsample=parent.downsample
        self.affine_type=affine_type
        
        self.batchnorms=nn.ModuleList([nn.BatchNorm2d(out_channels, affine=False) for _ in range(2)])
        self.relu=nn.ReLU()
        self.affine_size = out_channels
        self.alpha = nn.Parameter(torch.zeros(self.affine_size))
        self.beta = nn.Parameter(torch.zeros(self.affine_size))
        if affine_type is 'linear':
            self.affine_a = nn.Sequential(nn.Linear(2*out_channels, out_channels//2), nn.Linear(out_channels//2, 2*out_channels))
            self.affine_b=nn.Sequential(nn.Linear(2*out_channels, out_channels//2), nn.Linear(out_channels//2, 2*out_channels))
        elif affine_type is 'gru':
            self.affine_a = nn.GRU(2*out_channels,2*out_channels,1)
            self.affine_b = nn.GRU(2*out_channels,2*out_channels,1)
            
    def forward(self, x):
        (x, bn1_affines, bn2_affines)=x
        bn1_wb = bn1_affines.split(self.affine_size)
        bn2_wb = bn2_affines.split(self.affine_size)
        residual=x
        if self.downsample is not None:
            residual=self.relu(self.downsample(x))
        out=self.batchnorms[0](self.convs[0](x))
        out=self.relu(out*bn1_wb[0].view(1,-1,1,1)+bn1_wb[1].view(1,-1,1,1))
        out=self.batchnorms[1](self.convs[1](out))
        out=self.relu(out*bn2_wb[0].view(1,-1,1,1)+bn2_wb[1].view(1,-1,1,1))
        out = out + residual
        if self.affine_type is 'gru':
            (new_bn1_affines, _) =self.affine_a(bn1_affines.view(1,1,-1))
            (new_bn2_affines, _) = self.affine_b(bn2_affines.view(1,1,-1))
            new_bn1_affines = new_bn1_affines.squeeze()
            new_bn2_affines = new_bn2_affines.squeeze()
        else:
            new_bn1_affines = self.affine_a(bn1_affines)
            new_bn2_affines = self.affine_b(bn2_affines)
            self.alpha = nn.Parameter(new_bn1_affines)
            self.beta = nn.Parameter(new_bn2_affines)
        return (out, new_bn1_affines, new_bn2_affines)
    
class AffineRecnet(nn.Module):
    def __init__(self, block, layers, affine_type, num_classes=100):
        super(AffineRecnet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.sizes = [16,32,64]
        self.alphas = nn.ParameterList([nn.Parameter(torch.FloatTensor(2*sz)) for sz in self.sizes])
        self.betas = nn.ParameterList([nn.Parameter(torch.FloatTensor(2*sz)) for sz in self.sizes])
        self.affine_type=affine_type
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
        
        for i in range(len(self.sizes)):
            (alpha_w, alpha_b)=self.alphas[i].split(self.sizes[i])
            (beta_w, beta_b)=self.betas[i].split(self.sizes[i])
            alpha_w.fill_(1)
            alpha_b.fill_(0)
            beta_w.fill_(1)
            beta_b.fill_(0)
            self.alphas[i] = nn.Parameter(torch.cat((alpha_w, alpha_b)))
            self.betas[i] = nn.Parameter(torch.cat((beta_w, beta_b)))
        
    def get_affine_parameters(self):
        for (name, param) in self.named_parameters():
            if 'linear' in name or 'alphas' in name or 'betas' in name:
                yield param
    def get_all_but_affine_params(self):
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
        layers.append(block(self.inplanes, planes, self.affine_type, stride, downsample))
        self.inplanes = planes
        parent=block(self.inplanes, planes, self.affine_type, downsample=None)
        layers.append(parent)
        
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, self.affine_type, parent=parent))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        (x, _, _) = self.layer1((x, self.alphas[0], self.betas[0]))
        (x, _, _) = self.layer2((x, self.alphas[1], self.betas[1]))
        (x, _, _) = self.layer3((x, self.alphas[2], self.betas[2]))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    

    
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
        for i in range(1, blocks-1):
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

def recnet_postrelu(**kwargs):
    model = RecNet(RecNetBlock_postrelu, [9, 9, 9], **kwargs)
    return model

def recnet_affine(**kwargs):
    model = AffineRecnet(AffineRecnetBlock, [9,9,9], 'linear', **kwargs)
    return model

def recnet_affine_gru(**kwargs):
    model = AffineRecnet(AffineRecnetBlock, [9,9,9], 'gru', **kwargs)
    return model


def recnet_affine_modular(**kwargs):
    model = ModularAffineRecnet([9,9,9], **kwargs)
    return model
if __name__ is '__main__':
    net = preact_resnet110_cifar()
    y = net(torch.autograd.Variable(torch.randn(1, 3, 32, 32)))
    print(net)
    print(y.size())

