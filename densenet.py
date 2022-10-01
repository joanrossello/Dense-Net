import torch
import torch.nn as nn
import torch.nn.functional as F
from dense_block import dense_block

# Note: in our DenseNet3 class we do not implement bottleneck layers because I don't think it is needed,
# since the performance of DenseNet3 is already better than the perdormace of the cnn used in the img_cls tutorial.

class DenseBlock(nn.Module):
    def __init__(self, n_in, k):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(n_in) # BN of dense layer 1 in dense block
        self.conv1 = nn.Conv2d(n_in, k, 3, 1, 1) # (3x3 conv. stride 1, padding 1, dense layer 1)
        self.norm2 = nn.BatchNorm2d(n_in + k) # BN of dense layer 2 in dense block
        self.conv2 = nn.Conv2d(n_in + k, k, 3, 1, 1) # (3x3 conv. stride 1, padding 1, dense layer 2)
        self.norm3 = nn.BatchNorm2d(n_in + 2*k) # BN of dense layer 3 in dense block
        self.conv3 = nn.Conv2d(n_in + 2*k, k, 3, 1, 1) # (3x3 conv. stride 1, padding 1, dense layer 3)
        self.norm4 = nn.BatchNorm2d(n_in + 3*k) # BN of dense layer 4 in dense block
        self.conv4 = nn.Conv2d(n_in + 3*k, k, 3, 1, 1) # (3x3 conv. stride 1, padding 1, dense layer 4)
        
    def forward(self, x):
        x = dense_block(x, self.norm1, self.conv1, self.norm2, self.conv2, self.norm3, self.conv3, self.norm4, self.conv4)
        return x

class DenseNet3(nn.Module): # note that this is specific for input images [3x32x32] and dense blocks of 4 layers
    def __init__(self, n_in, k):
        super().__init__()
        self.conv0 = nn.Conv2d(3, n_in, 1) # (1x1 conv. to change depth to chose value 6)
        self.denseblock1 = DenseBlock(n_in, k) # dense block 1
        self.norm1 = nn.BatchNorm2d(n_in + 4*k) # BN of transitional layer 1
        self.conv1 = nn.Conv2d(n_in + 4*k, n_in + 2*k, 1) # (1x1 conv. to reduce the depth of the feature space)
                                                          # this reduction has been chosen arbitrarily to a number that 
                                                          # seems to yield good results for the network
        self.denseblock2 = DenseBlock(n_in + 2*k, k) # dense block 2
        self.norm2 = nn.BatchNorm2d(n_in + 6*k) # BN of transitional layer 2
        self.conv2 = nn.Conv2d(n_in + 6*k, n_in + 3*k, 1) # (1x1 conv. to reduce the depth of the feature space)
                                                          # this reduction has been chosen arbitrarily to a number that 
                                                          # seems to yield good results for the network
        self.denseblock3 = DenseBlock(n_in + 3*k, k) # dense block 3
        self.norm3 = nn.BatchNorm2d(n_in + 7*k) # BN of transitional layer 1
        self.pool = nn.AvgPool2d(2, 2) # (2x2, stride 2, average pooling, to downsample feature map in transitional layer)
        self.lin1 = nn.Linear(4*4*(n_in + 7*k), int(4*4*(n_in + 7*k)/2))
        self.lin2 = nn.Linear(int(4*4*(n_in + 7*k)/2), int(4*4*(n_in + 7*k)/4))
        self.lin3 = nn.Linear(int(4*4*(n_in + 7*k)/4), 10)

    def forward(self, x):
        x = self.conv0(x) # initial convolution
        x = self.denseblock1(x) # dense block 1
        x = self.pool(self.conv1(self.norm1(x))) # transitional layer 1
        x = self.denseblock2(x) # dense block 2
        x = self.pool(self.conv2(self.norm2(x))) # transitional layer 2
        x = self.denseblock3(x) # dense block 3
        x = self.pool(self.norm3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x) # remember there is no need to apply SoftMax, because CrossEntropy loss function already does it
        return x

def print_net(n_in, k):
    print('                        *** NETWORK SUMMARY ***')
    print(' ')
    print('     Layer & Type     |        Input Shape         |        Output Shape')
    print('                      |                            | ')
    print('1)   Conv.2D          |      [-1, 3, 32, 32]       |       [-1, %d, 32, 32]' % n_in)
    print('                            * DENSE BLOCK 1 *')
    print('                              Dense Layer 1 ')
    print('2)   BatchNorm.2D     |      [-1, %d, 32, 32]      |       [-1, %d, 32, 32]' % (n_in, n_in))
    print('3)   ReLU             |      [-1, %d, 32, 32]      |       [-1, %d, 32, 32]' % (n_in, n_in))
    print('4)   Conv.2D          |      [-1, %d, 32, 32]      |        [-1, %d, 32, 32]' % (n_in, k))
    print('                              Dense Layer 2 ')
    print('5)   BatchNorm.2D     |      [-1, %d, 32, 32]      |       [-1, %d, 32, 32]' % (n_in+k, n_in+k))
    print('6)   ReLU             |      [-1, %d, 32, 32]      |       [-1, %d, 32, 32]' % (n_in+k, n_in+k))
    print('7)   Conv.2D          |      [-1, %d, 32, 32]      |        [-1, %d, 32, 32]' % (n_in+k, k))
    print('                              Dense Layer 3 ')
    print('8)   BatchNorm.2D     |      [-1, %d, 32, 32]      |       [-1, %d, 32, 32]' % (n_in+2*k, n_in+2*k))
    print('9)   ReLU             |      [-1, %d, 32, 32]      |       [-1, %d, 32, 32]' % (n_in+2*k, n_in+2*k))
    print('10)  Conv.2D          |      [-1, %d, 32, 32]      |        [-1, %d, 32, 32]' % (n_in+2*k, k))
    print('                              Dense Layer 4 ')
    print('11)  BatchNorm.2D     |      [-1, %d, 32, 32]      |       [-1, %d, 32, 32]' % (n_in+3*k, n_in+3*k))
    print('12)  ReLU             |      [-1, %d, 32, 32]      |       [-1, %d, 32, 32]' % (n_in+3*k, n_in+3*k))
    print('13)  Conv.2D          |      [-1, %d, 32, 32]      |        [-1, %d, 32, 32]' % (n_in+3*k, k))
    print('                         * TRANSITION LAYER 1 *')
    print('14)  BatchNorm.2D     |      [-1, %d, 32, 32]      |       [-1, %d, 32, 32]' % (n_in+4*k, n_in+4*k))
    print('15)  Conv.1D          |      [-1, %d, 32, 32]      |       [-1, %d, 32, 32]' % (n_in+4*k, n_in+2*k))
    print('16)  AvgPool.2D       |      [-1, %d, 32, 32]      |       [-1, %d, 16, 16]' % (n_in+2*k, n_in+2*k))
    print('                            * DENSE BLOCK 2 *')
    print('                              Dense Layer 1 ')
    print('17)  BatchNorm.2D     |      [-1, %d, 16, 16]      |       [-1, %d, 16, 16]' % (n_in+2*k, n_in+2*k))
    print('18)  ReLU             |      [-1, %d, 16, 16]      |       [-1, %d, 16, 16]' % (n_in+2*k, n_in+2*k))
    print('19)  Conv.2D          |      [-1, %d, 16, 16]      |        [-1, %d, 16, 16]' % (n_in+2*k, k))
    print('                              Dense Layer 2 ')
    print('20)  BatchNorm.2D     |      [-1, %d, 16, 16]      |       [-1, %d, 16, 16]' % (n_in+3*k, n_in+3*k))
    print('21)  ReLU             |      [-1, %d, 16, 16]      |       [-1, %d, 16, 16]' % (n_in+3*k, n_in+3*k))
    print('22)  Conv.2D          |      [-1, %d, 16, 16]      |        [-1, %d, 16, 16]' % (n_in+3*k, k))
    print('                              Dense Layer 3 ')
    print('23)  BatchNorm.2D     |      [-1, %d, 16, 16]      |       [-1, %d, 16, 16]' % (n_in+4*k, n_in+4*k))
    print('24)  ReLU             |      [-1, %d, 16, 16]      |       [-1, %d, 16, 16]' % (n_in+4*k, n_in+4*k))
    print('25)  Conv.2D          |      [-1, %d, 16, 16]      |        [-1, %d, 16, 16]' % (n_in+4*k, k))
    print('                              Dense Layer 4 ')
    print('26)  BatchNorm.2D     |      [-1, %d, 16, 16]      |       [-1, %d, 16, 16]' % (n_in+5*k, n_in+5*k))
    print('27)  ReLU             |      [-1, %d, 16, 16]      |       [-1, %d, 16, 16]' % (n_in+5*k, n_in+5*k))
    print('28)  Conv.2D          |      [-1, %d, 16, 16]      |        [-1, %d, 16, 16]' % (n_in+5*k, k))
    print('                         * TRANSITION LAYER 2 *')
    print('29)  BatchNorm.2D     |      [-1, %d, 16, 16]      |       [-1, %d, 16, 16]' % (n_in+6*k, n_in+6*k))
    print('30)  Conv.1D          |      [-1, %d, 16, 16]      |       [-1, %d, 16, 16]' % (n_in+6*k, n_in+3*k))
    print('31)  AvgPool.2D       |      [-1, %d, 16, 16]      |         [-1, %d, 8, 8]' % (n_in+3*k, n_in+3*k))
    print('                            * DENSE BLOCK 3 *')
    print('                              Dense Layer 1 ')
    print('32)  BatchNorm.2D     |        [-1, %d, 8, 8]      |         [-1, %d, 8, 8]' % (n_in+3*k, n_in+3*k))
    print('33)  ReLU             |        [-1, %d, 8, 8]      |         [-1, %d, 8, 8]' % (n_in+3*k, n_in+3*k))
    print('34)  Conv.2D          |        [-1, %d, 8, 8]      |          [-1, %d, 8, 8]' % (n_in+3*k, k))
    print('                              Dense Layer 2 ')
    print('35)  BatchNorm.2D     |        [-1, %d, 8, 8]      |         [-1, %d, 8, 8]' % (n_in+4*k, n_in+4*k))
    print('36)  ReLU             |        [-1, %d, 8, 8]      |         [-1, %d, 8, 8]' % (n_in+4*k, n_in+4*k))
    print('37)  Conv.2D          |        [-1, %d, 8, 8]      |          [-1, %d, 8, 8]' % (n_in+4*k, k))
    print('                              Dense Layer 3 ')
    print('38)  BatchNorm.2D     |        [-1, %d, 8, 8]      |         [-1, %d, 8, 8]' % (n_in+5*k, n_in+5*k))
    print('39)  ReLU             |        [-1, %d, 8, 8]      |         [-1, %d, 8, 8]' % (n_in+5*k, n_in+5*k))
    print('40)  Conv.2D          |        [-1, %d, 8, 8]      |          [-1, %d, 8, 8]' % (n_in+5*k, k))
    print('                              Dense Layer 4 ')
    print('41)  BatchNorm.2D     |        [-1, %d, 8, 8]      |         [-1, %d, 8, 8]' % (n_in+6*k, n_in+6*k))
    print('42)  ReLU             |        [-1, %d, 8, 8]      |         [-1, %d, 8, 8]' % (n_in+6*k, n_in+6*k))
    print('43)  Conv.2D          |        [-1, %d, 8, 8]      |          [-1, %d, 8, 8]' % (n_in+6*k, k))
    print('                         * TRANSITION LAYER 3 *')
    print('44)  BatchNorm.2D     |        [-1, %d, 8, 8]      |         [-1, %d, 8, 8]' % (n_in+7*k, n_in+7*k))
    print('45)  AvgPool.2D       |        [-1, %d, 8, 8]      |         [-1, %d, 4, 4]' % (n_in+7*k, n_in+7*k))
    print('46)  Linear           |             [-1, %d]      |              [-1, %d]' % (4*4*(n_in+7*k), 
                                                                                            int(4*4*(n_in + 7*k)/2)))
    print('47)  ReLU             |             [-1, %d]      |              [-1, %d]' % (int(4*4*(n_in + 7*k)/2), 
                                                                                            int(4*4*(n_in + 7*k)/2)))
    print('48)  Linear           |             [-1, %d]      |              [-1, %d]' % (int(4*4*(n_in + 7*k)/2), 
                                                                                            int(4*4*(n_in + 7*k)/4)))
    print('49)  ReLU             |             [-1, %d]      |              [-1, %d]' % (int(4*4*(n_in + 7*k)/4), 
                                                                                            int(4*4*(n_in + 7*k)/4)))
    print('50)  Linear           |             [-1, %d]      |              [-1, 10]' % int(4*4*(n_in + 7*k)/4))
