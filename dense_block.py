import torch
import torch.nn as nn
import torch.nn.functional as F

# dense_block function of 4 dense layers to be used in the DenseBlock class

def dense_block(x, N1, C1, N2, C2, N3, C3, N4, C4):

    ''' 
    Inputs:
    x: input data to the dense block
    N1: batch normalisation function of dense layer 1
    C1: convolution fucntion of dense layer 1
    N2: batch normalisation function of dense layer 2
    C2: convolution fucntion of dense layer 2
    N3: batch normalisation function of dense layer 3
    C3: convolution fucntion of dense layer 3
    N4: batch normalisation function of dense layer 4
    C4: convolution fucntion of dense layer 4

    Output:
    x: data 'x' after it has gone thorough the dense block of 4 dense layers

    '''

    x = torch.cat((x, C1(F.relu(N1(x)))), 1) # dense layer 1 + concatenate
    x = torch.cat((x, C2(F.relu(N2(x)))), 1) # dense layer 2 + concatenate
    x = torch.cat((x, C3(F.relu(N3(x)))), 1) # dense layer 3 + concatenate
    x = torch.cat((x, C4(F.relu(N4(x)))), 1) # dense layer 4 + concatenate

    return x