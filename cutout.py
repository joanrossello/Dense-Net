import torch
import numpy as np

def cutout(image, s):
    im_size_0 = image.shape[1] # images have shape (D,H,W) --> here we want H
    im_size_1 = image.shape[2] # images have shape (D,H,W) --> here we want W

    size = np.random.randint(low=0, high=s+1)
    origin = np.random.randint(low=0, high=4)
    # origin = 0 --> top left pixel
    # origin = 1 --> top right pixel
    # origin = 2 --> bottom left pixel
    # origin = 3 --> bottom right pixel
    mask = torch.zeros([image.shape[0],size,size])

    # x1 and x2 are the image pixel coordinates where we place the mask origin pixel
    if origin == 0:
        x1 = np.random.randint(low=0, high=im_size_0 - size + 1)
        x2 = np.random.randint(low=0, high=im_size_1 - size + 1)
    if origin == 1:
        x1 = np.random.randint(low=0, high=im_size_0 - size + 1)
        x2 = np.random.randint(low=size - 1, high=im_size_1)
    if origin == 2:
        x1 = np.random.randint(low=size - 1, high=im_size_0)
        x2 = np.random.randint(low=0, high=im_size_1 - size + 1)
    if origin == 3:
        x1 = np.random.randint(low=size - 1, high=im_size_0)
        x2 = np.random.randint(low=size - 1, high=im_size_1)


    if origin == 0: 
        image[:, x1:x1+size, x2:x2+size] = mask
    elif origin == 1:
        image[:, x1:x1+size, x2-size+1:x2+1] = mask
    elif origin == 2:
        image[:, x1-size+1:x1+1, x2:x2+size] = mask
    elif origin == 3:
        image[:, x1-size+1:x1+1, x2-size+1:x2+1] = mask

    return image