import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from cutout import cutout
from PIL import Image
from densenet import DenseNet3, print_net

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# TASK 2: REGULARISED DENSENET
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20 # mini-batch size
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # ----------------------------------------------------------------
    # CUTOUT IMAGES EXAMPLES
    # ----------------------------------------------------------------
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    im_cutout = images[:16,:,:,:] # only take 16 images

    tester = iter(im_cutout) # iterator

    # specify the masimum size of the cutout mask. 
    # We will set it to 16 for now --> since the images are 32x32, the maximum cutout will be half the size of the image
    s = 16

    for i in range(im_cutout.shape[0]):
        nextItem = next(tester) # iterable
        im_cutout[i,:,:,:] = cutout(nextItem, s)

    im = Image.fromarray((torch.cat(im_cutout.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("cutout.png")
    print('cutout.png saved.')

    # ----------------------------------------------------------------
    # TRAINING CODE
    # ----------------------------------------------------------------

    ## cnn

    n_in, k = 10, 6 # these hyperparameters are chosen arbitrarily
                    # I chose these values because they are similar to the ones used in the DenseNet paper,
                    # they are not too large, and the resultant network already performs better than the
                    # network used in the image classification tutorial.
    net = DenseNet3(n_in, k)
    print_net(n_in, k) # print the network architecture

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    n_epochs = 10
    n_steps = len(trainloader)

    ## train
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): # every mini-batch of 20 images each
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            for j in range(inputs.shape[0]):
                inputs[j,:,:,:] = cutout(inputs[j,:,:,:], s=16)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i+1) % 2000 == 0:    # print every 2000 mini-batches
                print (f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_steps}], Loss: {running_loss/2000:.4f}')
                running_loss = 0.0
        

        # ----------------------------------------------------------------
        # TESTING CODE
        # ----------------------------------------------------------------

        with torch.no_grad():
            # for the visualising results section, create a lists where you will store your images, labels and predictions
            # only do this at the last epoch when training is done
            if epoch == n_epochs-1:
                images_list = []
                labels_list = []
                pred_list = []
                iter = 0 # set iterator to 0, and stop when it reaches the number of images we want to visualise
                n_visualise = 36

            n_correct = 0
            n_samples = 0
            for images, labels in testloader: # we iterate over all the batches
                outputs = net(images) 
                # our output is a vector with a value for each class
                # the class with the highest value is the predicted class
                # max function returns (max value ,index of max value)
                _, predicted = torch.max(outputs, 1) # we are only interested in the index --> the class
                n_samples += labels.size(0) # sum total number of images in each batch, for every batch
                n_correct += (predicted == labels).sum().item() # total number of predictions in the batch that were 
                                                                # the same as the ground truth labels
                                                                # we sum for all the batches to get the grand total
                
                # The following is for visualising results later on:
                if epoch == n_epochs-1:
                    for i in range(images.shape[0]):
                        iter += 1
                        if iter <= n_visualise:
                            images_list.append(images[i,:,:,:])
                            labels_list.append(labels[i])
                            pred_list.append(predicted[i])

            # Calculate accuracy of the network:
            accuracy = (n_correct / n_samples) * 100 # we want to express it as a percentage (%)

            print(f'The accuracy of the network after {epoch+1} epochs is: {accuracy:.2f} %')

    print('Training done.')

    # save trained model
    torch.save(net.state_dict(), 'saved_DenseNet3_model.pt')
    print('DenseNet3 model saved.')

    
    # ----------------------------------------------------------------
    # VISUALISE RESULTS
    # ----------------------------------------------------------------
    
    # Prepare the images arrays to be used by Pillow
    for i in range(n_visualise):
        images_list[i] = (images_list[i].squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8')

    
    im = Image.fromarray(np.concatenate(images_list)) # concatenate images vertically
    im.save("result.png")
    print('result.png saved.')

    # Grond truth labels and predictions in order of appearance in the .png file from top to bottom:
    for i in range(n_visualise):
        print(f'Image {i+1} -- Ground truth: {classes[labels_list[i]]}, Prediction: {classes[pred_list[i]]}')

# Note: our DenseNet already performs better than the cnn used in the img_cls tutorial,
# where after 10 epochs, the accuracy of the network was around 60%.

# ----------------------------------------------------------------
# END OF TASK 2
# ----------------------------------------------------------------
