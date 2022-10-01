import torch 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import timeit

from network_pt import Net
from network_leaky_pt import NetLeaky

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# TASK 3: ABLATION USING CROSS-VALIDATION
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Choice of modification: 
# Difference between using ReLU and leaky ReLU (with a negative slope 
# alpha = 0.1), as activation functions throughout the network.

# Note that in this task we refer to "model 1" the model using ReLU
# and "model 2" the model using Leaky ReLU
# ----------------------------------------------------------------------

if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Development set (50000 images) --> we use this for cross-validation
    devset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Holdout test set (10000 images)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # ----------------------------------------------------------------------
    # IMPLEMENT 3-FOLD CROSS-VALIDATION USING DEVELOPMENT SET
    # ----------------------------------------------------------------------

    k_folds = 3 # number of subsamples

    # We train and validate both models at the same time, with the same loops

    rem = len(devset) % k_folds

    div = int((len(devset) - rem)/k_folds) * np.ones([k_folds], dtype=int)
    div[:rem] += 1

    folds = torch.utils.data.random_split(devset, div) # list of subsets for cross-validation

    # Print summary of each fold:
    print('-------------------------------------------------------------------')
    print('*** FOLDS SPLIT DATA SUMMARY ***')
    print(' ')
    for k in range(k_folds):
        print(f'** Fold nº {k+1} **')
        print('1) Number of samples in the fold:')
        print(f'--> Fold {k+1} has {len(folds[k])} samples')
        print(f'2) Occurrence percentage per class in fold {k+1}:')
        f_labels = []
        for l in folds[k]:
            f_labels.append(l[1])
        f_labels = np.array(f_labels)
        max_perc = 0
        min_perc = 100
        for c in range(len(classes)):
            perc = np.count_nonzero(f_labels == c) / len(folds[k]) * 100
            print(f'--> {classes[c]}:  {perc:.2f} %')
            if perc > max_perc:
                max_perc = perc
                max_class = c
            if perc < min_perc:
                min_perc = perc
                min_class = c
        print(f'3) Class with the most occurreces in fold {k+1}: {classes[max_class]}, {max_perc:.2f} %')
        print(f'4) Class with the least occurreces in fold {k+1}: {classes[min_class]}, {min_perc:.2f} %')
        print(' ')


    # START CROSS-VALIDATION LOOP:

    print('-------------------------------------------------------------------')
    print('*** START OF CROSS-VALIDATION LOOPS ***')
    print('-------------------------------------------------------------------')

    # Note that this for loop works for 3-folds or more. 
    # If it was 2-folds, we would not need to concatenate, and the ConcatenateDataset function would be applied incorrectly.

    # Initialise lists where you will store monitoring metrics for each fold
    L1 = [] # Loss ReLU model at validation
    L2 = [] # Loss Leaky ReLU model at validation
    ACC1 = [] # Accuracy ReLU model at validation
    ACC2 = [] # Accuracy Leaky ReLU model at validation
    TT1 = [] # Training speed ReLU model
    TT2 = [] # Training speed Leaky ReLU model
    VT1 = [] # Validating speed ReLU model
    VT2 = [] # Validating speed Leaky ReLU model
    E1 = [] # Entropy ReLU model network at validation
    E2 = [] # Entropy Leaky ReLU model network at validation

    for k in range(k_folds):

        valset = folds[k]
        if k == 0:
            trainset = torch.utils.data.ConcatDataset(folds[k+1:])
        elif k == k_folds-1:
            trainset = torch.utils.data.ConcatDataset(folds[:k_folds-1])
        else:
            trainset = torch.utils.data.ConcatDataset(folds[:k] + folds[k+1:])


        # Print summary of the train set and the validation set:
        print(f'** CROSS-VALIDATION LOOP {k+1} **')
        print(' ')
        print('* TRAIN SET SUMMARY *')
        print(f'1) Number of samples: {len(trainset)}')
        print('2) Occurrence percentage per class')
        f_labels = []
        for l in trainset:
            f_labels.append(l[1])
        f_labels = np.array(f_labels)
        max_perc = 0
        min_perc = 100
        for c in range(len(classes)):
            perc = np.count_nonzero(f_labels == c) / len(trainset) * 100
            print(f'        - {classes[c]}:  {perc:.2f} %')
            if perc > max_perc:
                max_perc = perc
                max_class = c
            if perc < min_perc:
                min_perc = perc
                min_class = c
        print(f'3) Class with the most occurreces: {classes[max_class]}, {max_perc:.2f} %')
        print(f'4) Class with the least occurreces: {classes[min_class]}, {min_perc:.2f} %')
        print(' ')
        print('* VALIDATION SET SUMMARY *')
        print(f'1) Number of samples: {len(valset)}')
        print('2) Occurrence percentage per class')
        f_labels = []
        for l in valset:
            f_labels.append(l[1])
        f_labels = np.array(f_labels)
        max_perc = 0
        min_perc = 100
        for c in range(len(classes)):
            perc = np.count_nonzero(f_labels == c) / len(valset) * 100
            print(f'        - {classes[c]}:  {perc:.2f} %')
            if perc > max_perc:
                max_perc = perc
                max_class = c
            if perc < min_perc:
                min_perc = perc
                min_class = c
        print(f'3) Class with the most occurreces: {classes[max_class]}, {max_perc:.2f} %')
        print(f'4) Class with the least occurreces: {classes[min_class]}, {min_perc:.2f} %')
        print(' ')


        # Specidy batch sizes for train set and validation set
        batch_size = 20

        # Data loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

        ## cnn
        net = Net() # model 1
        net_leaky = NetLeaky() # model 2
        # this also initialises weights and overwrites the previously trained ones in the previous fold


        ## loss and optimiser
        criterion = torch.nn.CrossEntropyLoss()
        optimizer1 = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # model 1
        optimizer2 = optim.SGD(net_leaky.parameters(), lr=0.001, momentum=0.9) # model 2


        n_epochs = 5
        n_steps = len(trainloader)

        train_timer1 = 0.0 # Timer
        train_timer2 = 0.0 # Timer

        ## train
        for epoch in range(n_epochs):  # loop over the dataset multiple times

            running_loss1 = 0.0
            running_loss2 = 0.0

            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                # Model 1
                start_train1 = timeit.default_timer() # Timer
                optimizer1.zero_grad()
                outputs1 = net(inputs)
                loss1 = criterion(outputs1, labels)
                loss1.backward()
                optimizer1.step()
                end_train1 = timeit.default_timer() # Timer
                train_timer1 += end_train1 - start_train1 # Timer

                # Model 2
                start_train2 = timeit.default_timer() # Timer
                optimizer2.zero_grad()
                outputs2 = net_leaky(inputs)
                loss2 = criterion(outputs2, labels)
                loss2.backward()
                optimizer2.step()
                end_train2 = timeit.default_timer() # Timer
                train_timer2 += end_train2 - start_train2 # Timer

                # metrics
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                if (i+1) % 1000 == 0: # print every 1000 steps
                    print (f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_steps}], Loss --> ReLU: {running_loss1/1000:.4f}, '
                                                                                f'Leaky ReLU: {running_loss2/1000:.4f}')
                    running_loss1 = 0.0
                    running_loss2 = 0.0
                
        print(' ')
        print(f'Training time --> ReLU model: {train_timer1:.4f} s, Leaky ReLU model: {train_timer2:.4f} s')
        TT1.append(train_timer1), TT2.append(train_timer2)
        print(' ')
        print('Training done.')
        print(' ')


        with torch.no_grad():
            val_timer1 = 0.0 # Timer
            val_timer2 = 0.0 # Timer
            val_loss1 = 0.0 # Loss
            val_loss2 = 0.0 # Loss
            entropy1 = [] # Entropy
            entropy2 = [] # Entropy
            n_correct1 = 0
            n_correct2 = 0
            n_samples = 0
            n_class_correct1 = [0 for i in range(10)]
            n_class_correct2 = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            for inputs, labels in valloader: # we iterate over all the batches
                # Model 1
                start_val1 = timeit.default_timer() # Timer
                outputs1 = net(inputs)
                loss1 = criterion(outputs1, labels)
                end_val1 = timeit.default_timer() # Timer
                val_timer1 += end_val1 - start_val1 # Timer

                # Model 2
                start_val2 = timeit.default_timer() # Timer
                outputs2 = net_leaky(inputs)
                loss2 = criterion(outputs2, labels)
                end_val2 = timeit.default_timer() # Timer
                val_timer2 += end_val2 - start_val2 # Timer

                # Losses
                val_loss1 += loss1.item()
                val_loss2 += loss2.item()

                # Entropies
                e1 = F.softmax(outputs1, dim=1)
                e2 = F.softmax(outputs2, dim=1)
                entropy1.append(torch.mean(- torch.sum(e1*np.log(e1), 1)))
                entropy2.append(torch.mean(- torch.sum(e2*np.log(e2), 1)))

                # Calculate accuracy of the network
                _, predicted1 = torch.max(outputs1, 1)
                _, predicted2 = torch.max(outputs2, 1)
                n_samples += labels.size(0)
                n_correct1 += (predicted1 == labels).sum().item()
                n_correct2 += (predicted2 == labels).sum().item()
                
                # Calculate accuracy of each class
                for i in range(len(labels)):
                    label = labels[i]
                    pred1 = predicted1[i]
                    pred2 = predicted2[i]
                    if (label == pred1):
                        n_class_correct1[label] += 1
                    if (label == pred2):
                        n_class_correct2[label] += 1
                    n_class_samples[label] += 1

            print(f'Loss at validation --> RelU: {(val_loss1/len(valloader)):.4f}, '
                                        f'Leaky ReLU: {(val_loss2/len(valloader)):.4f}')
            L1.append(val_loss1/len(valloader)), L2.append(val_loss2/len(valloader))
            print(' ')

            acc1 = 100.0 * n_correct1 / n_samples
            acc2 = 100.0 * n_correct2 / n_samples

            print(f'Accuracy of network at validation --> ReLU: {acc1:.2f} %, Leaky ReLU: {acc2:.2f} %')
            ACC1.append(acc1), ACC2.append(acc2)
            print('Accuracy of each class:')

            for i in range(10):
                acc1 = 100.0 * n_class_correct1[i] / n_class_samples[i]
                acc2 = 100.0 * n_class_correct2[i] / n_class_samples[i]
                print(f'        - {classes[i]} --> ReLU: {acc1:.2f} %, Leaky ReLU: {acc2:.2f} %')

            print(' ')
            print(f'Validation time --> ReLU: {val_timer1:.4f} s, Leaky ReLU: {val_timer2:.4f} s')
            VT1.append(val_timer1), VT2.append(val_timer2)
            print(' ')
            print(f'Entropy of fold at validation --> ReLU: {np.mean(entropy1):.4f}, Leaky ReLU: {np.mean(entropy2):.4f}')
            E1.append(np.mean(entropy1)), E2.append(np.mean(entropy2))
            print('-------------------------------------------------------------------')
            print(' ')

    # SUMMARY OF AVERAGE RESULTS ACROSS FOLDS FOR EACH MODEL
    print('** SUMMARY OF AVERAGE CROSS-VALIDATION RESULTS PER MODEL **')
    print('Metric        |     ReLU model      |     Leaky ReLU model')
    print('Val. Loss     |        %.4f       |           %.4f' % (np.mean(L1), np.mean(L2)))
    print('Val. Acc.     |        %.2f %%      |           %.2f %%' % (np.mean(ACC1), np.mean(ACC2)))
    print('Train. Speed  |        %.4f s    |           %.4f s' % (np.mean(TT1), np.mean(TT2)))
    print('Val. Speed    |        %.4f s     |           %.4f s' % (np.mean(VT1), np.mean(VT2)))
    print('Val. Entropy  |        %.4f       |           %.4f' % (np.mean(E1), np.mean(E2)))
            

    # Comment on results:
    # Both models perform very similarly, with the only difference that the ReLU model has lower
    # computational speed for training and validation than the Leaky ReLU model.
    # Note that the entropy is the additional metric we have defined to monitor. It indicates how much
    # uncertainty there is in our network. If our network predicted every image with 100% certainty
    # i.e. the soft max function would return 1 for the predicted class, and 0 for the rest of the classes,
    # then the entropy would be minimised (= 0). Low entropy means good performance.


    # ----------------------------------------------------------------------
    # TRAIN AND TEST RELU AND LEAKY RELU MODELS USING ENTIRE DEVELOPMENT SET
    # ----------------------------------------------------------------------

    print(' ')
    print('-------------------------------------------------------------------')
    print('*** RESULTS USING ENTIRE DEVELOPMENT SET AND HOLDOUT SET ***')
    print('-------------------------------------------------------------------')
    print('** TRAINING **')
    print(' ')

    # Note that in this part we are not asked to monitor speed during training and testing.

    # Specidy batch sizes for train set and validation set
    batch_size = 20

    # Data loaders
    devloader = torch.utils.data.DataLoader(devset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    ## cnn
    net = Net() # model 1
    net_leaky = NetLeaky() # model 2

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # model 1
    optimizer2 = optim.SGD(net_leaky.parameters(), lr=0.001, momentum=0.9) # model 2

    n_epochs = 5
    n_steps = len(devloader)

    ## train
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss1 = 0.0
        running_loss2 = 0.0
        
        for i, data in enumerate(devloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            # forward + backward + optimize
            outputs1 = net(inputs)
            outputs2 = net_leaky(inputs)
            loss1 = criterion(outputs1, labels)
            loss2 = criterion(outputs2, labels)
            loss1.backward()
            loss2.backward()
            optimizer1.step()
            optimizer2.step()

            running_loss1 += loss1.item()
            running_loss2 += loss2.item()

            if (i+1) % 2000 == 0: # print every 2000 steps
                print (f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_steps}], Loss --> ReLU: {running_loss1/2000:.4f}, '
                                                                            f'Leaky ReLU: {running_loss2/2000:.4f}')
                running_loss1 = 0.0
                running_loss2 = 0.0
            
    print(' ')
    print('Training done.')

    # save trained models
    torch.save(net.state_dict(), 'saved_ReLU_model.pt')
    print('ReLU model saved.')
    torch.save(net_leaky.state_dict(), 'saved_LeakyReLU_model.pt')
    print('Leaky ReLU model saved.')
    print(' ')


    print('** TESTING **')
    print(' ')

    with torch.no_grad():
        test_loss1 = 0.0 # Loss
        test_loss2 = 0.0 # Loss
        entropy1 = [] # Entropy
        entropy2 = [] # Entropy
        n_correct1 = 0
        n_correct2 = 0
        n_samples = 0
        n_class_correct1 = [0 for i in range(10)]
        n_class_correct2 = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for inputs, labels in testloader: # we iterate over all the batches
            outputs1 = net(inputs)
            outputs2 = net_leaky(inputs)
            loss1 = criterion(outputs1, labels)
            loss2 = criterion(outputs2, labels)

            # Losses
            test_loss1 += loss1.item()
            test_loss2 += loss2.item()

            # Entropies
            e1 = F.softmax(outputs1, dim=1)
            e2 = F.softmax(outputs2, dim=1)
            entropy1.append(torch.mean(- torch.sum(e1*np.log(e1), 1)))
            entropy2.append(torch.mean(- torch.sum(e2*np.log(e2), 1)))

            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)
            n_samples += labels.size(0)
            n_correct1 += (predicted1 == labels).sum().item()
            n_correct2 += (predicted2 == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i]
                pred1 = predicted1[i]
                pred2 = predicted2[i]
                if (label == pred1):
                    n_class_correct1[label] += 1
                if (label == pred2):
                    n_class_correct2[label] += 1
                n_class_samples[label] += 1

        print(f'Loss --> ReLU: {(test_loss1/len(testloader)):.4f}, Leaky ReLU: {(test_loss2/len(testloader)):.4f}')
        print(' ')

        acc1 = 100.0 * n_correct1 / n_samples
        acc2 = 100.0 * n_correct2 / n_samples
        print(f'Accuracy of network --> ReLU: {acc1} %, Leaky ReLU: {acc2} %')
        print('Accuracy of each class:')

        for i in range(10):
            acc1 = 100.0 * n_class_correct1[i] / n_class_samples[i]
            acc2 = 100.0 * n_class_correct2[i] / n_class_samples[i]
            print(f'        - {classes[i]} --> ReLU: {acc1} %, Leaky ReLU: {acc2} %')
        
        print(' ')
        print(f'Entropy of network --> ReLU: {np.mean(entropy1):.4f}, Leaky ReLU: {np.mean(entropy2):.4f}')
        print(' ')


    print('-------------------------------------------------------------------')
    print('*** COMPARISON OF TESTING AND CROSS-VALIDATION RESULTS ***')
    print('-------------------------------------------------------------------')
    print('                  ** ReLU model **')
    print('Metric        |   Cross-Validation   |   Holdout Testing')
    print('Loss          |        %.4f        |           %.4f' % (np.mean(L1), test_loss1/len(testloader)))
    print('Accuracy      |        %.2f %%       |           %.2f %%' % (np.mean(ACC1), acc1))
    print('Entropy       |        %.4f        |           %.4f ' % (np.mean(E1), np.mean(entropy1)))
    print(' ')
    print('               ** Leaky ReLU model **')
    print('Metric        |   Cross-Validation   |   Holdout Testing')
    print('Loss          |        %.4f        |           %.4f' % (np.mean(L2), test_loss2/len(testloader)))
    print('Accuracy      |        %.2f %%       |           %.2f %%' % (np.mean(ACC2), acc2))
    print('Entropy       |        %.4f        |           %.4f ' % (np.mean(E2), np.mean(entropy2)))


    # Comment on results: 
    # Using the entire development set for training and the holdout test set for testing,
    # both models yield similar performance results, as we expected from cross-validation.
    # If we increase the number of epochs, these results will become even more similar.
    # Testing results are better because training was done using the entire development set.

# ----------------------------------------------------------------
# END OF TASK 3
# ----------------------------------------------------------------
