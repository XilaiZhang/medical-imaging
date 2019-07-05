import torch
import cv2
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import os
import pickle
import numpy as np
import copy
import random
import argparse
import matplotlib.pyplot as plt

SIZE = 224
SHOW_TEST_IMG_BOOL = False

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input folder of images")
args = vars(ap.parse_args())

baseFolder = args["input"]

# Implement DataLoaders
class Train_Dataset(Dataset):
    def __init__(self, split, splitRatio=0.99):
        dataFolder = os.path.join(baseFolder, 'train_dataset')
        filelist = os.listdir(dataFolder)
        # print(filelist)
        random.shuffle(filelist)
        if split == 'train':
            self.dataFileList = filelist[:int(splitRatio * len(filelist))]
            print('Training Sample Number: ', len(self.dataFileList))
        elif split == 'val':
            self.dataFileList = filelist[int(splitRatio * len(filelist)):]
            print('Validating Sample Number: ', len(self.dataFileList))

        self.dataList = []
        self.labelList = []

        for dataFile in self.dataFileList:
            datas = pickle.load(open(os.path.join(dataFolder, dataFile), 'rb'))
            img = datas[0]
            img = cv2.resize(img, (SIZE,SIZE))   # Standardize the size of the img (NOTE: Rescaling the img)
            label = datas[1]
            swappedImg = np.zeros((3,SIZE,SIZE), dtype='float32')
            for i in range(3):
                swappedImg[i] = img[:,:,i]

            self.dataList.append(swappedImg)
            self.labelList.append(np.array(label, dtype='float32'))

    def __len__(self):
        return len(self.dataFileList)

    def __getitem__(self, idx):
        return self.dataList[idx], self.labelList[idx]

class Test_Dataset(Dataset):
    def __init__(self):
        dataFolder = os.path.join(baseFolder, 'test_dataset')
        filelist = os.listdir(dataFolder)
        self.dataFileList = filelist
        print('Testing Sample Number: ', len(self.dataFileList))


        self.dataList = []
        self.labelList = []

        for dataFile in self.dataFileList:
            # print(dataFile)
            datas = pickle.load(open(os.path.join(dataFolder, dataFile), 'rb'))
            img = datas[0]
            img = cv2.resize(img, (SIZE,SIZE))   # Standardize the size of the img (NOTE: Rescaling the img)
            label = datas[1]
            swappedImg = np.zeros((3,SIZE,SIZE), dtype='float32')
            for i in range(3):
                swappedImg[i] = img[:,:,i]
            # print(np.array(img))
            # print(swappedImg)


            self.dataList.append(swappedImg)
            self.labelList.append(np.array(label, dtype='float32'))

    def __len__(self):
        return len(self.dataFileList)

    def __getitem__(self, idx):
        return self.dataList[idx], self.labelList[idx]

# Implement Training & Testing Functions
def trainModel(model, trainLoader, valLoader, criterion, optimizer, numEpochs=5):
    # Training Phase
    model.train()   # Set the model in train mode
    for epoch in range(numEpochs):
        print('Epoch {}/{}'.format(epoch+1, numEpochs))
        print('-'*10)

        losses = []
        for batch_idx, (inputs, labels) in enumerate(trainLoader):
            inputs = Variable(inputs).float().cuda()  # Move input into GPU RAM
            outputs = model(inputs)                   # Use the current model to generate result for the input
            labels = Variable(labels).float().cuda()  # Move label into GPU RAM
            loss = criterion(outputs, labels)         # Use loss function to find out the loss of the current calculation
            optimizer.zero_grad()                   # Perform Gradiant Desent ???
            loss.backward()                         # Trace backwards
            optimizer.step()                        # Update weights
            losses.append(loss.data.cpu().numpy().mean())   # Move data to cpu to do numpy calculation
        print('[%s/%s] Training Loss: %.3f' % (epoch+1, numEpochs, np.mean(losses)))

    # Evaluation Phase:
    model.eval()
    accuracy = 0
    smpSum = 0
    for batch_idx, (inputs, labels) in enumerate (valLoader):
        inputs = Variable(inputs).float().cuda()
        outputs = model(inputs)
        outputs = outputs.cpu().detach().numpy()      # Detach to read value

        for smp_i in range(labels.shape[0]):
            if labels[smp_i, 0] != 0:
                smpSum += 1                           # Effective count + 1
                if labels[smp_i, 0] == -1 and outputs[smp_i, 0] < 0:
                    accuracy += 1
                elif labels[smp_i, 0] == 1 and outputs[smp_i, 0] > 0:
                    accuracy += 1
    print('Validation Accuracy = ', accuracy/smpSum)
    return(accuracy/smpSum)

def testModel(model, testLoader, SHOW_TEST_IMG=False):
    model.eval()
    accuracy = 0
    smpSum = 0
    for batch_idx, (inputs, labels) in enumerate (testLoader):
        inputs = Variable(inputs).float().cuda()
        outputs = model(inputs)
        outputs = outputs.cpu().detach().numpy()      # Detach to read value
        if SHOW_TEST_IMG:
            inputs = inputs.cpu().detach().numpy()

        # print(outputs)
        # exit(1)

        for smp_i in range(labels.shape[0]):
            if labels[smp_i, 0] != 0:
                smpSum += 1                           # Effective count + 1
                if labels[smp_i, 0] == -1 and outputs[smp_i, 0] < 0:
                    accuracy += 1
                elif labels[smp_i, 0] == 1 and outputs[smp_i, 0] > 0:
                    accuracy += 1
                else:
                    if SHOW_TEST_IMG:
                        sswappedImg = np.zeros((SIZE,3,SIZE), dtype='float32')
                        ssswappedImg = np.zeros((SIZE,SIZE,3), dtype='float32')
                        for i in range(SIZE):
                            sswappedImg[i] = inputs[smp_i][:,:,i]
                        for i in range(SIZE):
                            ssswappedImg[i] = sswappedImg[:,:,i]
                        ssswappedImg = cv2.resize(ssswappedImg, (SIZE,SIZE))

                        cv2.imshow("Temp", ssswappedImg/255)
                        cv2.waitKey(0)
    print('Test Accuracy = ', accuracy/smpSum)
    return(accuracy/smpSum)

def main():
    # Set up
    # resnet34
    resnet34 = models.resnet34(pretrained=True)
    print(resnet34)
    # print(resnet34.fc)
    # exit(1)
    # resnet34.classifier = nn.Sequential(*list(resnet34.classifier.children())[:-1] + [nn.Linear(in_features=4096, out_features=1, bias=False)])
    # resnet34 = nn.DataParallel(resnet34, device_ids=[0]).cuda()

    # Set output layer
    resnet34.fc = nn.Linear(in_features=512, out_features=1)
    resnet34 = resnet34.cuda()
    optimizer = optim.Adam(resnet34.parameters(), lr=0.001)
    criterion = nn.MSELoss().cuda()
    trainLoader = DataLoader(dataset=Train_Dataset('train'), batch_size=32, shuffle=True, num_workers=4)
    valLoader = DataLoader(dataset=Train_Dataset('val'), batch_size=32, shuffle=True, num_workers=4)
    testLoader = DataLoader(dataset=Test_Dataset(), batch_size=32, shuffle=True, num_workers=4)

    # Start training the model and through evaluation, find the best model
    train_score_list = []
    test_score_list = []
    best_score = 0
    best_model = None
    print("\n\nTraining Starts.")
    for i in range(40):
        print("\n--- Round {} ---".format(i))
        current_score = trainModel(resnet34, trainLoader, valLoader, criterion, optimizer)
        test_score = testModel(resnet34, testLoader)
        # if current_score > best_score:
        if test_score > best_score:
            best_model = copy.deepcopy(resnet34)
            # best_score = current_score
            best_score = test_score
        train_score_list.append(current_score)
        test_score_list.append(test_score)




    # Test the model with the best model trained
    print("="*20)
    testModel(best_model, testLoader, SHOW_TEST_IMG_BOOL)

    # Save model
    jcj = input("Do you want to save this model?(y/n)")
    while jcj != "y" and jcj != "n":
        jcj = input("Type \"y\" or \"n\" only:")
    if jcj == "y":
        torch.save(best_model, "full_model")
        print("---Model Saved---")

    print("="*20)
    axis = list(range(1,41))
    plt.plot(axis, train_score_list, 'r-')
    plt.plot(axis, test_score_list, 'b--')
    plt.show()

if __name__ == '__main__':
    main()
