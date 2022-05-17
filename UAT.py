import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Function import function
from WideModel import *
from DeepModel import *
import matplotlib.pyplot as plt


#Initialization for which Device to use GPU or CPU
def initialization():
    flagCuda = torch.cuda.is_available()
    if not flagCuda:
        print('Using CPU')
    else:
        print('Using GPU')
    return flagCuda


#Creating the Dataset for sin() function
def creattingDataset():
    #x = np.linspace(-30,30,100)
    x = np.linspace(-np.pi, np.pi, 201)
    y = function(x)
    return x,y

#Training the Wide Network
def wideTest(flagCuda, hiddenNeurons, learningRate, stepSize, epochs, x, y, inputs, labels):
    wideModel = WideNet(hiddenNeurons)
    print(wideModel)

    if flagCuda:
        wideModel.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(wideModel.parameters(), lr=learningRate)

    #Learning Rate Decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stepSize, gamma=0.1)

    for epoch in range(1, epochs):
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = wideModel(inputs.float())
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        #scheduler.step()
        print("EPOCHS: {}".format(epoch))

    with torch.no_grad():
        test_inputs = torch.tensor(x).view(len(x),-1).float()
        y_hat = wideModel(test_inputs)
        y_hat = y_hat.detach().numpy()

    # ******************************************************  #

    ### Plot results: Actual vs Model Prediction ***********  #
    plt.scatter(x,y,label='Actual Function')
    plt.scatter(x,y_hat,label="Predicted Function")
    plt.title(f'WIDE MODEL----Number of neurons: {hiddenNeurons}')
    plt.xlabel('Input Variable (x)')
    plt.ylabel('Output Variable (y)')
    plt.legend()
    plt.savefig('images/WideNet.png')
    #plt.show()



#Training the Deep Network
def deepTest(flagCuda, hiddenNeurons, learningRate, stepSize, epochs, x, y, inputs, labels):
    deepModel = DeepNet(hiddenNeurons)
    print(deepModel)

    if flagCuda:
        deepModel.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(deepModel.parameters(), lr=learningRate)

    #Learning Rate Decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stepSize, gamma=0.1)

    for epoch in range(1, epochs):
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = deepModel(inputs.float())
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        #scheduler.step()
        print("EPOCHS: {}".format(epoch))

    with torch.no_grad():
        test_inputs = torch.tensor(x).view(len(x),-1).float()
        y_hat = deepModel(test_inputs)
        y_hat = y_hat.detach().numpy()

    # ******************************************************  #

    ### Plot results: Actual vs Model Prediction ***********  #
    plt.scatter(x,y,label='Actual Function')
    plt.scatter(x,y_hat,label="Predicted Function")
    plt.title(f'DEEP MODEL----Number of neurons: {hiddenNeurons}')
    plt.xlabel('Input Variable (x)')
    plt.ylabel('Output Variable (y)')
    plt.legend()
    plt.savefig('images/DeepNet.png')
    #plt.show()


#Calling the sub-routines
def main():

    flagCuda = initialization()
    x, y = creattingDataset()
    inputs = torch.tensor(x).view(-1,1)
    labels = torch.tensor(y).view(-1,1)

    #Adjusting the hyperparameters
    hiddenNeurons = 5
    learningRate = 0.001
    stepSize = 1000
    epochs = 1500

    wide = 0
    if wide:
        wideTest(flagCuda, hiddenNeurons, learningRate, stepSize, epochs, x, y, inputs, labels)
    else:
        deepTest(flagCuda, hiddenNeurons, learningRate, stepSize, epochs, x, y, inputs, labels)


#Calling the main function
if __name__=="__main__":
    main()