# Import libraries
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from model import Net
from data import mnist

# Set up GPU acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using GPU!')
else:
    device = torch.device("cpu")
    print('Using CPU!')

# Set up the path
os.chdir('./corruptmnist')

# Set up argpase to get experiental setup
def get_setup():
    """ Description: Gets hyper-parameters from command line and stores all hyper-parameters in an object
        Return: config-object """

    # Get varying hyper-parameters from command line
    argparser = argparse.ArgumentParser(description='Get hyper-parameters')

    argparser.add_argument('-m', type=str, default="train", help='train or weights')
    argparser.add_argument('-lr', type=float, default="1e-3", help='learning rate')

    args = argparser.parse_args()

    return args

    # Create hyper-parameter object
    class config():
        def __init__(self, data, model, epochs, sl, s_train, s_test, wu, bn):
            self.data = data
            self.model = model
            self.train_bs = 256
            self.test_bs = 256
            self.epochs = epochs
            self.sl = sl
            self.s_train = s_train
            self.s_test = s_test

    config = config(args.data, args.model, args.epochs, args.sl, args.s_train, args.s_test, args.wu, args.bn)

# Set up config with hyperparameters
class hyperparameters():
    def __init__(self, args):
        self.batch_size = 64
        self.epochs = 10
        self.lr = args.lr

# Show images
def show_raw_data():
    # Import example of data from the corruptmnist folder
    data = np.load('./train_0.npz')
    images = data['images']
    labels = data['labels']

    # Show 5 example images
    fig, ax = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        ax[i].imshow(images[i], cmap='gray')
        ax[i].set_title(labels[i])
        ax[i].axis('off')

    plt.show()

def show_transformed_data(train_dl):
    # Plot example images from the dataloader
    imgs, labels = next(iter(train_dl))

    fig, ax = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        ax[i].imshow(imgs[i].numpy().squeeze(), cmap='gray')
        ax[i].set_title(labels[i])
        ax[i].axis('off')
    # plt.show()

# Train model
def train(config, model, criterion, optimizer, train_dl, eval_dl):
    model.train()

    train_loss = []
    eval_loss = []

    for epoch in range(config.epochs):
        running_loss = 0
        for i, data in enumerate(train_dl, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Plot loss
    plt.plot(train_loss, label='train')
    plt.plot(eval_loss, label='eval')
    plt.legend()
    plt.show()

    # Save model weights TODO: Change to best model (compared to eval loss)
    torch.save(model.state_dict(), 'model_weights.pth')

# Test model
def test(model, test_dl):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dl:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print test accuracy
    print(correct/total)

# Main function
def main():
    # Get experimental setup and create hyperparameter config
    args = get_setup()
    config = hyperparameters(args)

    # Sshow examples of raw images
    show_raw_data()

    # Initialize dataset
    train_set = mnist('./train_0.npz', './train_1.npz', './train_2.npz', './train_3.npz')
    eval_set = mnist('./train_4.npz')
    test_set = mnist('test.npz')

    # Initialize dataloader
    train_dl = DataLoader(train_set, batch_size=64, shuffle=True)
    eval_dl = DataLoader(eval_set, batch_size=64, shuffle=False)
    test_dl = DataLoader(test_set, batch_size=64, shuffle=True)

    # Show examples of transformed images
    show_transformed_data(train_dl)

    # Initialize model
    model = Net().to('cuda')

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Train or evaluate model
    if args.m == 'train':
        train(config, model, criterion, optimizer, train_dl, eval_dl)
    else:
        model.load_state_dict(torch.load(args.m))
        test(model, test_dl)

if __name__ == '__main__':
    main()




