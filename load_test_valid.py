# @title
from torchvision.transforms.functional import to_pil_image
import PIL
import torch
from torchvision import *
import torchvision.transforms as transforms
import torchvision
import random
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import time

import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

import matplotlib
import matplotlib.pyplot as plt

import torch.nn.functional as F
import cv2
import CNNModel
import plotting_utils

def main():
    # Insert the directory path in here
    # choices are diamonds, clubs, spades, hearts
    SUIT = "diamonds"
    SUIT_CAPS = SUIT.capitalize()
    print(SUIT_CAPS)
    
    save_path   =  SUIT_CAPS+"_output.pth"
    # train_path   =  "datasets/big_set/"+SUIT+"_train"
    # valid_path   =  "datasets/big_set/"+SUIT+"_valid"
    save_path   =  "HEARTSWITHSPADES_output.pth"
    SUIT = "hearts"
    SUIT_CAPS = SUIT.capitalize()
    print(SUIT_CAPS)
    train_path   =  "datasets/onlyheartsifspades/train"
    valid_path   =  "datasets/onlyheartsifspades/valid"

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel.CNNmodel().to(device)
    # model = torchvision.models.resnet18(num_classes=2).cuda()
    model.load_state_dict(  torch.load(save_path)['model_state_dict']  )
    ## END: GET RID OF THIS IF IT'S UR FIRST TIME TRAINING ##

       # freeze_support()
    ######## DATA LOADING ########


    # batch size
    BATCH_SIZE = 64

    # the training transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    #   transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
    #  transforms.Normalize(
    #     mean=[0.5, 0.5, 0.5],
        #    std=[0.5, 0.5, 0.5]
        # )
    ])

    # the validation transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.5, 0.5, 0.5],
    #     std=[0.5, 0.5, 0.5]
    # )
    ])

    # training dataset
    train_set = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )

    print('Train length is', len(train_set))
    n = len(train_set)
    i = random.randrange(n)
    item = train_set[i]
    print(i,'th item: ')
    # Display the PIL image and the class name directly.
    to_pil_image(item[0]).show()
    print('Class name is ', train_set.classes[item[1]])

    # validation dataset
    val_set = datasets.ImageFolder(
        root=valid_path,
        transform=valid_transform
    )

    print('Valid length is', len(val_set))
    n = len(val_set)
    i = random.randrange(n)
    item = val_set[i]
    print(i,'th item: ')
    # Display the PIL image and the class name directly.
    to_pil_image(item[0]).show()
    print('Class name is ', val_set.classes[item[1]])

        # Sanity-checking the model
    model.eval()

    # This is me collecting all the mislabeled images from the validation set
    actuallynohearts = []
    actuallyhearts = []
    actuallyright = []
    val_set_old = val_set
    val_set = train_set
    print('Valid length is', len(val_set))
    n = len(val_set)


    for j in range(n):
        # i = random.randrange(n)
        i = j
        item = val_set[i]
        i = item[0].to('cuda')
        # print(i,'th item is a pair', item)
        # Display the PIL image and the class name directly.
        img = to_pil_image(i)
        # display(img)
        print('Class name is', val_set.classes[item[1]])
        # batch size of 1, so we have to tack in a new dimension out front
        i = i[None,:,:,:]

        y_pred = model(i)
        prediction = torch.argmax(y_pred)

        if prediction == 0:
            print('Prediction is', SUIT)
            if SUIT != val_set.classes[item[1]]:
                print('A MISMATCH! There are actually NO', SUIT)
                actuallynohearts.append(img)
                print(f'Prediction: ',{y_pred})
            else:
                actuallyright.append(img)
        else:
            neglabel = 'no'+ SUIT
            print('Prediction is', neglabel)
            if neglabel != val_set.classes[item[1]]:
                print('A MISMATCH!  There actually IS', SUIT)
                actuallyhearts.append(img)
            else:
                actuallyright.append(img)
                print(f'Prediction: ',{y_pred})

        # Extra code to display which datapoints got mislabeled
    noheartsimgs = []
    heartsimgs = []
    actuallyrightimgs = []

    anh_num = len(actuallynohearts)
    ah_num = len(actuallyhearts)
    ar_num = len(actuallyright)

    print("FALSE POSITIVES CARDINALITY:", anh_num)
    print("FALSE NEGATIVES CARDINALITY:", ah_num)
    print("CORRECT CARDINALITY:", ar_num)

    def image_grid(imgs, rows, cols):

        w, h = imgs[0].size
        grid = PIL.Image.new('RGB', size=(cols*(w+5), rows*(h+5)))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*(w+5), i//cols*(h+5)))
        return grid

    print("Actually NO", SUIT, "examples: ")
    for i in range(25):
        i = random.randrange(len(actuallynohearts))
        img = actuallynohearts[i]
        # i = actuallynohearts[i]
        # item = val_set[i]
        # i = item[0].to('cuda')
        # Display the PIL image and the class name directly.
        # p_img = to_pil_image(i)
        noheartsimgs.append(img)

        grid1 = image_grid(noheartsimgs, rows=5, cols=5)
    grid1.show()

    print("Actually YES", SUIT, "examples: ")
    for i in range(25):
        i = random.randrange(len(actuallyhearts))
        img = actuallyhearts[i]
        # item = val_set[i]
        # i = item[0].to('cuda')
        # Display the PIL image and the class name directly.
        # p_img = to_pil_image(i)
        heartsimgs.append(img)

        grid2 = image_grid(heartsimgs, rows=5, cols=5)
    grid2.show()

    print("Actually RIGHT examples: ")
    for i in range(25):
        i = random.randrange(len(actuallyright))
        img = actuallyright[i]
        # item = val_set[i]
        # i = item[0].to('cuda')
        # Display the PIL image and the class name directly.
        # p_img = to_pil_image(i)
        actuallyrightimgs.append(img)

        grid3 = image_grid(actuallyrightimgs, rows=5, cols=5)
    grid3.show()



if __name__ == "__main__":
    main()
