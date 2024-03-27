import os
import gc
import cv2
import time
import tqdm
import random
import collections
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tq
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# ablumentations for easy image augmentation for input as well as output
import albumentations as albu
# from albumentations import torch as AT
plt.style.use('bmh')

from utils import *
from cloudSet import CloudDataset
from loss import *
from optimizer import *
from models.UNet import UNet
from models.PSPNet import PSPNet


def make_img(df: pd.DataFrame,  shape: tuple = (350, 525), path="understanding_cloud_organization/"):
    """
    create 350,525 img for later dataloader
    """
    # print(df.head())
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    for idx, im_name in tq(enumerate(df["im_id"].values)):
        for classidx, classid in enumerate(["Fish", "Flower", "Gravel", "Sugar"]):
            enocded = df[df.Image_Label == im_name + "_" + classid]
            enocded = enocded.EncodedPixels
            encoded = enocded.iloc[0]
            mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
            if encoded is not np.nan:   
                mask = rle_decode(encoded)
                if mask[:, :].shape != (350, 525):
                    mask = cv2.resize(mask, (525, 350))
                masks[:, :, classidx] = mask
                # mask*=225
            cv2.imwrite(path+"train_images_525_withcolor/"+im_name[:-4]+classid+".jpg",mask)
    return masks


def main():
    save_path = "datasaved/Unet/"
    path = "understanding_cloud_organization/"
    img_paths = "understanding_cloud_organization/train_image/"
    train_on_gpu = torch.cuda.is_available()
    SEED = 42
    MODEL_NO = 0 # in K-fold
    N_FOLDS = 10 # in K-fold
    seed_everything(SEED)
    torch.cuda.set_device(0)
    print(os.listdir(path))
    print(torch.cuda.device_count())
    print(train_on_gpu)

    """## Make split in train test validation"""

    train = pd.read_csv(f"{path}/train.csv")
    train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
    train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])

    # split data
    print(train.head())
    id_mask_count = (
        train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"]
        .apply(lambda x: x.split("_")[0])
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"index": "img_id", "Image_Label": "count"})
    )
    print(id_mask_count.info())
    print(id_mask_count.head())
    test_id_mask_count = id_mask_count[:int(len(id_mask_count) / 5)]
    id_mask_count = id_mask_count[int(len(id_mask_count) / 5):]

    ids = id_mask_count["img_id"].values
    li = [
        [train_index, test_index]
        for train_index, test_index in StratifiedKFold(
            n_splits=N_FOLDS, random_state=SEED
        ,shuffle=True).split(ids, id_mask_count["count"])
    ]
    train_ids, valid_ids = ids[li[MODEL_NO][0]], ids[li[MODEL_NO][1]]
    test_ids = test_id_mask_count['img_id'].values

    print(f"training set   {train_ids[:5]}.. with length {len(train_ids)}")
    print(f"validation set {valid_ids[:5]}.. with length {len(valid_ids)}")
    print(f"testing set    {test_ids[:5]}.. with length {len(test_ids)}")

    # define dataset and dataloader
    num_workers = 0
    bs = 2
    train_dataset = CloudDataset(
        df=train,
        datatype="train",
        img_ids=train_ids,
        transforms=get_training_augmentation(),
    )
    valid_dataset = CloudDataset(
        df=train,
        datatype="valid",
        img_ids=valid_ids,
        transforms=get_validation_augmentation(),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers
    )


    test_dataset = CloudDataset(df=train,
                                datatype='test', 
                                img_ids=ids,
                                transforms=get_validation_augmentation())
    test_loader = DataLoader(test_dataset, batch_size=bs,
                            shuffle=False, num_workers=0)


    # # Debug
    # sample_size = 25
    # indices = random.sample(range(len(train_dataset)), sample_size)
    # sampler = SubsetRandomSampler(indices)
    # train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler, num_workers=num_workers)

    # indices = random.sample(range(len(valid_dataset)), sample_size)
    # sampler = SubsetRandomSampler(indices)
    # valid_loader = DataLoader(valid_dataset, batch_size=bs, sampler=sampler, num_workers=num_workers)


    # indices = random.sample(range(len(test_dataset)), sample_size)
    # sampler = SubsetRandomSampler(indices)
    # test_loader = DataLoader(test_dataset, batch_size=bs, sampler=sampler, num_workers=num_workers)

    model = UNet(n_channels=3, n_classes=4).float()
    if train_on_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    criterion = BCEDiceLoss(eps=1.0, activation=None)
    optimizer = RAdam(model.parameters(), lr = 0.005)
    current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)

    """## Training loop"""

    # number of epochs to train the model
    n_epochs = 4
    train_loss_list = []
    valid_loss_list = []
    dice_score_list = []
    lr_rate_list = []
    valid_loss_min = np.Inf # track change in validation loss
    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        dice_score = 0.0
        print(f"Training {epoch} Start")
        model.train()
        bar = tq(train_loader, postfix={"train_loss":0.0})
        for data, target in bar:
            # move tensors to GPU if CUDA is available
            # print(data.size(),target.size())
            if train_on_gpu:
                data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            #print(loss)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            bar.set_postfix(ordered_dict={"train_loss":loss.item()})
            # print(f"train epoch {epoch}: loss : {loss}")
  
        # validate the model #
        model.eval()
        del data, target
        with torch.no_grad():
            bar = tq(valid_loader, postfix={"valid_loss":0.0, "dice_score":0.0})
            for data, target in bar:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss 
                valid_loss += loss.item()*data.size(0)
                dice_cof = dice_no_threshold(output.to(device), target.to(device)).item()
                dice_score +=  dice_cof * data.size(0)
                bar.set_postfix(ordered_dict={"valid_loss":loss.item(), "dice_score":dice_cof})  
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        dice_score = dice_score/len(valid_loader.dataset)
        print(f"train loss: {train_loss}, valid loss: {valid_loss}, dice score: {dice_score}")
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        dice_score_list.append(dice_score)
        lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])
        
        # print training/validation statistics 
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model_cifar.pt')
            valid_loss_min = valid_loss
        
        scheduler.step(valid_loss)

    """## Ploting Metrics"""
    # Get the current working directory
    cwd = os.getcwd()

    # List the contents of the directory
    contents = os.listdir(cwd)

    # Search for the model file
    for file in contents:
        if file == 'model_cifar.pt':
            print(f"Found model file at: {os.path.join(cwd, file)}")


    plt.figure(figsize=(10,10))
    plt.plot([i[0] for i in lr_rate_list])
    plt.ylabel('learing rate during training', fontsize=22)
    plt.savefig(save_path+"learning rate.png")
    plt.show()

    plt.figure(figsize=(10,10))
    plt.plot(train_loss_list,  marker='o', label="Training Loss")
    plt.plot(valid_loss_list,  marker='o', label="Validation Loss")
    plt.ylabel('loss', fontsize=22)
    plt.legend()
    plt.savefig(save_path+"loss.png")
    plt.show()

    plt.figure(figsize=(10,10))
    plt.plot(dice_score_list)
    plt.ylabel('Dice score')
    plt.savefig(save_path+"Dice Score.png")
    plt.show()

    # load best model
    model.load_state_dict(torch.load('model_cifar.pt'))
    model.eval()

    valid_masks = []
    count = 0
    tr = min(len(valid_ids)*4, 2000)
    probabilities = np.zeros((tr, 350, 525), dtype = np.float32)
    for data, target in tq(valid_loader):
        if train_on_gpu:
            data = data.cuda()
        target = target.cpu().detach().numpy()
        outpu = model(data).cpu().detach().numpy()
        for p in range(data.shape[0]):
            output, mask = outpu[p], target[p]
            for m in mask:
                valid_masks.append(resize_it(m))
            for probability in output:
                probabilities[count, :, :] = resize_it(probability)
                count += 1
            if count > tr - 1:
                break
        if count > tr - 1:
            break

    """## Grid Search for best Threshold"""

    class_params = {}
    with torch.no_grad():
        for class_id in range(4):
            print(class_id)
            attempts = []
            for t in range(0, 100, 5):
                t /= 100
                # for ms in [0, 100, 1200, 5000, 10000, 30000]:
                for ms in [0, 100, 1200, 5000]:
                    masks, d = [], []
                    for i in range(class_id, len(probabilities), 4):
                        probability = probabilities[i]
                        predict, num_predict = post_process(probability, t, ms)
                        masks.append(predict)
                    for i, j in zip(masks, valid_masks[class_id::4]):
                        if (i.sum() == 0) & (j.sum() == 0):
                            d.append(1)
                        else:
                            d.append(dice(i, j))
                    attempts.append((t, ms, np.mean(d)))

            attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
            attempts_df = attempts_df.sort_values('dice', ascending=False)
            print(attempts_df.head())
            best_threshold = attempts_df['threshold'].values[0]
            best_size = attempts_df['size'].values[0]
            class_params[class_id] = (best_threshold, best_size)

    del masks
    del valid_masks
    del probabilities
    gc.collect()

    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
    print(class_params)

    attempts_df.groupby(['threshold'])['dice'].max()

    attempts_df.groupby(['size'])['dice'].max()

    attempts_df = attempts_df.sort_values('dice', ascending=False)
    attempts_df.head(10)

    sns.lineplot(x='threshold', y='dice', hue='size', data=attempts_df)
    plt.title('Threshold and min size vs dice')
    plt.savefig(save_path+"threshold.jpg")
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['size'].values[0]


    # class_params = {0:(0.3,1200),1:(0.3,1200),2:(0.3,1200),3:(0.3,1200)}
    print("visualize starting...")
    with torch.no_grad():
        for i, (data, target) in enumerate(valid_loader):
            if train_on_gpu:
                data = data.cuda()
            print(data.shape)
            output = ((model(data))[0]).cpu().detach().numpy()
            print(model(data).shape)
            print(output.shape)
            image  = data[0].cpu().detach().numpy()
            mask   = target[0].cpu().detach().numpy()
            output = output.transpose(1 ,2, 0)
            image_vis = image.transpose(1, 2, 0)

            mask = mask.astype('uint8').transpose(1, 2, 0)
            pr_mask = np.zeros((350, 525, 4))
            for j in range(4):
                probability = resize_it(output[:, :, j])
                pr_mask[:, :, j], _ = post_process(probability,
                                                class_params[j][0],
                                                class_params[j][1])
            visualize_with_raw(image=image_vis, mask=pr_mask,
                            original_image=image_vis, original_mask=mask,
                            raw_image=image_vis, raw_mask=output,iter=i)
            if i >= 6:
                break

    # torch.cuda.empty_cache()
    # gc.collect()

    # del train_dataset, train_loader
    # del valid_dataset, valid_loader
    # gc.collect()


    test_loss = 0
    dice_score = 0
    print("testing start...")
    pred = np.zeros((bs,4,350,525))
    masks = np.zeros((bs,4,350,525))

    with torch.no_grad():
        for data, targ in tq(test_loader):
            # print(data.shape,targ.shape)
            if train_on_gpu:
                data = data.cuda()
                targ = targ.cuda()
            outp = model(data) 
            loss = criterion(outp, targ)
            # print(outp.shape,targ.shape)
            test_loss += loss.item()*data.shape[0]

            outp = outp.cpu().detach().numpy()
            targ = targ.cpu().detach().numpy()
            # print(outp.shape,targ.shape)
            for p in range(data.shape[0]):
                outpu, targe = outp[p], targ[p]
                # print(outpu.shape,targe.shape)
                for j in range(4):
                    output = resize_it(outpu[j])
                    target = resize_it(targe[j])
                    # print(output.shape)
                    output,_ = post_process(output,
                                                    class_params[j][0],
                                                    class_params[j][1])
                    pred[p,j,:,:] = output
                    masks[p,j,:,:] = target

            dice_cof = dice(pred, masks).item()
            dice_score += dice_cof*data.size(0)
            bar.set_postfix(ordered_dict={"valid_loss":loss.item(), "dice_score":dice_cof})  
    # calculate average losses
    # test_loss = test_loss/len(test_loader.sampler)
    # dice_score = dice_score/len(test_loader.sampler)
    test_loss = test_loss/len(test_loader.dataset)
    dice_score = dice_score/len(test_loader.dataset)
    print(f"test loss: {test_loss}, dice score: {dice_score}")