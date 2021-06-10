from  dataset import RGB3DBlocks
from model import Resnet3D, ResidualBlock
from torchvision import utils, transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, Resize, RandomHorizontalFlip, RandomApply, RandomCrop, RandomResizedCrop
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd

path_to_training = '../data/mini_UCF/train.txt'
path_to_validation = "../data/mini_UCF/validation.txt"
path_to_classes ='../data/mini_UCF/classes.txt'
root_rgb = '../data/mini_UCF/'
root_flow = '../data/mini-ucf101_flow_img_tvl1_gpu'

# rgb3d dataset
rgb3d_train = RGB3DBlocks(16,path_to_training, path_to_classes, root_rgb ,transform = Compose([RandomApply([RandomResizedCrop(256,(0.1,1.0)),RandomCrop(256), RandomHorizontalFlip(0.5)],0.9), transforms.Resize((224,224)), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) ]))
rgb3d_validation = RGB3DBlocks(16,path_to_validation, path_to_classes, root_rgb, transform = Compose([Resize((224,224)), Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) ]))

# train and test dataloaders
dataloader_train = DataLoader(rgb3d_train, batch_size=8, shuffle=True, num_workers=8)
dataloader_validation = DataLoader(rgb3d_validation, batch_size=8, shuffle=True, num_workers=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get labels from id
def get_labels(path_to_classes):
    classes ={}
    with open(path_to_classes) as f:
        c = f.readlines()
    for x in c:
        classes[int(x.strip().split(" ")[0])] =  x.strip().split(" ")[1]
    return classes

# save model
def save_model(model,stats, model_name):
    model_dict = {"model":model, "stats":stats}
    torch.save(model_dict, "../models/" + model_name + ".pth")


@torch.no_grad()
def eval_model(model):
    """ Computing model accuracy """
    correct = 0
    total = 0
    loss_list = []
    pred_list = []
    label_list = []

    criterion = torch.nn.CrossEntropyLoss().to(device) # cross entropy loss

    for batch in dataloader_validation:

        snippets = batch[0].to(device)  # get snippets
        labels = batch[1].to(device)  # get label
        label_list.append(labels)

        # Forward pass only to get logits/output
        outputs = model(snippets) # forward pass

        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Get predictions from the maximum value
        preds = torch.argmax(outputs, dim=1) # get prediction
        pred_list.append(preds)
        correct += len(torch.where(preds == labels)[0])
        total += len(labels)

    # get classwise accuracy
    predictions = torch.cat(pred_list)
    labels = torch.cat(label_list)
    classes = get_labels(path_to_classes)
    classwise_accuracy = {}
    for i in range(25):
        classwise_accuracy[classes[i]] = len(torch.where((predictions == i) & (predictions == labels))[0]) / len(
            torch.where(labels == i)[0])

    # Total correct predictions and loss
    accuracy = correct / total * 100
    loss = np.mean(loss_list)
    return accuracy, loss, classwise_accuracy

def train_model():
    LR = 1e-4 # learning rate
    EPOCHS = 10
    EVAL_FREQ = 1
    SAVE_FREQ = 10

    # initialize model
    resnet3D = Resnet3D(ResidualBlock, [2,2,2,2], [64, 128, 256, 512], 25)
    resnet3D = resnet3D.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device) # cross entropy loss
    optimizer = torch.optim.Adam(params=resnet3D.parameters(), lr = LR) # define optimizer

    stats = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "accuracy": []
    }
    init_epoch = 0
    loss_hist = []
    for epoch in range(init_epoch,EPOCHS): # iterate over epochs
        loss_list = []
        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for i, batch in progress_bar: # iterate over batches
            snippets = batch[0].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad() # remove old grads
            y = resnet3D(snippets) # get predictions
            loss = criterion(y, labels) # find loss
            loss_list.append(loss.item())

            loss.backward() # find gradients
            optimizer.step()  # update weights
            progress_bar.set_description(f"Epoch {0 + 1} Iter {i + 1}: loss {loss.item():.5f}. ")
        # update stats
        loss_hist.append(np.mean(loss_list))
        stats['epoch'].append(epoch)
        stats['train_loss'].append(loss_hist[-1])

        if epoch % EVAL_FREQ == 0:
            accuracy, valid_loss = eval_model(resnet3D)
            print(f"Accuracy at epoch {epoch}: {round(accuracy, 2)}%")
        else:
            accuracy, valid_loss = -1, -1
        stats["accuracy"].append(accuracy)
        stats["valid_loss"].append(valid_loss)

        # saving checkpoint
        if epoch % SAVE_FREQ == 0:
            save_model(resnet3D, stats, "3D_resnet_inflated")
        save_model(resnet3D, stats, "3D_resnet_inflated")
    save_model(resnet3D, stats, "3D_resnet_inflated")

# method to generate results after training
def compare_results(model_list, labels,dataloaders):
    train_loss = {}
    validation_loss = {}
    accuracy = {}
    stat_list = [model["stats"] for model in model_list]
    models = [m["model"] for m in model_list]
    for i,stat in enumerate(stat_list):
        accuracy[labels[i]] =  stat['accuracy']
        validation_loss[labels[i]] = stat['valid_loss']
        train_loss[labels[i]] = stat['train_loss']
    for i,label in enumerate(labels):
        plt.plot(accuracy[label], label=label)
    plt.suptitle("Accuracy comparision")
    plt.legend()
    plt.show()

    for i,label in enumerate(labels):
        plt.plot(train_loss[label], label=label)
    plt.suptitle("Train  Loss comparision")
    plt.legend()
    plt.show()

    for i,label in enumerate(labels):
        plt.plot(validation_loss[label], label=label)
    plt.suptitle("Val Loss comparision")
    plt.legend()
    plt.show()

    testing_accuracy = {}
    classwise = {}
    for i, model in enumerate(models):
        testing_accuracy[labels[i]],t, classwise[labels[i]] = eval_model(model.eval())
    plt.bar(range(len(testing_accuracy)), list(testing_accuracy.values()), align='center')
    plt.xticks(range(len(testing_accuracy)), list(testing_accuracy.keys()))
    plt.legend()
    plt.show()
    acc_class = np.array([np.fromiter(classwise[model].values(), dtype=float) for model in classwise.keys()]).T
    df  = pd.DataFrame(classwise)
    ax = df.plot.bar()
    plt.show()



train_model() # use this to train model
# model_inflated = torch.load('../models/3D_resnet_inflated.pth')
# model_random = torch.load('../models/3D_resnet_random.pth')
# compare_results([model_inflated,model_random],["Resnet3D Inflated", "Resnet3D Random" ],[dataloader_validation,dataloader_validation])
