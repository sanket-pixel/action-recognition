from dataset import RGBFrames, OpticalFlowStack
from model import TSN
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

# rgb dataset
rgb_dataset_train= RGBFrames(path_to_training,path_to_classes,root_rgb , mode = "train", transform = Compose([RandomApply([RandomResizedCrop(256,(0.1,1.0)),RandomCrop(256), RandomHorizontalFlip(0.5)],0.9), Resize((224,224)), Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) ]))
rgb_dataset_validation = RGBFrames(path_to_validation,path_to_classes,root_rgb,mode = "test", transform = transforms.Compose([Resize((224,224)), Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) ]))

# optical flow dataset
optical_flow_train = OpticalFlowStack(path_to_training,path_to_classes,root_flow,mode = "train",transform = transforms.Compose([Resize((224,224)), Normalize(mean=[0.449, 0.449, 0.449, 0.449, 0.449, 0.449, 0.449, 0.449, 0.449,
       0.449],std=[0.226, 0.226, 0.226, 0.226, 0.226, 0.226, 0.226, 0.226, 0.226,
       0.226])]))
optical_flow_test = OpticalFlowStack(path_to_validation,path_to_classes,root_flow,mode = "test",transform = transforms.Compose([Resize((224,224)), Normalize(mean=[0.449, 0.449, 0.449, 0.449, 0.449, 0.449, 0.449, 0.449, 0.449,
       0.449],std=[0.226, 0.226, 0.226, 0.226, 0.226, 0.226, 0.226, 0.226, 0.226,
       0.226])]))

# general dataloader
dataloader_train = DataLoader(rgb_dataset_train, batch_size=16, shuffle=True, num_workers=16)
dataloader_validation = DataLoader(rgb_dataset_validation, batch_size=16, shuffle=True, num_workers=16)
# get cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataloader testing for optical flow
dataloader_validation_optical = DataLoader(optical_flow_test, batch_size=8, shuffle=False, num_workers=8)
# dataloader testing for rgb
dataloader_validation_rgb = DataLoader(rgb_dataset_validation, batch_size=8, shuffle=False, num_workers=8)


# get class names from id
def get_labels(path_to_classes):
    classes ={}
    with open(path_to_classes) as f:
        c = f.readlines()
    for x in c:
        classes[int(x.strip().split(" ")[0])] =  x.strip().split(" ")[1]
    return classes


# function to save model and stats
def save_model(model,stats, model_name):
    model_dict = {"model":model, "stats":stats}
    torch.save(model_dict, "../models/" + model_name + ".pth")

# function to eval model and calculate classwise accuracy
@torch.no_grad()
def eval_model(model,dataloader_validation=dataloader_validation):
    """ Computing model accuracy """
    correct = 0
    total = 0
    loss_list = []
    pred_list = []
    label_list = []

    criterion = torch.nn.CrossEntropyLoss().to(device) # cross entropy loss

    for batch in dataloader_validation:

        snippets = batch[0].to(device) # get snippets
        labels = batch[1].to(device) # get label
        label_list.append(labels)

        # Forward pass only to get logits/output
        outputs = model(snippets) # forward pass

        loss = criterion(outputs, labels) # find loss
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

# function to calculate late fusion
@torch.no_grad()
def late_fusion_eval(model_optical_flow, model_rgb):
    """ Computing model accuracy """
    correct = 0
    total = 0
    loss_list = []
    pred_list = []
    label_list = []

    criterion = torch.nn.CrossEntropyLoss().to(device)

    for batch_optical, batch_rgb in zip(dataloader_validation_optical,dataloader_validation_rgb):

        snippets_optical= batch_optical[0].to(device) # get optical flow snippets
        labels_o = batch_optical[1].to(device)

        snippets_rgb = batch_rgb[0].to(device) # get rgb snippets
        labels = batch_rgb[1].to(device)
        label_list.append(labels)

        # Forward pass optical flow and rgb models
        outputs_optical_flow = model_optical_flow(snippets_optical)
        output_rgb = model_rgb(snippets_rgb)

        # take average of both models
        outputs = torch.add(outputs_optical_flow,output_rgb).divide(2)

        # find loss
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Get predictions from the maximum value
        preds = torch.argmax(outputs, dim=1)
        pred_list.append(preds)
        correct += len(torch.where(preds == labels)[0])
        total += len(labels)

    # classwise accuracy
    predictions = torch.cat(pred_list)
    labels = torch.cat(label_list)
    classes = get_labels(path_to_classes)
    classwise_accuracy = {}
    for i in range(25):
        classwise_accuracy[classes[i]] = len(torch.where((predictions == i) & (predictions == labels))[0]) / len(torch.where(labels == i)[0])

    # Total correct predictions and loss
    accuracy = correct / total * 100
    loss = np.mean(loss_list)
    return accuracy, loss,classwise_accuracy
def train_model():
    LR = 1e-4 # learning rate
    EPOCHS = 20
    EVAL_FREQ = 1
    SAVE_FREQ = 5

    # initialize model
    # for using with optical flow change modality to "optical_flow"
    tsn = TSN(4, 25, modality="rgb")
    tsn = tsn.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device) # cross entropy loss
    optimizer = torch.optim.Adam(params=tsn.parameters(), lr = LR) # define optimizer

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
            y = tsn(snippets) # get predictions
            loss = criterion(y, labels) # find loss
            loss_list.append(loss.item())

            loss.backward() # find gradients
            optimizer.step() # update weights
            progress_bar.set_description(f"Epoch {0 + 1} Iter {i + 1}: loss {loss.item():.5f}. ")
        # update stats
        loss_hist.append(np.mean(loss_list))
        stats['epoch'].append(epoch)
        stats['train_loss'].append(loss_hist[-1])

        if epoch % EVAL_FREQ == 0:
            accuracy, valid_loss, _ = eval_model(tsn)
            print(f"Accuracy at epoch {epoch}: {round(accuracy, 2)}%")
        else:
            accuracy, valid_loss = -1, -1
        stats["accuracy"].append(accuracy)
        stats["valid_loss"].append(valid_loss)

        # saving checkpoint
        if epoch % SAVE_FREQ == 0:
            save_model(tsn,stats, "RGB_tsn")
        save_model(tsn, stats, "RGB_tsn")
    save_model(tsn, stats, "RGB_tsn")

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
        testing_accuracy[labels[i]],t, classwise[labels[i]] = eval_model(model.eval(),dataloaders[i])
    plt.bar(range(len(testing_accuracy)), list(testing_accuracy.values()), align='center')
    plt.xticks(range(len(testing_accuracy)), list(testing_accuracy.keys()))
    plt.legend()
    plt.show()
    acc_class = np.array([np.fromiter(classwise[model].values(), dtype=float) for model in classwise.keys()]).T
    df  = pd.DataFrame(classwise)
    ax = df.plot.bar()
    plt.show()


train_model() # call this to train model
# model_optical_flow = torch.load('../models/optical_flow_pre_trained.pth')
# model_optical_flow_random = torch.load('../models/optical_flow_random.pth')
# model_rgb = torch.load('../models/RGB_tsn.pth')
# result_dict = {}
# result_dict['rgb'] = eval_model(model_rgb['model'].eval(),dataloader_validation_rgb)
# result_dict['optical'] = eval_model(model_optical_flow['model'].eval(),dataloader_validation_optical)
# result_dict['fusion'] = late_fusion_eval(model_optical_flow['model'].eval(), model_rgb['model'].eval())
# torch.save(result_dict,'../result_compare.pth')



# call this to see plots
# compare_results([model_optical_flow_random,model_optical_flow],["OpticalFlowRandom","OpticalFlow"],[dataloader_validation_optical,dataloader_validation_optical])