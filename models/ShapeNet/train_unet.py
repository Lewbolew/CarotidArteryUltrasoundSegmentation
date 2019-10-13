import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from data_loader import UltrasoundData
from unet import UNet


from tqdm import tqdm
import time
import os
import skimage.io as io
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmax(memory_available)

def save_epoch_predictions(predicted_log_logits, corresponding_masks, epoch, folder_to_save, phase):
    if phase == 'val':
        x_folder_name = 'x_val'
        y_folder_name = 'y_val'
    else:
        x_folder_name = 'x_train'
        y_folder_name = 'y_train'

    if not os.path.exists(folder_to_save):
        os.mkdir(folder_to_save)

    if not os.path.exists(os.path.join(folder_to_save, str(epoch))):
        os.mkdir(os.path.join(folder_to_save, str(epoch)))

    if not os.path.exists(os.path.join(folder_to_save, str(epoch), x_folder_name)):
        os.mkdir(os.path.join(folder_to_save, str(epoch), x_folder_name))

    if not os.path.exists(os.path.join(folder_to_save, str(epoch), y_folder_name)):
        os.mkdir(os.path.join(folder_to_save, str(epoch), y_folder_name))

    for i in range(len(predicted_log_logits)):
        np.save(os.path.join(folder_to_save, str(epoch), x_folder_name, str(epoch)+'_'+str(i)+'.npy'),
                  predicted_log_logits[i])
        np.save(os.path.join(folder_to_save, str(epoch), y_folder_name, str(epoch)+'_'+str(i)+'.npy'), 
                  corresponding_masks[i])

def train(model, train_loader, val_loader, optimizer, 
        criterion, num_epochs, path_to_save_best_weights):
    model.train()

    log_softmax = nn.LogSoftmax(dim=1)# Use for NLLLoss()

    since = time.time()

    best_model_weights = model.state_dict()
    best_acc = 0.0
    best_val_loss = 1000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)



        for phase in ['train', 'val']:

            predicted_log_logits = [] # store all predictions for saving 
            corresponding_masks = []
            
            if phase == 'train':
                # scheduler.step() TODO the learning rate plato
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader
            running_loss = 0.0
            # running_corrects = 0 TODO add IoU

            # Iterate over data.
            for imgs, masks in tqdm(data_loader):
                imgs, masks = imgs.to(device), masks.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                logits = model(imgs)

                log_logits = log_softmax(logits).to(device)

                # Save the results of the epoch
                # output = np.argmax(logits.detach().cpu().numpy(), axis=1)
                output = logits.detach().cpu().numpy()

                corr_masks = masks.detach().cpu().numpy()
                for i in range(len(output)):
                    predicted_log_logits.append(output[i])
                    corresponding_masks.append(corr_masks[i])

                loss = criterion(log_logits, masks)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.detach().item()
                # running_corrects += TODO
            epoch_loss = running_loss / len(data_loader)
            # epoch_acc = running_corrects / dataset_sizes[phase] # TODO add IoU

            print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #         phase, epoch_loss, epoch_acc)) TODO add IoU

            # Save Predictions
            folder_to_save = '/local/temporary/petryshak/pytorch_unet_predictions/'
            save_epoch_predictions(predicted_log_logits, corresponding_masks, epoch, folder_to_save, phase)
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_val_loss: # TODO add IoU
                # best_acc = epoch_acc
                best_val_loss = epoch_loss
                best_model_weights = model.state_dict()
    
    # Saving best model
    torch.save(best_model_weights, 
        os.path.join(path_to_save_best_weights, 'Unet_{}_epoch_{:.2f}_loss.pt'.format(str(num_epochs), best_val_loss)))

    # Show the timing and final statistics
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_val_loss)) # TODO add IoU


# Choose free GPU
device = torch.device("cuda:{}".format(str(get_freer_gpu())))

ROOT_DIR = '../../dataset/pytorch_data/color_non_rigid_aug'

# Create Data Loaders
partition = 'train'
ultrasound_train = UltrasoundData(ROOT_DIR, partition)
train_loader = torch.utils.data.DataLoader(ultrasound_train,
                                             batch_size=5, 
                                             shuffle=True,
                                            )
partition = 'val'
ultrasound_val = UltrasoundData(ROOT_DIR, partition)
val_loader = torch.utils.data.DataLoader(ultrasound_val,
                                        batch_size=1,
                                        shuffle=False
                                        )
# Create model
model = UNet((3, 544,544))

# model.load_state_dict(torch.load('weights_unet/Unet_35_epoch_0.12_loss.pt'))
model.to(device)

# Specify optimizer and criterion
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.NLLLoss().to(device)

NUM_OF_EPOCHS = 35
train(model, train_loader, val_loader, optimizer, criterion, NUM_OF_EPOCHS, 'weights_unet/')



# masks = []
# model.eval()
# for batch_idx, (val_imgs, val_masks) in enumerate(val_loader):
#     val_imgs, val_masks = val_imgs.to(device), val_masks.to(device)

#     output = model(val_imgs)
#     softmax = nn.Softmax(dim=1)
#     output = softmax(output)
#     output = np.argmax(output.detach().cpu().squeeze().numpy(), axis=0)
#     masks.append(output)

# for ind, mask in enumerate(masks):
#     io.imsave('{}.png'.format(ind), mask)




