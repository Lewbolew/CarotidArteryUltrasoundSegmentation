import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from data_loader import ShapeData
from shape_net import ShapeUNet


from tqdm import tqdm
import time
import numpy as np
import os
import cv2
import skimage.io as io
import numpy as np
from metrics_evaluator import PerformanceMetricsEvaluator
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmax(memory_available)

def train(model, train_loader, val_loader, optimizer, num_epochs, path_to_save_best_weights):
    model.train()

    log_softmax = nn.LogSoftmax(dim=1)# Use for NLLLoss()
    softmax = nn.Softmax(dim=1)

    criterion_nlloss = nn.NLLLoss()
    criterion_mseloss = nn.MSELoss().to(device)

    metrics_evaluator = PerformanceMetricsEvaluator()

    to_tensor = transforms.ToTensor()

    writer = SummaryWriter('runs/shape_net_with_only_crossentropy/')

    since = time.time()

    best_model_weights = model.state_dict()
    best_IoU = 0.0 
    best_val_loss = 1000000000

    curr_val_loss = 0.0
    curr_training_loss = 0.0
    curr_training_IoU = 0.0
    curr_val_IoU = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':
                # scheduler.step(best_val_loss)
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_IoU = 0 

            # Iterate over data.
            ind = 0
            for imgs, masks in tqdm(data_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                masks_for_shape = masks.clone().unsqueeze(1).float()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                logits, encoded_shape = model(imgs)
                _, encoded_mask = model(masks_for_shape)
                
                log_logits = log_softmax(logits).to(device)

                nll_loss = criterion_nlloss(log_logits, masks)
                # mse_loss = criterion_mseloss(encoded_shape, encoded_mask)
                loss = nll_loss #+ 0.5*mse_loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                softmax_logits = softmax(logits)
                collapsed_softmax_logits = np.argmax(softmax_logits.detach(), axis=1)
                for i in range(len(collapsed_softmax_logits)):
                    if phase == 'val':
                        writer.add_image('ValidationEpoch: {}'.format(str(epoch)), 
                            vutils.make_grid([imgs[i].cpu(),masks[i].cpu().float().unsqueeze(0), collapsed_softmax_logits[i].float().unsqueeze(0)]), epoch)
                    else:
                        writer.add_image('TrainingEpoch: {}'.format(str(epoch)), 
                            vutils.make_grid([imgs[i].cpu(),masks[i].cpu().float().unsqueeze(0), collapsed_softmax_logits[i].float().unsqueeze(0)]), epoch)

                # statistics
                running_loss += loss.detach().item()
                running_IoU += metrics_evaluator.mean_IU(collapsed_softmax_logits.numpy()[0], masks.cpu().numpy()[0])
                ind+=1
            epoch_loss = running_loss / len(data_loader)
            epoch_IoU = running_IoU / len(data_loader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_IoU))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_val_loss: # TODO add IoU
                best_val_loss = epoch_loss
                best_IoU = epoch_IoU
                best_model_weights = model.state_dict()
    
            if phase == 'val':
                # print(optimizer.param_groups[0]['lr'])
                curr_val_loss = epoch_loss
                curr_val_IoU = epoch_IoU
            else:
                curr_training_loss = epoch_loss
                curr_training_IoU = epoch_IoU

        writer.add_scalars('TrainValIoU', 
                            {'trainIoU': curr_training_IoU,
                             'validationIoU': curr_val_IoU
                            },
                            epoch
                           )
        writer.add_scalars('TrainValLoss', 
                            {'trainLoss': curr_training_loss,
                             'validationLoss': curr_val_loss
                            },
                            epoch
                           ) 
    # Saving best model
    torch.save(best_model_weights, 
        os.path.join(path_to_save_best_weights, 'shape_net_with_only_crossentropy{:2f}.pth'.format(best_val_loss)))

    # Show the timing and final statistics
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_val_loss)) # TODO add IoU



# Choose free GPU
device = torch.device("cuda:{}".format(str(get_freer_gpu())))

ROOT_DIR = 'dataset'

# Create Data Loaders
partition = 'train'
shape_train = ShapeData(ROOT_DIR, partition)
train_loader = torch.utils.data.DataLoader(shape_train,
                                             batch_size=10, 
                                             shuffle=True,
                                            )
partition = 'val'
shape_val = ShapeData(ROOT_DIR, partition)
val_loader = torch.utils.data.DataLoader(shape_val,
                                        batch_size=10,
                                        shuffle=False
                                        )
# Create model

model = ShapeUNet((1, 200,200))
model.to(device)

# Specify optimizer and criterion
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
NUM_OF_EPOCHS = 50

#training
train(model, train_loader, val_loader, optimizer, NUM_OF_EPOCHS, 'weights/')

# PATH_TO_SAVE = 'predictions/'

# preds = []
# masks = []
# imgs = []

# model.eval()
# with torch.no_grad():
#     for batch_idx, (val_imgs, val_masks) in enumerate(val_loader):
#         val_imgs = val_imgs.to(device)
#         print(batch_idx)
#         output, _ = model(val_imgs)
#         softmax = nn.Softmax(dim=1)
#         output = softmax(output)
#         # output = np.argmax(output.detach().cpu().squeeze().numpy(), axis=0)
#         preds.append(output.detach().cpu().squeeze(0).numpy())
#         masks.append(val_masks.detach().cpu().squeeze(0).numpy())
#         imgs.append(val_imgs.detach().cpu().squeeze(0).numpy())

# save_predictions(imgs, masks, preds, PATH_TO_SAVE)
























































# import torch
# import torch.nn as nn
# from torch.optim import Adam
# import torch.nn.functional as F

# from data_loader import UltrasoundData
# from shape_net import ShapeUNet


# from tqdm import tqdm
# import time
# import os
# import skimage.io as io
# import numpy as np

# from metrics_evaluator import PerformanceMetricsEvaluator
# from tensorboardX import SummaryWriter
# import torchvision.utils as vutils
# import warnings
# warnings.filterwarnings("ignore")

# def get_freer_gpu():
#     os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#     memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
#     os.remove('tmp')
#     return np.argmax(memory_available)

# def train(model, train_loader, val_loader, optimizer, num_epochs, path_to_save_best_weights):
#     model.train()

#     log_softmax = nn.LogSoftmax(dim=1)# Use for NLLLoss()
#     softmax = nn.Softmax(dim=1)

#     criterion_nlloss = nn.NLLLoss().to(device)
#     criterion_mseloss = nn.MSELoss().to(device)

#     metrics_evaluator = PerformanceMetricsEvaluator()
#     writer = SummaryWriter('LessCorrupted')

#     since = time.time()

#     best_model_weights = model.state_dict()
#     best_acc = 0.0
#     curr_val_loss = 0.0
#     curr_training_loss = 0.0
#     best_val_loss = 100000000

#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)

#         for phase in ['train', 'val']:

#             if phase == 'train':
#                 # scheduler.step() TODO the learning rate plato
#                 model.train()
#                 data_loader = train_loader
#             else:
#                 model.eval()
#                 data_loader = val_loader
#             running_loss = 0.0
#             # running_corrects = 0 TODO add IoU

#             # Iterate over data.
#             for imgs, masks in tqdm(data_loader):

#                 imgs, masks = imgs.to(device), masks.to(device)
#                 masks_for_shape = masks.clone().unsqueeze(1).float()
#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 logits, encoded_shape = model(imgs)
#                 _, encoded_mask = model(masks_for_shape)
                
#                 log_logits = log_softmax(logits).to(device)

#                 nll_loss = criterion_nlloss(log_logits, masks)
#                 mse_loss = criterion_mseloss(encoded_shape, encoded_mask)
#                 loss = nll_loss + 0.5*mse_loss

#                 # ================================================================== #
#                 #                        Tensorboard Logging                         #
#                 # ================================================================== #
#                 softmax_logits = softmax(logits)
#                 collapsed_softmax_logits = np.argmax(softmax_logits.detach(), axis=1)
#                 for i in range(len(collapsed_softmax_logits)):
#                     if phase == 'val':
#                         writer.add_image('ValidationEpoch: {}'.format(str(epoch)), 
#                             vutils.make_grid([imgs[i].cpu(),masks[i].cpu().float().unsqueeze(0), collapsed_softmax_logits[i].float().unsqueeze(0)]), epoch)
#                     else:
#                         writer.add_image('TrainingEpoch: {}'.format(str(epoch)), 
#                             vutils.make_grid([imgs[i].cpu(),masks[i].cpu().float().unsqueeze(0), collapsed_softmax_logits[i].float().unsqueeze(0)]), epoch)

#                 # backward + optimize only if in training phase
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
#                 # statistics
#                 running_loss += loss.detach().item()
#                 # running_corrects += TODO
#             epoch_loss = running_loss / len(data_loader)
#             # epoch_acc = running_corrects / dataset_sizes[phase] # TODO add IoU

#             print('{} Loss: {:.4f}'.format(
#                     phase, epoch_loss))
#             # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#             #         phase, epoch_loss, epoch_acc)) TODO add IoU

#             # deep copy the model
#             if phase == 'val' and epoch_loss < best_val_loss: # TODO add IoU
#                 # best_acc = epoch_acc
#                 best_val_loss = epoch_loss
#                 best_model_weights = model.state_dict()

#             if phase == 'val':
#                 curr_val_loss = epoch_loss
#             else:
#                 curr_training_loss = epoch_loss

#         writer.add_scalars('data/TrainValLoss', 
#                             {'trainLoss': curr_training_loss,
#                              'validationLoss': curr_val_loss
#                             },
#                             epoch
#                            ) 

#     # Show the timing and final statistics
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Loss: {:4f}'.format(best_val_loss)) # TODO add IoU


# # Choose free GPU
# device = torch.device("cuda:{}".format(str(get_freer_gpu())))
# print(device)
# ROOT_DIR = 'dataset/'

# # Create Data Loaders
# partition = 'train'
# ultrasound_train = UltrasoundData(ROOT_DIR, partition)
# train_loader = torch.utils.data.DataLoader(ultrasound_train,
#                                              batch_size=1, 
#                                              shuffle=True,
#                                             )
# partition = 'val'
# ultrasound_val = UltrasoundData(ROOT_DIR, partition)
# val_loader = torch.utils.data.DataLoader(ultrasound_val,
#                                         batch_size=1,
#                                         shuffle=False
#                                         )
# # # Create model
# model = ShapeUNet((1, 200,200))
# model.to(device)

# # Specify optimizer and criterion
# lr = 1e-4
# optimizer = Adam(model.parameters(), lr=lr)

# NUM_OF_EPOCHS = 50
# train(model, train_loader, val_loader, optimizer, NUM_OF_EPOCHS, 'weights_shape_net/')


# # PATH_TO_SAVE = 'predictions/'

# # preds = []
# # masks = []
# # imgs = []

# # model.eval()
# # with torch.no_grad():
# #     for batch_idx, (val_imgs, val_masks) in enumerate(val_loader):
# #         val_imgs = val_imgs.to(device)
# #         print(batch_idx)
# #         output, _ = model(val_imgs)
# #         softmax = nn.Softmax(dim=1)
# #         output = softmax(output)
# #         # output = np.argmax(output.detach().cpu().squeeze().numpy(), axis=0)
# #         preds.append(output.detach().cpu().squeeze(0).numpy())
# #         masks.append(val_masks.detach().cpu().squeeze(0).numpy())
# #         imgs.append(val_imgs.detach().cpu().squeeze(0).numpy())

# # save_predictions(imgs, masks, preds, PATH_TO_SAVE)




