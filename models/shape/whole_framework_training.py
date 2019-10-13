import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from data_loader import ShapeData
from shape_net import ShapeUNet
from SR_Unet import SH_UNet

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
    criterion_mseloss = nn.MSELoss(size_average=False).to(device)

    metrics_evaluator = PerformanceMetricsEvaluator()

    to_tensor = transforms.ToTensor()

    writer = SummaryWriter('runs/unet_output_connected_with_less_pretrained_shapenet/')

    since = time.time()

    best_model_weights = model.state_dict()
    best_IoU = 0.0 
    best_val_loss = 1000000000

    curr_val_loss = 0.0
    curr_training_loss = 0.0
    curr_training_IoU = 0.0
    curr_val_IoU = 0.0
    curr_unet_training_IoU = 0.0
    curr_unet_val_IoU = 0.0

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
            running_IoU = 0.0
            unet_IoU = 0.0
            # Iterate over data.
            for imgs, masks in tqdm(data_loader):
                mask_to_encode = (np.arange(2) == masks.numpy()[...,None]).astype(float)
                mask_to_encode = torch.from_numpy(np.moveaxis(mask_to_encode, 3, 1)).float().to(device)
                imgs, masks = imgs.to(device), masks.to(device)
                # masks_for_shape = masks.clone().unsqueeze(1).float()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                unet_prediction, shape_net_encoded_prediction, shape_net_final_prediction = model(imgs)
                encoded_mask = model(mask_to_encode, only_encode=True)
                
                log_softmax_unet_prediction = log_softmax(unet_prediction)
                softmax_unet_prediction = softmax(unet_prediction)
                softmax_shape_net_final_prediction = softmax(shape_net_final_prediction)

                first_term = criterion_mseloss(softmax_unet_prediction, softmax_shape_net_final_prediction)
                second_term = criterion_mseloss(encoded_mask, shape_net_encoded_prediction)
                third_term = criterion_mseloss(softmax_shape_net_final_prediction, mask_to_encode)

                # print('First term: ', first_term.data.cpu().numpy(), 'Second term: ', second_term.data.cpu().numpy(), 'Third term: ', 
                #     third_term.data.cpu().numpy())
                # print()
                lambda_1 = 0.5
                lambda_2 = 0.5
                loss = first_term + lambda_1*third_term + lambda_2*third_term
                # one = criterion_mseloss(softmax_shape_net_final_prediction, mask_to_encode)
                # two = criterion_mseloss(soft)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                collapsed_softmax_logits = np.argmax(softmax_shape_net_final_prediction.detach(), axis=1)
                collapsed_softmax_unet = np.argmax(softmax_unet_prediction.detach(), axis=1)

                for i in range(len(collapsed_softmax_logits)):
                    if phase == 'val':
                        writer.add_image('ValidationEpoch: {}'.format(str(epoch)), 
                            vutils.make_grid([imgs[i].cpu(),masks[i].cpu().float().unsqueeze(0), collapsed_softmax_unet[i].float().unsqueeze(0),
                                collapsed_softmax_logits[i].float().unsqueeze(0)]), epoch)
                    else:
                        writer.add_image('TrainingEpoch: {}'.format(str(epoch)), 
                            vutils.make_grid([imgs[i].cpu(),masks[i].cpu().float().unsqueeze(0), collapsed_softmax_unet[i].float().unsqueeze(0),
                                collapsed_softmax_logits[i].float().unsqueeze(0)]), epoch)

                # statistics
                running_loss += loss.detach().item()

                batch_IoU = 0.0
                for k in range(len(imgs)):
                    batch_IoU += metrics_evaluator.mean_IU(collapsed_softmax_logits.numpy()[k], masks.cpu().numpy()[k])
                batch_IoU /= len(imgs)

                unet_batch_IoU = 0.0
                for j in range(len(imgs)):
                    unet_batch_IoU += metrics_evaluator.mean_IU(collapsed_softmax_logits.numpy()[k], masks.cpu().numpy()[k])
                unet_batch_IoU /= len(imgs)

                running_IoU += batch_IoU
                unet_IoU += unet_batch_IoU

            epoch_loss = running_loss / len(data_loader)
            epoch_IoU = running_IoU / len(data_loader)
            epoch_unet_IoU = unet_IoU / len(data_loader)
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
                curr_unet_val_IoU = epoch_unet_IoU
            else:
                curr_training_loss = epoch_loss
                curr_training_IoU = epoch_IoU
                curr_unet_training_IoU = epoch_unet_IoU
        writer.add_scalars('TrainValIoU', 
                            {'trainIoU': curr_training_IoU,
                             'validationIoU': curr_val_IoU,
                             'trainUnetIoU': curr_unet_training_IoU,
                             'validationUnetIoU': curr_unet_val_IoU
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
        os.path.join(path_to_save_best_weights, 'unet_output_connected_with_less_pretrained_shapenet{:2f}.pth'.format(best_val_loss)))

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

model = SH_UNet('/mnt/datagrid/personal/qqpetrys/carotidArteryProject/shapesBaseline/weights/pretrained_shape_net_less_epochs113.171801.pth')
# model = SH_UNet()
model.to(device)

# Specify optimizer and criterion
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
NUM_OF_EPOCHS = 100

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


