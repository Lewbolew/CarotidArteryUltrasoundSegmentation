import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from data_loader import UltrasoundData
from unet import UNet
from shape_net import ShapeUNet

from tqdm import tqdm
import time
import os
import skimage.io as io
import numpy as np
from metrics_evaluator import PerformanceMetricsEvaluator
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import warnings
warnings.filterwarnings("ignore")

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmax(memory_available)

def train(unet, shapeNet, train_loader, val_loader, optimizer1, optimizer2,num_epochs, path_to_save_best_weights):
    unet.train()
    shapeNet.train()

    log_softmax = nn.LogSoftmax(dim=1)# Use for NLLLoss()
    softmax = nn.Softmax(dim=1)

    criterion_nlloss = nn.NLLLoss().to(device)
    criterion_mseloss1 = nn.MSELoss().to(device)
    criterion_mseloss2 = nn.MSELoss().to(device)

    metrics_evaluator = PerformanceMetricsEvaluator()
    writer = SummaryWriter('FrameworkTensorboard/exp9/')

    since = time.time()

    best_model_weights = shapeNet.state_dict()
    best_IoU = 0.0 # TODO
    curr_val_loss = 0.0
    curr_training_loss = 0.0
    best_val_loss = 1000000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':
                # scheduler.step() TODO the learning rate plato
                unet.train()
                shapeNet.train()
                data_loader = train_loader
            else:
                unet.eval()
                shapeNet.eval()
                data_loader = val_loader

            running_loss = 0.0
            # running_corrects = 0 TODO add IoU

            # Iterate over data.
            for imgs, masks in tqdm(data_loader):
                mask_to_encode = masks.numpy()
                mask_to_encode = (np.arange(4) == mask_to_encode[...,None]).astype(float)
                mask_to_encode = torch.from_numpy(np.moveaxis(mask_to_encode, 3, 1)).float().to(device)
                imgs, masks = imgs.to(device), masks.to(device)
               
                # zero the parameter gradients
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                # forward
                unet_prediction = unet(imgs)
                shape_net_final_prediction, shape_net_encoded_prediction  = shapeNet(softmax(unet_prediction))
                _, encoded_mask = shapeNet(mask_to_encode)

                log_softmax_unet_prediction = log_softmax(unet_prediction)
                softmax_unet_prediction = softmax(unet_prediction)
                softmax_shape_net_final_prediction = softmax(shape_net_final_prediction)
                first_term = criterion_mseloss1(mask_to_encode, shape_net_final_prediction)
                # second_term = criterion_mseloss(encoded_mask, shape_net_encoded_prediction)
                third_term = criterion_mseloss2(unet_prediction, mask_to_encode)
                lambda_1 = 0.5
                lambda_2 = 0.5
                #loss = third_term + 0.5*first_term
                # loss = third_term + 0.2*second_term
                # loss = second_term + third_term
                # loss = second_term + third_term
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #
                unet_softmax_collupsed = np.argmax(softmax_unet_prediction.detach(), axis=1)
                unet_softmax_uncollupsed = (np.arange(4) == unet_softmax_collupsed[...,None]).numpy().astype(np.uint8)
                unet_softmax_uncollupsed = torch.from_numpy(np.moveaxis(unet_softmax_uncollupsed, 3, 1)).float()

                softmax_shape_net_final_collupsed = np.argmax(softmax_shape_net_final_prediction.detach(), axis=1)
                softmax_shape_net_final_uncollupsed = (np.arange(4) == softmax_shape_net_final_collupsed[...,None]).numpy().astype(np.uint8)
                softmax_shape_net_final_uncollupsed = torch.from_numpy(np.moveaxis(softmax_shape_net_final_uncollupsed, 3, 1)).float()

                for i in range(len(unet_softmax_uncollupsed)):
                    empty_channel = np.zeros((544, 544), dtype=np.uint64)
                    if phase == 'val':
                        img_name = 'ValidationEpoch: {}'.format(str(epoch))
                    else:
                        img_name = 'TrainingEpoch: {}'.format(str(epoch))
                    writer.add_image(img_name, 
                            vutils.make_grid([
                                imgs[i].detach().cpu(), 
                                unet_softmax_uncollupsed[i][1:, :, :],
                                softmax_shape_net_final_uncollupsed[i][1:,:,:].cpu()
                                ]), epoch)
                # backward + optimize only if in training phase
                if phase == 'train':
                    # first_term.backward(retain_graph=True)
                    third_term.backward(retain_graph=True)
                    first_term.backward()
                    optimizer1.step()
                    optimizer2.step()
                # statistics
                running_loss += first_term.detach().item() + third_term.detach().item()
                # running_corrects += TODO
            epoch_loss = running_loss / len(data_loader)
            # epoch_acc = running_corrects / dataset_sizes[phase] # TODO add IoU

            print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))
            # print('{} L1: {:.4f}'.format(
            #         phase, first_term))
            # print('{} L2: {:.4f}'.format(
            #         phase, second_term))
            # print('{} L3: {:.4f}'.format(
            #         phase, third_term))
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #         phase, epoch_loss, epoch_acc)) TODO add IoU
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_val_loss: # TODO add IoU
                # best_acc = epoch_acc
                best_val_loss = epoch_loss
                best_model_weights = shapeNet.state_dict()
    
            if phase == 'val':
                curr_val_loss = epoch_loss
            else:
                curr_training_loss = epoch_loss

        writer.add_scalars('TrainValLoss', 
                            {'trainLoss': curr_training_loss,
                             'validationLoss': curr_val_loss
                            },
                            epoch
                           ) 
    # Saving best model
    torch.save(best_model_weights, 
        os.path.join(path_to_save_best_weights, 'Unet_manually_corrupted_lumenTissue{:2f}.pth'.format(best_val_loss)))

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
                                             batch_size=1, 
                                             shuffle=True,
                                            )
partition = 'val'
ultrasound_val = UltrasoundData(ROOT_DIR, partition)
val_loader = torch.utils.data.DataLoader(ultrasound_val,
                                        batch_size=1,
                                        shuffle=False
                                        )
# # Create models
PATH_TO_SHAPE_WEIGHT_MODEL = 'weights_shape_net/Unet_manually_corrupted_lumen0.113253.pth'

# Specify optimizer and criterion

NUM_OF_EPOCHS = 50
unet = UNet((3, 544,544))
shapeUNet = ShapeUNet((4, 544,544))
shapeUNet.load_state_dict(torch.load(PATH_TO_SHAPE_WEIGHT_MODEL))

unet.to(device)
shapeUNet.to(device)
lr = 1e-4
optimizer1 = Adam(shapeUNet.parameters(), lr=lr)
optimizer2 = Adam(unet.parameters(), lr=lr)
train(unet, shapeUNet, train_loader, val_loader, optimizer1, optimizer2, NUM_OF_EPOCHS, 'weights_shape_net/')
