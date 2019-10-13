import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from data_loader import UltrasoundDataShapeNet
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

def train(model, train_loader, val_loader, optimizer, num_epochs, path_to_save_best_weights):
    model.train()

    log_softmax = nn.LogSoftmax(dim=1)# Use for NLLLoss()
    softmax = nn.Softmax(dim=1)

    criterion_nlloss = nn.NLLLoss().to(device)
    criterion_mseloss = nn.MSELoss().to(device)

    metrics_evaluator = PerformanceMetricsEvaluator()
    writer = SummaryWriter('LumenTissueCorruption')

    since = time.time()

    best_model_weights = model.state_dict()
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
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            # running_corrects = 0 TODO add IoU

            # Iterate over data.
            for imgs, masks in tqdm(data_loader):
                mask_to_encode = masks.numpy()
                mask_to_encode = (np.arange(4) == mask_to_encode[...,None]).astype(float)
                mask_to_encode = np.moveaxis(mask_to_encode, 3, 1)
                mask_to_encode = torch.from_numpy(mask_to_encode).float().to(device)
                imgs, masks = imgs.to(device), masks.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                print(imgs.shape)
                return
                logits, encoded_shape = model(imgs)
                _, encoded_mask = model(mask_to_encode)

                log_logits = log_softmax(logits)
                softmax_logits = softmax(logits).detach().cpu()

                nll_loss = criterion_nlloss(log_logits, masks)
                # mse_loss1 = criterion_mseloss(softmax_logits, imgs)
                mse_loss2 = criterion_mseloss(encoded_shape, encoded_mask)
                loss = nll_loss + 0.5*mse_loss2

                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #
                collapsed_softmax_logits = np.argmax(softmax_logits.detach(), axis=1)
                uncollupsed_images = (np.arange(4) == collapsed_softmax_logits[...,None]).numpy().astype(np.uint8)
                uncollupsed_images = torch.from_numpy(np.moveaxis(uncollupsed_images, 3, 1)).float()
                for i in range(len(softmax_logits)):
                    empty_channel = np.zeros((544, 544), dtype=np.uint64)
                    if phase == 'val':
                        img_name = 'ValidationEpoch: {}'.format(str(epoch))
                    else:
                        img_name = 'TrainingEpoch: {}'.format(str(epoch))
                    writer.add_image(img_name, 
                            vutils.make_grid([
                                imgs[i][1:,:,:].detach().cpu(), 
                                softmax_logits[i][1:, :, :],
                                uncollupsed_images[i][1:,:,:]
                                ]), epoch)
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
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_val_loss: # TODO add IoU
                # best_acc = epoch_acc
                best_val_loss = epoch_loss
                best_model_weights = model.state_dict()
    
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

ROOT_DIR = '../../dataset/pytorch_data/manually_corrupted_data/'

# Create Data Loaders
partition = 'train'
ultrasound_train = UltrasoundDataShapeNet(ROOT_DIR, partition)
train_loader = torch.utils.data.DataLoader(ultrasound_train,
                                             batch_size=2, 
                                             shuffle=True,
                                            )
partition = 'val'
ultrasound_val = UltrasoundDataShapeNet(ROOT_DIR, partition)
val_loader = torch.utils.data.DataLoader(ultrasound_val,
                                        batch_size=1,
                                        shuffle=False
                                        )
# # Create model
model = ShapeUNet((4, 544,544))
weights = torch.load('weights_shape_net/Unet_manually_corrupted_lumen0.113253.pth')
model.load_state_dict(weights)
model.to(device)

# Specify optimizer and criterion
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)

NUM_OF_EPOCHS = 50
train(model, train_loader, val_loader, optimizer, NUM_OF_EPOCHS, 'weights_shape_net/')




















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

#     criterion_nlloss = nn.NLLLoss()
#     criterion_mseloss = nn.MSELoss()

#     metrics_evaluator = PerformanceMetricsEvaluator()
#     writer = SummaryWriter('Lol')

#     since = time.time()

#     best_model_weights = model.state_dict()
#     best_IoU = 0.0 # TODO
#     curr_val_loss = 0.0
#     curr_training_loss = 0.0
#     best_val_loss = 1000000000

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

#                 imgs = imgs.permute(0,2,3,1)
#                 imgs = F.softmax(imgs, dim=1)
#                 mask_to_encode = masks.numpy()
#                 mask_to_encode = (np.arange(4) == mask_to_encode[...,None]).astype(float)
#                 mask_to_encode = np.moveaxis(mask_to_encode, 3, 1)
#                 mask_to_encode = torch.from_numpy(mask_to_encode).float().to(device)
#                 imgs, masks = imgs.to(device), masks.to(device)
                
#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 logits, encoded_shape = model(imgs)
#                 _, encoded_mask = model(mask_to_encode)

#                 log_logits = log_softmax(logits)
#                 softmax_logits = softmax(logits)

#                 nll_loss = criterion_nlloss(log_logits.cpu(), masks.cpu())
#                 # mse_loss1 = criterion_mseloss(softmax_logits, imgs)
#                 mse_loss2 = criterion_mseloss(encoded_shape.cpu(), encoded_mask.cpu())
#                 loss = nll_loss + 0.5*mse_loss2

#                 # ================================================================== #
#                 #                        Tensorboard Logging                         #
#                 # ================================================================== #
#                 softmax_logits = softmax(logits)
#                 collapsed_softmax_logits = np.argmax(softmax_logits.detach(), axis=1)
#                 collapsed_images = np.argmax(imgs.detach(), axis=1)
#                 for i in range(len(collapsed_softmax_logits)):
#                     if phase == 'val':
#                         writer.add_image('ValidationEpoch: {}'.format(str(epoch)), 
#                             vutils.make_grid([collapsed_images[i].cpu().float().unsqueeze(0),masks[i].cpu().float().unsqueeze(0), collapsed_softmax_logits[i].float().unsqueeze(0)]), epoch)
#                     else:
#                         writer.add_image('TrainingEpoch: {}'.format(str(epoch)), 
#                             vutils.make_grid([collapsed_images[i].cpu().float().unsqueeze(0),masks[i].cpu().float().unsqueeze(0), collapsed_softmax_logits[i].float().unsqueeze(0)]), epoch)

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

#         writer.add_scalars('TrainValLoss', 
#                             {'trainLoss': curr_training_loss,
#                              'validationLoss': curr_val_loss
#                             },
#                             epoch
#                            ) 
#     # Saving best model
#     torch.save(best_model_weights, 
#         os.path.join(path_to_save_best_weights, 'Unet_cross_entropy_part_and_2d_part_of_loss{:2f}.pth'.format(best_val_loss)))

#     # Show the timing and final statistics
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Loss: {:4f}'.format(best_val_loss)) # TODO add IoU


# # Choose free GPU
# device = torch.device("cuda:{}".format(str(get_freer_gpu())))

# ROOT_DIR = '../../dataset/pytorch_data/data_shape_net/'

# # Create Data Loaders
# partition = 'train'
# ultrasound_train = UltrasoundData(ROOT_DIR, partition)
# train_loader = torch.utils.data.DataLoader(ultrasound_train,
#                                              batch_size=2, 
#                                              shuffle=True,
#                                             )
# partition = 'val'
# ultrasound_val = UltrasoundData(ROOT_DIR, partition)
# val_loader = torch.utils.data.DataLoader(ultrasound_val,
#                                         batch_size=1,
#                                         shuffle=False
#                                         )
# # # Create model
# model = ShapeUNet((4, 544,544))
# # weights = torch.load('weights_shape_net/Unet_2_parts_of_loss_the_same_as_paper3.721386.pth')
# # model.load_state_dict(weights)
# model.to(device)

# # Specify optimizer and criterion
# lr = 1e-4
# optimizer = Adam(model.parameters(), lr=lr)

# NUM_OF_EPOCHS = 50
# train(model, train_loader, val_loader, optimizer, NUM_OF_EPOCHS, 'weights_shape_net/')


