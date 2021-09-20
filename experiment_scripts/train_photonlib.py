import sys
import os
import time
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import loss_functions, modules, training, utils, dataio
from torch.utils.data import DataLoader

from functools import partial
import configargparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from photon_library import PhotonLibrary


# Example syntax:
# run train_photonlib.py --output_dir results --experiment_name mse_long --batch_size 10 --num_epochs 100000

# Configure Arguments
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--output_dir', type=str, default='./results', help='root for logging outputs')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4') #5e-6 for FH
p.add_argument('--num_epochs', type=int, default=20000,
               help='Number of epochs to train for.')
p.add_argument('--kl_weight', type=float, default=1e-1,
               help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')
p.add_argument('--sample_frac', type=float, default=1,
               help='What fraction of detector pixels to sample in each batch (default is full detector)')
p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

opt = p.parse_args()

device = list(range(torch.cuda.device_count()))
start = time.time()

# Load plib dataset
print('Load data ...')
plib = PhotonLibrary()
full_data = plib.numpy() 

output_dir = os.path.join(opt.output_dir, opt.experiment_name)


data = full_data
# data_shape = tuple(data.shape)
# data_shape = data.shape[0:-1]
# data_shape = list(data_shape)
# data_shape = tuple(data_shape)

data = np.expand_dims(np.sum(data, -1), axis=-1)
# data = -np.log(data+1e-7)
# data = np.squeeze(data)
data = np.uint8((data*1+0.5)*255)
data_shape = tuple(data.shape)

detector_dataset = dataio.Detector(data)
coord_dataset = dataio.PhotonWrapper(detector_dataset, sidelength=data.shape)

print('at the dataloader')
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=False, num_workers=0)

print('Assigning Model...')
model = modules.SingleBVPNet(type='sine', in_features=3, out_features=detector_dataset.channels,
                                 mode='mlp', hidden_features=512, num_hidden_layers=1)
model = model.float()
model.cuda()

loss = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_detector_summary, detector_dataset)
img_summary = partial(utils.make_images, detector_dataset)

print('Training...')
training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
           steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
           model_dir=output_dir, loss_fn=loss, summary_fn=summary_fn, img_summary=img_summary, data_shape=data_shape)


end = time.time()
print('Delta Time: {}'.format(end-start))

print('Complete :)')
