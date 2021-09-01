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
# run train_photonlib.py --output_dir test --experiment_name log_mse --batch_size 10

# Configure Arguments
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--output_dir', type=str, default='./results', help='root for logging outputs')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--min_x', type=int, default=2,
                    help='Minimum number of slices in x axis')
p.add_argument('--max_x', type=int, default=3,
                    help='Maximum number of slices in x axis')
p.add_argument('--skip_x', type=int, default=2,
                    help='Number of slices skipped in x axis')
p.add_argument('--batch_size', type=int, default=100)
p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5') #5e-6 for FH
p.add_argument('--num_epochs', type=int, default=5000,
               help='Number of epochs to train for.')
p.add_argument('--kl_weight', type=float, default=1e-1,
               help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
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
data_shape = data.shape[0:-1]
data_shape = list(data_shape)
data_shape = tuple(data_shape)

data_image = []
data = np.expand_dims(np.sum(data, -1), axis=-1)
data = -np.log(data+1e-7)
data = np.squeeze(data)
data = np.uint8((data*1+0.5)*255)


for i in range(data_shape[0]):
    data_image.append(Image.fromarray(np.uint8(data[i])).convert('RGB'))
    
image_resolution = (data_shape[1], data_shape[2])
train_data = dataio.Implicit2DWrapperPhoton(data_image, data_shape[1:])

print('at the dataloader')
dataloader = DataLoader(train_data, shuffle=True, batch_size=opt.batch_size, pin_memory=False, num_workers=0)

print('Assigning Model...')
model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=image_resolution, hidden_features=154)
model = model.float()
model.cuda()

loss = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_image_summary, image_resolution)

print('Training...')
training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
           steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
           model_dir=output_dir, loss_fn=loss, summary_fn=summary_fn)
    
end = time.time()
print('Delta Time: {}'.format(end-start))
