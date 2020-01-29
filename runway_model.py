import numpy as np
import runway
import torch

from stylegan2 import utils, models

np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
directions = torch.load('checkpoint/directions.pth')

colors = ['aqua', 'black', 'blue', 'brown',  'green', 'grey', 'lavender', 'light_brown', 'multicolored', 'orange', 'pink', 'purple', 'red', 'silver', 'white', 'yellow']
switches = ['open', 'closed', 'covered']
adjs = ['glowing', 'gradient', 'reflective', 'ringed', 'rolling', 'rubbing', 'shading', 'sparkling']

# generate composition of elements
components = ['eyes', 'eyes']

eyes = [f'{color}_eyes' for color in colors if f'{color}_eyes' in directions]
hairs = [f'{color}_hair' for color in colors if f'{color}_hair' in directions]

coeffs = ['hair_coeff', 'eyes_coeff']

def shift_latents(latents, inputs):
    z = latents.clone().repeat(1, 16, 1) # repeat self
    for key in coeffs:
        color = key.replace('coeff', 'color')
        coeff = inputs[key]
        if coeff != 0:
            direction = directions[inputs[color]]
            z = inject_coeff(z, direction, coeff)
    return z

def inject_coeff(latent_vector, direction, coeff):
    new_latent_vector = latent_vector.clone()
    new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
    return new_latent_vector

@runway.setup(options={'checkpoint': runway.file(extension='.pth')})
def setup(opts):
    global Gs
    if opts['checkpoint'] is None or opts['checkpoint'] == '':
        opts['checkpoint'] = 'checkpoint/Gs.pth'
    state = torch.load(opts['checkpoint'], map_location=device)
    Gs = models.load(state, device)
    Gs.to(device)
    return Gs

generate_inputs = {
    'z': runway.vector(512, sampling_std=0.5),
    'truncation': runway.number(min=0, max=1, default=0.8, step=0.01),
    'hair_color': runway.category(choices=hairs),
    'hair_coeff': runway.number(min=-5, max=5, default=0, step=0.01),
    'eyes_color': runway.category(choices=eyes),
    'eyes_coeff': runway.number(min=-5, max=5, default=0, step=0.01),
}

@runway.command('generate', inputs=generate_inputs, outputs={'image': runway.image})
def convert(model, inputs):
    truncation = inputs['truncation']
    Gs.set_truncation(truncation_psi=truncation)
    qlatents = torch.Tensor(inputs['z']).reshape(1, 512).to(device=device, dtype=torch.float32)
    dlatents = Gs.G_mapping(qlatents)
    swifted_dlatents = shift_latents(dlatents, inputs)
    generated = Gs(dlatents=swifted_dlatents)
    images = utils.tensor_to_PIL(generated)
    return {'image': images[0]}


if __name__ == '__main__':
    runway.run()