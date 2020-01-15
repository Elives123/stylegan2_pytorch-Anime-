import numpy as np
import runway
import torch

from stylegan2 import utils, models

np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@runway.setup(options={'checkpoint': runway.file(extension='.pth')})
def setup(opts):
    global Gs
    if opts['checkpoint'] is None:
      opts['checkpoint'] = '00006000.pth'
    state = torch.load(opts['checkpoint'], map_location=device)
    Gs = models.load(state['Gs'], device)
    Gs.to(device)
    return Gs


generate_inputs = {
    'z': runway.vector(512, sampling_std=0.5),
    'truncation': runway.number(min=0, max=1, default=0.8, step=0.01)
}

@runway.command('generate', inputs=generate_inputs, outputs={'image': runway.image})
def convert(model, inputs):
    z = inputs['z']
    truncation = inputs['truncation']
    latents = torch.from_numpy(z.reshape((1, 512))).to(device=device, dtype=torch.float32)
    Gs.set_truncation(truncation_psi=truncation)
    generated = Gs(latents=latents)
    images = utils.tensor_to_PIL(generated)
    # print(images.shape)
    return {'image': images[0]}


if __name__ == '__main__':
    runway.run()