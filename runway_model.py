import numpy as np
import runway
import torch
import stylegan2.models

np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
    global Gs
    state = torch.load(opts['checkpoint'], map_location=device)
    Gs = stylegan2.models.load(state['Gs'], device)
    Gs.to(device)
    Gs.fixed_noise()
    return Gs


generate_inputs = {
    'z': runway.vector(512, sampling_std=0.5),
    'truncation': runway.number(min=0, max=1, default=0.8, step=0.01)
}

@runway.command('generate', inputs=generate_inputs, outputs={'image': runway.image})
def convert(model, inputs):
    z = inputs['z']
    truncation = inputs['truncation']
    latents = z.reshape((1, 512))
    Gs.set_truncation(truncation_psi=truncation)
    images = Gs(latents)
    output = np.clip(images[0], 0, 255).astype(np.uint8)
    return {'image': output}


if __name__ == '__main__':
    runway.run()