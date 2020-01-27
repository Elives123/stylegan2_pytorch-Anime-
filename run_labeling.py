import warnings
import argparse
import os
from PIL import Image
import numpy as np
import torch
import cntk
import torch.nn.functional as F
import pickle

import stylegan2
from stylegan2 import utils

#----------------------------------------------------------------------------

_description = """StyleGAN2 labeling.
Run 'python %(prog)s <subcommand> --help' for subcommand help."""

#----------------------------------------------------------------------------

_examples = """examples:
  # Train a network or convert a pretrained one.
  # Example of converting pretrained ffhq model:
  python run_convert_from_tf --download ffhq-config-f --output G.pth D.pth Gs.pth

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate_images --network=Gs.pth --seeds=6600-6625 --truncation_psi=0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate_images --network=Gs.pth --seeds=66,230,389,1518 --truncation_psi=1.0

  # Example of converting pretrained car model:
  python run_convert_from_tf --download car-config-f --output G_car.pth D_car.pth Gs_car.pth

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate_images --network=Gs_car.pth --seeds=6000-6025 --truncation_psi=0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style_mixing_example --network=Gs.pth --row_seeds=85,100,75,458,1500 --col_seeds=55,821,1789,293 --truncation_psi=1.0
"""

#----------------------------------------------------------------------------

## only takes these tags
colors = ['aqua', 'black', 'blue', 'brown',  'green', 'grey', 'lavender', 'light_brown', 'multicolored', 'orange', 'pink', 'purple', 'red', 'silver', 'white', 'yellow']
switches = ['open', 'closed', 'covered']
adjs = ['glowing', 'gradient', 'reflective', 'ringed', 'rolling', 'rubbing', 'shading', 'sparkling']

# generate composition of elements
whitelist = []
components = ['eyes', 'hair']

for component in components:
    whitelist = whitelist + [f'{color}_{component}' for color in colors]
    whitelist = whitelist + [f'{switch}_{component}' for switch in switches]
    whitelist = whitelist + [f'{adj}_{component}' for adj in adjs]

#----------------------------------------------------------------------------

def get_arg_parser():
    parser = argparse.ArgumentParser(
        description=_description,
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--network',
        help='Network file path',
        required=True,
        metavar='FILE'
    )

    parser.add_argument(
        '--label_project',
        help='Labeling Network file path',
        # required=True,
        metavar='FILE'
    )

    parser.add_argument(
        '--output',
        help='Root directory for run results. Default: %(default)s',
        type=str,
        default='./results',
        metavar='DIR'
    )

    parser.add_argument(
        '--pixel_min',
        help='Minumum of the value range of pixels in generated images. ' + \
            'Default: %(default)s',
        default=-1,
        type=float,
        metavar='VALUE'
    )

    parser.add_argument(
        '--pixel_max',
        help='Maximum of the value range of pixels in generated images. ' + \
            'Default: %(default)s',
        default=1,
        type=float,
        metavar='VALUE'
    )

    parser.add_argument(
        '--gpu',
        help='CUDA device indices (given as separate ' + \
            'values if multiple, i.e. "--gpu 0 1"). Default: Use CPU',
        type=int,
        default=[],
        nargs='*',
        metavar='INDEX'
    )

    parser.add_argument(
        '--truncation_psi',
        help='Truncation psi. Default: %(default)s',
        type=float,
        default=0.5,
        metavar='VALUE'
    )

    parser.add_argument(
        '--seed',
        help='random seed for generating images.',
        type=int,
        default=0,
        metavar='VALUE'
    )

    parser.add_argument(
        '--iter',
        help='How many images will generated',
        type=int,
        default=10,
        metavar='VALUE'
    )

    return parser

#----------------------------------------------------------------------------

def transform_labels(tags, predicted):
    result = {}
    for i in range(len(tags)):
        tag = tags[i]
        if tag in whitelist:
            result[tag] = predicted[i]
    return result

def run_labeling(G, C, tags, args):
    threshold = 0.5
    latent_size, label_size = G.latent_size, G.label_size
    device = torch.device(args.gpu[0] if args.gpu else 'cpu')
    if device.index is not None:
        torch.cuda.set_device(device.index)
    G.to(device)

    if args.truncation_psi != 1:
      G.set_truncation(truncation_psi=args.truncation_psi)

    if len(args.gpu) > 1:
        warnings.warn(
            'Noise can not be randomized based on the seed ' + \
            'when using more than 1 GPU device. Noise will ' + \
            'now be randomized from default random state.'
        )
        G.random_noise()
        G = torch.nn.DataParallel(G, device_ids=args.gpu)
    else:
        noise_reference = G.static_noise()

    rnd = np.random.RandomState(args.seed)

    noise_tensors = None
    if len(args.gpu) <= 1:
      noise_tensors = [[] for _ in noise_reference]
      for i, ref in enumerate(noise_reference):
          noise_tensors[i].append(torch.from_numpy(rnd.randn(*ref.size()[1:])))
      noise_tensors = [
          torch.stack(noise, dim=0).to(device=device, dtype=torch.float32)
          for noise in noise_tensors
      ]
      G.static_noise(noise_tensors=noise_tensors)

    progress = utils.ProgressWriter(args.iter)
    progress.write('Generating images...', step=False)

    qlatents = []
    dlatents = []
    labels_data = []
    for i in range(0, args.iter):
        qlatent = torch.from_numpy(rnd.randn(latent_size)).reshape(1, latent_size).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            dlatent = G.G_mapping(latents=qlatent)
            dlatent = dlatent.unsqueeze(1).repeat(1, len(G.G_synthesis), 1)
            generated = G.G_synthesis(dlatent)
            images = generated.clamp_(min=0, max=1)
            # 299 is the input size of the model
            images = F.interpolate(images, size=(299, 299), mode='bilinear')
            predicted_labels = C.eval(images[0].cpu().numpy()).reshape(tags.shape[0])  # array of tag score
            # transform labels to dict
            labels = transform_labels(tags, predicted_labels)
            # [image] = utils.tensor_to_PIL(generated, pixel_min=args.pixel_min, pixel_max=args.pixel_max)
            # image.save(os.path.join(args.output, 'seed%05d-resized.png' % i))

            # store the result
            qlatents.append(qlatent)
            dlatents.append(dlatent)
            labels_data.append(labels)

            progress.step()

    out_path = os.path.join(args.output, 'result.pkl')
    with open(out_path, 'wb') as f:
      pickle.dump((qlatent, dlatents, labels_data), f)

    progress.write('Done!', step=False)
    progress.close()

#----------------------------------------------------------------------------

def main():
    args = get_arg_parser().parse_args()
    assert os.path.isdir(args.output) or not os.path.splitext(args.output)[-1], \
        '--output argument should specify a directory, not a file.'
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    G = stylegan2.models.load(args.network)
    assert isinstance(G, stylegan2.models.Generator), 'Model type has to be ' + \
        'stylegan2.models.Generator. Found {}.'.format(type(G))

    tags_path = os.path.join(args.label_project, 'tags.txt')
    model_path = os.path.join(args.label_project, 'model.cntk')

    with open(tags_path, 'r') as tags_stream:
        tag_array = np.array([tag for tag in (tag.strip()
                                              for tag in tags_stream) if tag])

    C = cntk.load_model(model_path)
    
    run_labeling(G, C, tag_array, args)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
