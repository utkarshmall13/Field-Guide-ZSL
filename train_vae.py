# execute this function to train and test the vae-model
# Modified code from https://github.com/edgarschnfld/CADA-VAE-PyTorch

from vaemodel import Model
import torch
from os.path import join, isdir
from os import mkdir
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument('--dataset')
parser.add_argument('--mode', default='')
parser.add_argument('--discrete', default=False, action='store_true')
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--fraction', type=float, default=1.0)
parser.add_argument('--test_mode', default='standard')
parser.add_argument('--train_mode', default='standard')
args = parser.parse_args()


if 'pt' in args.mode:
    lr = 0.0001
else:
    lr = 0.001

########################################
# the basic hyperparameters
########################################
hyperparameters = {
    'num_shots': 0,
    'device': 'cuda',
    'model_specifics': {
        'cross_reconstruction': True,
        'name': 'CADA',
        'distance': 'wasserstein',
        'warmup': {
            'beta': {'factor': 0.25, 'end_epoch': 93, 'start_epoch': 0},
            'cross_reconstruction': {'factor': 2.37, 'end_epoch': 75, 'start_epoch': 21},
            'distance': {'factor': 8.13, 'end_epoch': 22, 'start_epoch': 6}}},

    'lr_gen_model': 0.00015,
    'generalized': True,
    'batch_size': 50,
    'xyu_samples_per_class': {'SUN': (200, 0, 400, 0),
                              'APY': (200, 0, 400, 0),
                              'CUB': (200, 0, 400, 0),
                              'AWA2': (200, 0, 400, 0),
                              'FLO': (200, 0, 400, 0),
                              'AWA1': (200, 0, 400, 0)},
    'epochs': 100,
    'loss': 'l1',
    'auxiliary_data_source': 'attributes',
    'lr_cls': lr,
    'dataset': 'CUB',
    'hidden_size_rule': {'resnet_features': (1560, 1660), 'attributes': (1450, 665), 'sentences': (1450, 665)},
    'latent_size': 64,
    'discrete': args.discrete,
    'run': args.run,
    'mode': args.mode
}

# The training epochs for the final classifier, for early stopping,
# as determined on the validation spit

cls_train_steps = [
    {'dataset': 'SUN', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 21},
    {'dataset': 'SUN', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 30},
    {'dataset': 'SUN', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 22},
    {'dataset': 'SUN', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
    {'dataset': 'SUN', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 29},
    {'dataset': 'SUN', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 78},
    {'dataset': 'SUN', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 29},
    {'dataset': 'SUN', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 61},
    {'dataset': 'SUN', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 79},
    {'dataset': 'SUN', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 94},
    {'dataset': 'AWA1', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 33},
    {'dataset': 'AWA1', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 25},
    {'dataset': 'AWA1', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 40},
    {'dataset': 'AWA1', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 81},
    {'dataset': 'AWA1', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 89},
    {'dataset': 'AWA1', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 62},
    {'dataset': 'AWA1', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 56},
    {'dataset': 'AWA1', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 59},
    {'dataset': 'AWA1', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 100},
    {'dataset': 'AWA1', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 50},
    {'dataset': 'CUB', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 50},
    {'dataset': 'CUB', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 50},
    {'dataset': 'CUB', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
    {'dataset': 'CUB', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
    {'dataset': 'CUB', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
    {'dataset': 'CUB', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
    {'dataset': 'CUB', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
    {'dataset': 'CUB', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
    {'dataset': 'CUB', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
    {'dataset': 'CUB', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 67},
    {'dataset': 'AWA2', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 29},
    {'dataset': 'AWA2', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 39},
    {'dataset': 'AWA2', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 44},
    {'dataset': 'AWA2', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
    {'dataset': 'AWA2', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 99},
    {'dataset': 'AWA2', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 100},
    {'dataset': 'AWA2', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 69},
    {'dataset': 'AWA2', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 79},
    {'dataset': 'AWA2', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 86},
    {'dataset': 'AWA2', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 78},
    {'dataset': 'APY', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 78},
    {'dataset': 'APY', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 78}
]

##################################
# change some hyperparameters here
##################################
hyperparameters['dataset'] = args.dataset
hyperparameters['fraction'] = args.fraction
hyperparameters['test_mode']= args.test_mode
hyperparameters['train_mode']= args.train_mode

if args.train_mode=='missingatts':
    hyperparameters['epochs']=200

hyperparameters['cls_train_steps'] = [
    x['cls_train_steps'] for x in cls_train_steps if all([hyperparameters['dataset']==x['dataset'], hyperparameters['num_shots']==x['num_shots'], hyperparameters['generalized']==x['generalized']])][0]

print('***')
if hyperparameters['generalized']:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {
            'CUB': (200, 0, 400, 0), 'SUN': (200, 0, 400, 0),
            'APY': (200, 0, 400, 0), 'AWA1': (200, 0, 400, 0),
            'AWA2': (200, 0, 400, 0), 'FLO': (200, 0, 400, 0)
        }
    else:
        hyperparameters['samples_per_class'] = {
            'CUB': (200, 0, 200, 200), 'SUN': (200, 0, 200, 200),
            'APY': (200, 0, 200, 200), 'AWA1': (200, 0, 200, 200),
            'AWA2': (200, 0, 200, 200), 'FLO': (200, 0, 200, 200)
        }
else:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {
            'CUB': (0, 0, 200, 0), 'SUN': (0, 0, 200, 0),
            'APY': (0, 0, 200, 0), 'AWA1': (0, 0, 200, 0),
            'AWA2': (0, 0, 200, 0), 'FLO': (0, 0, 200, 0)
        }
    else:
        hyperparameters['samples_per_class'] = {
            'CUB': (0, 0, 200, 200), 'SUN': (0, 0, 200, 200),
            'APY': (0, 0, 200, 200), 'AWA1': (0, 0, 200, 200),
            'AWA2': (0, 0, 200, 200), 'FLO': (0, 0, 200, 200)
        }


model = Model(hyperparameters)
model.to(hyperparameters['device'])

losses = model.train_vae()
state = {
    'state_dict': model.state_dict(),
    'hyperparameters': hyperparameters,
    'encoder': {},
    'decoder': {}
}

for d in model.all_data_sources:
    state['encoder'][d] = model.encoder[d].state_dict()
    state['decoder'][d] = model.decoder[d].state_dict()


model_dir = 'models'
if args.train_mode=='standard':
    if not isdir(model_dir):
        mkdir(model_dir)
    torch.save(state, join(model_dir, 'CADA_trained_'+args.dataset+'_'+args.mode+'_'+str(args.discrete)+'_'+str(args.run)+'_'+str(args.fraction)+'.pth.tar'))
else:
    if not isdir(model_dir):
        mkdir(model_dir)
    torch.save(state, join(model_dir, 'CADA_trained_'+args.dataset+'_'+args.mode+'_'+args.train_mode+'_'+str(args.discrete)+'_'+str(args.run)+'_'+str(args.fraction)+'.pth.tar'))
print('>> Model saved')
