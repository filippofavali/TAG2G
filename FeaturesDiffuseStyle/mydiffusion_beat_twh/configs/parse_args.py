import configargparse
import argparse

# edited Favali, 03-01-2024

def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='DiffuseStyleGesture')
    parser.add_argument('--config', default='./configs/DiffuseStyleGesture.yml')
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--no_cuda', type=list, default=['2'])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--use_vqvae_decoder', action='store_true', default=False)
    
    args = parser.parse_args()
    return args
