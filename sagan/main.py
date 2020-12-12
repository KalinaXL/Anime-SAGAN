from train import SAGAN
import argparse

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--channels', default = 3, type = int, help = 'channels of the image')
    ap.add_argument('-s', '--image-size', default = 32, type = int, help = 'size of the image')
    ap.add_argument('-l', '--latent-dim', default = 100, type = int, help = 'latent dim')
    ap.add_argument('--ngf', default = 32, type = int)
    ap.add_argument('--ndf', default = 32, type = int)
    ap.add_argument('--epochs', default = 50, type = int)
    ap.add_argument('--batch-size', default = 64, type = int)
    ap.add_argument('--device', default = 'cpu', type = str)
    ap.add_argument('--base-image-path', type = str, required = True, help = 'path of the training image folder')
    ap.add_argument('--beta1', default = 0.0, type = float)
    ap.add_argument('--beta2', default = 0.9, type = float)
    ap.add_argument('--gen-lr', default = 1e-4, type = float)
    ap.add_argument('--dis-lr', default = 4e-4, type = float)
    ap.add_argument('--weight-decay', default = 1e-6, type = float)
    ap.add_argument('--loss-smooth', default = .9, type = float)
    ap.add_argument('--checkpoints-path', default = 'checkpoints', type = str)

    return ap.parse_args()

def main(args):
    sagan = SAGAN(args)
    sagan.train()

if __name__ == "__main__":
    main(get_args())
