from model import Generator, Discriminator
from dataset import AnimeDataset
from losses import *
import os
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

class SAGAN:
    def __init__(self, args):
        self.args = args

        self.gen_model = Generator(args.channels, args.image_size, args.latent_dim, args.ngf)
        self.dis_model = Discriminator(args.channels, args.image_size, args.ndf)
        self.gen_opt = torch.optim.Adam(self.gen_model.parameters(), lr = args.gen_lr, betas = (args.beta1, args.beta2), weight_decay = args.weight_decay)
        self.dis_opt = torch.optim.Adam(self.dis_model.parameters(), lr = args.dis_lr, betas = (args.beta1, args.beta2), weight_decay = args.weight_decay)
        self.anime_dataset = AnimeDataset(args.base_image_path)
        self.train_loader = DataLoader(self.anime_dataset, batch_size = args.batch_size, shuffle = True, drop_last = False)
    
    def train_one_epoch(self, epoch):
        self.gen_model.train()
        self.dis_model.train()
        
        print('[INFO] Epoch:', epoch)

        pbar = tqdm(self.train_loader, total = len(self.train_loader))
        acc_d_loss, acc_g_loss = 0, 0
        for images in pbar:
            images = images.to(self.args.device).float()
            latents = torch.randn(self.args.batch_size, self.args.latent_dim, 1, 1, device = self.args.device).float()
            fake_images = self.gen_model(latents)
            dis_real = self.dis_model(images)
            dis_fake = self.dis_model(fake_images.detach())

            self.dis_opt.zero_grad()
            dis_loss = dis_hinge_loss(dis_fake, dis_real)
            dis_loss.backward()
            self.dis_opt.step()

            latents = torch.randn(self.args.batch_size, self.args.latent_dim, 1, 1, device = self.args.device).float()
            fake_images = self.gen_model(latents)
            dis_fake = self.dis_model(fake_images)

            self.gen_opt.zero_grad()
            gen_loss = gen_hinge_loss(dis_fake)
            gen_loss.backward()
            self.gen_opt.step()

            acc_d_loss = acc_d_loss * self.args.loss_smooth + dis_loss.detach().cpu().item() * (1 - self.args.loss_smooth)
            acc_g_loss = acc_g_loss * self.args.loss_smooth * gen_loss.detach().cpu().item() * (1 - self.args.loss_smooth)

    def visualize(self, num_samples = 20):
        self.gen_model.eval()
        self.dis_model.eval()

        latents = torch.randn(num_samples, self.args.latent_dim, 1, 1)
        fake_images = self.gen_model(latents).detach().cpu().numpy()
        fake_images = fake_images * .5 + .5
        fake_images = np.transpose(fake_images, (0, 2, 3, 1))
        plt.figure(figsize = (10, 10))
        for i in range(num_samples):
            plt.subplot(4, num_samples // 4, i + 1)
            plt.imshow(fake_images[i])
        plt.savefig(str(time()) + '.jpg')

    def save_checkpoints(self, epoch):
        if not os.path.exists(self.args.checkpoints_path):
            os.mkdir(self.args.checkpoints_path)
        torch.save({
            'gen_model': self.gen_model.state_dict(),
            'dis_model': self.dis_model.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'dis_opt': self.dis_model.state_dict()
        }, f'{self.args.checkpoints_path}/epoch_{epoch}_{time()}.tar')

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch + 1)
            self.visualize()
            if epoch == 0 or (epoch + 1) % self.args.checkpoint_step:
                self.save_checkpoints(epoch + 1)