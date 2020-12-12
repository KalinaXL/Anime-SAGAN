import torch
import torch.nn.functional as F

def dis_hinge_loss(dis_fake, dis_real):
    return torch.mean(F.relu(1 - dis_real)) + torch.mean(F.relu(1 + dis_fake))
def gen_hinge_loss(dis_fake):
    return -torch.mean(dis_fake)