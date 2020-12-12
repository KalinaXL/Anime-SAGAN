import sys
sys.path.append('../../')
from sagan import Generator
import os
import torch

checkpoint = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint_30.tar'), map_location = 'cpu')
gen = Generator()
gen.load_state_dict(checkpoint['g_model'])
gen = gen.eval()