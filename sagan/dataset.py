import glob
import os
from functools import reduce
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset

class AnimeDataset(Dataset):
    def __init__(self, base_path, image_exts = ['.jpg'], is_cached = False):
        self.image_paths = reduce(lambda acc, ext: acc + glob.glob(os.path.join(base_path, '*' + ext)), image_exts, [])
        self.cached = {}
        self.is_cached = is_cached
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        path = self.image_paths[index]
        if self.is_cached and path in self.cached:
            return self.cached[path]
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image / 127.5 - 1.
        image = np.transpose(image, (2, 0, 1))
        if self.is_cached:
            self.cached[path] = image
        return image
