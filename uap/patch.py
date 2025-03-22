import numpy as np
import torch

class PatchManager:
    def __init__(self, config, device):
        self.device = device
        self.patch = None
        self.config = config
        self.patch_file = config['patch_file']
        self.init(patch_file=self.patch_file)

    def init(self, patch_file=None):
        if patch_file is None:
            self.generate()
        else:
            self.read(patch_file)
        self.patch.requires_grad = True

    def read(self, patch_file):
        print('Reading patch from file: ' + patch_file)
        if patch_file.endswith('.pth'):
            patch = torch.load(patch_file, map_location=self.device)
            # patch.new_tensor(patch)
            print(patch.shape, patch.requires_grad, patch.is_leaf)
        self.patch = patch.to(self.device)

    def generate(self, init_mode='random'):
        channel = self.config['channel']
        height = self.config['height']
        width = self.config['width']
        if init_mode.lower() == 'random':
            print('Random initializing a universal patch')
            patch = torch.rand((channel, height, width))
        self.patch = patch.to(self.device)

    def total_variation(self):
        adv_patch = self.patch[0]
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)

    def update_(self, patch_new):
        del self.patch
        self.patch = patch_new.detach()
        self.patch.requires_grad = True

    def save(self, save_file):
        torch.save(self.patch, save_file)

    def freeze(self):
        self.patch.requires_grad = False
        
    @torch.no_grad()
    def clamp_(self, p_min=0, p_max=1):
        torch.clamp_(self.patch, min=p_min, max=p_max)