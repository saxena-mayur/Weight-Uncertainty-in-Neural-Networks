import torch


class MNISTTransform(object):

    def __init__(self, mean=0.1307, std=0.3081):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = torch.from_numpy(data)
        tensor = tensor.float().div(126)
        #tensor.sub_(self.mean).div_(self.std)
        return tensor
