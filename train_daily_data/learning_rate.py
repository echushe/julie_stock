import torch.optim as optim
import random

from torch.optim.lr_scheduler import _LRScheduler

class RandomLRSchedulerV1(_LRScheduler):
    def __init__(self, optimizer, min_lr=1e-7, max_lr=1e-4, last_epoch=-1):
        self.min_lr = min_lr
        self.max_lr = max_lr

        # generate learning rates between min_lr and max_lr
        lr_list = []
        lr = self.min_lr
        max_min = self.max_lr / self.min_lr
        for i in range(100):
            lr_list.append(lr * (max_min ** (i / 99.0)))
        self.lr_list = lr_list

        super(RandomLRSchedulerV1, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # randomly select a learning rate from the list
        random_lr = random.choice(self.lr_list)
        return [random_lr for _ in self.optimizer.param_groups]