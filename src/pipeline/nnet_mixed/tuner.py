import collections
import shutil
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def gpu_accelerated(*tensors):
    if not torch.cuda.is_available():
        return tensors

    return tuple(t.cuda() for t in tensors)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=20):
        self.reset(window_size)

    def reset(self, window_size):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window = collections.deque([], window_size)

    @property
    def mavg(self):
        return np.mean(self.window)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.window.append(self.val)


class Tuner:
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 epochs=200,
                 early_stopping=None,
                 tag=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.start_epoch = 0
        self.best_score = -float('Inf')
        self.tag = tag

    def restore_checkpoint(self, checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))

        checkpoint = torch.load(checkpoint_file)
        self.start_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file,
                                                            checkpoint['epoch']))

    def save_checkpoint(self, validation_score, epoch):
        checkpoint_filename = ('checkpoint.pth.tar' if not self.tag else
                               'checkpoint_{}.pth.tar'.format(self.tag))
        best_model_filename = ('model_best.pth.tar' if not self.tag else
                               'model_best_{}.pth.tar'.format(self.tag))

        is_best = validation_score > self.best_score
        self.best_score = max(validation_score, self.best_score)

        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_score': self.best_score,
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state, checkpoint_filename)
        if is_best:
            shutil.copyfile(checkpoint_filename, best_model_filename)

    def run(self, train_loader, val_loader):
        self.train_nnet(train_loader, val_loader)

    def train_nnet(self, train_loader, val_loader):

        scheduler = ReduceLROnPlateau(
            self.optimizer,
            'min',
            threshold_mode='rel',
            threshold=0.002,
            patience=3,
            min_lr=1e-7,
            verbose=True,
        )

        for epoch in range(self.start_epoch, self.epochs):
            self.train_epoch(train_loader, epoch)
            val_score = self.validate(val_loader, epoch)

            scheduler.step(val_score)

            if self.early_stopping:
                if self.early_stopping.should_trigger(
                        epoch,
                        val_score,
                ):
                    break

            self.save_checkpoint(val_score, epoch)

    def compute_batch(self, inputs, target, optimizer=None):
        end = time.time()

        if optimizer:
            optimizer.zero_grad()

        inputs, target = gpu_accelerated(inputs, target)
        output = self.model(inputs)
        loss = self.criterion(output, target)

        if optimizer:
            loss.backward()
            optimizer.step()

        loss_value = loss.data.item()
        batch_time = time.time() - end
        return loss_value, batch_time

    def train_epoch(self, train_loader, epoch):
        loss_value_meter = AverageMeter()
        batch_time_meter = AverageMeter()

        self.model.train()

        tq = tqdm(total=len(train_loader))
        tq.set_description('{:16}'.format(f'Epoch #{epoch}'))

        for inputs, target in train_loader:
            loss_value, batch_time = self.compute_batch(inputs, target, self.optimizer)

            loss_value_meter.update(loss_value)
            batch_time_meter.update(batch_time)

            tq.set_postfix(
                batch_time='{:.3f}'.format(batch_time_meter.mavg),
                loss='{:.3f}'.format(loss_value_meter.mavg),
            )
            tq.update()

        tq.close()

    def validate(self, val_loader, epoch):
        loss_value_meter = AverageMeter()
        batch_time_meter = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        tq = tqdm(total=len(val_loader))
        tq.set_description('{:16}'.format(f'Validating #{epoch}'))

        for inputs, target in val_loader:
            loss_value, batch_time = self.compute_batch(inputs, target)

            loss_value_meter.update(loss_value)
            batch_time_meter.update(batch_time)

            tq.set_postfix(
                batch_time='{:.3f}'.format(batch_time_meter.mavg),
                loss='{:.3f}'.format(loss_value_meter.mavg),
            )
            tq.update()

        tq.close()

        print(f'Validation results (avg): loss = {loss_value_meter.avg:.3f}\n')
        return loss_value_meter.avg
