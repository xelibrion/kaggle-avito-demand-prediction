import collections
import json
import os
import shutil
import time
from datetime import datetime

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


class Emitter:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, event):
        with open(self.path, 'a') as out_file:
            event.update({'timestamp': datetime.utcnow().isoformat()})
            out_file.write(json.dumps(event))
            out_file.write('\n')


class Tuner:
    def __init__(self,
                 model,
                 criterion,
                 bootstrap_optimizer,
                 optimizer,
                 bootstrap_epochs=1,
                 epochs=200,
                 early_stopping=None,
                 tag=None):
        self.model = model
        self.criterion = criterion
        self.bootstrap_optimizer = bootstrap_optimizer
        self.optimizer = optimizer
        self.bootstrap_epochs = bootstrap_epochs
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.start_epoch = 0
        self.best_score = -float('Inf')
        self.tag = tag
        self.emit = Emitter('./logs/events.json'
                            if not tag else './logs/events_{}.json'.format(tag))

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
        self.bootstrap(train_loader, val_loader)
        self.train_nnet(train_loader, val_loader)

    def train_nnet(self, train_loader, val_loader):

        scheduler = ReduceLROnPlateau(
            self.optimizer,
            'max',
            threshold_mode='rel',
            threshold=0.002,
            patience=3,
            min_lr=1e-7,
            verbose=True,
        )

        for epoch in range(self.start_epoch, self.epochs):
            self.train_epoch(train_loader, self.optimizer, epoch, 'training',
                             'Epoch #{epoch}')

            val_score = self.validate(val_loader, epoch, 'validation',
                                      'Validating #{epoch}')

            scheduler.step(val_score)

            if self.early_stopping:
                if self.early_stopping.should_trigger(
                        epoch,
                        val_score,
                ):
                    break

            self.save_checkpoint(val_score, epoch)

    def bootstrap(self, train_loader, val_loader):
        if self.start_epoch:
            return

        for epoch in range(self.bootstrap_epochs):
            self.train_epoch(train_loader, self.bootstrap_optimizer, epoch, 'bootstrap',
                             'Bootstrapping #{epoch}')
            self.validate(val_loader, epoch, 'bootstrap-val', 'Validating #{epoch}')

    def train_epoch(self, train_loader, optimizer, epoch, stage, format_str):
        batch_time = AverageMeter()
        losses = AverageMeter()

        self.model.train()

        tq = tqdm(total=len(train_loader))
        description = format_str.format(**locals())
        tq.set_description('{:16}'.format(description))

        batch_idx = -1
        end = time.time()
        for _, (inputs, target) in enumerate(train_loader):
            batch_idx += 1

            inputs, target = gpu_accelerated(inputs, target)

            output = self.model(inputs)
            loss = self.criterion(output, target)

            batch_size = inputs.size(0)
            losses.update(loss.data.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)

            tq.set_postfix(
                batch_time='{:.3f}'.format(batch_time.mavg),
                loss='{:.3f}'.format(losses.mavg),
            )
            tq.update()

            self.emit({
                'stage': stage,
                'epoch': epoch,
                'batch': batch_idx,
                'loss': losses.val
            })

            end = time.time()

        tq.close()

    def validate(self, val_loader, epoch, stage, format_str):
        batch_time = AverageMeter()
        losses = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        tq = tqdm(total=len(val_loader))
        description = format_str.format(**locals())
        tq.set_description('{:16}'.format(description))

        batch_idx = -1
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            batch_idx += 1

            inputs, target = gpu_accelerated(inputs, target)

            output = self.model(inputs)
            loss = self.criterion(output, target)

            batch_size = inputs.size(0)
            losses.update(loss.data.item(), batch_size)

            batch_time.update(time.time() - end)

            tq.set_postfix(
                batch_time='{:.3f}'.format(batch_time.mavg),
                loss='{:.3f}'.format(losses.mavg),
            )
            tq.update()

            self.emit({
                'stage': stage,
                'epoch': epoch,
                'batch': batch_idx,
                'loss': losses.val
            })
            end = time.time()

        tq.close()

        print(f'Validation results (avg): loss = {losses.avg:.3f}\n')
        return losses.avg
