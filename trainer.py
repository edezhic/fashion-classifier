import torch
import time
import math
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils import one_hot, group_weights_by_weight_decay, initialize_weights
from amsgradw import AMSGradW

class Trainer(object):
    cuda = torch.cuda.is_available()

    def __init__(self, model, optimizer=None, loss_f=F.cross_entropy,
                lr=0.001, decay_lr=True, weight_decay=0.003,
                grow_batch_size=True, grow_step=5, min_batch_size=16, max_batch_size=128,
                write_log=False, log_dir=None):
        self.model = model
        self.loss_f = loss_f
        self.initial_lr = lr
        self.decay_lr = decay_lr
        self.initial_wd = weight_decay
        self.grow_batch_size = grow_batch_size
        self.write_log = write_log

        self.model.apply(initialize_weights)
        if self.cuda:
            self.model.cuda()

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = AMSGradW(group_weights_by_weight_decay(model), lr=lr, weight_decay=weight_decay)

        if self.grow_batch_size:
            self.grow_batch_size = True
            self.batch_size_grow_step = grow_step
            self.min_batch_size = min_batch_size
            self.max_batch_size = max_batch_size

        if self.write_log:
            self.writer = SummaryWriter(log_dir=log_dir)

    def _loop(self, data_loader, train_mode=True):
        loop_loss = []
        correct = []
        for idx, (data, target) in enumerate(data_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if train_mode:
                self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data[0] / len(data_loader))
            correct.append((output.data.max(1)[1] == target.data).sum() / len(data_loader.dataset))
            if train_mode:
                loss.backward()
                self.optimizer.step()
        return sum(loop_loss), sum(correct) * 100

    def train(self, data_loader):
        self.model.train()
        return self._loop(data_loader)

    def validate(self, data_loader):
        self.model.eval()
        return self._loop(data_loader, train_mode=False)

    def fit(self, epochs, train_data, val_data):
        print("Starting training...")
        val_accuracy_log = []
        total_time = 0

        if self.grow_batch_size:
            train_data.batch_sampler.batch_size = self.min_batch_size

        # Weight Decay normalization according to number of steps per epoch
        for param_group in self.optimizer.param_groups:
            if param_group['weight_decay'] != 0:
                param_group['weight_decay'] /= math.sqrt(len(train_data))

        for epoch in range(epochs):
            start = time.time()
            train_loss, train_acc = self.train(train_data)
            val_loss, val_acc = self.validate(val_data)

            print("Epoch {:2} in {:.0f}s || Train loss = {:.3f}, accuracy = {:.2f} | Validation loss = {:.3f}, accuracy = {:.2f}".format(epoch, time.time() - start, train_loss, train_acc, val_loss, val_acc))

            # decay Learning Rate linearly
            if self.decay_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.initial_lr * (epochs - (epoch+1)) / epochs

            # Weight Decay coefficient normalization according to the number of training steps per epoch
            for param_group in self.optimizer.param_groups:
                if param_group['weight_decay'] != 0:
                        param_group['weight_decay'] = self.initial_wd / math.sqrt(len(train_data))

            # Double batch_size each batch_size_grow_step epochs
            if self.grow_batch_size and epoch % self.batch_size_grow_step == 0 and train_data.batch_sampler.batch_size < self.max_batch_size:
                train_data.batch_sampler.batch_size *= 2

            # Write summary for TensorBoard
            if self.writer:
                self.writer.add_scalar('accuracy/train', train_acc, epoch)
                self.writer.add_scalar('accuracy/validation', val_acc, epoch)
                self.writer.add_scalar('loss/train', train_loss, epoch)
                self.writer.add_scalar('loss/validation', val_loss, epoch)

            total_time += time.time() - start
            val_accuracy_log.append(val_acc)

        print("Done in {:.0f}s with best validation accuracy: {:.2f}".format(total_time, max(val_accuracy_log)))
