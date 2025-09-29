import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class DRAGO(Optimizer):

    def __init__(self, params,  data_len, batch_size, lr=required, weight_decay=1e-3, freq=10, pi_decay=1e-3, eps=np.finfo(np.float32).eps, pi_lr=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, freq=freq)
        self.counter = 0
        self.counter2 = 0
        self.flag = False
        self.freq = freq
        if pi_lr is None:
            self.pi_lr = lr
        else:
            self.pi_lr = pi_lr
        self.weight_decay = weight_decay
        self.loss_scale = data_len / batch_size
        self.pi = torch.ones(data_len, requires_grad=False) / data_len
        self.pi_reg = torch.tensor(1. / data_len, requires_grad=False)
        self.pi_decay = pi_decay
        self.__eps = eps
        self.pi_buf1 = torch.zeros_like(self.pi, requires_grad=False)
        self.pi_buf2 = torch.zeros_like(self.pi, requires_grad=False)

        super(DRAGO, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DRAGO, self).__setstate__(state)

    def __move_pi_to_device(self, device):
        self.pi = self.pi.to(device)
        self.pi_reg = self.pi_reg.to(device)
        self.pi_buf1 = self.pi_buf1.to(device)
        self.pi_buf2 = self.pi_buf2.to(device)
    
    @torch.no_grad()
    def __update_pi(self, losses):
        pi_grad = -losses
        pi_new_log = torch.log(self.pi + self.__eps) + self.pi_decay * self.pi_lr * torch.log(self.pi_reg) - self.pi_lr * pi_grad
        pi_new_log /= 1 + self.pi_decay * self.pi_lr
        self.pi = torch.nn.functional.softmax(pi_new_log, dim=-1)

    def step(self, closure, dataset_indexes):
        self.__move_pi_to_device(closure.device)
        pi_selected = self.pi[dataset_indexes]
        losses, loss = closure(pi_selected, self.loss_scale)

        if self.counter == self.freq:
            self.pi_buf1 = torch.zeros_like(self.pi_buf1, requires_grad=False)
            self.pi_buf1[dataset_indexes] = losses
            self.pi_buf2 = torch.zeros_like(self.pi_buf2, requires_grad=False)
        if self.counter2 == 1:
            self.pi_buf2[dataset_indexes] += losses
        if self.counter != self.freq and self.flag != False:
            losses_full = torch.zeros_like(self.pi, requires_grad=False)
            losses_full[dataset_indexes] = losses
            self.__update_pi(losses_full - self.pi_buf2 + self.pi_buf1)


        for group in self.param_groups:
            freq = group['freq']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                
                if 'large_batch' not in param_state:
                    buf = param_state['large_batch'] = torch.zeros_like(p.data)
                    buf.add_(d_p)
                    buf2 = param_state['small_batch'] = torch.zeros_like(p.data)

                buf = param_state['large_batch']
                buf2 = param_state['small_batch']

                if self.counter == freq:
                    buf.data = d_p.clone()
                    temp = torch.zeros_like(p.data)
                    buf2.data = temp.clone()
                    
                if self.counter2 == 1:
                    buf2.data.add_(d_p)
                if self.counter != freq and self.flag != False:
                    p.data.add_(-group['lr'], (d_p - buf2 + buf))


        self.flag = True
        
        if self.counter == freq:
            self.counter = 0
            self.counter2 = 0

        self.counter += 1    
        self.counter2 += 1

        return loss


class DRAGOForClasses(Optimizer):

    def __init__(self, params,  n_classes, batch_size, lr=required, weight_decay=1e-3, freq=10, pi_decay=1e-3, eps=np.finfo(np.float32).eps, pi_lr=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, freq=freq)
        self.counter = 0
        self.counter2 = 0
        self.flag = False
        self.freq = freq
        if pi_lr is None:
            self.pi_lr = lr
        else:
            self.pi_lr = pi_lr
        self.weight_decay = weight_decay
        self.loss_scale = n_classes / batch_size
        self.pi = torch.ones(n_classes, requires_grad=False) / n_classes
        self.pi_reg = torch.tensor(1. / n_classes, requires_grad=False)
        self.pi_decay = pi_decay
        self.__eps = eps
        self.pi_buf1 = torch.zeros_like(self.pi, requires_grad=False)
        self.pi_buf2 = torch.zeros_like(self.pi, requires_grad=False)

        super(DRAGOForClasses, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DRAGOForClasses, self).__setstate__(state)

    def __move_pi_to_device(self, device):
        self.pi = self.pi.to(device)
        self.pi_reg = self.pi_reg.to(device)
        self.pi_buf1 = self.pi_buf1.to(device)
        self.pi_buf2 = self.pi_buf2.to(device)
    
    @torch.no_grad()
    def __update_pi(self, losses):
        pi_grad = -losses
        pi_new_log = torch.log(self.pi + self.__eps) + self.pi_decay * self.pi_lr * torch.log(self.pi_reg) - self.pi_lr * pi_grad
        pi_new_log /= 1 + self.pi_decay * self.pi_lr
        self.pi = torch.nn.functional.softmax(pi_new_log, dim=-1)

    def step(self, closure, classes):
        self.__move_pi_to_device(closure.device)
        pi_selected = self.pi[classes]
        losses, loss = closure(pi_selected, self.loss_scale)

        if self.counter == self.freq:
            self.pi_buf1 = torch.zeros_like(self.pi_buf1, requires_grad=False)
            self.index_add_(0, classes, losses)
            self.pi_buf2 = torch.zeros_like(self.pi_buf2, requires_grad=False)
        if self.counter2 == 1:
            self.index_add_(0, classes, losses)
        if self.counter != self.freq and self.flag != False:
            losses_full = torch.zeros_like(self.pi, requires_grad=False)
            losses_full.index_add_(0, classes, losses)
            self.__update_pi(losses_full - self.pi_buf2 + self.pi_buf1)


        for group in self.param_groups:
            freq = group['freq']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                
                if 'large_batch' not in param_state:
                    buf = param_state['large_batch'] = torch.zeros_like(p.data)
                    buf.add_(d_p)
                    buf2 = param_state['small_batch'] = torch.zeros_like(p.data)

                buf = param_state['large_batch']
                buf2 = param_state['small_batch']

                if self.counter == freq:
                    buf.data = d_p.clone()
                    temp = torch.zeros_like(p.data)
                    buf2.data = temp.clone()
                    
                if self.counter2 == 1:
                    buf2.data.add_(d_p)
                if self.counter != freq and self.flag != False:
                    p.data.add_(-group['lr'], (d_p - buf2 + buf))


        self.flag = True
        
        if self.counter == freq:
            self.counter = 0
            self.counter2 = 0

        self.counter += 1    
        self.counter2 += 1

        return loss
