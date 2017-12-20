import math
#from ..torch_imports import *
import torch
from torch.optim import Optimizer


class AMSGradW(Optimizer):
    """Implements AMSGrad with fixed weight decay (based on fast.ai implementation of AdamW)"""

    def __init__(self, params, lr=0.0005, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AMSGradW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # to preserve maximal second moments
                exp_avg_sq_old = exp_avg_sq.clone()

                # for Weight Decay
                data_old = p.data.clone()

                # Decay the first and second moment running average coefficient
                # m_t = B1 * m_t-1 + (1 - B1) * grad
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t = B2 * v_t-1 + (1 - B2) * grad^^2
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Apply bias_correction1
                step_size = group['lr'] / bias_correction1

                # Apply bias_correction2 and take the max values
                # v_t_hat = v_t / bias_correction2
                exp_avg_sq.div(bias_correction2)

                # Keep the maximum of current and past squared gradients (main feature of AMSGrad)
                # v_t_hat = max(v_t_hat, v_t-1_hat)
                exp_avg_sq = torch.max(exp_avg_sq, exp_avg_sq_old)

                # denominator = sqrt(v_t_hat) + epsilon
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                # Change weights
                p.data.addcdiv_(-step_size, exp_avg, denom)

                # group['weight_decay'] is externally decayed
                p.data = p.data.add(-group['weight_decay'], data_old)

        return loss
