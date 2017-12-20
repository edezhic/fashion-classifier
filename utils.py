import torch
import torch.nn.init as init
from torch.nn.modules.module import _addindent
import numpy as np
import random
import math
import notify2

def notify(message):

    notify2.init("Experiment Status")
    # Создаем Notification-объект
    n = notify2.Notification("Experiment Status")
    # Устанавливаем уровень срочности
    n.set_urgency(notify2.URGENCY_NORMAL)
    # Устанавливаем задержку
    n.set_timeout(1000)
    # Обновляем содержимое
    n.update("Status: ", message)
    # Показываем уведомление
    n.show()

def summarize_model(model, show_weights=False, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    total_params = 0
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = summarize_model(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(str(modstr), 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        total_params += params
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'

    print (tmpstr)
    print("Total learnable parameters number: {}".format(total_params))
    #return tmpstr


def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight, gain=init.calculate_gain('relu'))
        #init.xavier_normal(m.weight, gain=init.calculate_gain('relu'))
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1 or classname.find('BatchReNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

    elif classname.find('Linear') != -1:
        init.kaiming_uniform(m.weight, mode='fan_out')
        #init.kaiming_normal(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant(m.bias, 0)

def group_weights_by_weight_decay(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif classname.find('Conv') != -1:
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif classname.find('BatchNorm') != -1 or classname.find('BatchReNorm') != -1:
            group_no_decay.append(m.weight)
            group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

def one_hot(batch, depth):
    # seq_batch.size() should be [seq,batch] or [batch,]
    # return size() would be [seq,batch,depth] or [batch,depth]
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,batch)
