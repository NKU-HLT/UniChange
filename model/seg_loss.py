import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):
   
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='mean')

    def forward(self, inputs, targets):
        
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    
    pos = (truth>0.5).float()
    neg = (truth<0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos*pos*loss/pos_num + weight_neg*neg*loss/neg_num).sum()

    return loss


class ChangeSimilarity(nn.Module):
    
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)
        
    def forward(self, x1, x2, label_change):
    
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x1 = torch.reshape(x1,[b*h*w,c])
        x2 = torch.reshape(x2,[b*h*w,c])
        
        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target,[b*h*w])
        
        loss = self.loss_f(x1, x2, target)
        return loss


class ChangeSalience(nn.Module):
  
    def __init__(self, reduction='mean'):
        super(ChangeSalience, self).__init__()
        self.loss_f = nn.MSELoss(reduction=reduction)
        
    def forward(self, x1, x2, label_change):
   
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)[:,0,:,:]  
        x2 = F.softmax(x2, dim=1)[:,0,:,:]  
                
        loss = self.loss_f(x1, x2.detach()) + self.loss_f(x2, x1.detach())
        return loss*0.5


def create_loss_functions():
    
    
    criterion = CrossEntropyLoss2d(ignore_index=0)
    
    
    criterion_sc = ChangeSimilarity()
    
    return criterion, criterion_sc



