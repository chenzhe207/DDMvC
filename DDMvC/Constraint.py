import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class D_constraint1(torch.nn.Module):

    def __init__(self):
        super(D_constraint1, self).__init__()

    def forward(self, d):
        I = torch.eye(d.shape[1]).cuda()
        loss_d1_constraint = torch.norm(torch.mm(d.t(),d) * I - I)
        return loss_d1_constraint

   
class D_constraint2(torch.nn.Module):

    def __init__(self):
        super(D_constraint2, self).__init__()

    def forward(self, d, k, n_clusters):
        S = torch.ones(d.shape[1],d.shape[1]).cuda()
        zero = torch.zeros(k, k)
        for i in range(n_clusters):
            S[i*k:(i+1)*k, i*k:(i+1)*k] = zero
        loss_d2_constraint = torch.norm(torch.mm(d.t(),d) * S)
        return loss_d2_constraint

