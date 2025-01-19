from collections import defaultdict
import numpy as np
import torch


def seperate(Z, y_pred, n_clusters):
    Z_seperate = defaultdict(list)
    for i in range(n_clusters):
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                Z_seperate[i].append(Z[j].cpu().detach().numpy())
    return Z_seperate

def Initialization_D(Z, y_pred, n_view, n_z, n_clusters, d):
    Z_seperate = seperate(Z, y_pred, n_clusters)
    U = np.zeros([n_view * n_z, n_clusters * d])
    print("Initialize D")
    for i in range(n_clusters):
        Z_seperate[i] = np.array(Z_seperate[i])
        u, ss, v = np.linalg. svd(Z_seperate[i].transpose())
        U[:,i*d:(i+1)*d] = u[:,0:d]
    D = U
    return D
 
