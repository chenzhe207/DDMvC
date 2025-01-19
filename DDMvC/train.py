import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
from function import initialize_D
from function import contrastive_learning
from function import pretrain
import os



# MNIST-USPS
# Fashion
# Caltech-5V
# NUSWIDE
# YouTubeVideo
Dataname = 'Caltech-5V'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--low_feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--lambda_1", default=1)
parser.add_argument("--lambda_2", default=1)
parser.add_argument("--show_interval", default=200)
parser.add_argument("--dict_size", default=5)
args = parser.parse_args()

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not found!")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"





def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if not os.path.exists('./models'):
    os.makedirs('./models')

dataset, dims, view, data_size, class_num = load_data(args.dataset)





for args.mse_epochs in [200]:
    for args.con_epochs in [200]:
        for args.dict_size in [5]:
            for args.lambda1 in [1e-3]:  # Q
                for args.lambda2 in [10]:  # # D#

                    setup_seed(5)
                    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                                              drop_last=False)
                    data_loader_all = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=True,
                                                                  drop_last=False)
                    model = Network(view, dims, args.low_feature_dim, args.high_feature_dim, class_num, args.dict_size)
                    # print(model)
                    model = model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
                    zeros = torch.zeros(args.batch_size, args.batch_size).to(device)

                    # pretrain
                    epoch = 1
                    while epoch <= args.mse_epochs:
                        pretrain(epoch, model, view, device, data_loader, optimizer)
                        epoch += 1

                    # initialize D

                    D = initialize_D(model, view, class_num, args.high_feature_dim,
                                     args.dict_size, device, data_loader_all)

                    model.D.data = D.to(device)


                    losss = []
                    losss1 = []
                    losss2 = []


                    while epoch <= args.mse_epochs + args.con_epochs:
                        loss, loss1, loss2 = contrastive_learning(epoch, model, view, class_num, args.dict_size,
                                                                  args.lambda1, args.lambda2, criterion, device,
                                                                  data_loader, optimizer)
                        losss.append(loss)
                        losss1.append(loss1)
                        losss2.append(loss2)
                        if (epoch - args.mse_epochs) % args.show_interval == 0:
                            acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, epoch)

                        epoch += 1



