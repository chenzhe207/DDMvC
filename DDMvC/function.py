import torch
from InitializeD import Initialization_D
from sklearn.cluster import KMeans
from Constraint import D_constraint1, D_constraint2

def refined_subspace_affinity(s):
    weight = s**2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain(epoch, model, view, device, data_loader, optimizer):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def initialize_D(model, view, class_num, high_feature_dim, dict_size, device, data_loader_all):

    for batch_idx, (xs, _, _) in enumerate(data_loader_all):
        for v in range(view):
            xs[v] = xs[v].to(device)
    _, fs = model(xs)
    F = torch.cat(fs, dim=1)
    kmeans = KMeans(n_clusters=class_num, n_init=10)
    y_pred = kmeans.fit_predict(F.data.cpu().numpy())
    print("Initial Cluster Centers: ", y_pred)
    D = Initialization_D(F, y_pred, view, high_feature_dim, class_num, dict_size)
    D = torch.tensor(D).to(torch.float32)
    print("D has been initialized")
    return D




def contrastive_learning(epoch, model, view, class_num, dict_size, lambda1, lambda2, criterion, device, data_loader, optimizer):
    tot_loss = 0.
    tot_loss1 = 0
    tot_loss2 = 0

    mse = torch.nn.MSELoss()
    d_cons1 = D_constraint1()
    d_cons2 = D_constraint2()


    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, fs = model(xs)
        ss = model.calculate_s(xs)
        loss_list = []
        loss1_list = []
        loss2_list = []
        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))
            # loss_list.append(args.lambda1 * torch.nn.functional.kl_div(ss[v].log(), refined_subspace_affinity(ss[v].data)))
            for w in range(v + 1, view):
                loss_list.append(lambda1 * criterion.forward_label(ss[v], ss[w]))
                loss1_list.append(lambda1 * criterion.forward_label(ss[v], ss[w]))
        loss_list.append(lambda2 * d_cons1(model.D))
        loss2_list.append(lambda2 * d_cons1(model.D))
        loss_list.append(lambda2 * d_cons2(model.D, dict_size, class_num))
        loss2_list.append(lambda2 * d_cons2(model.D, dict_size, class_num))
        loss = sum(loss_list)
        loss1 = sum(loss1_list)
        loss2 = sum(loss2_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        tot_loss1 += loss1.item()
        tot_loss2 += loss2.item()



    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

    return tot_loss, tot_loss1, tot_loss2