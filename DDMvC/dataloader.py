from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from torch.nn.functional import normalize



class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Prokaryotic(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'prokaryotic.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'prokaryotic.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'prokaryotic.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'prokaryotic.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Synthetic3d(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'synthetic3d.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'synthetic3d.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'synthetic3d.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'synthetic3d.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class SUNRGBD(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'SUNRGBD.mat')['X1'].astype(np.float32).transpose()
        data2 = scipy.io.loadmat(path+'SUNRGBD.mat')['X2'].astype(np.float32).transpose()
        labels = scipy.io.loadmat(path+'SUNRGBD.mat')['Y']
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = scaler.fit_transform(np.load(path+'SIFT.npy').astype(np.float32))
        self.data3 = scaler.fit_transform(np.load(path+'MFCC.npy').astype(np.float32))
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Noisy_MNIST(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'NoisyMNIST-30000.mat')['Y'].astype(np.int32).reshape(30000,)
        self.V1 = scipy.io.loadmat(path + 'NoisyMNIST-30000.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'NoisyMNIST-30000.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 30000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Cifar100(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'Cifar100.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'Cifar100.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Cifar100.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'Cifar100.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Scene15(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'scene15_deep.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'scene15_deep.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'scene15_deep.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'scene15_deep.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Cifar10(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'cifar10.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'cifar10.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'cifar10.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'cifar10.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class Youtubeface(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'YoutubeFace_sel_fea.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'YoutubeFace_sel_fea.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'YoutubeFace_sel_fea.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'YoutubeFace_sel_fea.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'YoutubeFace_sel_fea.mat')['X5'].astype(np.float32)
        labels = scipy.io.loadmat(path+'YoutubeFace_sel_fea.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx]), torch.from_numpy(
           self.x4[idx]), torch.from_numpy(
           self.x5[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class NUSWIDE(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'NUSWIDE.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'NUSWIDE.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X5'].astype(np.float32)
        labels = scipy.io.loadmat(path+'NUSWIDE.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx]), torch.from_numpy(
           self.x4[idx]), torch.from_numpy(
           self.x5[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class TinyImageNet(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'TinyImageNet_4Views.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'TinyImageNet_4Views.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'TinyImageNet_4Views.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'TinyImageNet_4Views.mat')['X4'].astype(np.float32)
        labels = scipy.io.loadmat(path+'YoutubeFace_sel_fea.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx]), torch.from_numpy(
           self.x4[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class YouTubeVideo(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'Video-3V.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'Video-3V.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Video-3V.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'Video-3V.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class YTF10(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'YTF10_deep.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'YTF10_deep.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'YTF10_deep.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'YTF10_deep.mat')['X4'].astype(np.float32)
        labels = scipy.io.loadmat(path+'YTF10_deep.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx]), torch.from_numpy(
           self.x4[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Hdigit():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][0][1].T.astype(np.float32)
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()



class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "YouTubeVideo":
        dataset = YouTubeVideo('./data/')
        dims = [512, 647, 838]
        view = 3
        data_size = 101499
        class_num = 31
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "NUSWIDE":
        dataset = NUSWIDE('./data/')
        dims = [65, 226, 145, 74, 129]
        view = 5
        data_size = 5000
        class_num = 5
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
