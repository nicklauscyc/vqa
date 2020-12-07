import h5py
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F
import config
import data
import utils

from torch.nn import Linear, Sequential, Conv2d, MaxPool2d, BatchNorm2d, ReLU

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 0),
            BatchNorm2d(64),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True),

            Conv2d(64, 128, kernel_size =  7, stride = 2, padding = 0),
            BatchNorm2d(128),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True),
        )

        self.linear_layers = Sequential(
            Linear(86528, 2048) #Linear(50176, 8)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x 


def create_vqa_loader(*paths):
    transform = utils.get_transform(config.image_size, config.central_fraction)
    datasets = [data.VSQImages(path, transform=transform) for path in paths]
    dataset = data.Composite(*datasets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers=config.data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


def main():
    cudnn.benchmark = True

    net = Net().cuda()
    net.eval()

    loader = create_vqa_loader(config.train_path, config.val_path)
    features_shape = (
        len(loader.dataset),
        config.output_features
    )

    with h5py.File(config.preprocessed_path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        vsq_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = j = 0
        for ids, imgs in tqdm(loader):
            imgs = Variable(imgs).cuda(device=None, non_blocking=False), volatile=True)
            out = net(imgs)

            j = i + imgs.size(0)
            features[i:j, :] = out.data.cpu().numpy().astype('float16')
            vsq_ids[i:j] = ids.numpy().astype('int32')
            i = j


if __name__ == '__main__':
    main()