from torchvision import transforms
import os
from dataloader import Dataset, ToTensor, Normalization, RandomFilp
from torch.utils.data import DataLoader
from train import save_model, load_model
import torch
from model import UNet
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def test_data():
    transform = transforms.Compose(
        [Normalization(), RandomFilp(), ToTensor()])

    data_dir = './datasets'
    batch_size = 4

    dataset_test = Dataset(data_dir=os.path.join(
        data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size,
                             shuffle=False, num_workers=8)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

    result_dir = './results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')
    lr = 1e-4
    net = UNet().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    net, optim, st_epoch = load_model('./checkpoint', net, optim)

    def fn_tonumpy(x): return x.to(
        'cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_loss = nn.BCEWithLogitsLoss().to(device)
    def fn_denorm(x, mean, std): return (x * std) + mean

    def fn_class(x): return 1.0 * (x > 0.5)
    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            label = (data['label']/255.0).to(device)
            input_ = data['input'].to(device)

            output = net(input_)

            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print('TEST: BATCH %04d/%04d | LOSS %.4f' %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # save result
            label = fn_tonumpy(label)
            input_ = fn_tonumpy(fn_denorm(input_, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            if os.path.exists(os.path.join(result_dir, 'numpy')) == False:
                os.makedirs(os.path.join(result_dir, 'numpy'))
            for j in range(label.shape[0]):
                id_ = batch_size*(batch-1) + j

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' %
                        id_), label[j])
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' %
                        id_), input_[j])
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' %
                        id_), output[j])


if __name__ == '__main__':
    test_data()
    result_dir = './results'
    list_data = os.listdir(os.path.join(result_dir, 'numpy'))

    list_label = [f for f in list_data if f.startswith('label')]
    list_input = [f for f in list_data if f.startswith('input')]
    list_output = [f for f in list_data if f.startswith('output')]

    list_label.sort()
    list_input.sort()
    list_output.sort()

    id = 0
    label = np.load(os.path.join(result_dir, 'numpy', list_label[id]))
    input_ = np.load(os.path.join(result_dir, 'numpy', list_input[id]))
    output = np.load(os.path.join(result_dir, 'numpy', list_output[id]))

    plt.figure(figsize=(8, 6))
    plt.subplot(131)
    plt.imshow(input_, cmap='gray')
    plt.title('Input')

    plt.subplot(132)
    plt.imshow(label, cmap='gray')
    plt.title('Label')

    plt.subplot(133)
    plt.imshow(output, cmap='gray')
    plt.title('Output')
    plt.show()
