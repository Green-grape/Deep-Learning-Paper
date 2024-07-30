import os
import torch
from torch import nn
from model import UNet
from dataloader import Dataset, ToTensor, Normalization, RandomFilp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np


def save_model(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict()
    }, './%s/model_epoch%d.pth' % (ckpt_dir, epoch))


def load_model(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    return net, optim, epoch


if __name__ == "__main__":
    lr = 1e-3
    batch_size = 4
    num_epoch = 80

    data_dir = './datasets'
    ckpt_dir = './checkpoint'
    log_dir = './log'

    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')

    transform = transforms.Compose([Normalization(), RandomFilp(), ToTensor()])

    dataset_train = Dataset(os.path.join(
        data_dir, 'train'), transform=transform)
    loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

    net = UNet().to(device)
    fn_loss = nn.BCEWithLogitsLoss().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

    def fn_tonumpy(x): return x.to(
        'cpu').detach().numpy().transpose(0, 2, 3, 1)

    def fn_denorm(x, mean, std): return (x * std) + mean
    def fn_class(x): return 1.0 * (x > 0.5)

    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    st_epoch = 0
    net, optim, st_epoch = load_model(ckpt_dir, net, optim)
    for epoch in range(st_epoch+1, num_epoch+1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            label = (data['label']/255.0).to(device)
            input_ = data['input'].to(device)

            output = net(input_)
            optim.zero_grad()  # 이전 루프에서 grad에 저장된 값이 있을 수 있으므로 초기화

            loss = fn_loss(output, label)
            loss.backward()  # loss를 통해 grad 계산(각 파라미터의 grad값에 변화도 저장)
            optim.step()

            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            output = fn_tonumpy(fn_class(output))
            label = fn_tonumpy(label)
            input_ = fn_tonumpy(fn_denorm(input_, mean=0.5, std=0.5))

            writer_train.add_images(
                'label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_images(
                'input', input_, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_images(
                'output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                label = (data['label']/255.0).to(device)
                input_ = data['input'].to(device)

                output = net(input_)

                loss = fn_loss(output, label)
                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                output = fn_tonumpy(fn_class(output))
                label = fn_tonumpy(label)
                input_ = fn_tonumpy(fn_denorm(input_, mean=0.5, std=0.5))

                writer_val.add_images(
                    'label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_images(
                    'input', input_, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_images(
                    'output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 10 == 0:
            save_model(ckpt_dir, net, optim, epoch)

    writer_train.close()
    writer_val.close()
