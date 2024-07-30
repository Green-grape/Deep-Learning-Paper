import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# divide the images into directories


def unzip_images(data_dir, name_label, name_input):
    img_label = Image.open(os.path.join(data_dir, name_label))
    img_input = Image.open(os.path.join(data_dir, name_input))

    ny, nx = img_label.size
    nframe = img_label.n_frames  # number of frames

    nframe_train = 24
    nframe_test = 3
    nframe_val = 3

    # create directories
    dir_save_train = os.path.join(data_dir, 'train')
    dir_save_test = os.path.join(data_dir, 'test')
    dir_save_val = os.path.join(data_dir, 'val')

    if not os.path.exists(dir_save_train):
        os.makedirs(dir_save_train)
    if not os.path.exists(dir_save_test):
        os.makedirs(dir_save_test)
    if not os.path.exists(dir_save_val):
        os.makedirs(dir_save_val)

    id_frame = np.arange(nframe)
    np.random.shuffle(id_frame)

    for i in range(nframe):
        img_label.seek(id_frame[i])
        img_input.seek(id_frame[i])
        target_dir = dir_save_train if i < nframe_train else dir_save_test if i < nframe_train + \
            nframe_test else dir_save_val
        label_ = np.array(img_label)
        input_ = np.array(img_input)

        np.save(os.path.join(
            target_dir, 'label_{:03d}'.format(i)), label_)
        np.save(os.path.join(
            target_dir, 'input_{:03d}'.format(i)), input_)

    plt.subplot(1, 2, 2)
    plt.hist(label_.flatten(), bins=20)
    plt.title('label')

    plt.subplot(1, 2, 1)
    plt.hist(input_.flatten(), bins=20)
    plt.title('input')

    plt.show()


if __name__ == '__main__':
    data_dir = './datasets'
    name_label = 'train-labels.tif'
    name_input = 'train-volume.tif'
    unzip_images(data_dir, name_label, name_input)
    print('Data unzip done!')
