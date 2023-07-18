
# 读取模型 并把权重参数保存下来
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.transforms import ToTensor
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
the_model = torch.load("mylenet.pth")


# for key, value in the_model.items(): # 由于最邻近差值没有参数，只需要将其他参数赋予给新模型即可
#     the_model[key] = value
#     f = open('./parameter/'+key+".txt", "w")
#     value_cpu = value.cpu()
#     data_arr = value_cpu.numpy()
#     data_arr = data_arr.flatten()
#     np.savetxt('./parameter/'+key+'.txt', data_arr, fmt='%f', delimiter='\r\n')
#     # f.write(data_arr)

# 把mnist保存成图片
import os
from skimage import io
import torchvision.datasets.mnist as mnist
import numpy as np
import torch
root = "C:/Users/18062/PycharmProjects/pythonProject_0326_test4/data/FashionMNIST/raw/"

train_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
)

test_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
print("train set:", train_set[0].size())
print("test set:", test_set[0].size())

def convert_to_img(train=True):
    if (train):
        f = open(root + 'train.txt', 'w')
        data_path = root + '/train/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            # f.write(img_path + ' ' + str(label) + '\n')
            num = label.numpy()
            f.write( str(num) + '\n')
        f.close()


convert_to_img(True)
# convert_to_img(False)