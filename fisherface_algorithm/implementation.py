import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from collections import defaultdict
dir_path = os.getcwd()
file_path = '/Fisher-Train/'

train_names = glob.glob(dir_path + file_path + '*.jpg')
IMG_CLASS = defaultdict(list)

for idx, image in enumerate(train_names):
    image_name = image.split('.')[0]
    IMG_CLASS[image_name].append(idx)

HEIGHT = 231
WIDTH = 195

test_names = glob.glob(dir_path + '/Train-Dataset/*.jpg')

def convert_to_mat(img_names):
    img_mat = np.ndarray(shape=(len(img_names), HEIGHT * WIDTH), dtype=np.float64)
    
    for i in range(len(img_mat)):
        img = plt.imread(img_names[i])
        img_mat[i, :] = np.array(img, dtype=np.float64).flatten()
        
    return img_mat

def calc_mean(data):
    return data.mean(axis=0)

def calc_mean_face(data):
    return data - calc_mean(data)

def pca(data):
    mean_face = calc_mean(data)
    data -= mean_face
    cov = np.cov(data)
    svd = np.linalg.svd(cov)
    K = svd[0][:, :data.shape[0]]
    
    eigen_space = np.dot(K.T, data)
    
    return (np.dot(data, eigen_space.T), eigen_space)


def get_class_mean(data, names):
    class_data = defaultdict(list)
    class_mean = {}
    
    for name in IMG_CLASS:
        for i in range(len(names)):
            img_name = names[i].split('.')[0]
            if img_name == name:
                class_data[name].append(data[i])
        class_data[name] = np.array(class_data[name])
        class_mean[name] = np.mean(a=class_data[name], axis=0)
    
    return class_data, class_mean

train_img_mat, eigen_space = pca(convert_to_mat(train_names))
class_data, class_mean = get_class_mean(train_img_mat, train_names)

def calc_scatter_mat(data, class_data, class_mean):
    Sw = np.zeros((data.shape[1], data.shape[1]))
    Sb = np.zeros((data.shape[1], data.shape[1]))
    MEAN = calc_mean(data)
    
    for name in IMG_CLASS:
        N = len(IMG_CLASS[name])
        
        scatter_val = class_data[name] - class_mean[name]
        Sw += np.dot(scatter_val.T, scatter_val)
        
        mean_scatter_val = class_mean[name] - MEAN
        Sb += mean_scatter_val
    
    return Sw, Sb
Sw, Sb = calc_scatter_mat(train_img_mat, class_data, class_mean)
def calc_scatter_val(Sw, Sb):
    Sw_inv = np.linalg.inv(Sw)
    
    return np.dot(Sw_inv, Sb)


Sval = calc_scatter_val(Sw, Sb)

def calc_eigen_values(X):
    eigen_values, eigen_vectors = np.linalg.eigh(X)
    
    eig = [(eigen_values[i], eigen_vectors[i]) for i in range(len(eigen_vectors))]

    eig.sort(reverse=True)
    eigval = [eig[i][0] for i in range(len(eig))]
    eigvec = [eig[i][1] for i in range(len(eig))]
    
    return eigvec, eigval

eigen_value, eigen_vector = calc_eigen_values(Sval)
fisher_faces = np.dot(calc_mean_face(train_img_mat), eigen_space)

for i in range(len(fisher_faces)):
    img = fisher_faces[i].reshape(HEIGHT, WIDTH)
    plt.subplot(2, 4, 1+i)
    plt.imshow(img, cmap="jet")
    plt.tick_params(which="both")

plt.show()