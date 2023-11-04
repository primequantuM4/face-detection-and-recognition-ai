import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob

from matplotlib.image import imread

path = os.getcwd()
train_names = glob.glob(path + "/Train-Dataset/*.jpg")
test_names = glob.glob(path + '/Test-Dataset/*.jpg')

HEIGHT = 231
WIDTH = 195

def convert_to_mat(img_names):
    
    image_mat = np.ndarray(shape=(len(img_names), HEIGHT * WIDTH), dtype=np.float64)
    
    for i in range(len(img_names)):
        img = plt.imread(img_names[i])
        image_mat[i, :] = np.array(img, dtype=np.float64).flatten()
    
    return image_mat

train_image_mat = convert_to_mat(train_names)
test_image_mat = convert_to_mat(test_names)

def calculate_mean_face(data):
    mean_face = np.zeros((1, HEIGHT * WIDTH))
    M = len(data)
    for i in data:
        mean_face = np.add(mean_face, i)
    
    mean_face = np.divide(mean_face, float(M)).flatten()
    return mean_face


def normalize_face(data):
    mean_face = calculate_mean_face(data)
    normalized_faces = data - mean_face
    
    return normalized_faces

def show_avg_face(data):
    mean_face = calculate_mean_face(data).reshape(HEIGHT, WIDTH)
    
    plt.imshow(mean_face, cmap="gray")
    plt.tick_params(which="both")
    
    plt.show()

def show_normalized_face(data):
    norm_faces = normalize_face(data) 
    for i in range(len(train_names)):
        img = norm_faces[i].reshape(HEIGHT, WIDTH)
        plt.subplot(2, 4, 1+i)
        plt.imshow(img, cmap="jet")
        plt.tick_params(which='both')
    
    plt.show()

def calculate_covariance(data):
    norm = normalize_face(data)
    cov = np.cov(norm)
    N = 8.0
    return np.divide(cov, N)

def calc_eigen_values(data):
    cov_mat = calculate_covariance(data)
    eigen = np.linalg.eigh(cov_mat)
    
    return eigen

def get_eigen_values_and_vectors(data): 
    eig_val, eig_vec = calc_eigen_values(data)
    eig = [(eig_val[i], eig_vec[i]) for i in range(len(eig_vec))]

    eig.sort(reverse=True)
    eigval = [eig[i][0] for i in range(len(eig))]
    eigvec = [eig[i][1] for i in range(len(eig))]
    
    return eigvec, eigval

def project_faces(data):
    _, eigen_vectors = get_eigen_values_and_vectors(data)
    N = 7
    
    reduced_data = np.array(eigen_vectors[:N]).T
    
    proj_face = np.dot(data.T, reduced_data)
    proj_face = proj_face.T
    
    return proj_face

def show_eigen_faces(data):
    eigen_face = project_faces(data)
    
    for i in range(eigen_face.shape[0]):
        img = eigen_face[i].reshape(HEIGHT, WIDTH)
        plt.subplot(2, 4, 1+i)
        plt.imshow(img, cmap="gray")
        plt.tick_params(which='both')
    
    plt.show()
    
        
show_normalized_face(train_image_mat)