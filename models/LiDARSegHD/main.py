import math
import torch
import torchhd
from dataloader import read_points
from sklearn import preprocessing

def create_classes(num_classes):
    "Class hypervectors which don't have relationships between each other" 
    classes_hv = torchhd.random(num_classes, d)
    return classes_hv

if __name__ == "__main__":
    d = 500 # dimensions
    #classes = torchhd.random(5, d) # alive, 1h, 10h, 100h, 1000h
    i_emb = torchhd.embeddings.Level(10, d, low=0, high=1) # Intensity
    x_emb = torchhd.embeddings.Level(100, d, low=-15, high=15)
    y_emb = torchhd.embeddings.Level(100, d, low=-15, high=15) 
    z_emb = torchhd.embeddings.Level(100, d, low=0, high=2)
    var = torchhd.random(4, d) # x, y, z, i

    points = read_points('Dataset_TLS/dense_dataset_semantic/sequences/00/velodyne/000001.bin')
    points[:,3] = preprocessing.normalize([points[:,3]]) #Normalize intensity

    enc_points = var[0] * x_emb(torch.tensor(points[:, 0])) + var[1] * y_emb(torch.tensor(points[:, 1])) + var[2] * z_emb(torch.tensor(points[:, 2])) + var[3] * i_emb(torch.tensor(points[:, 3]))
    

    print(enc_points)