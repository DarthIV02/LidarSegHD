import numpy as np
import torch

def read_points(bin_file):
    points = np.fromfile(bin_file, dtype = np.float32)
    points = np.reshape(points,(-1,4)) # x,y,z,intensity
    return torch.from_numpy(points)

def read_semlabels(label_file):
    semlabels = np.fromfile(label_file, dtype = np.uint32) & 0xffff
    semlabels = semlabels.astype(np.uint8)
    return torch.from_numpy(semlabels)

if __name__ == "__main__":
    points = read_points('Dataset_TLS/dense_dataset_semantic/sequences/00/velodyne/000001.bin')
    