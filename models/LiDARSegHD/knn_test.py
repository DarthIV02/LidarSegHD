#from tool import DataProcessing as DP
import torchhd
import torch
from dataloader import read_points, read_semlabels

points = read_points(f'Dataset_TLS/dense_dataset_subsample_2/sequences/00/velodyne/000001.bin')[:,:3]

emb = torchhd.embeddings.Random(4, 5000)
print(emb(torch.Tensor([0])))