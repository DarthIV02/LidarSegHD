from dataloader import read_points, read_semlabels
from sklearn.neighbors import radius_neighbors_graph
import torch
import numpy as np
from scipy.sparse import bsr_array
from tqdm import tqdm

def dfs(graph, visited, node, component):
    visited[node] = True
    component[node] = component["count"]
    
    for neighbor in range(len(graph[node])):
        if graph[node][neighbor] == 1 and not visited[neighbor]:
            dfs(graph, visited, neighbor, component)

def connected_components(graph):
    num_points = len(graph)
    visited = [False] * num_points
    component = [-1] * num_points
    component["count"] = 0
    
    for i in range(num_points):
        if not visited[i]:
            dfs(graph, visited, i, component)
            component["count"] += 1
    
    return component

if __name__ == '__main__':
    # Read files
    points = read_points(f'Dataset_TLS/dense_dataset_subsample_2/sequences/00/velodyne/000001.bin')[:,:3]
    #labels = read_semlabels(f'Dataset_TLS/dense_dataset_subsample_2/sequences/00/labels/000001.label')

    # Temperature hyperparameter
    t = 0.1

    dist = radius_neighbors_graph(points, 0.4, mode='distance', include_self=False) # Get points closer to 0.1
    dis_center = torch.sqrt((points[:,0])**2 + (points[:,1])**2 + (points[:,2])**2) # Get distances to center of scan
    assigned = torch.zeros(dist.shape[0]) # Everyone starts with component = 0 (None)
    connected = torch.zeros(dist.shape[0], dist.shape[0], dtype=torch.bool)

    for i in tqdm(range(dist.shape[0])):
        close_points_i = dist[i].indices # indices of the points that are less than radius
        connected_s = torch.tensor(dist[i].data) <= torch.max(dis_center[i], dis_center[close_points_i]) * t
        close_points_i = close_points_i[connected_s]
        connected[i][close_points_i] = 1

    components = connected_components(connected)
    print("Connected components:", components[:-1])

            

            

    