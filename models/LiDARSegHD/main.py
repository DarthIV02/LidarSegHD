import torchhd
from dataloader import read_points, read_semlabels
import numpy as np
import torch
from tqdm import tqdm
import os
from scipy.stats import norm
from scipy.special import softmax
import argparse
import tensorboard_logger as tb_logger

class SegmentHD:

    def __init__(self, opt, logger) -> None:
        self.d = opt.d
        self.num_classes = opt.classes
        self.num_closest_points = opt.number_close_points
        self.training_samples_per_class = opt.number_of_choices_training
        self.testing_samples_per_class = opt.number_of_choices_testing
        self.lr = opt.lr
        self.val_rate = opt.val_rate
        self.logger = logger
        self.opt = opt
        #self.i_emb = torchhd.embeddings.Level(1000, d, low=0, high=1) # Intensity
        #self.i_emb = torchhd.embeddings.Projection(self.num_closest_points, d) # Random Projection
        #self.x_emb = torchhd.embeddings.Level(1000, d, low=-0.01, high=0.01)
        #self.y_emb = torchhd.embeddings.Level(1000, d, low=-0.01, high=0.01) 
        #self.z_emb = torchhd.embeddings.Level(1000, d, low=-0.01, high=0.01)
        self.x_emb = torchhd.embeddings.Projection(self.num_closest_points, self.d)
        self.y_emb = torchhd.embeddings.Projection(self.num_closest_points, self.d) 
        self.z_emb = torchhd.embeddings.Projection(self.num_closest_points, self.d)
        self.var = torchhd.random(3, self.d) # x, y, z, i 
        self.classes_hv = torch.zeros(self.num_classes, self.d) # alive, 1h, 10h, 100h, 1000h
        self.count_matrix = torch.zeros(self.num_classes, self.num_classes)
        self.softmax = torch.nn.Softmax(dim=0)
    
    def to(self, *args):
        '''
        Moves data to the device specified, e.g. cuda, cpu or changes
        dtype of the data representation, e.g. half or double.
        Because the internal data is saved as torch.tensor, the parameter
        can be anything that torch accepts. The change is done in-place.

        Args:
            device (str or :class:`torch.torch.device`) Device to move data.

        Returns:
            :class:`SegmentHD`: self
        '''

        self.classes_hv = self.classes_hv.to(*args)
        self.var = self.var.to(*args)
        self.count_matrix = self.count_matrix.to(*args)
        self.x_emb = self.x_emb.to(*args)
        self.y_emb = self.y_emb.to(*args)
        self.z_emb = self.z_emb.to(*args)
        return self

    def encode_points(self, x, y, z, int=None):
        # Normalize xyz
        #x /= 0.4
        #y /= 0.4
        #z /= 0.3

        # Encode points x,y,z,int = Torch vectors
        enc_points = (self.var[0] * self.x_emb(torch.tensor(x))) + (self.var[1] * self.y_emb(torch.tensor(y))) + (self.var[2] * self.z_emb(torch.tensor(z))) # + (self.var[3] * self.i_emb(torch.tensor(int)))
        return enc_points.sign()
    
    def encode_graph(self, id_point, point, points, distances=None):
        # close_points = self.get_close_points_sklearn(id_point, A) # <- With sklearn Get indeces for the closest points including itself
        
        # Test to encode the vectors based on the closest points
        if (distances == None):
            close_points = self.get_close_points_euclidean(point, points)
            #new_id = np.where(close_points == id_point) # In only closest points
            #close_points = np.delete(close_points, new_id)
        else:
            close_points = distances.indices
        
        vectors_closest_points = points[close_points] - point

        enc_vectors = self.encode_points(vectors_closest_points[:, 0], vectors_closest_points[:, 1], vectors_closest_points[:, 2]) # In case of intensity points[close_points][:, 3]
        #full_point = enc_vectors
        
        #edges = torch.zeros(self.num_closest_points, d)
        #for i in close_points.shape[0]:
        #    if(i != id_point):
        #        edges[i] = torchhd.bind(enc_points[i], enc_points[id_point])
        
        # If multiple points
        #full_point = torch.zeros(d) # Create a 0 vector which is going to be added to each one of the encoded points
        #for e, label in zip(enc_vectors, labels[close_points]): # add edges together
        #    if labels[id_point] == label:
        #        full_point = torchhd.bundle(full_point, e)
        #full_point = torchhd.hard_quantize(full_point) # Fully encoded hv
        
        #classes[label-1] = torchhd.bundle(classes[label-1], full_point) # Add to class hv
        return enc_vectors
    
    def similarity(self, point):
        cos_sim_one = torchhd.cosine_similarity(point, self.classes_hv)
        cos_sim_one = self.softmax(cos_sim_one)
        return cos_sim_one
    
    def get_close_points_euclidean(self, point, points): # TODO HD KNN? -> HD clustering by whose work has studied 
        squared_dist = torch.sum((point-points)**2, 1)
        dist = torch.sqrt(squared_dist)
        index_closest_points = torch.topk(dist, self.num_closest_points+1, largest=False)
        return index_closest_points.indices[1:]
    
    #def get_close_points_number(self, i, A):
        
    #    return A.indices
    
    #def get_close_points_radius(self, i, A):
        
    #    return A.indices

    def train(self, points, labels, file_idx):
        #A = neighbors.kneighbors_graph(points, self.num_closest_points, mode='connectivity', include_self=False) #<- With sklearn
        #A = neighbors.radius_neighbors_graph(points, 0.5, mode='connectivity', include_self=False)
        for label in range(1, self.num_classes+1):
            classes_id = torch.where(labels == label)
            if(classes_id[0].shape[0] != 0):
                ids = np.random.choice(range(classes_id[0].shape[0]), size=self.training_samples_per_class, replace=False)
                training_samples = classes_id[0][ids] # Choose 100 points for training
                for i, point in zip(training_samples, points[training_samples]): # id, xyz of points
                    #current_point_graph = model.encode_graph(i, point, points, labels, A[i]) #<- With sklearn
                    current_point_graph = self.encode_graph(i, point, points)
                    cos_sim_one = self.similarity(current_point_graph)
                    max_sim_id = torch.argmax(cos_sim_one).item()
                    if (max_sim_id != label-1):
                        self.classes_hv[max_sim_id] -= current_point_graph*self.lr*(1-cos_sim_one[max_sim_id])
                    self.classes_hv[label-1] += current_point_graph*self.lr*(1-cos_sim_one[label-1])
    

    def test(self, points, labels, mode="Verbose"):

        for label in range(1, self.num_classes+1):
            classes_id = torch.where(labels == label)
            if(classes_id[0].shape[0] != 0):
                if classes_id[0].shape[0] > self.testing_samples_per_class:
                    ids = np.random.choice(range(classes_id[0].shape[0]), size=self.testing_samples_per_class, replace=False)
                else:
                    ids = np.random.choice(range(classes_id[0].shape[0]), size=classes_id[0].shape[0], replace=False)
                training_samples = classes_id[0][ids]
                for i, point in zip(training_samples, points[training_samples]):
                    enc_point = self.encode_graph(i, point, points)
                    sim = self.similarity(enc_point)
                    pred_class = torch.argmax(sim).item() #<- How you determine the class it belongs to
                    self.count_matrix[int(label-1), pred_class] += 1

    def evaluate(self, idx = 0, epoch = 0):
        label_sum = torch.sum(self.count_matrix, 0)
        pred_sum = torch.sum(self.count_matrix, 1)

        IoU = torch.zeros(self.num_classes)

        for c in range(0, self.num_classes):
            IoU[c] = self.count_matrix[c,c] / (label_sum[c] + pred_sum[c] - self.count_matrix[c,c])
        
        mIoU = torch.mean(IoU)
        with open(os.path.join(self.opt.save_folder, 'result.txt'), 'a+') as f:
            f.write('{epoch},{idx},{class_iou},{mIoU}\n'.format(
                epoch=epoch, idx=idx, class_iou=IoU, mIoU=mIoU
            ))

        # tensorboard logger
        for c in range(self.num_classes):
            self.logger.log_value(f'Class {c} IoU', IoU[c], idx)
        self.logger.log_value('Mean IoU', mIoU, idx)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--classes', type=int, default=5,
                        help='Classes for segmentation without unlabeled')
    parser.add_argument('--d', type=int, default=5000,
                        help='Dimensionality of the class hypervectors')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning Rate for HD training')
    parser.add_argument('--number_close_points', type=int, default=200,
                        help='Number of KNN to consider for point encoding')
    parser.add_argument('--number_of_choices_training', type=int, default=200,
                        help='Number of points per class per file for training')
    parser.add_argument('--number_of_choices_testing', type=int, default=100,
                        help='Number of points per class per file for testing')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs or number of passes on dataset')
    parser.add_argument('--val_rate', type=int, default=5,
                        help='Test the model every val_rate files')
    
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--trial', type=int, default=0,
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    opt.method = 'LidarSeg'
    opt.model_path = './save/{}/{}_models/'.format(opt.method, 'TLS')
    opt.tb_path = './save/{}/{}_tensorboard/'.format(opt.method, 'TLS')

    opt.model_name = '{}_{}_{}_{}_{}_{}_epoch_{}_trial_{}'.format(
        opt.classes, opt.d, opt.lr, opt.number_close_points,
        opt.number_of_choices_training, opt.number_of_choices_testing, 
        opt.epochs, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def main():

    opt = parse_option()
    
    print("============================================")
    print(opt)
    print("============================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    
    np.random.seed(opt.trial)
    torch.manual_seed(opt.trial)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    
    model = SegmentHD(opt, logger)

    model = model.to(device)

    for e in range(opt.epochs):
        for i, file in tqdm(enumerate(os.listdir('Dataset_TLS/dense_dataset_semantic/sequences/00/velodyne/'))):
            file = file[:-4]
            points = read_points(f'Dataset_TLS/dense_dataset_semantic/sequences/00/velodyne/{file}.bin')[:,:3]
            labels = read_semlabels(f'Dataset_TLS/dense_dataset_semantic/sequences/00/labels/{file}.label')

            #Normalize intensity
            #mu, std = norm.fit([points[:,3]])
            #points[:, 3][points[:, 3] >= std*2] = std*2
            #points[:,3] /= std*2 

            #Normalize xyz
            #points[:, :3] /= 0.02
            points = points.to(device)
            labels = labels.to(device)
            model.train(points, labels, i)

            if i % opt.val_rate == 0:
                model.count_matrix = torch.zeros(model.num_classes, model.num_classes)
                for file_val in tqdm(os.listdir('Dataset_TLS/dense_dataset_semantic/sequences/02/velodyne/')):
                    file_val = file_val[:-4]
                    points_val = read_points(f'Dataset_TLS/dense_dataset_semantic/sequences/02/velodyne/{file_val}.bin')[:,:3]
                    labels_val = read_semlabels(f'Dataset_TLS/dense_dataset_semantic/sequences/02/labels/{file_val}.label')

                    points_val = points_val.to(device)
                    labels_val = labels_val.to(device)

                    model.test(points_val, labels_val)

                model.evaluate(i, e)

    # Testing
    for file in tqdm(os.listdir('Dataset_TLS/dense_dataset_semantic/sequences/01/velodyne/')):
        file = file[:-4]
        points = read_points(f'Dataset_TLS/dense_dataset_semantic/sequences/01/velodyne/{file}.bin')[:,:3]
        labels = read_semlabels(f'Dataset_TLS/dense_dataset_semantic/sequences/01/labels/{file}.label')

        points = points.to(device)
        labels = labels.to(device)

        model.test(points, labels)

    model.evaluate()

if __name__ == '__main__':
    main()