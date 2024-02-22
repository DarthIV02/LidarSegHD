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
import random
import laspy

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
        #self.i_emb = torchhd.embeddings.Projection(self.num_closest_points, self.d) # Random Projection
        #self.x_emb = torchhd.embeddings.Level(1000, d, low=-0.01, high=0.01)
        #self.y_emb = torchhd.embeddings.Level(1000, d, low=-0.01, high=0.01) 
        #self.z_emb = torchhd.embeddings.Level(1000, d, low=-0.01, high=0.01)
        self.x_emb = torchhd.embeddings.Sinusoid(self.num_closest_points, self.d)
        self.y_emb = torchhd.embeddings.Sinusoid(self.num_closest_points, self.d) 
        self.z_emb = torchhd.embeddings.Sinusoid(self.num_closest_points, self.d)
        self.h_emb = torchhd.embeddings.Sinusoid(self.num_closest_points, self.d)
        self.var_emb = torchhd.embeddings.Sinusoid(3, self.d)
        self.mean_emb = torchhd.embeddings.Sinusoid(3, self.d)
        self.std_emb = torchhd.embeddings.Sinusoid(3, self.d)
        self.skew_emb = torchhd.embeddings.Sinusoid(3, self.d)
        self.kurt_emb = torchhd.embeddings.Sinusoid(3, self.d)
        #self.var = torchhd.random(4, self.d) # x, y, z, i 
        self.classes_hv = torch.zeros(self.num_classes, self.d) # alive, 1h, 10h, 100h, 1000h
        self.count_matrix = torch.zeros(self.num_classes, self.num_classes)
        self.softmax = torch.nn.Softmax(dim=0)
        self.test_id = 0
    
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
        #self.var = self.var.to(*args)
        self.count_matrix = self.count_matrix.to(*args)
        self.x_emb = self.x_emb.to(*args)
        self.y_emb = self.y_emb.to(*args)
        self.z_emb = self.z_emb.to(*args)
        self.h_emb = self.h_emb.to(*args)
        self.var_emb = self.var_emb.to(*args)
        self.mean_emb = self.mean_emb.to(*args)
        self.std_emb = self.std_emb.to(*args)
        self.skew_emb = self.skew_emb.to(*args)
        self.kurt_emb = self.kurt_emb.to(*args)
        #self.i_emb = self.i_emb.to(*args)
        return self

    def encode_points(self, x, y, z, input=None, int=None, features = [False]*7):
        # Encode points x,y,z,int = Torch vectors
        x_vec = x
        y_vec = y
        z_vec = z
        x = input[:,0]
        y = input[:,1]
        z = input[:,2]

        enc_points = torch.zeros((1,self.d), device=self.opt.device)
        mean = torch.tensor([torch.mean(x), torch.mean(y), torch.mean(z)], device=self.opt.device)
        diffs = input - mean
        var = torch.mean(torch.pow(diffs, 2.0), 0)
        std = torch.pow(var, 0.5)
        zscores = diffs / std

        if features[0]:
            enc_points += self.x_emb(torch.tensor(x_vec)) + self.y_emb(torch.tensor(y_vec)) + self.z_emb(torch.tensor(z_vec))
        if features[1]:
            enc_points += self.h_emb(torch.tensor(z)) # Height enc_points
        if features[2]:
            enc_points += self.mean_emb(mean) # Mean of x,y,z
        if features[3]:
            enc_points += self.var_emb(var)
        if features[4]:
            enc_points += self.std_emb(std)
        if features[5]:
            skews = torch.mean(torch.pow(zscores, 3.0), 0)
            enc_points = enc_points + self.skew_emb(skews)
        if features[6]:
            kurtoses = torch.mean(torch.pow(zscores, 4.0), 0) - 3.0
            enc_points = enc_points + self.kurt_emb(kurtoses)

        #enc_points = self.x_y_z_emb(torch.tensor(input[:,2])) # Actually just height
        #e_final = torch.zeros(self.d, device=self.opt.device)
        #for p, e in enumerate(enc_points):
        #    e_final += torchhd.permute(e,shifts=p)
        #enc_points = self.x_emb(torch.tensor(x)) + self.y_emb(torch.tensor(y)) + self.z_emb(torch.tensor(z)) + enc_points

        return enc_points[0].sign()
    
    def encode_graph(self, id_point, point, points, distances=None, labels=None):
        # close_points = self.get_close_points_sklearn(id_point, A) # <- With sklearn Get indeces for the closest points including itself
        
        # Test to encode the vectors based on the closest points
        if (distances == None):
            close_points = self.get_close_points_euclidean(point[:3], points[:,:3])
            #new_id = np.where(close_points == id_point) # In only closest points
            #close_points = np.delete(close_points, new_id)
        else:
            close_points = distances.indices
        if labels != None:
            classes_count = torch.bincount(labels[close_points])
        
        #If we vecotrize based on the current point
        vectors_closest_points = points[:,:3][close_points] - point[:3]
            
        #If we vectorize based on the center of the points selected
        #max_x, min_x = torch.max(points[:,0][close_points]), torch.min(points[:,0][close_points])
        #max_y, min_y = torch.max(points[:,1][close_points]), torch.min(points[:,1][close_points])
        #max_z, min_z = torch.max(points[:,2][close_points]), torch.min(points[:,2][close_points])
        #vectors_closest_points = points[:,:3][close_points] - torch.tensor([((max_x-min_x)/2)+min_x, ((max_y-min_y)/2)+min_y, ((max_z-min_z)/2)+min_z], device=self.opt.device)
        features = [False]*7
        features[self.opt.features] = True
        enc_vectors = self.encode_points(vectors_closest_points[:, 0], vectors_closest_points[:, 1], vectors_closest_points[:, 2], points[:,:3][close_points], features=features) # In case of intensity points[close_points][:, 3]
        #enc_vectors = self.encode_points(vectors_closest_points, input=points[:,:3][close_points])
        
        if labels != None:
            return enc_vectors, classes_count
        else:
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
                if classes_id[0].shape[0] > self.training_samples_per_class:
                    ids = np.random.choice(range(classes_id[0].shape[0]), size=self.training_samples_per_class, replace=False)
                else:
                    ids = np.random.choice(range(classes_id[0].shape[0]), size=classes_id[0].shape[0], replace=False)
                training_samples = classes_id[0][ids] # Choose 100 points for training
                for i, point in zip(training_samples, points[training_samples]): # id, xyz of points
                    #current_point_graph = model.encode_graph(i, point, points, labels, A[i]) #<- With sklearn
                    current_point_graph, purity = self.encode_graph(i, point, points, labels = labels)
                    cos_sim_one = self.similarity(current_point_graph)
                    max_sim_id = torch.argmax(cos_sim_one).item()
                    if (max_sim_id != label-1):
                        self.classes_hv[max_sim_id] -= current_point_graph*self.lr*(1-cos_sim_one[max_sim_id])#*(purity[label-1]/self.num_closest_points)
                    self.classes_hv[label-1] += current_point_graph*self.lr*(1-cos_sim_one[label-1]) #*(purity[label-1]/self.num_closest_points)
    
    def test(self, points, labels, mode="Infer"):

        if mode=="Visualize":
            full_testing_samples = []
            full_labels_samples = []

        for label in range(1, self.num_classes+1):
            classes_id = torch.where(labels == label)
            if(classes_id[0].shape[0] != 0):
                if classes_id[0].shape[0] > self.testing_samples_per_class:
                    ids = np.random.choice(range(classes_id[0].shape[0]), size=self.testing_samples_per_class, replace=False)
                else:
                    ids = np.random.choice(range(classes_id[0].shape[0]), size=classes_id[0].shape[0], replace=False)
                training_samples = classes_id[0][ids]
                if mode=="Visualize":
                    full_testing_samples.extend(points[training_samples].tolist())
                for i, point in zip(training_samples, points[training_samples]):
                    enc_point = self.encode_graph(i, point, points)
                    sim = self.similarity(enc_point)
                    pred_class = torch.argmax(sim).item() #<- How you determine the class it belongs to
                    self.count_matrix[int(label-1), pred_class] += 1
                    if mode=="Visualize":
                        full_labels_samples.append(pred_class)
        
        if mode=="Visualize":
            self.points_to_las(full_testing_samples, full_labels_samples)

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

    def points_to_las(self, points, labels):
        output_file_name = os.path.join(self.opt.vis_path, f'Classified_scan_{self.test_id}.las')
        header = laspy.LasHeader(point_format=3, version="1.2")
        outfile = laspy.LasData(header)

        # Add points to the LAS file
        outfile.x, outfile.y, outfile.z = [i[0] for i in points], [i[1] for i in points], [i[2] for i in points]
        outfile.classification = labels

        # Write to LAS file
        outfile.write(output_file_name)
        self.test_id += 1

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--classes', type=int, default=5,
                        help='Classes for segmentation without unlabeled')
    parser.add_argument('--d', type=int, default=10000,
                        help='Dimensionality of the class hypervectors')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning Rate for HD training')
    parser.add_argument('--number_close_points', type=int, default=50,
                        help='Number of KNN to consider for point encoding')
    parser.add_argument('--number_of_choices_training', type=int, default=300,
                        help='Number of points per class per file for training')
    parser.add_argument('--number_of_choices_testing', type=int, default=200,
                        help='Number of points per class per file for testing')
    parser.add_argument('--features', type=int, default=0,
                        help='Features missing')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs or number of passes on dataset')
    parser.add_argument('--val_rate', type=int, default=5,
                        help='Test the model every val_rate files')
    
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--trial', type=int, default=0,
                        help='id for recording multiple runs')

    parser.add_argument('--dataset_path', type=str, default="Dataset_TLS/dense_dataset_subsample/sequences/",
                        help='Path to the folder sequences')
    

    opt = parser.parse_args()

    opt.method = 'LidarSeg'
    opt.model_path = './save/{}/{}_models/'.format(opt.method, 'TLS')
    opt.tb_path = './save/{}/{}_tensorboard/'.format(opt.method, 'TLS')
    opt.vis_path = './save/{}/{}_vis/'.format(opt.method, 'TLS')

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

    opt.vis_path = os.path.join(opt.vis_path, opt.model_name)
    if not os.path.isdir(opt.vis_path):
        os.makedirs(opt.vis_path)

    return opt

def main():

    opt = parse_option()
    
    print("============================================")
    print(opt)
    print("============================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = device
    print("Using {} device".format(device))
    
    np.random.seed(opt.trial)
    torch.manual_seed(opt.trial)
    random.seed(opt.trial)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    
    model = SegmentHD(opt, logger)

    model = model.to(device)

    for e in range(opt.epochs):
        files = os.listdir(os.path.join(opt.dataset_path, f'00/velodyne/'))
        random.shuffle(files)
        for i, file in tqdm(enumerate(files)):
            file = file[:-4]
            points = read_points(os.path.join(opt.dataset_path, f'00/velodyne/{file}.bin'))[:,:3]
            labels = read_semlabels(os.path.join(opt.dataset_path, f'00/labels/{file}.label'))

            #Normalize intensity
            #mu, std = norm.fit(points[:,3])
            #points[:, 3][points[:, 3] >= std*2] = std*2
            #points[:,3] /= std*2 

            #Normalize xyz
            #points[:, :3] /= 0.02
            points = points.to(device)
            labels = labels.to(device)
            model.train(points, labels, i)

            if i % opt.val_rate == 0 and i != 0:
                model.count_matrix = torch.zeros(model.num_classes, model.num_classes)
                for file_val in tqdm(os.listdir(os.path.join(opt.dataset_path, '02/velodyne/'))):
                    file_val = file_val[:-4]
                    points_val = read_points(os.path.join(opt.dataset_path, f'02/velodyne/{file_val}.bin'))[:,:3]
                    labels_val = read_semlabels(os.path.join(opt.dataset_path, f'02/labels/{file_val}.label'))

                    points_val = points_val.to(device)
                    labels_val = labels_val.to(device)

                    model.test(points_val, labels_val)

                model.evaluate(i, e)

    # Testing
    model.count_matrix = torch.zeros(model.num_classes, model.num_classes)
    for file in tqdm(os.listdir(os.path.join(opt.dataset_path, '01/velodyne/'))):
        file = file[:-4]
        points = read_points(os.path.join(opt.dataset_path, f'01/velodyne/{file}.bin'))[:,:3]
        labels = read_semlabels(os.path.join(opt.dataset_path, f'01/labels/{file}.label'))

        points = points.to(device)
        labels = labels.to(device)

        model.test(points, labels, "Visualize")

    model.evaluate()

if __name__ == '__main__':
    main()