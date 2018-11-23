"ModelNet40 and ModelNet10 dataset"
import os
import os.path
import json
import numpy as np
import sys
#Lay dia chi thu muc hien tai
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
#import provider

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetDataset():
    def __init__(self, root, batch_size = 32, npoints = 1024, split='train', normalize=True, normal_channel=False, modelnet10=False, cache_size=15000, shuffle=None):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'shape_names.txt')
        #Tên các categories của file. Ví dụ: ['airplane', 'bathtub']
        self.cat = [line.rstrip() for line in open(self.catfile)]
        #Đặt index cho các class. Ví dụ: {'airplane': 0, 'bathtub': 1}
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.normal_channel = normal_channel

        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        

if __name__ == '__main__':
    d = ModelNetDataset(root = './data/modelnet40_ply_hdf5_2048', )
    print(d.shape_ids)