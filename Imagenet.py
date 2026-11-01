import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import numpy as np
from PIL import Image
from glob import glob
from odl.contrib import torch as odl_torch
from CTslice import CTSlice_Provider


class Imagenet(Dataset):
    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, num_view, startidx=0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.num_view = num_view
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d, num_view:%d' % (
            mode, batchsz, n_way, k_shot, k_query, resize, num_view))
        self.root = root
        

        self.folders = [item for item in os.listdir(root)
                   if os.path.isdir(os.path.join(root, item))]
        #print(self.folders)
        #print('self.folders:', len(self.folders))
        self.cls_num = len(self.folders)

        self.data = []
        folders = [item for item in os.listdir(root)
                   if os.path.isdir(os.path.join(root, item))]
        for folder in folders:
            folder_path = os.path.join(root, folder)
            files = [item for item in os.listdir(folder_path)
                     if os.path.isfile(os.path.join(folder_path, item))]
            self.data.append(files)

        self.create_batch(self.batchsz)

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False) #False means sampling without replacement
                np.random.shuffle(selected_imgs_idx)
                #print("selected_imgs_idx:", selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                #print("indexDtrain:", indexDtrain) #[1745 1893 2848 1996  213]
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())
                #print("support_x:", support_x) #[['L229_1-094.dcm', 'L024_1-117.dcm', 'L071_1-059.dcm', 'L071_1-149.dcm', 'L058_1-174.dcm']]
                #print("query_x:", query_x)

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets
            #print(self.support_x_batch)
            #nps = np.array(self.support_x_batch)
            #print(nps.shape) #(10, 1, 5)
            #print("self.support_x_batch:", len(self.support_x_batch))  #len is batchsz = 10
            #print("self.query_x_batch:", len(self.query_x_batch))

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 1, self.resize, self.resize)
        # [setsz]
        support_y = torch.FloatTensor(self.setsz, 1, self.resize, self.resize)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 1, self.resize, self.resize)
        # [querysz]
        query_y = torch.FloatTensor(self.querysz, 1, self.resize, self.resize)

        flatten_support_x = [os.path.join(self.root, item[:1], item)
                             for sublist in self.support_x_batch[index] for item in sublist]
    
        flatten_query_x = [os.path.join(self.root, item[:1], item)
                             for sublist in self.query_x_batch[index] for item in sublist]
        # if index == 0:
        #     print("flatten_support_x:", flatten_support_x)
        #     print("len(flatten_support_x):", len(flatten_support_x)) #5
        #     print("flatten_query_x:", flatten_query_x)
        
        #print(f'flatten_support_x:{flatten_support_x}')
        #print(f'flatten_query_x:{flatten_query_x}')   ### ---------------check when visual

        ctslice = CTSlice_Provider(num_view= self.num_view)
        for i, path in enumerate(flatten_support_x):
            #print("path:", path)
            support_x[i], support_y[i]= ctslice.transform_read(path)
            #print(support_x[i])

        for i, path in enumerate(flatten_query_x):
            # if i ==0:
            #     print(len(flatten_query_x))
            #     print(f'the {i}-th query;{path}')
            query_x[i], query_y[i]= ctslice.transform_read(path)

        return support_x, support_y, query_x, query_y

if __name__ == '__main__':
    imagenet = Imagenet(root= "/data1/gdli/Dataset/LDCT_Full_Dose/CLN_copy",
                        mode= 'train', batchsz = 10, n_way = 1, k_shot = 1, k_query = 2, resize = 256, startidx=0)  #512
    
    db = DataLoader(imagenet, 1, shuffle=True, num_workers=0, pin_memory=False) 
    #batch_size = num_workers  #num_workers=1, pin_memory=True
    #print(imagenet[0])

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        if step ==3:
            #pass
            print(x_spt.shape) #torch.Size([1, 5, 1, 512, 512])
            print(y_spt.shape)
            print(x_qry.shape) #torch.Size([1, 15, 1, 512, 512])
            print(y_qry.shape)


    mini_test = Imagenet("/data1/gdli/Dataset/Phantom_MAML", mode='test', n_way=1, k_shot=5,
                             k_query=15,
                             batchsz=6, resize=256) #batchsz=100
    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True)
    device = torch.device('cuda:3')
    for x_spt, y_spt, x_qry, y_qry in db_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
        print('x_spt.shape_StaMLtest:', x_spt.shape)
                    