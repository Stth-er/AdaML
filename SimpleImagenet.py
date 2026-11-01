import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import random
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from odl.contrib import torch as odl_torch
from CTslice import CTSlice_Provider

class SimpleImagenet(Dataset):
    def __init__(self,root, mode= 'train', batchsz = 0, n_way = 0, k_shot = 0, k_query = 0, resize = 0, num_view = 0, startidx=0):
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
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d, num_view:%d, baseline' % (
            mode, batchsz, n_way, k_shot, k_query, resize, num_view))
        self.root = root
        self.data_slice = glob(os.path.join(self.root,'**/*.dcm'), recursive=True)
        # for i in range(15000):
        #     if i %100==0:
        #         print(self.data_slice[i])
        

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
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
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
        #return self.batchsz
        return len(self.data_slice)


    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        ctslice = CTSlice_Provider(num_view = self.num_view)
        slice_path = self.data_slice[index]
        #print("path:", slice_path)
        x,y= ctslice.transform_read(slice_path)

        return x,y
    


if __name__ == '__main__':
    imagenet = SimpleImagenet(root= "/data1/gdli/Dataset/LDCT_Full_Dose/CLN_copy",
                        mode= 'train', batchsz = 10, n_way = 1, k_shot = 20, k_query = 3, resize = 512, startidx=0)
    #print(len(imagenet)) #15000
    db = DataLoader(imagenet, 1, shuffle=True, num_workers=0, pin_memory=False) 
    #batch_size = num_workers  #num_workers=1, pin_memory=True

    for step, (x,y) in enumerate(db):
        if step ==3:
            #pass
            print(x.shape) #torch.Size([1, 5, 1, 512, 512])       #torch.Size([2, 1, 512, 512])   (bathch_size, C, H, W)
            print(y.shape)
            #print(x_qry.shape) #torch.Size([1, 15, 1, 512, 512]) 
            #print(y_qry.shape)

            y = y[0,...].numpy()
            x = x[0,...].numpy()
                      
            gt = sitk.GetImageFromArray(y)
            #sitk.WriteImage(gt, "./show/y.nii.gz")
            fbpu = sitk.GetImageFromArray(x)
            #sitk.WriteImage(fbpu, "./show/x.nii.gz")
           
                    