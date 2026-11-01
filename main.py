import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import torch
import shutil 
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader
from sklearn import metrics 
from Imagenet import Imagenet
from SimpleImagenet import SimpleImagenet
from learner import Learner
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
from sklearn import metrics  
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from  utils import  parse_args 
from Imagenet_no_replace import Imagenet_No_replace

class Trainer(object):
    def __init__(self, is_restart=False, max_epoch= 50):
        self.is_restart=is_restart
        self.max_epoch=max_epoch
        self.lr = args.base_lr
        self.update_step_test =  args.update_step_test

        torch.manual_seed(222) #Sets the seed for generating random numbers on all devices
        torch.cuda.manual_seed_all(222)  #Set the seed for generating random numbers on all GPUs
        np.random.seed(222)

        config = [
        # Encoder
        # Block 1_1
        ('conv2d', [64, 1, 3, 3, 1, 1]),  # in_channels=1, out_channels=64
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),  # in_channels=64, out_channels=64
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),  # in_channels=64, out_channels=64
        ('relu', [True]),
        ('bn_1_1', [64]),

        # Block 2_1
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [128, 64, 3, 3, 1, 1]),  # in_channels=64, out_channels=128
        ('relu', [True]),
        ('bn', [128]),
        ('conv2d', [128, 128, 3, 3, 1, 1]),  # in_channels=128, out_channels=128
        ('relu', [True]),
        ('bn_2_1', [128]),

        # Block 3_1
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [256, 128, 3, 3, 1, 1]),  # in_channels=128, out_channels=256
        ('relu', [True]),
        ('bn', [256]),
        ('conv2d', [256, 256, 3, 3, 1, 1]),  # in_channels=256, out_channels=256
        ('relu', [True]),
        ('bn_3_1', [256]),

        # Block 4_1
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [512, 256, 3, 3, 1, 1]),  # in_channels=256, out_channels=512
        ('relu', [True]),
        ('bn', [512]),
        ('conv2d', [512, 512, 3, 3, 1, 1]),  # in_channels=512, out_channels=512
        ('relu', [True]),
        ('bn_4_1', [512]),

        # Block 5
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [1024, 512, 3, 3, 1, 1]),  # in_channels=512, out_channels=1024
        ('relu', [True]),
        ('bn', [1024]),
        ('conv2d', [1024, 1024, 3, 3, 1, 1]),  # in_channels=1024, out_channels=1024
        ('relu', [True]),
        ('bn', [1024]),
        ('convt2d', [1024, 512, 3, 3, 2, 1, 1]),  # in_channels=1024, out_channels=512, stride=2, padding=1, output_padding=1

        # Decoder
        ('concate_4_2', [0, 0]),

        # Block 4_2
        ('conv2d', [512, 1024, 3, 3, 1, 1]),  # in_channels=1024, out_channels=512
        ('relu', [True]),
        ('bn', [512]),
        ('conv2d', [512, 512, 3, 3, 1, 1]),  # in_channels=512, out_channels=512
        ('relu', [True]),
        ('bn', [512]),
        ('convt2d', [ 512, 256, 3, 3, 2, 1, 1]),  # in_channels=512, out_channels=256, stride=2, padding=1, output_padding=1

        ('concate_3_2', [0, 0]),
        # Block 3_2
        ('conv2d', [256, 512, 3, 3, 1, 1]),  # in_channels=512, out_channels=256
        ('relu', [True]),
        ('bn', [256]),
        ('conv2d', [256, 256, 3, 3, 1, 1]),  # in_channels=256, out_channels=256
        ('relu', [True]),
        ('bn', [256]),
        ('convt2d', [256, 128,  3, 3, 2, 1, 1]),  # in_channels=256, out_channels=128, stride=2, padding=1, output_padding=1

        ('concate_2_2', [0, 0]),
        # Block 2_2
        ('conv2d', [128, 256, 3, 3, 1, 1]),  # in_channels=256, out_channels=128
        ('relu', [True]),
        ('bn', [128]),
        ('conv2d', [128, 128, 3, 3, 1, 1]),  # in_channels=128, out_channels=128
        ('relu', [True]),
        ('bn', [128]),
        ('convt2d', [128,64,  3, 3, 2, 1, 1]),  # in_channels=128, out_channels=64, stride=2, padding=1, output_padding=1

        ('concate_1_2', [0, 0]),
        # Block 1_2
        ('conv2d', [64, 128, 3, 3, 1, 1]),  # in_channels=128, out_channels=64
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),  # in_channels=64, out_channels=64
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [1, 64, 1, 1, 1, 0]),  # in_channels=64, out_channels=1, kernel=1, stride=1, padding=0
        ('skip_connection', [0])
        ]

        self.device = torch.device(args.device_baseline)
        print(f"Base line device: {self.device}")
        train1 = '/data1/gdli/Dataset/Meta_data/meta_train'
        test1 = '/data1/gdli/Dataset/Meta_data/meta_test_1'

        root = train1
        mini = SimpleImagenet(root, num_view= args.num_view) 
        self.db = DataLoader(mini, args.task_num, shuffle=True, num_workers= 0 , pin_memory= True ) #batch_size = args.task_num = 1  num_workers=1, pin_memory=True
        
        torch.manual_seed(3) #Sets the seed for generating random numbers on all devices
        torch.cuda.manual_seed_all(3)  #Set the seed for generating random numbers on all GPUs
        np.random.seed(3)
        self.meta_test_path =   test1
        test_data_slice = glob(os.path.join(self.meta_test_path,'**/*.dcm'), recursive=True)
        self.batchsz_test = len(test_data_slice)// (args.k_spt+args.k_qry) #len(test_data_slice)  
        
        mini_test = Imagenet_No_replace(self.meta_test_path, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                k_query=args.k_qry,
                                batchsz = self.batchsz_test, resize=args.imgsz, num_view= args.num_view)
        self.db_test = DataLoader(mini_test, 1, shuffle= False, num_workers=0, pin_memory=True) #shuffle=True, num_workers=1, pin_memory=True
         
        
        self.net = Learner(config, args.imgc, args.imgsz).to(self.device) 
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.lr) 

        num_param = sum(p.numel() for p in self.net.parameters() if p.requires_grad) 
        #print('Number of parameters in model: {}'.format(num_param))  
        
        if not os.path.exists('./results/models'):
            os.mkdir('./results/models')
        if not os.path.exists('./results/visualization'):
            os.mkdir('./results/visualization')
        if self.is_restart:
            shutil.rmtree('./results/visualization') #Recursively remove the visualization directory and its contents on restart.
            os.mkdir('./results/visualization')
            self.epoch= 0
            print('Training process started')
        else:
            try:    
                state=torch.load('./results/models/epoch_13iter22999.pth.tar', weights_only=True) #------------the start state
                self.epoch=state['epoch']+1
                self.net.load_state_dict(state['reconstructor_state'])
                self.meta_optim.load_state_dict(state['reconstructor_optimizer'])
                print('Saved ckpt is loaded successfully, epoch {}.'.format(state['epoch']))
            except:
                self.epoch= 0
                print('There is no saved ckpt file to load, the training process is restarted')

    def train(self):
        print(f"--------------------------------------- method: base line ---- train ------------------------------------------")
        for e in range(self.epoch, self.max_epoch):
            start_time = time.time()
            for num_iter, (x, y) in enumerate(self.db):
                x, y = x.to(self.device), y.to(self.device)     
                pred_q = self.net(x, self.net.parameters(), bn_training=True)  
                #print(pred_q.shape)
                left_loss = self.loss(pred_q,  y) 
                loss_q =  torch.mean(left_loss)  
                loss_recon=loss_q  
                loss_reconstructor=loss_recon

                self.meta_optim.zero_grad()   
                loss_reconstructor.backward()    
                self.meta_optim.step()  
                
                # calculate the psnr/ssim of the current training images, and feedback the average vale
                if num_iter % 1000 ==0:
                    reconstructed_image = pred_q
                    curr_psnr, curr_ssim, _=self.calculate_metric(reconstructed_image, y)
                    curr_psnr = np.array(curr_psnr)
                    print('This is the {}-th epoch {}-th iteration, the psnr and ssim is {:.2f} {:.4f}'.format(e, num_iter, curr_psnr, curr_ssim))
                    print('The loss of sinos, ril, recon, sum is {:.4f}'.format(loss_reconstructor))
            #end_time = time.time()
            #print( f"This epoch training  time: {(end_time - start_time)/60 :.1f} minutes")    
            if e%2==1:
                state={'epoch': e,
                'reconstructor_state': self.net.state_dict(),
                'reconstructor_optimizer': self.meta_optim.state_dict()}
                self.save_checkpoint(e, state, num_iter)
               
    def save_checkpoint(self, num_epoch, state, num_iter):
        base_path='./results/models'
        save_path=base_path+'/'+'epoch_'+str(num_epoch)+'iter'+str(num_iter)+'.pth.tar'
        torch.save(state, save_path)
        print('Save model after {}-th epoch'.format(num_epoch))

    def calculate_metric(self, pred, gt):
            assert len(pred.shape)==4 and pred.shape==gt.shape   
            #print(pred.shape, gt.shape) #torch.Size([2, 1, 256, 256])
            pred=torch.clamp(pred, 0.0, 1.0).cpu().data.numpy()  #Clamp predictions to [0, 1] and move them to the CPU before conversion to NumPy.
            #.data accesses the tensor data without tracking gradients.
            gt=torch.clamp(gt, 0.0, 1.0).cpu().data.numpy()
            #curr_psnr, curr_ssim=0.0, 0.0
            average_psnr, average_ssim, average_rmse = [], [], []
            for i in range(pred.shape[0]):
                for j in range(pred.shape[1]):                   
                    curr_psnr = peak_signal_noise_ratio(pred[i, j, ...], gt[i, j, ...], data_range=1.0)
                    #print(f'curr_psnr:{curr_psnr}')
                    curr_ssim = structural_similarity(pred[i, j, ...], gt[i, j, ...], gaussian_weights=True, win_size=11, data_range=1.0, sigma=1.5)
                    ##Compare using an 11-pixel sliding window and Gaussian weights with sigma=1.5.
                    #print(f'curr_ssim:{curr_ssim}')
                    curr_rmse=np.sqrt(metrics.mean_squared_error(pred[i,j,...], gt[i,j,...]))  #Root mean squared error
                    #print(f'curr_rmse:{curr_rmse}')
                    average_psnr.append(curr_psnr)
                    average_ssim.append(curr_ssim)
                    average_rmse.append(curr_rmse)
            average_psnr = np.array(average_psnr).mean(axis = 0)
            average_ssim = np.array(average_ssim).mean(axis = 0)
            average_rmse = np.array(average_rmse).mean(axis = 0)
            #print(f'average_psnr:{average_psnr}')
            #print(f'average_ssim:{average_ssim}')
            #print(f'average_rmse:{average_rmse}')
            return average_psnr, average_ssim, average_rmse

    def loss(self, input, target):
        return F.mse_loss(input, target)
        #return F.l1_loss(input, target) 

    def inference_FBP(self):
        print(f"--------------------------------------inference_FBP-----{self.meta_test_path}----------------------------------")
        aver_psnr, aver_ssim, aver_rmse=0.0, 0.0, 0.0
        aver_time=0.0
        time_start=time.time()
        if True:
            for num_iter, (x_spt, y_spt, x_qry, y_qry) in enumerate(self.db_test):
                x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(self.device), y_spt.squeeze(0).to(self.device), \
                                            x_qry.squeeze(0).to(self.device), y_qry.squeeze(0).to(self.device)
                x, y = x_qry, y_qry
            
                time_start_in=time.time()
                time_end_in=time.time()
                curr_psnr, curr_ssim, curr_rmse = self.calculate_metric(x, y)
                if num_iter  % 200 == 0 or num_iter == self.batchsz_test - 1:
                    print('This is the {}-th infering slice, the psnr and ssim is {:.2f} {:.4f}'.format(num_iter, curr_psnr, curr_ssim))
                aver_psnr+=curr_psnr
                aver_ssim+=curr_ssim
                aver_rmse+=curr_rmse
                aver_time+=(time_end_in-time_start_in)

                if num_iter in [0,1,2,3,4]:  # save the first slice of num_iter   
                    reconstructed_image =  x[0:1,...]
                    #print(reconstructed_image.shape) # torch.Size([1, 1, 256, 256])
                    y = y[0:1,...]
                    #print(y.shape) #torch.Size([1, 1, 256, 256])
                    curr_psnr, curr_ssim, curr_rmse = self.calculate_metric(reconstructed_image, y)
                    #print(type(reconstructed_image)) #<class 'torch.Tensor'>
                    to_save_image=reconstructed_image[0,0,...].detach().cpu().numpy()  #Extract the first image and channel, detach from autograd, and convert to NumPy.
                    target_immage = y[0,0,...].detach().cpu().numpy()
                    #print(to_save_image.shape)  #(256, 256)
                    to_save_image_=sitk.GetImageFromArray(to_save_image)  #Get a SimpleITK Image from a numpy array
                    target_immage = sitk.GetImageFromArray(target_immage)
                    sitk.WriteImage(target_immage, './results/visualization/iter_{}_{:.2f}_{:.4f}_{:.4f}_target.nii.gz'.format( num_iter, 0, 0, 0))
                    #print("type(to_save_image):", type(to_save_image_)) # <class 'SimpleITK.SimpleITK.Image'>
                    sitk.WriteImage(to_save_image_, './results/visualization/iter_{}_{:.2f}_{:.4f}_{:.4f}_infer_FBP.nii.gz'.format(num_iter, curr_psnr, curr_ssim, curr_rmse))

        
            time_end=time.time()
            aver_psnr=aver_psnr/(num_iter+1)  
            aver_ssim=aver_ssim/(num_iter+1)
            aver_rmse=aver_rmse/(num_iter+1)
            aver_time=aver_time/(num_iter+1)  
            aver_time_out=(time_end-time_start)/(num_iter+1)   
            print('the average psnr and ssim rmse is {:.2f} {:.4f} {:.4f}'.format(aver_psnr, aver_ssim, aver_rmse))
            #print('the average time of inside and outside is {:.2f} and {:.2f}'.format(aver_time, aver_time_out))


    def inference(self):
        train0 = "/data1/gdli/Dataset/LDCT_Full_Dose/CLN_copy"
        train1 = '/data1/gdli/Dataset/Meta_data/meta_train'
        test0 = "/data1/gdli/Dataset/test"
        test1 = '/data1/gdli/Dataset/Meta_data/meta_test_1'
        test2 = '/data1/gdli/Dataset/Meta_data/meta_test_2'
            # meta-testing data
        root = test2  
        mini_test = SimpleImagenet(root) 
        self.db_test = DataLoader(mini_test, args.task_num, shuffle=True, num_workers=0, pin_memory=False) #batch_size = task_num = 1  num_workers=1, pin_memory=True
        
        state=torch.load('./results/models/epoch_49iter18999.pth.tar', weights_only=True)  
        self.net.load_state_dict(state['reconstructor_state'])

        aver_psnr, aver_ssim, aver_rmse=0.0, 0.0, 0.0
        aver_time=0.0
        time_start=time.time()
        with torch.no_grad():
            for num_iter, (x, y) in enumerate(self.db_test):   
                x, y = x.to(self.device), y.to(self.device)
            
                time_start_in=time.time()
                pred_q = self.net(x, self.net.parameters(), bn_training=True) ##
                reconstructed_image = pred_q 
                time_end_in=time.time()
                curr_psnr, curr_ssim, curr_rmse = self.calculate_metric(reconstructed_image, y)
                if num_iter  % 1000 == 0:
                    print('This is the {}-th infering slice, the psnr and ssim is {:.2f} {:.4f}'.format(num_iter, curr_psnr, curr_ssim))
                aver_psnr+=curr_psnr
                aver_ssim+=curr_ssim
                aver_rmse+=curr_rmse
                aver_time+=(time_end_in-time_start_in)
                
        
            time_end=time.time()
            aver_psnr=aver_psnr/(num_iter+1) #Average PSNR per iteration
            aver_ssim=aver_ssim/(num_iter+1)
            aver_rmse=aver_rmse/(num_iter+1)
            aver_time=aver_time/(num_iter+1) ##Average reconstruction time per iteration
            aver_time_out=(time_end-time_start)/(num_iter+1)  #Average reconstruction and loss calculation time per iteration
            print('the average psnr and ssim rmse is {:.2f} {:.4f} {:.4f}'.format(aver_psnr, aver_ssim, aver_rmse))
            #print('the average time of inside and outside is {:.2f} and {:.2f}'.format(aver_time, aver_time_out))


    def inference_no_fine(self, pth = None):
        print(f"--------------------------------------inference_no_fine-----{self.meta_test_path}----------------------------------")
        print("pth:", pth)
        self.test_state=torch.load(pth, weights_only=True)
        self.net.load_state_dict(self.test_state['reconstructor_state'])

        aver_psnr, aver_ssim, aver_rmse=0.0, 0.0, 0.0
        aver_time=0.0
        time_start=time.time()
        with torch.no_grad():
            for num_iter, (x_spt, y_spt, x_qry, y_qry) in enumerate(self.db_test):
                # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape) #torch.Size([1, 2, 1, 256, 256])
                x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(self.device), y_spt.squeeze(0).to(self.device), \
                                            x_qry.squeeze(0).to(self.device), y_qry.squeeze(0).to(self.device)
                x, y = x_qry, y_qry
                #print('x_shape, y_shape:', x.shape, y.shape) #x_shape, y_shape: torch.Size([2, 1, 256, 256]) torch.Size([2, 1, 256, 256])
            
                time_start_in=time.time()
                pred_q = self.net(x, self.net.parameters(), bn_training=True) ##
                reconstructed_image = pred_q 
                time_end_in=time.time()
                curr_psnr, curr_ssim, curr_rmse = self.calculate_metric(reconstructed_image, y)
                if num_iter  % 200 == 0 or num_iter == self.batchsz_test - 1:
                    print('This is the {}-th infering slice, the psnr and ssim is {:.2f} {:.4f}'.format(num_iter, curr_psnr, curr_ssim))
                aver_psnr+=curr_psnr
                aver_ssim+=curr_ssim
                aver_rmse+=curr_rmse
                aver_time+=(time_end_in-time_start_in)
                if num_iter in [0,1,2,3,4]:
                    #print(reconstructed_image.shape) #torch.Size([2, 1, 256, 256])
                    reconstructed_image =  reconstructed_image[0:1,...]
                    #print(reconstructed_image.shape) # torch.Size([1, 1, 256, 256])
                    y = y[0:1,...]
                    #print(y.shape) #torch.Size([1, 1, 256, 256])
                    curr_psnr, curr_ssim, curr_rmse = self.calculate_metric(reconstructed_image, y)
                    #print(type(reconstructed_image)) #<class 'torch.Tensor'>
                    to_save_image=reconstructed_image[0,0,...].detach().cpu().numpy()  #Extract the first image and channel, detach from autograd, and convert to NumPy.
                    #print(to_save_image.shape)  #(256, 256)
                    to_save_image_=sitk.GetImageFromArray(to_save_image)  #Get a SimpleITK Image from a numpy array
                    #print("type(to_save_image):", type(to_save_image_)) # <class 'SimpleITK.SimpleITK.Image'>
                    sitk.WriteImage(to_save_image_, './results/visualization/iter_{}_{:.2f}_{:.4f}_{:.4f}_infer_query.nii.gz'.format( num_iter, curr_psnr, curr_ssim, curr_rmse))

            time_end=time.time()
            aver_psnr=aver_psnr/(num_iter+1) #Average PSNR per iteration
            aver_ssim=aver_ssim/(num_iter+1)
            aver_rmse=aver_rmse/(num_iter+1)
            aver_time=aver_time/(num_iter+1) ##Average reconstruction time per iteration
            aver_time_out=(time_end-time_start)/(num_iter+1)  #Average reconstruction and loss calculation time per iteration
            print('the average psnr and ssim rmse is {:.2f} {:.4f} {:.4f}'.format(aver_psnr, aver_ssim, aver_rmse))
            #print('the average time of inside and outside is {:.2f} and {:.2f}'.format(aver_time, aver_time_out))

    def inference_fine_tune(self, pth = None):
        print(f"--------------------------------inference_fine_tune-----{self.meta_test_path}----------------------------------")
        #self.update_step_test = 50  #args.update_step_test

        print("pth:", pth)
        psnr_s_all_test = []
        ssim_s_all_test = []
        rmse_s_all_test = []

        #aver_psnr, aver_ssim, aver_rmse=0.0, 0.0, 0.0
        for num_iter, (x_spt, y_spt, x_qry, y_qry) in enumerate(self.db_test):
        #print(x_qry.shape)
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(self.device), y_spt.squeeze(0).to(self.device), \
                                        x_qry.squeeze(0).to(self.device), y_qry.squeeze(0).to(self.device)
            #print(x_qry.shape) #torch.Size([2, 1, 256, 256])
            #state=torch.load('./results/models/epoch_13iter22999.pth.tar', weights_only=True)
            self.test_state=torch.load(pth, weights_only=True)
            self.net.load_state_dict(self.test_state['reconstructor_state'])
            self.meta_optim.load_state_dict(self.test_state['reconstructor_optimizer'])
            
            PSNR_s = [0 for _ in range(self.update_step_test + 1)] 
            SSIM_s = [0 for _ in range(self.update_step_test + 1)]
            RMSE_s = [0 for _ in range(self.update_step_test + 1)]
            with torch.no_grad():
                pred_q = self.net(x_qry, self.net.parameters(), bn_training=True)  
                reconstructed_image = pred_q    
                curr_psnr, curr_ssim, rmse = self.calculate_metric(reconstructed_image, y_qry)
                PSNR_s[0] = PSNR_s[0] + curr_psnr  
                #print("PSNR_s[0]:", PSNR_s[0])
                SSIM_s[0] = SSIM_s[0] + curr_ssim
                RMSE_s[0] = RMSE_s[0] + rmse

            for k in range(0, self.update_step_test):  #50    args.update_step_test   #corresponding the update_step_test for finetuning
                pred_s = self.net(x_spt, self.net.parameters(), bn_training=True)                
                left_loss = self.loss(pred_s,  y_spt )  
                loss_q =  torch.mean(left_loss)   
                loss_recon=loss_q  
                loss_reconstructor=loss_recon                 
                self.meta_optim.zero_grad()  
                loss_reconstructor.backward()    
                self.meta_optim.step()   
                
                with torch.no_grad():
                    pred_q = self.net(x_qry, self.net.parameters(), bn_training=True)
                    if k== 0 or (k+1)%5 ==0:
                        curr_psnr, curr_ssim, rmse=self.calculate_metric(pred_q, y_qry)
                        PSNR_s[k + 1] = PSNR_s[k + 1] + curr_psnr
                        SSIM_s[k + 1] = SSIM_s[k + 1] + curr_ssim
                        RMSE_s[k + 1] = RMSE_s[k + 1] + rmse

            PSNR_ = np.array(PSNR_s)
            SSIM_ = np.array(SSIM_s)
            RMSE_ = np.array(RMSE_s)

            psnr_s_all_test.append(PSNR_)
            ssim_s_all_test.append(SSIM_)
            rmse_s_all_test.append(RMSE_)

            if num_iter % 200 ==0 or num_iter == self.batchsz_test - 1:
                print(f"This is the {num_iter}-th inferencing slice, the current average psnr: {np.array(psnr_s_all_test).mean(axis=0).astype(np.float16)}")
            
            if num_iter in [0,1,2,3,4]:
                    #print(reconstructed_image.shape) #torch.Size([2, 1, 256, 256])
                    reconstructed_image =  pred_q[0:1,...]
                    #print(reconstructed_image.shape) # torch.Size([1, 1, 256, 256])
                    y = y_qry[0:1,...]
                    #print(y.shape) #torch.Size([1, 1, 256, 256])
                    curr_psnr, curr_ssim, curr_rmse = self.calculate_metric(reconstructed_image, y)
                    #print(type(reconstructed_image)) #<class 'torch.Tensor'>
                    to_save_image=reconstructed_image[0,0,...].detach().cpu().numpy()  #Extract the first image and channel, detach from autograd, and convert to NumPy.
                    #print(to_save_image.shape)  #(256, 256)
                    to_save_image_=sitk.GetImageFromArray(to_save_image)  #Get a SimpleITK Image from a numpy array
                    sitk.WriteImage(to_save_image_, './results/visualization/iter_{}_{:.2f}_{:.4f}_{:.4f}_fine_tune.nii.gz'.format(num_iter, curr_psnr, curr_ssim, curr_rmse))

        PSNR = np.array(psnr_s_all_test).mean(axis=0).astype(np.float16)  #de the mean of the per clown
        SSIM = np.array(ssim_s_all_test).mean(axis=0).astype(np.float16)
        RMSE = np.array(rmse_s_all_test).mean(axis=0).astype(np.float16)
        #PSNR = [float(f"{x:.2f}") for x in PSNR]  #float
        PSNR = [(f"{x:.2f}") for x in PSNR]  #str
        #SSIM = [float(f"{x:.4f}") for x in SSIM]
        SSIM = [(f"{x:.4f}") for x in SSIM]
        #RMSE = [float(f"{x:.4f}") for x in RMSE]
        RMSE = [(f"{x:.4f}") for x in RMSE]
        # print('Testing PSNR:', *PSNR)
        # print('Testing SSIM:', *SSIM)
        # print('Testing RMSE:', *RMSE)
        for i in range(self.update_step_test+1):
            #print(i, len(PSNR), len(SSIM), len(RMSE))
            if float(PSNR[i]) !=0:
                print(f"Epoch:{self.test_state['epoch']+1}. The average PSNR, SSIM, RMSE of the {i}-th update step: {PSNR[i]} {SSIM[i]} {RMSE[i]}")    
            

if __name__=='__main__':
    args = parse_args('AdaML')  # 'StaML' or 'AdaML'
    print(args)
    trainer=Trainer()
    #trainer.train()

    #trainer.inference_FBP()
    path = './results/models/epoch_47iter18999.pth.tar'  

    trainer.inference_no_fine(pth = path)
   # trainer.inference_fine_tune(pth = path)

    # base_path = './results/models'
    # file_paths = []
    # path_list = os.listdir(base_path)
    # path_list.sort(key = lambda x: int(x.split('_')[1].split('iter')[0])) 
    # for file in path_list:
    #     if file.endswith('.pth.tar'):
    #        print (file)
    #        file_path = os.path.join(base_path, file)
    #        file_paths.append(file_path)
    # for index, path in enumerate(file_paths):
    #     if 2*index+1 > 50:
    #         print("index:", index, "path:", path)
    #         trainer.inference_fine_tune(pth = path)

  
