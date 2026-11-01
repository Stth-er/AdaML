import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"   
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import SimpleITK as sitk
from sklearn import metrics  
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from Imagenet import Imagenet
import scipy.stats
from torch.utils.data import DataLoader
from glob import glob
import time
import shutil
from meta_bool import Meta
from utils import  parse_args 
from Imagenet_no_replace import Imagenet_No_replace



class Trainer(object):
    def __init__(self, is_restart=False, max_epoch= 50 , num_view=96):
        self.num_view=num_view
        self.is_restart=is_restart
        self.max_epoch=max_epoch* args.task_num
        self.method = args.method
        
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

        self.device = torch.device(args.device)
        self.StaML = Meta(args, config)
        #StaML = nn.DataParallel(StaML, device_ids=[1, 2])  
        self.StaML.to(self.device)
        print("Model device:", next(self.StaML.parameters()).device)

        tmp = filter(lambda x: x.requires_grad, self.StaML.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp)) #np.prod():Calculate product  
        #print('Total trainable tensors:', num) 
                 
        train1 = '/data1/gdli/Dataset/Meta_data/meta_train'
        test1 = '/data1/gdli/Dataset/Meta_data/meta_test_1' #  '/data1/gdli/Dataset/Meta_data/meta_test_2'  #  

         
        meta_train_path  =  train1
        meta_test_path =   test1
        train_data_slice = glob(os.path.join(meta_train_path,'**/*.dcm'), recursive=True)
        test_data_slice = glob(os.path.join(meta_test_path,'**/*.dcm'), recursive=True)
        batchsz_train = len(train_data_slice)   
        print('len(train_data_slice)', len(train_data_slice), 'len(test_data_slice):', len(test_data_slice))
        
        # batchsz here means total episode number
        mini = Imagenet(meta_train_path, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry,
                            batchsz = batchsz_train, resize=args.imgsz, num_view= args.num_view)  ## n_way*(k_shot+k_query) per batchsz, thus may duplicate for some image

        self.db = DataLoader(mini, args.task_num, shuffle=True, num_workers = 0, pin_memory= True) #batch_size = task_num = 1  num_workers=1, pin_memory=True

        torch.manual_seed(3) #Sets the seed for generating random numbers on all devices
        torch.cuda.manual_seed_all(3)  #Set the seed for generating random numbers on all GPUs
        np.random.seed(3)
        self.meta_test_path =   test1
        test_data_slice = glob(os.path.join(self.meta_test_path,'**/*.dcm'), recursive=True)
        self.batchsz_test = len(test_data_slice)//(args.k_spt+args.k_qry) #1000 #len(test_data_slice)  
        
        mini_test = Imagenet_No_replace(self.meta_test_path, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                k_query=args.k_qry,
                                batchsz = self.batchsz_test, resize=args.imgsz, num_view= args.num_view)  
        self.db_test = DataLoader(mini_test, 1, shuffle= False, num_workers=0, pin_memory=True) #shuffle=True, num_workers=1, pin_memory=True
        
        folders = ['./results/models', './results/models_StaML', './results/visualization', './results/models_AdaML']
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

        if self.is_restart:
            shutil.rmtree('./results/visualization')  
            os.mkdir('./results/visualization')
            self.epoch=0
            self.global_iter=0   
            self.best_loss=np.inf  
            print('Training process started')
        else:
            try:
                print('method:', self.method)
                if self.method == 'StaML':    
                    state=torch.load('./results/models_StaML/StaML_epoch_49iter9499.pth.tar', weights_only=True) #StaML
                else:
                    #print('AdaML state is loaded')
                    state=torch.load('./results/models_AdaML/AdaML_epoch_23iter18999.pth.tar', weights_only=True)
                    print('AdaML state')
                
                #print(state['reconstructor_optimizer'])
                self.epoch=state['epoch']+1
                #print(self.epoch)
                self.StaML.net.load_state_dict(state['reconstructor_state'])
                self.StaML.meta_optim.load_state_dict(state['reconstructor_optimizer'])
                self.StaML.update_lr = state['final_update_lr']
                print('Saved ckpt is loaded successfully, the saved update_lr:', self.StaML.update_lr) 
            except:
                self.epoch=0
                print('There is no saved ckpt file to load, the training process is restarted')
        print('Settings are finished')

    def train_StaML(self):
        print(f"------------------------------------- method: {self.method} ---- train ----------------------------------------")
        for epoch in range(self.epoch, self.max_epoch): # args.task_num = 1       
            # fetch meta_batchsz num of episode each time
            start_time = time.time()
            for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(self.db):
                #print("x_spt.shape:", x_spt.shape) #torch.Size([1, 1, 1, 512, 512])
                # print(y_spt.shape)
                # print(x_qry.shape)
                #print(y_qry.shape)  #torch.Size([1, 3, 1, 512, 512])  (args.task_num, n_way*k_query, c,h,w)

                x_spt, y_spt, x_qry, y_qry = x_spt.to(self.device), y_spt.to(self.device), x_qry.to(self.device), y_qry.to(self.device)
                if step % 1000 == 0:     #1000
                    PSNR, SSIM, update_lr = self.StaML(x_spt, y_spt, x_qry, y_qry, psnr = True)
                    PSNR = [float(f"{x:.2f}") for x in PSNR]
                    SSIM = [float(f"{x:.4f}") for x in SSIM]
                    print('epoch:', epoch, 'step:', step, '\ttraining PSNR:', PSNR, '\ttraining SSIM:', SSIM)
                    #end_time = time.time()
                    #print( f"This epoch training  time: {(end_time - start_time)/60 :.1f} minutes")
                else: 
                    PSNR, SSIM, update_lr = self.StaML(x_spt, y_spt, x_qry, y_qry, psnr = False)  # must exist this line, can backward
                #print('main_StaML update_lr:', update_lr)
            #save model
            if epoch %2== 1:  # #step%100==0
                state={'epoch': epoch,
                        'reconstructor_state': self.StaML.net.state_dict(),
                        'reconstructor_optimizer': self.StaML.meta_optim.state_dict(),
                        'final_update_lr': update_lr,
                            }
                self.save_checkpoint(epoch, state, step)
                print(f"Saving model at epoch {epoch}, step {step}, update_lr: {update_lr}")
            end_time = time.time()
            print( f"This epoch training time: {(end_time - start_time)/60 :.1f} minutes")

    def save_checkpoint(self, num_epoch, state, num_iter):
        if self.method == 'StaML':
            base_path='./results/models_StaML'
            save_path=base_path+'/'+'StaML_epoch_'+str(num_epoch)+'iter'+str(num_iter)+'.pth.tar'
        else:
            base_path='./results/models_AdaML'
            save_path=base_path+'/'+'AdaML_epoch_'+str(num_epoch)+'iter'+str(num_iter)+'.pth.tar'
        torch.save(state, save_path)
        #print('Save model after {}-th epoch'.format(num_epoch))


    def calculate_metric(self, pred, gt):
            assert len(pred.shape)==4 and pred.shape==gt.shape  #Raise AssertionError if the shape requirements are not met
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
                    curr_ssim = structural_similarity(pred[i, j, ...], gt[i, j, ...], gaussian_weights=True, win_size=11, data_range=2.0, sigma=1.5)
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

    def inference_StaML(self, pth=None):
        # meta-testing
        print(f"----------------------------------inference_StaML-----{self.meta_test_path}----------------------------------")
        print('pth:', pth)  
        psnr_s_all_test = []
        ssim_s_all_test = []
        rmse_s_all_test = []

        #for x_spt, y_spt, x_qry, y_qry in self.db_test:
        for num_iter, (x_spt, y_spt, x_qry, y_qry) in enumerate(self.db_test):
            if self.method == 'StaML':
                state=torch.load(pth, weights_only=True)
            else:
                state=torch.load(pth, weights_only=True)
            self.StaML.net.load_state_dict(state['reconstructor_state'])
            self.StaML.update_lr = state['final_update_lr']
            if num_iter ==0:
                print('the saved update_lr:', self.StaML.update_lr)

            #torch.cuda.empty_cache()  #  CUDA out of memory
            #print('x_spt.shape:', x_spt.shape)
            #print('y_spt.shape', y_spt.shape)
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(self.device), y_spt.squeeze(0).to(self.device), \
                                            x_qry.squeeze(0).to(self.device), y_qry.squeeze(0).to(self.device)
            #print('x_spt.shape:', x_spt.shape) #torch.Size([1, 1, 256, 256])
            #print('y_qry.shape', y_qry.shape) #torch.Size([2, 1, 256, 256])
            
            PSNR, SSIM, RMSE, pred_q = self.StaML.finetunning(x_spt, y_spt, x_qry, y_qry)
            psnr_s_all_test.append(PSNR)
            ssim_s_all_test.append(SSIM)
            rmse_s_all_test.append(RMSE)
            if num_iter % 200 ==0 or num_iter == self.batchsz_test - 1:
                print(f"This is the {num_iter}-th inferencing slice, the current average psnr: {np.array(psnr_s_all_test).mean(axis=0).astype(np.float16)}")
            
            if num_iter <1:  # save the first slice of num_iter   
                reconstructed_image =  pred_q[0:1,...]
                #print(reconstructed_image.shape) # torch.Size([1, 1, 256, 256])
                y = y_qry[0:1,...]                  
                curr_psnr, curr_ssim, curr_rmse = self.calculate_metric(reconstructed_image, y)
                #print(type(reconstructed_image)) #<class 'torch.Tensor'>
                to_save_image=reconstructed_image[0,0,...].detach().cpu().numpy()  #Extract the first image and channel, detach from autograd, and convert to NumPy.                     
                to_save_image_=sitk.GetImageFromArray(to_save_image)  #Get a SimpleITK Image from a numpy array                    
                #print("type(to_save_image):", type(to_save_image_)) # <class 'SimpleITK.SimpleITK.Image'>
                if self.method == 'StaML':
                    sitk.WriteImage(to_save_image_, './results/visualization/iter_{}_{:.2f}_{:.4f}_{:.4f}_infer_StaML.nii.gz'.format(num_iter, curr_psnr, curr_ssim, curr_rmse))
                else:
                    sitk.WriteImage(to_save_image_, './results/visualization/iter_{}_{:.2f}_{:.4f}_{:.4f}_infer_AdaML.nii.gz'.format(num_iter, curr_psnr, curr_ssim, curr_rmse))

            #if num_iter ==2:
                # [b, update_step+1]                                     
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
        for i in range(args.update_step_test+1):
            if float(PSNR[i]) !=0:
                print(f"Epoch:{state['epoch']+1}. The average PSNR, SSIM, RMSE of the {i}-th update step: {PSNR[i]} {SSIM[i]} {RMSE[i]}")

if __name__ == '__main__':
    args = parse_args('AdaML')  # 'StaML' or 'AdaML'
    print(args)
    trainer = Trainer()
    #trainer.train_StaML()
    inference_StaML =  True # True or False, whether to inference   
    if inference_StaML:
        if args.method == 'StaML':
            trainer.inference_StaML(pth = './results/models_StaML/StaML_epoch_49iter18999.pth.tar')
        else:
            trainer.inference_StaML(pth = './results/models_StaML/StaML_epoch_49iter18999.pth.tar')

    # base_path = './results/models_StaML' if args.method == 'StaML' else './results/models_AdaML'
    # file_paths = []
    # path_list = os.listdir(base_path)
    # path_list.sort(key = lambda x: int(x.split('_')[2].split('iter')[0]) if args.method == 'StaML' else int(x.split('_')[3].split('iter')[0]))  # sort by epoch number
    # for file in path_list:
    #     if file.endswith('.pth.tar'):
    #        print (file)
    #        file_path = os.path.join(base_path, file)
    #        file_paths.append(file_path)
    # for index, path in enumerate(file_paths):
    #     if 2*index+1 > 64:
    #         print("index:", index, "path:", path)
    #         inference_StaML = True  # True or False, whether to inference   
    #         if inference_StaML:
    #             if args.method == 'StaML':
    #                 trainer.inference_StaML(pth = path) #'./results/models_StaML/StaML_epoch_49iter18999.pth.tar'
    #             else:
    #                 trainer.inference_StaML(pth = path) #'./results/models_AdaML/AdaML_epoch_49iter18999.pth.tar'
