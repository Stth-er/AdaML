import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import optim
import numpy as np
from learner import Learner
from copy import deepcopy
from sklearn import metrics  
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
 

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Meta, self).__init__()
#Namespace(epoch=60000, n_way=5, k_spt=1, k_qry=15, imgsz=84, imgc=3, task_num=4, meta_lr=0.001, update_lr=0.01, update_step=5, update_step_test=10)
        self.method = args.method
        if self.method == 'StaML':
            self.update_learning_rate = False # True   False
        else:
            self.update_learning_rate = True
        self.reset_lr_equal_initial = False  # True    False   # alpah is reset to the initial value, i.e 0.01
        
        self.hyper_lr = args.hyper_lr
        self.update_lr = args.update_lr
        self.args = args
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)  

    def loss(self, input, target):
        return F.mse_loss(input, target)
        #return F.l1_loss(input, target) 

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry, psnr = True):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
               
        task_num, setsz, c_, h, w = x_spt.size()  #4 i.e.  batch
        querysz = x_qry.size(1)  #querysz i.e. 5*15   task_num * k_qry  1*2
        #print('querysz_forward:', querysz)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i  
        #print('losses_q:', losses_q) ##[0, 0, 0, 0, 0, 0]
        #corrects = [0 for _ in range(self.update_step + 1)]
        #PSNR_s = [0 for _ in range(self.update_step + 1)] ##
        PSNR_s = [0 for _ in range(3)] #
        SSIM_s = [0 for _ in range(3)] #

        for i in range(task_num):   # task_num = 1
            if self.update_learning_rate == True:
                if self.reset_lr_equal_initial ==True:
                    self.update_lr = self.args.update_lr
                self.update_lr = torch.tensor([self.update_lr], requires_grad=True)
                #self.update_lr_intermediate_0 = torch.tanh(self.update_lr)   
                
                self.device = torch.device(self.args.device)
                self.update_lr = self.update_lr.to(self.device)
                #print(self.update_lr.device)  #cuda:2
                #self.update_lr_intermediate_0 = self.update_lr_intermediate_0.to(self.device)  
                #print('self.update_lr_intermediate_0:', self.update_lr_intermediate_0)
            #print('first--self.update_lr:', self.update_lr)

            # 1. run the i-th task and compute loss for k=0

            pred = self.net(x_spt[i], vars=None, bn_training=True)  
            #print('pred.shape:', pred.shape)
            left_loss = self.loss (pred, y_spt[i] )  
            #print('left_loss:', left_loss)
            loss =  torch.mean(left_loss)  ##
            #print("loss", loss)
            grad = torch.autograd.grad(loss, self.net.parameters())  #Compute and return the gradients of outputs with respect to the inputs.
            
            #print("self.net.parameters():", self.net.parameters())  #
            #print("list(self.net.parameters()):", list(self.net.parameters())) #
            #print("grad:", grad) #'tuple'
            #print(self.update_lr.device) # cpu
             
            # if self.update_learning_rate == False:
            #     fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters()))) #theta - lr * grad
            # else:
            #     fast_weights = list(map(lambda p: p[1] - self.update_lr_intermediate_0 * p[0], zip(grad, self.net.parameters())))
            # #print("fast_weights:", fast_weights) #<class 'list'>
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters()))) #theta - lr * grad

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                pred_q = self.net(x_qry[i], self.net.parameters(), bn_training=True) ##
                left_loss = self.loss (pred_q,  y_qry[i] ) ##
                loss_q =  torch.mean(left_loss)  ##
                
                losses_q[0] += loss_q
                if psnr == True:
                    curr_psnr, curr_ssim, _=self.calculate_metric(pred_q, y_qry[i]) ##--------------
                    PSNR_s[0] = PSNR_s[0] + curr_psnr  ##--------------
                    SSIM_s[0] = SSIM_s[0] + curr_ssim

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                #logits_q = self.net(x_qry[i], fast_weights, bn_training=True) #fast_weights is theta_i^'
                #loss_q = F.cross_entropy(logits_q, y_qry[i])

                pred_q = self.net(x_qry[i], fast_weights, bn_training=True)  ##
                left_loss = self.loss (pred_q,  y_qry[i] ) ##
                loss_q =  torch.mean(left_loss)  ##

                losses_q[1] += loss_q
                # [setsz]
                # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                # correct = torch.eq(pred_q, y_qry[i]).sum().item()
                # corrects[1] = corrects[1] + correct
                if psnr == True:
                    curr_psnr, curr_ssim, _=self.calculate_metric(pred_q, y_qry[i]) ## ##--------------
                    PSNR_s[1] = PSNR_s[1] + curr_psnr  ## ##--------------
                    SSIM_s[1] = SSIM_s[1] + curr_ssim

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                pred_q = self.net(x_spt[i], fast_weights, bn_training=True)   
                left_loss = self.loss (pred_q,  y_spt[i] )  
                loss =  torch.mean(left_loss) 
                #print(k, 'loss:', loss) #tensor(2.8576, device='cuda:0')
                #print('fast_weights:', fast_weights)
                del grad
                torch.cuda.empty_cache()
                
                if self.update_learning_rate == True:
                    grad_lr = torch.autograd.grad(loss, self.update_lr, allow_unused = True, retain_graph=True)
                    #print('grad_lr[0]:', grad_lr[0])
                    grad_lr = grad_lr[0]
                    #print(k, 'grad_lr[0]:', grad_lr)
                    grad_lr = torch.tanh(grad_lr)  
                    #print(k, 'grad_lr:', grad_lr) #grad_lr: tensor([-1.6674e-05], device='cuda:3')
                    #self.update_lr = self.update_lr - self.hyper_lr * grad_lr
                    self.update_lr = self.update_lr - self.hyper_lr * grad_lr
                    #print(f'{k}, self.update_lr:{self.update_lr}') #tensor([0.0004], device='cuda:3', grad_fn=<ToCopyBackward0>) 
                    #self.update_lr_intermediate_i.requires_grad_(True)
                    del grad_lr

                    # if k>1:
                    #     print('grad_loss_intermediate_i:', torch.autograd.grad(loss, self.update_lr_intermediate_i, allow_unused = True, retain_graph=True))
                    # grad_lr = torch.autograd.grad(loss, self.update_lr, allow_unused = True, retain_graph=True)
                    # #print('grad_lr[0]:', grad_lr[0])
                    # grad_lr = grad_lr[0]
                    # #print(k, 'grad_lr[0]:', grad_lr)
                    # #grad_lr = torch.tanh(grad_lr)  
                    # print(k, 'grad_lr:', grad_lr) #grad_lr: tensor([-1.6674e-05], device='cuda:3')
                    # #self.update_lr = self.update_lr - self.hyper_lr * grad_lr
                    # self.update_lr = self.update_lr - self.hyper_lr * grad_lr
                    # self.update_lr_intermediate_i = torch.tanh(self.update_lr)
                    # #self.update_lr = torch.tanh(self.update_lr)  
                    # #self.update_lr = torch.tanh(self.update_lr) 
                    # print(f'{k}, self.update_lr_intermediate_i:{self.update_lr_intermediate_i}') #tensor([0.0004], device='cuda:3', grad_fn=<ToCopyBackward0>) 
                    # #self.update_lr_intermediate_i.requires_grad_(True)
                    # self.update_lr_intermediate_i = self.update_lr_intermediate_i.to(self.device)
                    # print(self.update_lr.device) #cpu
                    # print(self.update_lr_intermediate_i.device)  #cuda:3
                    # del grad_lr

                # 2. compute grad on theta_pi
                #print('fast_weights_forward:', fast_weights)
                grad = torch.autograd.grad(loss, fast_weights)

                # 3. theta_pi = theta_pi - train_lr * grad
                # if self.update_learning_rate == False:
                #     fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                # else:
                #     fast_weights = list(map(lambda p: p[1] - self.update_lr_intermediate_i * p[0], zip(grad, fast_weights)))
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                pred_q = self.net(x_qry[i], fast_weights, bn_training=True) ##
                left_loss = self.loss (pred_q,  y_qry[i] ) ##
                loss_q =  torch.mean(left_loss)  ##
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    if psnr == True:
                        if k + 1 == self.update_step:
                            curr_psnr, curr_ssim, _=self.calculate_metric(pred_q, y_qry[i]) ##
                            #PSNR_s[k + 1] = PSNR_s[k + 1] + curr_psnr  ##
                            PSNR_s[-1] = PSNR_s[-1] + curr_psnr
                            SSIM_s[-1] = SSIM_s[-1] + curr_ssim
        #print('final self.update_lr:', self.update_lr)

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        #accs = np.array(corrects) / (querysz * task_num)
        #PSNR_ = np.array(PSNR_s) / (querysz * task_num) ###wait validate????????
        PSNR_ = np.array(PSNR_s) / (task_num)      #############
        SSIM_ = np.array(SSIM_s) / (task_num) 

        #print("PSNR_accs:", PSNR_)
        #print(self.update_lr.data) #tensor([0.0160], device='cuda:2')
        #print(type(self.update_lr.data)) #<class 'torch.Tensor'>
        #print(type(self.update_lr.item()))
        if self.update_learning_rate == True:
            self.update_lr = self.update_lr.item()
        return PSNR_, SSIM_, self.update_lr

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4
        querysz = x_qry.size(0)
        #print('querysz_finetunning:', querysz)

        if self.update_learning_rate == True:
            #torch.cuda.empty_cache()
            if self.reset_lr_equal_initial ==True:  # in  testing time, must initial to 0.01 
                self.update_lr = self.args.update_lr 
            # need introduction the saved update_lr, so we can modify the functon finetuning (..., first_lr = self.update_lr)
            # first_update_lr = first_lr
            # first_update_lr =  self.args.update_lr
            self.update_lr = torch.tensor([self.update_lr], requires_grad=True)
            self.device = torch.device(self.args.device)
            self.update_lr = self.update_lr.to(self.device)  
            #print(self.update_lr.device)  #cuda:2
        #print('first--self.update_lr:', self.update_lr)
        
        #corrects = [0 for _ in range(self.update_step_test + 1)]
        PSNR_s = [0 for _ in range(self.update_step_test + 1)] 
        SSIM_s = [0 for _ in range(self.update_step_test + 1)]
        RMSE_s = [0 for _ in range(self.update_step_test + 1)]
        #PSNR_s = [0 for _ in range(3)]  ##########----------
        #print(PSNR_s)
        #SSIM_s = [0 for _ in range(3)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        pred = net(x_spt)
        left_loss = self.loss (pred, y_spt) ##
        loss =  torch.mean(left_loss)  ##
        
        grad = torch.autograd.grad(loss, net.parameters())
        #print('grad00:', type(grad)) # <class 'tuple'>
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            #logits_q = net(x_qry, net.parameters(), bn_training=True)
            pred_q = net(x_qry, net.parameters(), bn_training=True)  
            curr_psnr, curr_ssim, rmse=self.calculate_metric(pred_q, y_qry) ##
            #print('curr_psnr_:', curr_psnr)  #5.06334566584
            PSNR_s[0] = PSNR_s[0] + curr_psnr  ##
            #print("PSNR_s[0]:", PSNR_s[0])
            SSIM_s[0] = SSIM_s[0] + curr_ssim
            RMSE_s[0] = RMSE_s[0] + rmse

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            pred_q = net(x_qry, fast_weights, bn_training=True)
            curr_psnr, curr_ssim, rmse=self.calculate_metric(pred_q, y_qry) ##---------------------PSNR_s[1] = 0
            PSNR_s[1] = PSNR_s[1] + curr_psnr  ##--------------------------PSNR_s[1] = 0
            SSIM_s[1] = SSIM_s[1] + curr_ssim
            RMSE_s[1] = RMSE_s[1] + rmse

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            pred = net(x_spt, fast_weights, bn_training=True)
            left_loss = self.loss(pred,  y_spt )
            #print('left_loss:', left_loss) 
            loss =  torch.mean(left_loss)   
            #print(k, 'loss:', loss)
            #print('fast_weights:', fast_weights)
            #print('l:', loss.device)
            del grad
            torch.cuda.empty_cache()

            if self.update_learning_rate == True:
                #pass
                grad_lr = torch.autograd.grad(loss, self.update_lr, allow_unused = True, retain_graph=True)
                #print('grad_lr[0]:', grad_lr[0])
                grad_lr = grad_lr[0]
                grad_lr = torch.tanh(grad_lr)
                #print(k, 'grad_lr:', grad_lr) #grad_lr: tensor([-1.6674e-05], device='cuda:3') 
                #print(type(grad_lr)) #<class 'tuple'>      [0]   <class 'torch.Tensor'>
                #print(f'grad_lr:{grad_lr}')
                self.update_lr = self.update_lr - self.hyper_lr * grad_lr 
                #print(f'{k}, self.update_lr:{self.update_lr}') #tensor([0.0004], device='cuda:3', grad_fn=<ToCopyBackward0>)
                self.update_lr.requires_grad_(True)
                del  grad_lr
                 
            # 2. compute grad on theta_pi
            #grad_0 = grad
            grad = torch.autograd.grad(loss, fast_weights)
            #print('grad:', grad.shape) 
             
            #print('grad_lr:', grad_lr)
            # grad_lr = grad_lr.item()
            #self.update_lr = self.update_lr.item()
            
            #print(self.update_lr)  
            #self.update_lr =  list(map(lambda p: p[1] - self.hyper_lr* p[0], zip(grad_lr, self.update_lr))) #0.0009
            #grad_lr = grad_lr.item()
            
                                    

            # 3. theta_pi = theta_pi - train_lr * grad
            
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            
            with torch.no_grad():
                pred_q = net(x_qry, fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                left_loss = self.loss (pred_q,  y_qry )  
                loss_q =  torch.mean(left_loss)  
            
                if (k+1)%5 ==0:
                    curr_psnr, curr_ssim, rmse=self.calculate_metric(pred_q, y_qry)
                    PSNR_s[k + 1] = PSNR_s[k + 1] + curr_psnr
                    SSIM_s[k + 1] = SSIM_s[k + 1] + curr_ssim
                    RMSE_s[k + 1] = RMSE_s[k + 1] + rmse
                 
                #if k + 1 == self.update_step_test:
                    #curr_psnr, curr_ssim, _=self.calculate_metric(pred_q, y_qry) ##
                #PSNR_s[k + 1] = PSNR_s[k + 1] + curr_psnr 
                    #PSNR_s[-1] = PSNR_s[-1] + curr_psnr  ##
                #print('k+1, PSNR_s[k + 1]:', k+1, PSNR_s[k + 1])
                    #SSIM_s[-1] = SSIM_s[-1] + curr_ssim
       
        #print('final self.update_lr:', self.update_lr)
        del net, self.update_lr
        #print('self.update_lr:', self.update_lr)  #tensor([0.0004], device='cuda:3', grad_fn=<ToCopyBackward0>)

        #accs = np.array(corrects) / querysz
        #print('PSNR_s:', PSNR_s)
        #PSNR_ = np.array(PSNR_s) / querysz
        PSNR_ = np.array(PSNR_s)
        SSIM_ = np.array(SSIM_s)
        RMSE_ = np.array(RMSE_s)
 
        return PSNR_, SSIM_, RMSE_, pred_q
    

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


    def save_checkpoint(self, num_epoch, state, num_iter):
        base_path='./results/meta_models'
        save_path=base_path+'/'+'epoch_'+str(num_epoch)+'iter'+str(num_iter)+'.pth.tar'
        torch.save(state, save_path)
        print('Save model after {}-th epoch'.format(num_epoch))

def main():
    pass


if __name__ == '__main__':
    main()
