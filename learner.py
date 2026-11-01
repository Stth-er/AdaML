import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np



class Learner(nn.Module):

    def __init__(self, config, imgc= 84, imgsz= 84):
        """
        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()
        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        #print(type(self.vars)) #<class 'torch.nn.modules.container.ParameterList'>
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            #if name is 'conv2d':
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                #print(w) #
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                
                w = nn.Parameter(torch.ones(*param[:4]))  #
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':          #'linear', [args.n_way, 32 * 5 * 5]
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w) #Scale factor
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #bias

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name == 'bn_4_1':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w) #Scale factor
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #bias

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name == 'bn_3_1':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w) #Scale factor
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #bias

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name == 'bn_2_1':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w) #Scale factor
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #bias

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name == 'bn_1_1':                
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w) #Scale factor
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #bias

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            #elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
            #              'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid','concate_4_2','concate_3_2','concate_2_2','concate_1_2','skip_connection']:  
                
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            #elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn', 'bn_4_1','bn_3_1','bn_2_1','bn_1_1',
                        'concate_4_2','concate_3_2','concate_2_2','concate_1_2','skip_connection']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            #else:
                #raise NotImplementedError
                #continue  #debug

        return info

    def forward(self, x, vars=None, bn_training=True):
        #print(x.shape) #torch.Size([5, 3, 84, 84])  #torch.Size([75, 3, 84, 84])
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        x_0 = x
        for name, param in self.config:
            #print(name, param)
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                #print("w, b:", w, b) #
                #print(x.shape) #torch.Size([5, 3, 84, 84])
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                
                idx += 2
                #print(name, param, '\tout:', x.shape) #conv2d [32, 3, 3, 3, 1, 0]      out: torch.Size([5, 32, 82, 82])
            elif name == 'convt2d':
                #('convt2d', [1024, 512, 3, 3, 2, 1, 1])
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                #x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])  
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5], output_padding = param[6]) ##
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training) 
                                   #Update running'mean and running-var with statistical information from the current batch
                                   # weight:scale factor, bias:
                                   #training:True Normalize using the mean and variance of the current batch, and update running'mean and running-var
                #print('bn shape:', x.shape)
                idx += 2
                bn_idx += 2
            elif name == 'bn_4_1':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training) 
                                   #Update running'mean and running-var with statistical information from the current batch
                                   # weight:scale factor, bias:
                                   #training:True Normalize using the mean and variance of the current batch, and update running'mean and running-var
                x_4 = x
                #print("x_4.shape:", x_4.shape) # torch.Size([5, 512, 64, 64])
                idx += 2
                bn_idx += 2
            elif name == 'bn_3_1':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training) 
                                   #Update running'mean and running-var with statistical information from the current batch
                                   # weight:scale factor, bias:
                                   #training:True Normalize using the mean and variance of the current batch, and update running'mean and running-var
                x_3 = x
                #print("x_3.shape:", x_3.shape) # torch.Size([5, 512, 64, 64])
                idx += 2
                bn_idx += 2
            elif name == 'bn_2_1':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training) 
                                   #Update running'mean and running-var with statistical information from the current batch
                                   # weight:scale factor, bias:
                                   #training:True Normalize using the mean and variance of the current batch, and update running'mean and running-var
                x_2 = x
                #print("x_2.shape:", x_2.shape) # torch.Size([5, 512, 64, 64])
                idx += 2
                bn_idx += 2
            elif name == 'bn_1_1':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training) 
                                   #Update running'mean and running-var with statistical information from the current batch
                                   # weight:scale factor, bias:
                                   #training:True Normalize using the mean and variance of the current batch, and update running'mean and running-var
                x_1 = x
                #print("x_1.shape:", x_1.shape) # torch.Size([5, 64, 512, 512])
                idx += 2
                bn_idx += 2

            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)    #(b, c, h, w)-->(b, c*h*w)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0]) #('relu', [True])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])  #('max_pool2d', [2, 2, 0])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            

            elif name == 'concate_4_2': #
                x = torch.cat((x_4, x), dim=1) ##
                #print('concate_4_2 shape:', x.shape)
            elif name == 'concate_3_2': #
                x = torch.cat((x_3, x), dim=1) ##
                #print('concate_3_2 shape:', x.shape)
            elif name == 'concate_2_2': #
                x = torch.cat((x_2, x), dim=1) ##
                #print('concate_2_2 shape:', x.shape)
            elif name == 'concate_1_2': #
                x = torch.cat((x_1, x), dim=1) ##
                #print('concate_1_2 shape:', x.shape)
            
            elif name == 'skip_connection':
                x =  x + x_0
                #print('output.shape:', x.shape)

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        #print(x.shape) #torch.Size([5, 5])   torch.Size([75, 5])
        return x


    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
    
if __name__ == "__main__":
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
    ('skip_connection', [0,0])
    ]
        
    net = Learner(config)
    print(net)
    input = torch.randn(5,1,512,512) 
    
    print('input.shape', input.shape) #torch.Size([5, 1, 512, 512])
    out = net(input, vars=None, bn_training=True)
    print('ouput.shape', out.shape)