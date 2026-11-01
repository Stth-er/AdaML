import argparse

def parse_args(script):
    argparser = argparse.ArgumentParser()
    #argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=1)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default= 1 ) # 1
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default= 2) #2
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=256 ) #512  256
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default= 1 )
    argparser.add_argument('--base_lr', type=float, help='base learning rate', default=0.0001)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default= 0.001 )
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default= 0.001 ) #0.01
    argparser.add_argument('--hyper_lr', type=float, help='learning rate of alpha in AdaML', default= 0.2 ) #0.1
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5) #default=5
    argparser.add_argument('--num_view', type=int, help='number of views', default=90) #default=90
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default = 10 ) #default=10
    
    argparser.add_argument('--device', type= str, help='the device for StaML', default = 'cuda:1') #StaML
    argparser.add_argument('--device_baseline', type= str, help='the device', default = 'cuda:0') #baseline   
    
    if script in ('StaML', 'AdaML'):
        argparser.add_argument('--method', choices=('StaML', 'AdaML'), default=script, help='method name')

    

    return argparser.parse_args()
