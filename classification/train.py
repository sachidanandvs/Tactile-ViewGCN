import sys; sys.path.insert(0, '.')
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
from Trainer import ModelNetTrainer
#from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from view_gcn import view_GCN, SVCNN
from ObjectClusterDataset import ObjectClusterDataset

def loadDatasets(split='train', shuffle=True, useClusterSampling=False,no_view = 12,batch_size = 20):
    return torch.utils.data.DataLoader(
        ObjectClusterDataset(split=split, doAugment=(split=='train'), doFilter = True, sequenceLength = no_view, metaFile=args.dataset, useClusters=useClusterSampling),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=1,drop_last=True)


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
# os.environ['CUDA_VISIBLE_DEVICES']='2'
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="view-gcn")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=20)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=0)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-4)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.001)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet18")
parser.add_argument("-num_views", type=int, help="number of views", default=8)
parser.add_argument("-dataset", type=str, default='../data/classification_lite/metadata.mat') # give path of the dataset
#parser.add_argument("-val_path", type=str, default="data/modelnet40v2png_ori4/*/test")
parser.set_defaults(train=False)

def create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    seed_torch()
    args = parser.parse_args()
    pretraining = not args.no_pretraining

    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()
    # STAGE 1
    log_dir = args.name+'_stage_1'
    create_folder(log_dir)
    cnet = SVCNN(args.name, nclasses=27, pretraining=False, cnn_name="resnet18").cuda()
    optimizer = optim.SGD(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    val_loader = loadDatasets('test', False, False,1,32) 
    train_loader = loadDatasets('train', True, False,1,32)   
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=args.num_views)  #for 2
    trainer.train(30)  

    # # # STAGE 2
    log_dir = args.name+'_stage_2'
    create_folder(log_dir)
    cnet_2 = view_GCN(args.name, cnet, nclasses=27, cnn_name=args.cnn_name, num_views=args.num_views).cuda()

    resnet_param = list(map(id, cnet_2.net_1.parameters()))
    rest_of_layer = filter(lambda p: id(p) not in resnet_param , cnet_2.parameters())
    params = [{'params': cnet_2.net_1.parameters(), 'lr': 0.0001},
            {'params': rest_of_layer }]

    optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    val_loader = loadDatasets('test', False, True,args.num_views,20)  
    train_loader = loadDatasets('train', True, True,args.num_views,20)
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'view-gcn', log_dir, num_views=args.num_views)
    trainer.train(15)

    val_loader_cluster = loadDatasets('test', False, True,args.num_views,20)
    trainer1 = ModelNetTrainer(cnet_2, train_loader, val_loader_cluster, optimizer, nn.CrossEntropyLoss(), 'view-gcn', log_dir, num_views=args.num_views)
    trainer1.update_validation_accuracy(1,True)
