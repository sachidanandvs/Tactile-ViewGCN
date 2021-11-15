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
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-dataset", type=str, default='/home/sachi/Desktop/college_project/data/classification_lite/metadata.mat')
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
    print(cnet)
    optimizer = optim.SGD(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    val_loader = loadDatasets('test', False, False,1,32)  #for 2
    train_loader = loadDatasets('train', True, False,1,32)   #for 2

    '''n_models_train = args.num_models*args.num_views
    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=400, shuffle=True, num_workers=4)
    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=400, shuffle=False, num_workers=4)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))'''
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=args.num_views)  #for 2
    trainer.train(30)   #for 2

    # # # STAGE 2
    log_dir = args.name+'_stage_2'
    create_folder(log_dir)
    cnet_2 = view_GCN(args.name, cnet, nclasses=27, cnn_name=args.cnn_name, num_views=args.num_views).cuda()
    optimizer = optim.SGD(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
    val_loader = loadDatasets('test', False, False,12,20)
    
    train_loader = loadDatasets('train', True, False,12,20)
    '''
    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views,test_mode=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)# shuffle needs to be false! it's done within the trainer
    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views,test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    '''
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'view-gcn', log_dir, num_views=args.num_views)
    #use trained_view_gcn
    #cnet_2.load_state_dict(torch.load('trained_view_gcn.pth'))
    #trainer.update_validation_accuracy(1)
    trainer.train(5)
    val_loader_cluster = loadDatasets('test', False, True,12,20)
    trainer1 = ModelNetTrainer(cnet_2, train_loader, val_loader_cluster, optimizer, nn.CrossEntropyLoss(), 'view-gcn', log_dir, num_views=args.num_views)
    trainer1.update_validation_accuracy(1)
