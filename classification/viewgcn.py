import argparse
import os
import shutil
import time, math, datetime, re
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.autograd.variable import Variable
import numpy as np
import Model
from view_gcn_utils import KNN_dist, View_selector, LocalGCN, NonLocalMP

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class SVCNN(Model):
    def __init__(self, name, nclasses=26, pretraining=True, cnn_name='resnet18'):
        super(SVCNN, self).__init__(name)
        '''
        self.classnames = ['Allen_key_set','Ball','Battery','Board_eraser','Bracket','Brain','Cat','Chain','Clip','Coin',
                            'Gel','Kiwano','Lotion','Mug','Multimeter','Pen','Safety_glasses']'''
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11_bn(pretrained=self.pretraining).features
                self.net_2 = models.vgg11_bn(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0], -1))

class view_GCN(Model):

    def __init__(self, name, model, nclasses=26, cnn_name='resnet18', num_views=12):
        super(view_GCN, self).__init__(name)
        self.nclasses = nclasses
        self.num_views = num_views
        self.use_resnet = cnn_name.startswith('resnet')
        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2
        if self.num_views == 20:
            phi = (1 + np.sqrt(5)) / 2
            vertices = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                        [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
                        [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
                        [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0]]
        elif self.num_views == 12:
            phi = np.sqrt(3)
            vertices = [[1, 0, phi/3], [phi/2, -1/2, phi/3], [1/2,-phi/2,phi/3],
                        [0, -1, phi/3], [-1/2, -phi/2, phi/3],[-phi/2, -1/2, phi/3],
                        [-1, 0, phi/3], [-phi/2, 1/2, phi/3], [-1/2, phi/2, phi/3],
                        [0, 1 , phi/3], [1/2, phi / 2, phi/3], [phi / 2, 1/2, phi/3]]
        self.vertices = torch.tensor(vertices).cuda()

        self.LocalGCN1 = LocalGCN(k=4,n_views=self.num_views)
        self.NonLocalMP1 = NonLocalMP(n_view=self.num_views)
        self.LocalGCN2 = LocalGCN(k=4, n_views=self.num_views//2)
        self.NonLocalMP2 = NonLocalMP(n_view=self.num_views//2)
        self.LocalGCN3 = LocalGCN(k=4, n_views=self.num_views//4)
        self.View_selector1 = View_selector(n_views=self.num_views, sampled_view=self.num_views//2)
        self.View_selector2 = View_selector(n_views=self.num_views//2, sampled_view=self.num_views//4)

        self.cls = nn.Sequential(
            nn.Linear(512*3,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout(),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512, self.nclasses)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        views = self.num_views
        y = self.net_1(x)
        y = y.view((int(x.shape[0] / views), views, -1))
        vertices = self.vertices.unsqueeze(0).repeat(y.shape[0], 1, 1)

        y = self.LocalGCN1(y,vertices)
        y2 = self.NonLocalMP1(y)
        pooled_view1 = torch.max(y, 1)[0]

        z, F_score, vertices2 = self.View_selector1(y2,vertices,k=4)
        z = self.LocalGCN2(z,vertices2)
        z2 = self.NonLocalMP2(z)
        pooled_view2 = torch.max(z, 1)[0]

        w, F_score2, vertices3 = self.View_selector2(z2,vertices2,k=4)
        w = self.LocalGCN3(w,vertices3)
        pooled_view3 = torch.max(w, 1)[0]

        pooled_view = torch.cat((pooled_view1,pooled_view2,pooled_view3),1)
        pooled_view = self.cls(pooled_view)
        return pooled_view,F_score,F_score2

    

class ClassificationModel(BaseModel):
    '''
    This class encapsulates the network and handles I/O.
    '''

    @property
    def name(self):
        return 'ClassificationModel'


    def initialize(self, numClasses, sequenceLength = 1, baseLr = 1e-3):
        BaseModel.initialize(self)

        print('Base LR = %e' % baseLr)
        self.baseLr = baseLr
        self.numClasses = numClasses
        self.sequenceLength = sequenceLength

        self.model = TouchNet(num_classes = self.numClasses, nFrames = self.sequenceLength)
        self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()
        cudnn.benchmark = True

        self.optimizer = torch.optim.Adam([
            {'params': self.model.module.parameters(),'lr_mult': 1.0},          
            ], self.baseLr)

        self.optimizers = [self.optimizer]

        self.criterion = nn.CrossEntropyLoss().cuda()

        self.epoch = 0
        self.error = 1e20 # last error
        self.bestPrec = 1e20 # best error

        self.dataProcessor = None

            
    def step(self, inputs, isTrain = True, params = {}):

        if isTrain:
            self.model.train()
            assert not inputs['objectId'] is None
        else:
            self.model.eval()


        image = torch.autograd.Variable(inputs['image'].cuda(async=True), requires_grad = (isTrain))
        pressure = torch.autograd.Variable(inputs['pressure'].cuda(async=True), requires_grad = (isTrain))
        objectId = torch.autograd.Variable(inputs['objectId'].cuda(async=True), requires_grad=False) if 'objectId' in inputs else None
    
        if isTrain:
            output = self.model(pressure)   
        else:
            with torch.no_grad():
                output = self.model(pressure)  
        
        _, pred = output.data.topk(1, 1, True, True)
        res = {
            'gt': None if objectId is None else objectId.data,
            'pred': pred,
            }

        if objectId is None:
            return res, {}

        loss = self.criterion(output, objectId.view(-1))

        (prec1, prec3) = self.accuracy(output, objectId, topk=(1, min(3, self.numClasses)))

        if isTrain:
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        losses = OrderedDict([
                            ('Loss', loss.data.item()),
                            ('Top1', prec1),
                            ('Top3', prec3),
                            ])

        return res, losses

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.data.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.data.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res[0], res[1]


    def importState(self, save):
        params = save['state_dict']
        if hasattr(self.model, 'module'):
            try:
                self.model.load_state_dict(params, strict=True)
            except:
                self.model.module.load_state_dict(params, strict=True)
        else:
            params = self._clearState(params)
            self.model.load_state_dict(params, strict=True)

        self.epoch = save['epoch'] if 'epoch' in save else 0
        self.bestPrec = save['best_prec1'] if 'best_prec1' in save else 1e20
        self.error = save['error'] if 'error' in save else 1e20
        print('Imported checkpoint for epoch %05d with loss = %.3f...' % (self.epoch, self.bestPrec))


    def _clearState(self, params):
        res = dict()
        for k,v in params.items():
            kNew = re.sub('^module\.', '', k)
            res[kNew] = v

        return res
        

    def exportState(self):
        dt = datetime.datetime.now()
        state = self.model.state_dict()
        for k in state.keys():
            #state[k] = state[k].share_memory_()
            state[k] = state[k].cpu()
        return {
            'state_dict': state,
            'epoch': self.epoch,
            'error': self.error,
            'best_prec1': self.bestPrec,
            'datetime': dt.strftime("%Y-%m-%d %H:%M:%S")
            }


    def updateLearningRate(self, epoch):
        self.adjust_learning_rate_new(epoch, self.baseLr)


    def adjust_learning_rate_new(self, epoch, base_lr, period = 100): # train for 2x100 epochs
        gamma = 0.1 ** (1.0/period)
        lr_default = base_lr * (gamma ** (epoch))
        print('New lr_default = %f' % lr_default)

        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr_mult'] * lr_default