import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

class ModelNetTrainer(object):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views=12):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views
        self.model.cuda()

    def train(self, n_epochs):
        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):
            if self.model_name == 'view-gcn':
                if epoch == 1:
                    for param_group in self.optimizer.param_groups:
                        #param_group['lr'] = lr
                        pass
                       
                if epoch > 1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5 * ( 1 + math.cos(epoch * math.pi / 15))
            else:
                if epoch > 0 and (epoch + 1) % 10 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(lr)
            for param_group in self.optimizer.param_groups:
                print('lr:', param_group['lr'])
            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):
                inputsDict = {
                'image': data[1],
                'pressure': data[2],
                'objectId': data[3],
                }
                image = Variable(inputsDict['image'].cuda())
                objectId = Variable(inputsDict['objectId'].cuda(),requires_grad=False)

                if self.model_name == 'view-gcn' and epoch == 0:
                    for param_group in self.optimizer.param_groups:
                        #param_group['lr'] = lr * ((i + 1) / (len(rand_idx) // 20))
                        pass

                if self.model_name == 'view-gcn':
                    N, V, H, W = inputsDict['pressure'].size()
                    in_data = Variable(inputsDict['pressure']).view(-1, 1, H, W).cuda()

                else:
                    pressure = Variable(inputsDict['pressure'].cuda())
                    in_data = pressure


                target = objectId.long().squeeze()
                target_ = target.unsqueeze(1).repeat(1, 3*self.num_views).view(-1)

                self.optimizer.zero_grad()
                if self.model_name == 'view-gcn':
                    out_data, F_score,F_score2= self.model(in_data)
                    out_data_ = torch.cat((F_score, F_score2), 1).view(-1, 27)

                    loss = self.loss_fn(out_data, target)+ self.loss_fn(out_data_, target_)
                else:
                    out_data = self.model(in_data)
                    #print(out_data.shape)
                    loss = self.loss_fn(out_data, target)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]

                loss.backward()
                self.optimizer.step()
                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch + 1, i + 1, loss, acc)
                if (i + 1) % 100 == 0:
                    print(log_str)
            i_acc += i
            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)

                self.model.save(self.log_dir, epoch)
            # save best model
                if val_overall_acc > best_acc:
                    best_acc = val_overall_acc


    def update_validation_accuracy(self, epoch, plot_confusion_matrix=False):
        all_correct_points = 0
        all_points = 0
        count = 0
        wrong_class = np.zeros(27)
        samples_class = np.zeros(27)
        all_loss = 0
        confusion_matrix = np.zeros((27, 27))
        self.model.eval()

        for _, data in enumerate(self.val_loader, 0):
            inputsDict = {
                'image': data[1],
                'pressure': data[2],
                'objectId': data[3],
                }
            image = Variable(inputsDict['image'].cuda())
            objectId = Variable(inputsDict['objectId'].cuda(),requires_grad=False)

            if self.model_name == 'view-gcn':
                N, V, H, W = inputsDict['pressure'].size()
                in_data = Variable(inputsDict['pressure']).view(-1, 1, H, W).cuda()
            else:  # 'svcnn'

                pressure = Variable(inputsDict['pressure'].cuda())
                in_data = pressure

            target = objectId.long().squeeze()
            if self.model_name == 'view-gcn':
                out_data,F1,F2=self.model(in_data)
            else:
                out_data = self.model(in_data)

            pred = torch.max(out_data, 1)[1]
            #confusion matrix
            if(plot_confusion_matrix):
                for t, p in zip(target.cpu().view(-1), pred.cpu().view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]
        
        #plotting confusion matrix
        if(plot_confusion_matrix):
            plt.figure(figsize=(13,13))

            class_names = ("empty_hand","full_can",'mug','tea_box','safety_glasses','multimeter','ball','empty_can','cat','lotion','gel','stapler'
                               ,'spray_can','kiwano','tape','board_eraser','allen_key_set','brain','pen','battery','bracket','scissors','screw_driver','clip',
                               'spoon','coin','chain')
            df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d",cmap="Blues")

            heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=12)
            heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=12)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig('../confusion_matrix.png')


        print('Total # of test models: ', all_points)
        class_acc = (samples_class - wrong_class) / samples_class
        val_mean_class_acc = np.mean(class_acc)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)
        print(class_acc)
        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc
