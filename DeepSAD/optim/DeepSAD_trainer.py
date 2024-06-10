from base.base_trainer import BaseTrainer
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import time
import torch
import torch.optim as optim
import numpy as np

from data_provider.data_factory import data_provider 

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0,
                 data: str='custom', timeenc: int=0, root_path: str='./dataset', seq_len: int=50, data_path: str='competition_data.csv'):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # data 
        self.data = data
        self.timeenc = timeenc
        self.root_path = root_path 
        self.seq_len = seq_len 
        self.data_path = data_path

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, net: BaseNet):

        # Get train data loader
        _, train_loader = data_provider(
            data=self.data, timeenc=self.timeenc, batch_size=self.batch_size
            , num_workers=self.n_jobs_dataloader, root_path=self.root_path
            , seq_len=self.seq_len, data_path=self.data_path, flag='train')

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            print('Center c initialized.')

        # Training
        print('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                seq_x, seq_y, seq_x_mark = data
                if self.timeenc == 0:
                    # inputs = torch.cat([seq_x, seq_x_mark], dim=2)
                    inputs = seq_x
                    inputs = inputs.float().to(self.device)
                elif self.timeenc == 1:
                    inputs = seq_x
                    inputs = inputs.float().to(self.device)
                
                semi_targets = seq_y.float().to(self.device)
                semi_targets = semi_targets.squeeze(-1)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs, h = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=2)
                losses = torch.where(semi_targets == -1
                                    ,self.eta * ((dist + self.eps) ** semi_targets.float()), dist)
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        print('Training Time: {:.3f}s'.format(self.train_time))
        print('Finished training.')

        return net

#     def test(self, net: BaseNet):

#         # Get test data loader
#         _, test_loader = data_provider(
#             data='pred', timeenc=self.timeenc, batch_size=self.batch_size
#             , num_workers=self.n_jobs_dataloader, root_path=self.root_path
#             , seq_len=self.seq_len, data_path=self.data_path, flag='test')
        
#         # Set device for network
#         net = net.to(self.device)

#         # Testing
#         print('Starting testing...')
#         epoch_loss = 0.0
#         n_batches = 0
#         start_time = time.time()
#         idx_label_score = []
#         net.eval()
#         total_prec = 0
#         total_rec = 0
#         total_f1 = 0
#         total_acc = 0
#         total_cnt = 0
#         with torch.no_grad():
#             for data in test_loader:
#                 seq_x, seq_y, seq_x_mark = data
#                 if self.timeenc == 0:
#                     inputs = torch.cat([seq_x, seq_x_mark], dim=2)
#                     inputs = inputs.float().to(self.device)
#                 elif self.timeenc == 1:
#                     inputs = seq_x
#                     inputs = inputs.float().to(self.device)
                
#                 semi_targets = seq_y.float().to(self.device)
#                 semi_targets = semi_targets.squeeze(-1)


#                 outputs, h = net(inputs)
#                 dist = torch.sum((outputs - self.c) ** 2, dim=2)
#                 losses = torch.where(semi_targets == -1
#                                     ,self.eta * ((dist + self.eps) ** semi_targets.float()), dist)
#                 loss = torch.mean(losses)
#                 scores = dist.cpu().data.numpy()

#                 # # Save triples of (idx, label, score) in a list
#                 # idx_label_score += list(zip(
#                 #                             semi_targets.cpu().data.numpy().tolist(),
#                 #                             scores.cpu().data.numpy().tolist())
#                 #                             )

#                 # metric 
#                 percentile = np.percentile(scores, 80) # dist 상위 80퍼센트 (짧은순 )
#                 pred = np.where(scores > percentile, -1, 1)
#                 labels = semi_targets.cpu().data.numpy()

#                 prec = 0
#                 rec = 0
#                 f1 = 0
#                 acc = 0
#                 cnt = 0
#                 for i in range(labels.shape[0]):
#                     cnt += 1
#                     prec += precision_score(labels[i], pred[i])
#                     rec += recall_score(labels[i], pred[i])
#                     f1 += f1_score(labels[i], pred[i])
#                     acc += accuracy_score(labels[i], pred[i])
#                 prec /= cnt
#                 rec /= cnt
#                 f1 /= cnt
#                 acc /= cnt

#                 total_prec += prec
#                 total_rec += rec
#                 total_f1 += f1
#                 total_acc += acc
#                 total_cnt += 1

#                 epoch_loss += loss.item()
#                 n_batches += 1

#         total_prec /= total_cnt
#         total_rec /= total_cnt
#         total_f1 /= total_cnt
#         total_acc /= total_cnt

#         self.test_time = time.time() - start_time
#         self.test_prec = total_prec
#         self.test_rec = total_rec
#         self.test_f1 = total_f1
#         self.test_acc = total_acc

#         # # Compute AUC
#         # _, labels, scores = zip(*idx_label_score)
#         # labels = np.array(labels)
#         # scores = np.array(scores)
#         # self.test_auc = roc_auc_score(labels, scores)

#         # Log results
#         print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
#         print('Test Accuracy: {:.2f}%'.format(100. * self.test_acc))
#         print('Test F1-score: {:.2f}%'.format(100. * self.test_f1))
#         print('Test Time: {:.3f}s'.format(self.test_time))
#         print('Finished testing.')

    def test(self, net: BaseNet):

        # Get test data loader
        test_dataset, test_loader = data_provider(
            data='pred2', timeenc=self.timeenc, batch_size=self.batch_size
            , num_workers=self.n_jobs_dataloader, root_path=self.root_path
            , seq_len=self.seq_len, data_path=self.data_path, flag='test')
        
        # Set device for network
        net = net.to(self.device)

        # Testing
        print('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        
        data_y = test_dataset.get_total_label() # test 전체 레이블 
        anomaly_score = np.zeros(data_y.shape[0])
        anomaly_idx = np.zeros(data_y.shape[0])
        anomaly_arr = np.arange(data_y.shape[0])

        with torch.no_grad():
            for data in test_loader:
                seq_x, seq_y, seq_x_mark, index = data
                if self.timeenc == 0:
                    # inputs = torch.cat([seq_x, seq_x_mark], dim=2)
                    inputs = seq_x
                    inputs = inputs.float().to(self.device)
                elif self.timeenc == 1:
                    inputs = seq_x
                    inputs = inputs.float().to(self.device)
                
                semi_targets = seq_y.float().to(self.device)
                semi_targets = semi_targets.squeeze(-1)


                outputs, h = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=2)
                losses = torch.where(semi_targets == -1
                                    ,self.eta * ((dist + self.eps) ** semi_targets.float()), dist)
                loss = torch.mean(losses)
                scores = dist.cpu().data.numpy()
                
                epoch_loss += loss.item()
                n_batches += 1

                lst = [anomaly_arr[i:i+self.seq_len] for i in index]
                for i in range(len(lst)):
                    anomaly_score[lst[i]] += scores[i]
                    anomaly_idx[lst[i]] += 1

        # metric 
        # anomaly_score += 0.01
        final_scores = anomaly_score / anomaly_idx
        percentile = np.percentile(final_scores, 90) # anomaly score threshold
        pred = np.where(final_scores > percentile, -1, 1)
        label = data_y.flatten()
        
        total_prec = precision_score(label, pred)
        total_rec = recall_score(label, pred)
        total_f1 = f1_score(label, pred)
        total_acc = accuracy_score(label, pred)

        self.test_time = time.time() - start_time
        self.test_prec = total_prec
        self.test_rec = total_rec
        self.test_f1 = total_f1
        self.test_acc = total_acc
        self.final_scores = final_scores.tolist()

        # Log results
        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        print('Test Accuracy: {:.2f}%'.format(100. * self.test_acc))
        print('Test F1-score: {:.2f}%'.format(100. * self.test_f1))
        print('Test Time: {:.3f}s'.format(self.test_time))
        print('Finished testing.')
        
    

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros((net.seq_len, net.rep_dim), device=self.device) # (seq_len, rep_dim)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                seq_x, seq_y, seq_x_mark = data
                if self.timeenc == 0:
                    # inputs = torch.cat([seq_x, seq_x_mark], dim=2)
                    inputs = seq_x
                    inputs = inputs.float().to(self.device)
                elif self.timeenc == 1:
                    inputs = seq_x
                    inputs = inputs.float().to(self.device)

                inputs = inputs.to(self.device)
                outputs, h = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
