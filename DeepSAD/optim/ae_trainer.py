from base.base_trainer import BaseTrainer
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from data_provider.data_factory import data_provider 

class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0, 
                 data: str='custom', timeenc: int=0, root_path: str='./dataset', seq_len: int=50, data_path: str='competition_data.csv'):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)
        # data 
        self.data = data
        self.timeenc = timeenc
        self.root_path = root_path 
        self.seq_len = seq_len 
        self.data_path = data_path

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None

    def train(self, ae_net: BaseNet):

        # Get train data loader
        _, train_loader = data_provider(
            data=self.data, timeenc=self.timeenc, batch_size=self.batch_size
            , num_workers=self.n_jobs_dataloader, root_path=self.root_path
            , seq_len=self.seq_len, data_path=self.data_path, flag='train')

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        print('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
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

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        print('Pretraining Time: {:.3f}s'.format(self.train_time))
        print('Finished pretraining.')

        return ae_net

    def test(self, ae_net: BaseNet):

        # Get test data loader
        _, test_loader = data_provider(
            data=self.data, timeenc=self.timeenc, batch_size=self.batch_size
            , num_workers=self.n_jobs_dataloader, root_path=self.root_path
            , seq_len=self.seq_len, data_path=self.data_path, flag='test')

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device for network
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        print('Testing autoencoder...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                seq_x, seq_y, seq_x_mark = data
                if self.timeenc == 0:
                    # inputs = torch.cat([seq_x, seq_x_mark], dim=2)
                    inputs = seq_x
                    labels = seq_y 
                elif self.timeenc == 1:
                    inputs = seq_x 
                    labels = seq_y 

                inputs, labels = inputs.float().to(self.device), labels.float().to(self.device)
                labels = labels.squeeze(-1)

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist())
                                            )

                loss = torch.mean(rec_loss)
                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time

        # # Compute AUC
        # labels, scores = zip(*idx_label_score)
        # labels = np.array(labels)
        # scores = np.array(scores)
        # self.test_auc = roc_auc_score(labels, scores)

        # Log results
        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        # print('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        print('Test Time: {:.3f}s'.format(self.test_time))
        print('Finished testing autoencoder.')