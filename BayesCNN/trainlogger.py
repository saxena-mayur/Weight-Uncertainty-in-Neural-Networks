from utils import DEVICE
import torch.optim as optim
import torch.nn as nn
import numpy as np


class TrainLogger:
    def __init__(self, name, model, train_loader, valid_loader, test_loader):
        self.name = name
        self.model = model
        self.log_dic = []
        self.epoch, self.n_epochs = 0, 0
        self.optimizer = optim.Adam(model.parameters())
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[33, 66], gamma=0.3)
        self.xent = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
    def _accuracy(self, outputs, labels, batch_size = batch_size):
        _, preds = outputs.max(1)
        correct = torch.eq(preds, labels).sum().item()
        return correct/len(labels)
    
    def _eval(self, loader):
        accs = []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                accs.append(self._accuracy(outputs, labels))
        return np.mean(accs)
        
    def loss_step(self, batch_id, outputs, labels):
        loss = self.xent(outputs[0], labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return dict(loss=loss.item())
        
    def train(self, n_epochs, n_samples = 1, callbacks = []):
        # call callbacks at the end of each epoch
        print('Training Model', self.name)
        self.n_epochs += n_epochs
        self.model.train()
        epoch_range = range(self.epoch, n_epochs)
        for self.epoch in epoch_range:
            self.scheduler.step(self.epoch)
            batch_log = {'train_acc':[]}
            for batch_id, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = [self.model(images) for _ in range(n_samples)]
                batch_log['train_acc'].append(self._accuracy(outputs[0], labels))
                loss_dic = self.loss_step(batch_id, outputs, labels)
                for loss_name, loss_value in loss_dic.items():
                    if not loss_name in batch_log:
                        batch_log[loss_name] = []
                    batch_log[loss_name].append(loss_value)

            means = {key:np.mean(values) for key, values in batch_log.items()}
            self.log(**means)
            self.log(val_acc=self._eval(self.valid_loader))

            self.display()
            for cb in callbacks:
                cb()
        self.epoch += 1
    
    def test(self):
        print('\n{0} Test Accuracy: {1:.2f} %'.format(self.name, 100*self._eval(self.test_loader)))
        
            
    def log(self, mode = 'epoch', **kwargs):
        while len(self.log_dic) < self.epoch+1:
            self.log_dic.append({})
        self.log_dic[self.epoch].update(kwargs)
        
    def display(self, mode='text'):
        print('\rEpoch {0}/{1} \t \t Loss: {2:.3f}, Train Acc: {3:.2f}%, Val Acc: {4:.2f}%'.format(
                    self.epoch+1, self.n_epochs, self.log_dic[self.epoch]['loss'], 
                    100*self.log_dic[self.epoch]['train_acc'], 100*self.log_dic[self.epoch]['val_acc']), end = '')
        
    def save(self, save_file):
        pd.DataFrame(self.log_dic).to_csv(save_file)