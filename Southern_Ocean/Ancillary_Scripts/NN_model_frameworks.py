# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:50:12 2021

@author: bcamc
"""

#%% Import packages
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchmetrics import R2Score
from tqdm import tqdm

#%% Set up model framework

#### ***For all models: assign device as cpu or gpu****
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class ANNRegressor(nn.Module):
    def __init__(self, nfeatures=1):
        super(ANNRegressor, self).__init__()
        self.layers = nn.Sequential(
            # Define our layers & activation functions
            nn.Linear(nfeatures, 30), # input: (# of predictors), output: hidden layer 1 (30 neurons)
            nn.Sigmoid(), # Activation function
            nn.Linear(30, 30),  # input: hidden layer 1 (30 neurons), output: hidden layer 2 (30 neurons)
            nn.Sigmoid(), # Activation function
            nn.Linear(30, 1), # input: hidden layer 2 (30 neurons), output: (1, for 1D array of predictions)
        )
        self.float()
        
    def forward(self, x):
        pred = self.layers(x.float()) # make a prediction in the forward pass
        return pred
    
    def validate_(self, val_dataloader, loss_fn, verbose=True):
        num_batches = len(val_dataloader)
        val_loss = 0
        self.eval() # set pytorch into "testing" mode
        y_val = []
        val_preds = []
        R2 = []
        with torch.no_grad(): # do not use backpropagation during evaluation
            for X, y in val_dataloader:
                X, y = X.to(device).float(), y.view(len(y),1).to(device).float()
                y_val.append(y)
                val_pred = self(X)
                val_preds.append(val_pred)
                val_loss += loss_fn(val_pred, y).item()
                r2score = R2Score()
                R2.append(r2score(val_pred, y))
        val_loss /= num_batches
        # convert output tensors to numpy for plotting
        y_val = np.concatenate([i.detach().numpy() for i in y_val])
        val_preds = np.concatenate([i.detach().numpy() for i in val_preds])
        R2 = np.array(R2).mean()
        return val_loss, y_val, val_preds, R2

    def train_(self, train_dataloader, loss_fn, optimizer, scheduler, verbose=True):
        num_batches = len(train_dataloader)
        # size = len(dataloader.dataset)
        train_loss = 0
        batch = 0
        self.train() # set pytorch into "training" mode
        y_train = []
        train_preds = []
        for batch, (X, y) in enumerate(train_dataloader):
            start = timeit.default_timer()
            X, y = X.to(device).float(), y.to(device).view(len(y),1).float()
            y_train.append(y)
            
            # Compute prediction error
            train_pred = self(X)
            train_preds.append(train_pred)
            loss = loss_fn(train_pred, y)
    
            # Backpropagation
            # optimizer.zero_grad()
            for param in self.parameters(): # more efficient then optimizer.zero_grad()
                param.grad = None
            loss.backward()
            optimizer.step()
            train_loss += loss
            
            if batch % 100 == 0:
                loss = loss_fn(train_pred,y)
                stop = timeit.default_timer()
                if verbose is True:
                    print(f"loss: {loss:>7f}  [{(stop-start)*1000:.2f}ms]")
        if scheduler != None:
            scheduler.step()
        train_loss /= num_batches
        y_train = np.concatenate([i.detach().numpy() for i in y_train])
        train_preds = np.concatenate([i.detach().numpy() for i in train_preds])
        return train_loss, y_train, train_preds
        
    def test_(self, test_dataloader, loss_fn, verbose=True):
        num_batches = len(test_dataloader)
        test_loss, correct = 0, 0
        self.eval() # set pytorch into "testing" mode
        correct = []
        with torch.no_grad(): # do not use backpropagation during evaluation
            for X, y in test_dataloader:
                X, y = X.to(device).float(), y.view(len(y),1).to(device).float()
                test_pred = self(X)
                test_loss += loss_fn(test_pred, y).item()
                r2score = R2Score()
                correct.append(r2score(test_pred, y))
        test_loss /= num_batches
        correct = np.array(correct).mean()
        if verbose is True:
            print(f"Test Error: \n Avg Accuracy: {correct*100:.2f}%, Avg loss: {test_loss:>8f} \n")
    
    def convert_to_datasets(self, data, batch_size, internal_validation=0.2):
        X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
        # convert to tensors
        X_test, y_test = torch.from_numpy(X_test.values), torch.from_numpy(y_test.values)
        X_valtrain, X_val, y_valtrain, y_val = train_test_split(X_train, y_train, test_size=internal_validation, random_state=0)
        X_valtrain, y_valtrain = torch.from_numpy(X_valtrain.values), torch.from_numpy(y_valtrain.values)
        X_val, y_val = torch.from_numpy(X_val.values), torch.from_numpy(y_val.values)
        # compile tensors together into datasets
        train_dataset = TensorDataset(X_valtrain, y_valtrain)
        test_dataset = TensorDataset(X_test, y_test)
        val_dataset = TensorDataset(X_val, y_val)
        # build dataloaders
        train_dataloader = DataLoader(
                                dataset=train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=0,
                                )
        test_dataloader = DataLoader(
                            dataset=test_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=0,
                            )
        val_dataloader = DataLoader(
                            dataset=val_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=0,
                            )
        return train_dataloader, test_dataloader, val_dataloader
    
    def model_performance_gif_maker(self, training_loss, validating_loss, train_pred, val_pred, y_train, y_val, R2, epoch):
        fig = plt.figure(figsize=(12,12))
        font={'family':'DejaVu Sans',
          'weight':'normal',
          'size':'22'} 
        plt.rc('font', **font) # sets the specified font formatting globally
        #------------------------
        # Scatter plot of fit
        ax = fig.add_subplot(211)
        ax.scatter(y_train,train_pred,s=10,c='k', label="Training")
        ax.scatter(y_val,val_pred,s=10,c='r', label=f"Validate (R2 = {R2:.2f})")
        l1 = np.min(ax.get_xlim())
        l2 = np.max(ax.get_xlim())
        ax.plot([l1,l2], [l1,l2], ls="--", c=".3", zorder=0)
        ax.set_xlim(0,6)
        ax.set_ylim(0,6)
        ax.set_ylabel(r'arcsinh(DMS$_{\rmmodel}$)')
        ax.set_xlabel(r'arcsinh(DMS$_{\rmmeasured}$)')
        ax.legend(loc='lower right', markerscale=3, fontsize=20, facecolor='none')
        ax.set_title('Base Model Training')
        #------------------------
        # Plot loss curve
        ax2 = fig.add_subplot(212)
        ax2.plot(np.append(np.empty(0),[val.detach().numpy() for val in training_loss]), 'b-', label='Train')
        ax2.plot(np.array(validating_loss), 'r-', label='Validate')
        ax2.plot([], [], ' ', label=f"Epoch: {epoch}")
        ax2.set_ylim(0.1, 1.4)
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right', markerscale=3, fontsize=20, facecolor='none')
        #-----------------------
        # Save plot canvas to export (see https://ndres.me/post/matplotlib-animated-gifs-easily/)
        fig.canvas.draw()       # draw the canvas & cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        
        return image
        
    
    def fit(self, input_data, batch_size, max_epochs, loss_fn, optimizer, scheduler, patience=0, internal_validation=0.2, early_stopping=False, fit_plot=False, verbose=True):
        train_dataloader, test_dataloader, val_dataloader = self.convert_to_datasets(input_data, batch_size, internal_validation)
        training_loss = []
        validating_loss = []
        my_images=[]
        for epoch in range(max_epochs):
            if verbose is True:
                print(f"Epoch {epoch+1}\n-------------------------------")
            train_loss, y_train, train_pred = self.train_(train_dataloader, loss_fn, optimizer, scheduler, verbose)
            val_loss, y_val, val_pred, R2 = self.validate_(val_dataloader, loss_fn)
            self.test_(test_dataloader, loss_fn, verbose)
            
            # append loss each epoch
            training_loss.append(train_loss)
            validating_loss.append(val_loss)
            
            # create a gif of training
            if fit_plot is True:
                image = self.model_performance_gif_maker(training_loss,
                                                         validating_loss,
                                                         train_pred,
                                                         val_pred,
                                                         y_train,
                                                         y_val,
                                                         R2,
                                                         epoch)
                my_images.append(image)
            
            if early_stopping is True: # see https://clay-atlas.com/us/blog/2021/08/25/pytorch-en-early-stopping/
                # patience is the threshold number of epochs to wait before stopping training
                loss_this_iter = val_loss # check loss on validation data
                if epoch == 0: # for the first iteration, set number/loss counters
                    self.loss_last_iter = loss_this_iter
                    self.counter = 0
                try:
                    if loss_this_iter > self.loss_last_iter: # check if loss is increasing
                        self.counter += 1
                        if self.counter >= patience: # check if counter has exceeded the defined num of epochs
                            raise StopIteration
                except StopIteration:
                    print(f"Early stopping: epoch {epoch}")
                    break
                self.loss_last_iter = loss_this_iter
        return training_loss, validating_loss, my_images
    
    def predict(self, X):
        X = torch.from_numpy(X)
        return self(X).detach().numpy().squeeze()
