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

class ANNRegressor(nn.Module):
    def __init__(self, nfeatures=1, neurons=30, cuda=False):
        super(ANNRegressor, self).__init__()
        self.layers = nn.Sequential(
            # Define our layers & activation functions
            nn.Linear(nfeatures, neurons), # input: (# of predictors), output: hidden layer 1 (30 neurons)
            nn.Sigmoid(), # Activation function
            nn.Linear(neurons, neurons),  # input: hidden layer 1 (30 neurons), output: hidden layer 2 (30 neurons)
            nn.Sigmoid(), # Activation function
            nn.Linear(neurons, 1), # input: hidden layer 2 (30 neurons), output: (1, for 1D array of predictions)
            # nn.ReLU(), # Activation function
        )
        self.device = torch.device("cuda" if cuda else "cpu")
        self.to(self.device).float()
        torch.manual_seed(0)
        
    def forward(self, x):
        pred = self.to(self.device).layers(x.float()) # make a prediction in the forward pass
        return pred

    def train_(self, train_dataloader, loss_fn, optimizer, scheduler, verbose=True):
        num_batches = len(train_dataloader)
        # size = len(dataloader.dataset)
        train_loss, batch = 0, 0
        self.train() # set pytorch into "training" mode
        y_train, train_preds = [], []
        for batch, (X, y) in enumerate(train_dataloader):
            start = timeit.default_timer()
            X, y = X.float().to(self.device), y.view(len(y),1).float().to(self.device)
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
            train_loss += loss
            optimizer.step()
            
            # model = self
            # params = self.parameters()
            # def closure():
            #     # Compute prediction error
            #     train_pred = model(X)
            #     # train_preds.append(train_pred)
            #     loss = loss_fn(train_pred, y)
        
            #     # Backpropagation
            #     optimizer.zero_grad()
            #     # for param in self.parameters(): # more efficient then optimizer.zero_grad()
            #     #     param.grad = None
            #     # for param in params:
            #     #     param.grad = None
            #     loss.backward()
            #     return loss
            
            # train_pred = model(X)
            # train_preds.append(train_pred)
            # loss = optimizer.step(closure)
            # train_loss += loss
            
            
            if batch % 100 == 0:
                loss = loss_fn(train_pred,y)
                stop = timeit.default_timer()
                if verbose is True:
                    print(f"loss: {loss:>7f}  [{(stop-start)*1000:.2f}ms]")
        # if scheduler != None:
        #     scheduler.step()
        train_loss /= num_batches
        y_train = torch.cat([i for i in y_train])
        train_preds = torch.cat([i for i in train_preds])
        return train_loss, y_train, train_preds
    
    def validate_(self, val_dataloader, loss_fn, verbose=True):
        num_batches = len(val_dataloader)
        val_loss = 0
        self.eval() # set pytorch into "testing" mode
        y_val, val_preds, R2 = [], [], []
        with torch.no_grad(): # do not use backpropagation during evaluation
            for X, y in val_dataloader:
                X, y = X.to(self.device).float(), y.to(self.device).view(len(y),1).float()
                y_val.append(y)
                val_pred = self(X)
                val_preds.append(val_pred)
                val_loss += loss_fn(val_pred, y).item()
                r2score = R2Score().to(self.device)
                R2.append(r2score(val_pred, y))
        val_loss /= num_batches
        # convert output tensors to numpy for plotting
        y_val = torch.cat([i for i in y_val])
        val_preds = torch.cat([i for i in val_preds])
        # R2 = np.array(R2).mean()
        R2 = torch.mean(torch.Tensor([i for i in R2]))
        if verbose is True:
            print(f"Valdiation Error: \n Avg Accuracy: {R2*100:.2f}%, Avg loss: {val_loss:>8f} \n")
        return val_loss, y_val, val_preds, R2
    
    def test_(self, test_dataloader, loss_fn, verbose=True):
        num_batches = len(test_dataloader)
        test_loss, correct = 0, 0
        self.eval() # set pytorch into "testing" mode
        correct = []
        with torch.no_grad(): # do not use backpropagation during evaluation
            for X, y in test_dataloader:
                X, y = X.to(self.device).float(), y.to(self.device).view(len(y),1).float()
                test_pred = self(X)
                test_loss += loss_fn(test_pred, y).item()
                r2score = R2Score().to(self.device)
                correct.append(r2score(test_pred, y))
        test_loss /= num_batches
        # correct = np.array(correct).mean()
        correct = torch.mean(torch.Tensor([i for i in correct]))
        if verbose is True:
            print(f"Test Error: \n Avg Accuracy: {correct*100:.2f}%, Avg loss: {test_loss:>8f} \n")
    
    def convert_to_datasets(self, data, batch_size, internal_testing=0.2):
        X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
        # convert to tensors
        X_test, y_test = torch.from_numpy(X_test.values), torch.from_numpy(y_test.values)
        if internal_testing > 0:
            X_valtrain, X_val, y_valtrain, y_val = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=internal_testing,
                                                                    random_state=0)
            X_valtrain, y_valtrain = torch.from_numpy(X_valtrain.values), torch.from_numpy(y_valtrain.values)
            X_val, y_val = torch.from_numpy(X_val.values), torch.from_numpy(y_val.values)
            # compile tensors together into datasets
            train_dataset = TensorDataset(X_valtrain, y_valtrain)
            val_dataset = TensorDataset(X_val, y_val)
            # build dataloaders
            val_dataloader = DataLoader(
                                dataset=val_dataset, 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=0,
                                )
        else:
            # compile tensors together into datasets
            X_train, y_train = torch.from_numpy(X_train.values), torch.from_numpy(y_train.values)
            train_dataset = TensorDataset(X_train, y_train)
            # validation dataloader is unused
            val_dataloader = None
        # compile tensors together into datasets
        test_dataset = TensorDataset(X_test, y_test)
        
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
        
    
    def fit(self,
            input_data,
            batch_size,
            max_epochs,
            loss_fn,
            optimizer,
            scheduler,
            patience=0,
            internal_testing=0.2,
            early_stopping=False,
            fit_plot=False,
            verbose=True
            ):
        train_dataloader, test_dataloader, val_dataloader = self.convert_to_datasets(input_data,
                                                                                     batch_size,
                                                                                     internal_testing)
        training_loss, validating_loss, my_images = [], [], []
        for epoch in range(max_epochs):
            if verbose is True:
                print(f"Epoch {epoch+1}\n-------------------------------")
            train_loss, y_train, train_pred = self.train_(train_dataloader, loss_fn, optimizer, scheduler, verbose)
            val_loss, y_val, val_pred, R2 = self.validate_(test_dataloader, loss_fn, verbose)
            if internal_testing > 0:
                self.test_(val_dataloader, loss_fn, verbose)
            
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
        return self(X.to(self.device)).cpu().detach().numpy().squeeze()

#%% Ensemble Model
from joblib import Parallel, delayed
from joblib import effective_n_jobs
import warnings
import threading
import sys
import os

def parallel_nn_train(scheduler,
                      estimator,
                      train_dataloader,
                      loss_fn,
                      optimizer,
                      current_lr,
                      device,
                      verbose,
                      ):
    if current_lr:
        for group in optimizer.param_groups:
            group["lr"] = current_lr
    
    num_batches = len(train_dataloader)
    # size = len(dataloader.dataset)
    train_loss, batch = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.float().to(device), y.view(len(y),1).float().to(device)
        
        # Compute prediction error
        train_pred = estimator(X)
        # train_preds.append(train_pred)
        loss = loss_fn(train_pred, y)
        
        # Backpropagation
        # optimizer.zero_grad()
        for param in estimator.parameters(): # more efficient then optimizer.zero_grad()
            param.grad = None
        loss.backward()
        optimizer.step()
        train_loss += loss
        
        if batch % 100 == 0:
            loss = loss_fn(train_pred,y)
      
    train_loss /= num_batches
    return estimator, optimizer, loss

class ANNEnsembler(ANNRegressor):
    # see this repo: https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/torchensemble/voting.py
    def __init__(self, 
                 base_estimator,
                 n_estimators,
                 optimizer,
                 loss_fn,
                 scheduler=None,
                 n_jobs=-1,
                 batch_size=32,
                 max_epochs=100,
                 patience=10,
                 tolerance=1e-4,
                 internal_testing=0.2,
                 early_stopping=True,
                 verbose=True,
                 cuda=False,
                 ):
        super(ANNEnsembler, self).__init__()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.tolerance = tolerance
        self.internal_testing = internal_testing
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.device = torch.device("cuda" if cuda else "cpu")
        print("Using {} device".format(self.device))
        self.to(self.device)
    
    def forward(self, x):
        ensemble_preds = [estimator(x.float()) for estimator in self.estimators_] # make a prediction in the forward pass
        mean_pred = sum(ensemble_preds) / len(ensemble_preds)
        return mean_pred
    
    
    def convert_to_datasets(self, data, batch_size, internal_testing=0.2):
        X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
        # convert to tensors
        X_test, y_test = torch.from_numpy(X_test.values), torch.from_numpy(y_test.values)
        if internal_testing > 0:
            X_valtrain, X_val, y_valtrain, y_val = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=internal_testing,
                                                                    random_state=0)
            X_valtrain, y_valtrain = torch.from_numpy(X_valtrain.values), torch.from_numpy(y_valtrain.values)
            X_val, y_val = torch.from_numpy(X_val.values), torch.from_numpy(y_val.values)
            # compile tensors together into datasets
            train_dataset = TensorDataset(X_valtrain, y_valtrain)
            val_dataset = TensorDataset(X_val, y_val)
            # build dataloaders
            val_dataloader = DataLoader(
                                dataset=val_dataset, 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=0,
                                )
        else:
            # compile tensors together into datasets
            X_train, y_train = torch.from_numpy(X_train.values), torch.from_numpy(y_train.values)
            train_dataset = TensorDataset(X_train, y_train)
            # validation dataloader is unused
            val_dataloader = None
        # compile tensors together into datasets
        test_dataset = TensorDataset(X_test, y_test)
        
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
        return train_dataloader, test_dataloader, val_dataloader
    
    def forward_(self, estimators, x):
        ensemble_preds = [estimator(x.float().to(self.device)) for estimator in estimators] # make a prediction in the forward pass
        mean_pred = sum(ensemble_preds) / len(ensemble_preds)
        return mean_pred
    
    def validate_(self, estimators, val_dataloader, loss_fn, verbose=True):
        num_batches = len(val_dataloader)
        val_loss = 0
        self.eval() # set pytorch into "testing" mode
        R2 = []
        with torch.no_grad(): # do not use backpropagation during evaluation
            for X, y in val_dataloader:
                X, y = X.to(self.device).float(), y.view(len(y),1).to(self.device).float()
                val_pred = self.forward_(estimators, X)
                val_loss += loss_fn(val_pred, y).item()
                r2score = R2Score().to(self.device)
                R2.append(r2score(val_pred, y))
        val_loss /= num_batches
        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(estimators)
        # convert output tensors to numpy for plotting
        R2 = torch.mean(torch.Tensor([i.to(self.device) for i in R2]))
        if verbose is True:
            print(f"Valdiation Error: \n Avg Accuracy: {R2*100:.2f}%, Avg loss: {val_loss:>8f} \n")
    
        return val_loss, R2, y
    
    def test_(self, estimators, test_dataloader, loss_fn, verbose=True):
        num_batches = len(test_dataloader)
        test_loss, R2 = 0, 0
        self.eval() # set pytorch into "testing" mode
        R2 = []
        with torch.no_grad(): # do not use backpropagation during evaluation
            for X, y in test_dataloader:
                X, y = X.to(self.device).float(), y.view(len(y),1).to(self.device).float()
                test_pred = self.forward_(estimators, X)
                test_loss += loss_fn(test_pred, y).item()
                r2score = R2Score().to(self.device)
                R2.append(r2score(test_pred, y))
        test_loss /= num_batches
        R2 = torch.mean(torch.Tensor([i.to(self.device) for i in R2]))
        if verbose is True:
            print(f"Test Error: \n Avg Accuracy: {R2*100:.2f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, R2
    
    def fit(self,input_data, log_interval=100):
        
        # Load in data
        train_dataloader, test_dataloader, val_dataloader = self.convert_to_datasets(input_data,
                                                                                     self.batch_size,
                                                                                     self.internal_testing)
        
        # call this during evaluation
        self.test_dataloader = test_dataloader
        
        # initialize ensemble models and optimizers
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self.base_estimator)
        
        optimizers = []
        for _ in range(self.n_estimators):
            optimizers.append(self.optimizer)
        
        # Initialize loss tracker
        validating_loss, training_loss = [], []
        
        # Assign chunk of networks to jobs
        # from sklearn implementation - provides ~4x speed up
        n_jobs = self._partition_estimators(self.n_estimators, self.n_jobs)
        
        # change to training mode
        self.train()
        
        # Main parellized training loop
        with Parallel(n_jobs=n_jobs) as parallel:
            for epoch in range(self.max_epochs):
                if self.scheduler is not None:
                    current_lr = self.scheduler.get_last_lr()[0]
                else:
                    current_lr = None
                
                if self.verbose is True:
                    print(f"\n===============================\
                          \nEpoch {epoch+1}\
                              \n-------------------------------")
                
                start = timeit.default_timer()
                nets = parallel(
                    delayed(parallel_nn_train)(
                        self.scheduler,
                        estimator,
                        train_dataloader,
                        self.loss_fn,
                        optimizer,
                        current_lr,
                        self.device,
                        self.verbose,
                        )
                    for idx, (estimator, optimizer) in enumerate(
                            zip(estimators, optimizers)
                    )
                 )
                    
                # Iterate and append
                estimators, optimizers, losses = [], [], []
                for estimator, optimizer, loss in nets:
                    estimators.append(estimator)
                    optimizers.append(optimizer)
                    losses.append(loss.item())
                
                mean_loss = sum(losses) / len(losses)
                stop = timeit.default_timer()
                if self.verbose is True:
                    # Print training status
                    training_time = stop-start
                    statement = f'[{(training_time)*1000:.2f}ms]' if training_time*1000<1000 else f'[{(training_time):.2f}s]'
                    print(f"Ensemble Training loss: {mean_loss:>7f}  "+statement)
                
                # Now validate models - use the test_dataloader (split designated by main script)
                val_loss, val_R2, y = self.validate_(estimators, test_dataloader, self.loss_fn, self.verbose)
                # Set shape of output (called later in predict)
                self.n_outputs_ = y.shape[1]
                
                # If internal split testing enabled, print the model fit on partitioned dataset
                if self.internal_testing > 0:
                    test_loss, test_R2 = self.test_(estimators, val_dataloader, self.loss_fn, self.verbose)
                    training_loss.append(test_R2)
                # append loss each epoch
                validating_loss.append(val_loss)
                
                # Early stopping - check if loss is still decreasing within patience limit
                if self.early_stopping is True: # see https://clay-atlas.com/us/blog/2021/08/25/pytorch-en-early-stopping/
                    # patience is the threshold number of epochs to wait before stopping training
                    if self.internal_testing > 0:
                        test_loss_this_iter = test_loss
                    loss_this_iter = val_loss # check loss on validation data
                    if epoch == 0: # for the first iteration, set number/loss counters
                        self.loss_last_iter = loss_this_iter
                        if self.internal_testing > 0:
                            self.test_loss_last_iter = test_loss_this_iter
                        self.counter = 0
                    try:
                        if self.internal_testing > 0:
                            if ((loss_this_iter - self.loss_last_iter) > self.tolerance) or ((test_loss_this_iter - self.test_loss_last_iter) > self.tolerance):
                                self.counter += 1
                                if self.verbose is True:
                                    countdown = self.patience-self.counter
                                    countdown_msg = "in "+str(countdown)+" epoch(s)"
                                    now = "now"
                                    print(f"~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~\
                                          \nEarly stopping: Loss not improved, \
                                              \nstopping {countdown_msg if countdown>0 else now}...\
                                              \n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~\n")
                                if self.counter >= self.patience: # check if counter has exceeded the defined num of epochs
                                    raise StopIteration
                        elif loss_this_iter - self.loss_last_iter > self.tolerance: # check if loss is increasing
                            self.counter += 1
                            if self.verbose is True:
                                countdown = self.patience-self.counter
                                countdown_msg = "in "+str(countdown)+" epoch(s)"
                                now = "now"
                                print(f"~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~\
                                      \nEarly stopping: Loss not improved, \
                                          \nstopping {countdown_msg if countdown>0 else now}...\
                                          \n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~\n")
                            if self.counter >= self.patience: # check if counter has exceeded the defined num of epochs
                                raise StopIteration
                    except StopIteration:
                        print(f"Early stopping: epoch {epoch}\n")
                        break
                    self.loss_last_iter = loss_this_iter
                    if self.internal_testing > 0:
                        self.test_loss_last_iter = test_loss_this_iter
                
                # Update the scheduler
                with warnings.catch_warnings():

                    # UserWarning raised by PyTorch is ignored because
                    # scheduler does not have a real effect on the optimizer.
                    warnings.simplefilter("ignore", UserWarning)

                    if self.scheduler is not None:
                        self.scheduler.step()                    
        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(estimators)
    
    def _partition_estimators(self, n_estimators, n_jobs):
        """
        Pulled from sklearn's randomforestregressor implementation
        refer here: https://github.com/scikit-learn/scikit-learn/blob/8f7bd0dcb3cf96412903be839d00f7b58aa05cad/sklearn/ensemble/_base.py#L206
        
        Private function used to partition estimators between jobs."""
        # Compute the number of jobs
        n_jobs = min(effective_n_jobs(n_jobs), n_estimators)
        return n_jobs
    
    def _accumulate_prediction(self, predict, x, out, lock):
        """
        Pulled directly from sklearn's randomforestregressor implementation
        refer here: https://github.com/scikit-learn/scikit-learn/blob/8f7bd0dcb3cf96412903be839d00f7b58aa05cad/sklearn/ensemble/_forest.py#L369
        
        This is a utility function for joblib's Parallel.
        It can't go locally in ForestClassifier or ForestRegressor, because joblib
        complains that it cannot pickle it when placed there.
        """
        prediction = predict(x)
        with lock:
            if len(out) == 1:
                out[0] += prediction
            else:
                for i in range(len(out)):
                    out[i] += prediction[i]
    
    def predict(self, x, verbose=False):
        """
        Pulled directly from sklearn's randomforestregressor implementation
        refer here: https://github.com/scikit-learn/scikit-learn/blob/8f7bd0dcb3cf96412903be839d00f7b58aa05cad/sklearn/ensemble/_forest.py#L369
        """
        # Assign chunk of networks to jobs
        n_jobs = self._partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((x.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((x.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=verbose, require="sharedmem")(
            delayed(self._accumulate_prediction)(estimator.predict, x, [y_hat], lock)
            for estimator in tqdm(self.estimators_)
        )
        print('\n')

        y_hat /= len(self.estimators_)

        return y_hat
    
    def save_ensemble(self, path, filename):
        state = {
               "n_estimators": len(self.estimators_),
               "estimators_": self.estimators_,
               "model": self.state_dict(),
               "n_outputs_": self.n_outputs_,
               "loss_fn": self.loss_fn
               }
    
        torch.save(state, os.path.join(path,f'{filename}.pth'))
    
    
    def load_ensemble(self, path, filename): 
        state = torch.load(os.path.join(path,f'{filename}.pth'))
    
        model_params = state['model']
        self.n_outputs_ = state['n_outputs_']
        self.loss_fn = state['loss_fn']
        self.estimators_ = state['estimators_']
    
        self.load_state_dict(model_params)
        self.eval()
     
        