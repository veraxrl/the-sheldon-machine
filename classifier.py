## Original classifier by author: need modification

import os
os.environ["THEANO_FLAGS"] = "floatX=float32"
import cPickle
import numpy as np
import collections
import theano
import theano.tensor as T
import re
import warnings
import sys
import pandas as pd
import logging
import math
import pickle
import os
import timeit
import time
import lasagne
from lasagne.layers import get_output_shape
from sklearn.base import BaseEstimator

from com.ccls.lstm.main.sarcasmLSTM import SarcasmLstm

from com.ccls.lstm.main.deep_mind import deep_mind 
from com.ccls.lstm.preprocess.utils import str_to_bool 
from com.ccls.lstm.preprocess.getData import split_train_test
from com.ccls.lstm.preprocess.getData import get_batch
from sklearn.metrics import precision_recall_fscore_support as score
from datetime import datetime


class SarcasmClassifier(BaseEstimator):
    def __init__(self,**kwargs):

        self.num_epochs = int(kwargs["num_epochs"])
        self.batch_size = int(kwargs["batch_size"])
        self.patience_increase = int(kwargs["patience"])
        self.num_epochs = 50
        self.classifier = SarcasmLstm(**kwargs) 

    def fit(self, X, y, log_file):

        print("starting training")
        early_stopping_heldout = .9
        X, X_heldout, y, y_heldout = split_train_test(X, y, train_size=early_stopping_heldout, random_state=123)
        train_size = X[0].shape[0]
        n_train_batches  = train_size  // self.batch_size
        best = 0

        patience = 1000  # look as this many examples regardless
        patience_increase = self.patience_increase  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995 
        validation_frequency = n_train_batches //4 

        best_validation_accuracy = -np.inf
        best_iter = 0
        start_time = timeit.default_timer()
        epoch = 0
        done_looping = False

        while (epoch < self.num_epochs) and (not done_looping):
            epoch = epoch + 1
            start_time_epoch = timeit.default_timer()
            print("Epoch number: {}\n".format(epoch))
            log_file.write("Epoch number: {}\n".format(epoch))
            log_file.flush()
            idxs = np.random.choice(train_size, train_size, False)

            for batch_num in range(n_train_batches):

                #print(batch_num)
                s = self.batch_size * batch_num
                e = self.batch_size * (batch_num+1)
                batch_idxs = idxs[s:e]
                X_batch, y_batch = get_batch(X, y, batch_idxs) 

                cost = self.classifier.train(*X_batch, y=y_batch)
                log_file.write("batch num: {}, cost: {}\n".format(batch_num, cost))


                # iteration number
                iter = (epoch - 1) * n_train_batches + batch_num

                if (iter + 1) % validation_frequency == 0:
                    this_validation_cost, this_validation_accuracy,_ = self.classifier.val_fn(*X_heldout, y=y_heldout)
                    log_file.write("this is the current validation lost {}\n".format(this_validation_cost))
                    log_file.write("this is the current validation accuracy {}\n".format(this_validation_accuracy))
                    
                    # if we got the best validation score until now
                    if this_validation_accuracy > best_validation_accuracy:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_accuracy > best_validation_accuracy *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        best_validation_accuracy = this_validation_accuracy

                        best_iter = iter
                        best_params = self.classifier.get_params()

                log_file.flush()
                if patience <= iter:
                    done_looping = True
                    break

            end_time_epoch = timeit.default_timer()
            total_time = (end_time_epoch - start_time_epoch) /60.
            print("Total time for epoch: " + str(total_time))

        
        log_file.flush()
        self.classifier.set_params(best_params)
        end_time = timeit.default_timer()
        print("the code trained for {}\n".format(((end_time-start_time)/60)))
        print("Optimization finished: the best validation accuracy of {} achieved at {}\n".format(best_validation_accuracy, best_iter))

        return self

    
    def predict(self, X, y):
        preds = self.classifier.pred(*X)
        precision, recall, fscore, support = score(y, preds)
        return preds,[precision, recall, fscore] 

    def save(self, outfilename):
        self.classifier.save(outfilename)