# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 09:02:36 2022

@author: SaeidNikbakht
"""

import os
import pandas as pd
import numpy as np
from DataPreprocessor import DataPreprocessor
from ModelCreator import ModelCreator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class ModelTrainer:
    
    def __init__(self, raw_model, callbacks_dict, trainer_dict):
        self.raw_model = raw_model
        self.model_name = raw_model.name
        self.monitoring_metric = callbacks_dict["monitoring_metric"]
        self.patience = callbacks_dict["patience"]
        self.model_path = callbacks_dict["model_path"]
        self.callback_verbose = callbacks_dict["callback_verbose"]
        self.batch_size = trainer_dict["batch_size"]
        self.epochs = trainer_dict["epochs"]
        self.training_verbose = trainer_dict["training_verbose"]
        self.validation_split = trainer_dict["validation_split"]
    
    def create_repo_if_not_exist(self):
        isExist = os.path.exists(self.model_path)
        if not isExist:
            os.makedirs(self.model_path)   
             
    def create_callbacks(self):
        early_stopping = EarlyStopping(monitor = self.monitoring_metric, patience = self.patience, restore_best_weights=True)
        checkpoint = ModelCheckpoint(filepath = self.model_path + self.model_name + ".h5", 
                             monitor = self.monitoring_metric,
                             verbose = self.callback_verbose, 
                             save_best_only = True,
                             mode = 'min')
        callbacks = [checkpoint, early_stopping]
        return callbacks
    
    def train_model(self, X_train, y_train):
        self.create_repo_if_not_exist()
        callbacks = self.create_callbacks()
        history = self.raw_model.fit(X_train, # input data
                  y_train, # target data
                  batch_size = self.batch_size, # Number of samples per gradient update. If unspecified, batch_size will default to 32.
                  epochs = self.epochs, # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
                  verbose = self.training_verbose, # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
                  callbacks = callbacks, # default=None, list of callbacks to apply during training. See tf.keras.callbacks
                  validation_split = self.validation_split, # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. 
                  #validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch. 
                  shuffle=True, # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
                  class_weight=None, # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                  sample_weight=None, # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
                  initial_epoch=0, # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
                  steps_per_epoch=None, # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. 
                  validation_steps=None, # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
                  validation_batch_size=None, # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
                  validation_freq=1, # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
                  max_queue_size=10, # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                  workers=1, # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
                  use_multiprocessing=False, # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. 
                  # experimental_relax_shapes=True,
                  )
        return self.raw_model, history


# Test the class        
if __name__ == "__main__":
    from DataCleaner import DataCleaner
    from DataPreprocessor import DataPreprocessor
    from ModelCreator import ModelCreator

    # Reading the Raw data
    real_pv_15min = pd.read_csv("datasets/Prod_real.csv", index_col = "Time", parse_dates = True)
    sim_pv_15min = pd.read_csv("datasets/Prod_sim.csv", index_col = "Time", parse_dates = True).loc[real_pv_15min.index]

    # Real PV Preparation
    ## Cleaning the Data
    cleaner_dict = {"value_type": "abs",
                    "resolution": "15min",
                    "column_name": "pv_sim",
                    "max_anomoly_margin": 99.9,
                    "min_anomoly_margin": 0.1,
                    "agg_func": "sum",
                    "hours_modifier":0}
    data_cleaner = DataCleaner(real_pv_15min, cleaner_dict)
    cln_real_pv_15min = data_cleaner.data_prep()

    ## Preprocessing the Data
    splitter_dict = {"past_steps":192,
                    "future_steps":16,
                    "test_size":0.2,
                    "random_state":1234,
                    "shuffle_flag":True}
    scaler_dict = {"scaler_type":"standard",
                   "scaler_path":"scalers"}
    data_preprocessor = DataPreprocessor(real_pv_15min, splitter_dict, scaler_dict)
    X_train_real_scaled_3d, X_test_real_scaled_3d, y_train, y_test = data_preprocessor.run_data_preprocessor()

    # Simulated PV Preparation
    ## Cleaning the Data
    cleaner_dict = {"value_type": "abs",
                    "resolution": "15min",
                    "column_name": "pv_sim",
                    "max_anomoly_margin": 99.9,
                    "min_anomoly_margin": 0.1,
                    "agg_func": "sum",
                    "hours_modifier":0}
    data_cleaner = DataCleaner(sim_pv_15min, cleaner_dict)
    cln_sim_pv_15min = data_cleaner.data_prep()

    ## Preprocessing the Data
    splitter_dict = {"past_steps":192,
                    "future_steps":16,
                    "test_size":0.2,
                    "random_state":1234,
                    "shuffle_flag":True}
    scaler_dict = {"scaler_type":"standard",
                   "scaler_path":"scalers"}
    data_preprocessor = DataPreprocessor(real_pv_15min, splitter_dict, scaler_dict)
    X_train_sim_scaled_3d, X_test_sim_scaled_3d, _, _ = data_preprocessor.run_data_preprocessor()

    # Combining input training data together for multivariate forecast
    X_train_scaled_3d = np.concatenate([X_train_real_scaled_3d, X_train_sim_scaled_3d],2)
    
    # Creating the model
    initializer_dict = {"past_steps": 192,
                        "future_steps": 16,
                        "model_name": "HummelHQ_stand_p192_f16_15min_v2"}
    compiler_dict = {"metrics":["mean_absolute_error"],
                     "loss": "mean_squared_error",
                     "optimizer": "adam",
                     "num_features":2}

    model_creator = ModelCreator(initializer_dict, compiler_dict)
    raw_model = model_creator.multi_step_model_7()

    # Training the Raw Model based on the combined training input data
    callbacks_dict = {"monitoring_metric": "val_loss",
                      "patience": 3,
                      "model_path": "Assets/Harrer/Models/",
                      "callback_verbose": 2}
    trainer_dict = {"batch_size": 32,
                    "epochs": 50,
                    "validation_split": 0.1,
                    "training_verbose": 2}
    model_trainer = ModelTrainer(raw_model, callbacks_dict, trainer_dict)
    trained_model = model_trainer.train_model(X_train_scaled_3d, y_train)
    