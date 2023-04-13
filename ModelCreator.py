# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 08:49:14 2022

@author: SaeidNikbakht
"""

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, SimpleRNN, GRU, Conv1D, MaxPooling1D, Flatten, RepeatVector, BatchNormalization, ReLU, AveragePooling1D
tensorflow.random.set_seed(1234)

class ModelCreator:
    
    def __init__(self, initializer_dict, compiler_dict):
        self.past_steps = initializer_dict["past_steps"]
        self.future_steps = initializer_dict["future_steps"]
        self.model_name = initializer_dict["model_name"]
        self.optimizer = compiler_dict["optimizer"]
        self.loss = compiler_dict["loss"]
        self.metrics = compiler_dict["metrics"]
        self.num_features = compiler_dict["num_features"]
    
    def multi_step_model_1(self):
        
        ##### Step 4 - Specify the structure of a Neural Network
        #cnn repeat vector
        raw_model = Sequential(name = self.model_name) # Model
        raw_model.add(Input(shape=(self.past_steps,self.num_features), name='Input-Layer'))
        raw_model.add(Conv1D(filters=16, kernel_size=3, activation='tanh'))
        raw_model.add(MaxPooling1D(pool_size=2))
        raw_model.add(Conv1D(filters=32, kernel_size=3, activation='tanh'))
        raw_model.add(MaxPooling1D(pool_size=2))
        raw_model.add(Flatten())
        raw_model.add(RepeatVector(16))
        raw_model.add(LSTM(32, activation='tanh'))
        raw_model.add(Dense(units = self.future_steps, name='Output-Layer'))
        ##### Step 5 - Compile keras model
        raw_model.compile(optimizer = self.optimizer, # default='rmsprop', an algorithm to be used in backpropagation
                      loss = self.loss, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                      metrics = self.metrics, # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
                      loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
                      weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                      run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                      #steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                     )
        return raw_model
    
    def multi_step_model_2(self):
        
        ##### Step 4 - Specify the structure of a Neural Network
        #cnn repeat vector
        raw_model = Sequential(name = self.model_name) # Model
        raw_model.add(Input(shape=(self.past_steps,self.num_features), name='Input-Layer'))
        raw_model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
        raw_model.add(MaxPooling1D(pool_size=2))
        raw_model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
        raw_model.add(MaxPooling1D(pool_size=2))
        raw_model.add(Flatten())
        raw_model.add(RepeatVector(64))
        raw_model.add(LSTM(32, activation='tanh'))
        raw_model.add(RepeatVector(32))
        raw_model.add(LSTM(16, activation='tanh'))
        raw_model.add(Dense(units = self.future_steps, name='Output-Layer'))
        ##### Step 5 - Compile keras model
        raw_model.compile(optimizer = self.optimizer, # default='rmsprop', an algorithm to be used in backpropagation
                      loss = self.loss, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                      metrics = self.metrics, # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
                      loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
                      weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                      run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                      #steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                     )
        return raw_model

    def multi_step_model_3(self):
        
        ##### Step 4 - Specify the structure of a Neural Network
        #cnn repeat vector
        raw_model = Sequential(name = self.model_name) # Model
        raw_model.add(Input(shape=(self.past_steps,self.num_features), name='Input-Layer'))
        raw_model.add(Conv1D(filters=64, kernel_size=5))
        raw_model.add(BatchNormalization())
        raw_model.add(ReLU())
        raw_model.add(Conv1D(filters=64, kernel_size=5))
        raw_model.add(ReLU())
        raw_model.add(AveragePooling1D(pool_size=5))
        raw_model.add(Flatten())
        raw_model.add(RepeatVector(64))
        raw_model.add(RNN(32, activation='tanh'))
        raw_model.add(RepeatVector(32))
        raw_model.add(RNN(32, activation='tanh'))
        raw_model.add(Dense(units = self.future_steps, name='Output-Layer'))
        
        ##### Step 5 - Compile keras model
        raw_model.compile(optimizer = self.optimizer, # default='rmsprop', an algorithm to be used in backpropagation
                      loss = self.loss, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                      metrics = self.metrics, # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
                      loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
                      weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                      run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                      #steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                     )
        
        return raw_model
    
    def multi_step_model_4(self):
        
        ##### Step 4 - Specify the structure of a Neural Network
        #cnn repeat vector
        raw_model = Sequential(name = self.model_name) # Model
        raw_model.add(Input(shape=(self.past_stepsself.num_features), name='Input-Layer'))
        raw_model.add(Conv1D(filters=64, kernel_size=7, padding = "same"))
        raw_model.add(BatchNormalization())
        raw_model.add(ReLU())
        raw_model.add(AveragePooling1D(pool_size=3, padding = "same"))
        raw_model.add(Conv1D(filters=64, kernel_size=7, padding = "same"))
        raw_model.add(BatchNormalization())
        raw_model.add(ReLU())
        raw_model.add(AveragePooling1D(pool_size=3, padding = "same"))
        raw_model.add(Flatten())
        raw_model.add(RepeatVector(64))
        raw_model.add(SimpleRNN(32, activation='tanh'))
        raw_model.add(RepeatVector(32))
        raw_model.add(SimpleRNN(16, activation='tanh'))
        raw_model.add(Dense(units = self.future_steps, name='Output-Layer'))
        ##### Step 5 - Compile keras model
        raw_model.compile(optimizer = self.optimizer, # default='rmsprop', an algorithm to be used in backpropagation
                      loss = self.loss, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                      metrics = self.metrics, # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
                      loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
                      weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                      run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                      #steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                     )
        return raw_model
    
    def multi_step_model_5(self):
        
        ##### Step 4 - Specify the structure of a Neural Network
        #cnn repeat vector
        raw_model = Sequential(name = self.model_name) # Model
        raw_model.add(Input(shape=(self.past_steps,self.num_features), name='Input-Layer'))
        raw_model.add(Conv1D(filters=64, kernel_size=7, padding = "same"))
        raw_model.add(BatchNormalization())
        raw_model.add(ReLU())
        raw_model.add(AveragePooling1D(pool_size=3, padding = "same"))
        raw_model.add(Conv1D(filters=64, kernel_size=7, padding = "same"))
        raw_model.add(BatchNormalization())
        raw_model.add(ReLU())
        raw_model.add(AveragePooling1D(pool_size=3, padding = "same"))
        raw_model.add(Flatten())
        raw_model.add(RepeatVector(64))
        raw_model.add(GRU(32, activation='tanh'))
        raw_model.add(RepeatVector(32))
        raw_model.add(GRU(16, activation='tanh'))
        raw_model.add(Dense(units = self.future_steps, name='Output-Layer'))
        ##### Step 5 - Compile keras model
        raw_model.compile(optimizer = self.optimizer, # default='rmsprop', an algorithm to be used in backpropagation
                      loss = self.loss, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                      metrics = self.metrics, # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
                      loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
                      weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                      run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                      #steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                     )
        return raw_model
    
    def multi_step_model_6(self):
        
        ##### Step 4 - Specify the structure of a Neural Network
        #cnn repeat vector
        raw_model = Sequential(name = self.model_name) # Model
        raw_model.add(Input(shape=(self.past_steps,self.num_features), name='Input-Layer'))
        raw_model.add(Conv1D(filters=64, kernel_size=7, padding = "same"))
        raw_model.add(BatchNormalization())
        raw_model.add(ReLU())
        raw_model.add(AveragePooling1D(pool_size=3, padding = "same"))
        raw_model.add(Conv1D(filters=64, kernel_size=7, padding = "same"))
        raw_model.add(BatchNormalization())
        raw_model.add(ReLU())
        raw_model.add(AveragePooling1D(pool_size=3, padding = "same"))
        raw_model.add(Flatten())
        raw_model.add(RepeatVector(64))
        raw_model.add(LSTM(32, activation='tanh'))
        raw_model.add(RepeatVector(32))
        raw_model.add(GRU(16, activation='tanh'))
        raw_model.add(Dense(units = self.future_steps, name='Output-Layer'))
        ##### Step 5 - Compile keras model
        raw_model.compile(optimizer = self.optimizer, # default='rmsprop', an algorithm to be used in backpropagation
                      loss = self.loss, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                      metrics = self.metrics, # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
                      loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
                      weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                      run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                      #steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                     )
        return raw_model
    
    def multi_step_model_7(self):
        
        ##### Step 4 - Specify the structure of a Neural Network
        #cnn repeat vector
        raw_model = Sequential(name = self.model_name) # Model
        raw_model.add(Input(shape=(self.past_steps,self.num_features), name='Input-Layer'))
        raw_model.add(Conv1D(filters=64, kernel_size=7, padding = "same"))
        raw_model.add(BatchNormalization())
        raw_model.add(ReLU())
        raw_model.add(AveragePooling1D(pool_size=3, padding = "same"))
        raw_model.add(Conv1D(filters=64, kernel_size=7, padding = "same"))
        raw_model.add(BatchNormalization())
        raw_model.add(ReLU())
        raw_model.add(AveragePooling1D(pool_size=3, padding = "same"))
        raw_model.add(Flatten())
        raw_model.add(RepeatVector(64))
        raw_model.add(GRU(32, activation='tanh'))
        raw_model.add(RepeatVector(32))
        raw_model.add(LSTM(16, activation='tanh'))
        raw_model.add(Dense(units = self.future_steps, name='Output-Layer'))
        ##### Step 5 - Compile keras model
        raw_model.compile(optimizer = self.optimizer, # default='rmsprop', an algorithm to be used in backpropagation
                      loss = self.loss, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                      metrics = self.metrics, # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
                      loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
                      weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                      run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                      #steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                     )
        return raw_model
# Test the class        
if __name__ == "__main__":

    initializer_dict = {"past_steps": 192,
                        "future_steps": 16,
                        "model_name": "HummelHQ_stand_p192_f16_15min_v2"}
    compiler_dict = {"metrics":["mean_absolute_error"],
                     "loss": "mean_squared_error",
                     "optimizer": "adam",
                     "num_features":2}

    model_creator = ModelCreator(initializer_dict, compiler_dict)
    raw_model = model_creator.multi_step_model_7()

