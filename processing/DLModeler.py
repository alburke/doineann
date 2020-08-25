from util.make_proj_grids import make_proj_grids, read_ncar_map_file
from sklearn.utils import shuffle
from os.path import exists
from glob import glob
import pandas as pd
import numpy as np
import random
random.seed(42)
import h5py

#Parallelization packages
from multiprocessing import Pool
import multiprocessing as mp
        
#Deep learning packages
import tensorflow as tf
from tensorflow.keras import layers,regularizers,models

import matplotlib.pyplot as plt


class DLModeler(object):
    def __init__(self,model_path,start_dates,end_dates,var_name,
        num_examples,class_percentages):
        
        self.model_path = model_path
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.var_name = var_name
        self.num_examples = num_examples
        self.class_percentages = class_percentages
        return

    def train_CNN(self,member,model_data,model_labels,valid_data,valid_labels):
        """
        Function to train a convolutional neural net (CNN) for random 
        training data and associated labels.

        Args:
            member (str): Ensemble member 
            model_data (ndarray): ensemble member data
            model_labels (ndarray): observation labels

        """
        print('\nTraining {0} models'.format(member))
        print('Training data shape {0}'.format(np.shape(model_data)))
        print('Training label data shape {0}\n'.format(np.shape(model_labels)))
        #print('Validation data shape {0}'.format(np.shape(valid_data)))
        #print('Validation label data shape {0}\n'.format(np.shape(valid_labels)))
        
        model_file = self.model_path+'/{0}_{1}_{2}_{3}_CNN_model.h5'.format(
            member,self.start_dates['train'].strftime('%Y%m%d'),
            self.end_dates['train'].strftime('%Y%m%d'),self.num_examples)
        if not exists(model_file):
            #Initiliaze Convolutional Neural Net (CNN)
            model = models.Sequential()
            #First layer, input shape (y,x,# variables) 
            input_shape = np.shape(model_data[0])
            model.add(layers.GaussianNoise(0.1, input_shape=(input_shape)))
            model.add(layers.Conv2D(32, (3,3),padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.3))
            model.add(layers.MaxPooling2D())
            #Second layer
            model.add(layers.Conv2D(64, (3,3),padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.3))
            #model.add(layers.Activation("relu"))
            model.add(layers.MaxPooling2D())
        
            #Third layer
            model.add(layers.Conv2D(128, (3,3),padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.3))
            #model.add(layers.Activation("relu"))
            model.add(layers.MaxPooling2D())
            
            #Flatten the last convolutional layer into a long feature vector
            model.add(layers.Flatten())
            #model.add(layers.Dropout(0.10))
            model.add(layers.Dense(512))
            model.add(layers.LeakyReLU(alpha=0.3))
            model.add(layers.Dense(4,activation='softmax'))
            #Compile neural net
            model.compile(optimizer='adam',loss='categorical_crossentropy',
                metrics=[tf.keras.metrics.AUC()])
            print(model.summary())
            #Fit neural net
            n_epochs = 40
            conv_hist = model.fit(model_data,model_labels,
                epochs=n_epochs,batch_size=512,verbose=1,
                class_weight=self.class_percentages)
            #Save trained model
            model.save(model_file)
            print('Writing out {0}'.format(model_file))
        else:
            print('Opening {0}'.format(model_file))
            model = models.load_model(model_file)

        del model_labels,model_data
        
        '''
        #Find the threshold with the highest AUC score 
        #on validation data
        threshold_file = self.model_path+'/{0}_{1}_{2}_{3}_CNN_model_threshold.h5'.format(
            member,self.start_dates['train'].strftime('%Y%m%d'),
            self.end_dates['train'].strftime('%Y%m%d'),self.num_examples)
        
        if exists(threshold_file): 
            del valid_data,valid_labels
            return

        cnn_preds = model.predict(valid_data)
        del valid_data
        sev_hail = cnn_preds[:,2]
        sig_hail = cnn_preds[:,3]
        sev_prob_preds = sev_hail+sig_hail
        print(np.nanmax(sev_prob_preds))
        df_best_score = pd.DataFrame(np.zeros((1, 1)),columns=['size_threshold'])
        thresholds = np.arange(0.01,0.97,0.01)
        true_preds = np.where(valid_labels >= 2, 1, 0)
        del valid_labels
        for prob_thresh in thresholds:
            threshold_preds = np.where(sev_prob_preds >= prob_thresh,1,0)
            print(np.count_nonzero(threshold_preds))
            if np.count_nonzero(threshold_preds) <= 50:
                df_best_score.loc[0,'size_threshold'] = prob_thresh
                break
        print(df_best_score)
        print('Writing out {0}'.format(threshold_file))
        df_best_score.to_csv(threshold_file)
        '''
        return 
                
    
    def predict_models(self,
        member,subset_map_data,total_map_data,date,dldataeng,
        patch_radius,map_conversion_inds,forecast_grid_path):
        """
        Function that opens a pre-trained convolutional neural net (cnn). 
        and predicts hail probability forecasts for a single ensemble member.
    
        Args:
        Right now only includes severe hail prediction, not sig-severe
        """
        
        #Extract forecast data (#hours, #patches, nx, ny, #variables)
        data = dldataeng.read_files('forecast',member,date)
        if data is None: 
            print('No forecast data found')
            return
        
        cnn_model_file = self.model_path+'/{0}_{1}_{2}_{3}_CNN_model.h5'.format(
            member,self.start_dates['train'].strftime('%Y%m%d'),
            self.end_dates['train'].strftime('%Y%m%d'),self.num_examples)
        cnn_model = models.load_model(model_file) 
        
        #Use minimum prob threshold chosen with validation data
        threshold_file = self.model_path+'/{0}_{1}_{2}_{3}_CNN_model_threshold.h5'.format(
            member,self.start_dates['train'].strftime('%Y%m%d'),
            self.end_dates['train'].strftime('%Y%m%d'),self.num_examples)
        if not os.path.exists(threshold_file):
            print('No thresholds found')
            return 
        prob_thresh = 0 #pd.read_csv(threshold_file).loc[0,'size_threshold']+0.05
        print(prob_thresh)    
        # Produce hail forecast using standardized forecast data every hour
        total_grid = np.zeros((data.shape[0],)+total_map_data['lat'].ravel().shape)
        total_count = 0
        for hour in np.arange(data.shape[0]):
            standard_forecast_data = dlprocess.standardize_data(member,data[hour])
            #Predict probability of severe hail
            cnn_preds = cnn_model.predict(standard_forecast_data)
            severe_proba_indices = np.where( (cnn_preds[:,2]+cnn_preds[:,3]) >= prob_thresh)[0]
            severe_patches = np.zeros_like(subset_map_data[0])
            #If no hourly severe hail predicted, continue
            if len(severe_proba_indices) <1 : continue
            severe_patches[severe_proba_indices] = np.full((patch_radius,patch_radius), 1)
            total_grid[hour,map_conversion_inds] = severe_patches.ravel()
            print(hour,len(severe_proba_indices),np.nanmax((cnn_preds[:,2]+cnn_preds[:,3])))
            total_count += len(severe_proba_indices)
        print('Total severe probs:',total_count)
        print()
    
        #Output gridded forecasts
        date_outpath = forecast_grid_path + '{0}/'.format(date[:-5])
        if not os.path.exists(date_outpath): os.makedirs(date_outpath)
        gridded_out_file = date_outpath + '{0}_{1}_forecast_grid.h5'.format(member,date)
        print('Writing out {0}'.format(gridded_out_file))
        with h5py.File(gridded_out_file, 'w') as hf: hf.create_dataset("data",
            data=total_grid.reshape((2,data.shape[0],)+total_map_data['lat'].shape),
            compression='gzip',compression_opts=6)
        return
