from processing.DLDataEngineering  import DLDataEngineering
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import h5py
import os

from scipy.ndimage import gaussian_filter
        
#Deep learning packages
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, UpSampling2D, GlobalMaxPooling2D, multiply
from tensorflow.keras.backend import max
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#from tensorflow import keras 
from sklearn.metrics import f1_score,roc_auc_score

import matplotlib.pyplot as plt
import cartopy.feature as cf 
import cartopy.crs as ccrs
import cartopy


from keras_unet_collection import models, base, utils





class DLModeler(object):
    def __init__(self,model_path,hf_path,num_examples,
        class_percentages,predictors,model_args,
        model_type):
        
        self.model_path = model_path
        self.hf_path = hf_path
        self.num_examples = num_examples
        self.class_percentages = class_percentages
        self.model_args = model_args 
        self.model_type = model_type
        
        long_predictors = []
        #Shorten predictor names
        
        for predictor in predictors:
            if "_" in predictor: 
                predictor_name = predictor.split('_')[0].upper() + predictor.split('_')[-1]
            elif " " in predictor: 
                predictor_name = ''.join([v[0].upper() for v in predictor.split()])
            else: predictor_name = predictor
            long_predictors.append(predictor_name)
         
        self.predictors = np.array(long_predictors)
    
        #Class to read data and standardize
        self.dldataeng = DLDataEngineering(self.model_path,self.hf_path, 
            self.num_examples,self.class_percentages,self.predictors,
            self.model_args)
        
        
        return
            

    def train_models(self,member,train_dates,valid_dates):
        """
        Function that reads and extracts pre-processed 2d member data 
        from an ensemble to train a convolutional neural net (cnn) or 
        UNET. 
        The model data is standardized before being input to the cnn, 
        with the observation data in the shape (# examples, # classes). 

        Args:
            member (str): ensemble member data that trains a DL model
        """
        train_data, train_label = self.dldataeng.extract_training_data(member,
            train_dates,self.model_type)
        
        #valid_data, valid_label = self.dldataeng.extract_validation_data(member,valid_dates,self.model_type)
        valid_data, valid_label = [],[]
    
        '''
        if self.model_type == 'CNN':
            onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
            encoded_label = onehot_encoder.fit_transform(train_label.reshape(-1, 1))
            self.train_CNN(member,train_data,encoded_label,valid_data,valid_label)

        elif self.model_type == 'UNET':
            self.train_UNET(member,train_data,train_label,valid_data,valid_label)
        '''
        return 

    def train_UNET(self,member,trainX,trainY,validX,validY):
        
        model_file = self.model_path + f'/{member}_{self.model_args}_UNET.h5'
        print(model_file)
        
        '''
        if os.path.exists(model_file):
            del trainX,trainY,validX,validY
            unet = tf.keras.models.load_model(model_file,compile=False)
            print(f'\nOpening {model_file}\n')
            #self.validate_UNET(model,validX,validY,threshold_file)
            return 
        '''
        print('\nTraining {0} models'.format(member))
        print('Training data shape {0}'.format(np.shape(trainX)))
        print('Training label data shape {0}\n'.format(np.shape(trainY)))
        #print('Validation data shape {0}'.format(np.shape(validX)))
        #print('Validation label data shape {0}\n'.format(np.shape(validY)))
        

        unet3plus = models.unet_3plus_2d(np.shape(trainX[0]), n_labels=1, 
                filter_num_down=[16, 32, 64, 128, 256], 
                filter_num_skip='auto', filter_num_aggregate='auto', 
                stack_num_down=2, stack_num_up=1, activation='LeakyReLU', 
                output_activation='ReLU', batch_norm=False, pool=True, 
                unpool=False, deep_supervision=True, name='unet3plus')
        
        unet3plus.compile(loss=[
            dice_loss,dice_loss,dice_loss,dice_loss,dice_loss],
            loss_weights=[0.25, 0.25, 0.25, 0.25, 1.0],
            optimizer=tf.keras.optimizers.Adam(lr=1e-4))

        print(unet3plus.summary())
         
        #Augment data
        aug = ImageDataGenerator(
                #rotation_range=10,zoom_range=0.15,
                width_shift_range=0.2,height_shift_range=0.2,
                fill_mode="nearest")
            
        #Fit UNET
        n_epochs = 2
        bs = 500
        
        train_generator = aug.flow(trainX,trainY,batch_size=bs)
        conv_hist = unet3plus.fit(train_generator,epochs=n_epochs,verbose=1) 
        
        '''
        pred_s = trainX[0].reshape(1,input_shape[0],
        input_shape[1],input_shape[2])

        prediction = unet.predict(pred_s)[0,:,:,:]
        print(prediction.shape)
        plt.imshow(prediction)
        plt.colorbar()
        plt.show()
        return
        '''
        #Save trained model
        unet3plus.save(model_file)
        print(f'Writing out {model_file}')
        
        #Clear graphs
        tf.keras.backend.clear_session()
        
        #self.validate_UNET(model,validX,validY,threshold_file)
        
        return 
    
    
    def train_CNN(self,member,input_data): 
        """
        Function to train a convolutional neural net (CNN) for random 
        training data and associated labels.

        Args:
            member (str): Ensemble member 
            trainX (tuple): Tuple of (train data, train labels, 
                validation data, validation labels) 
        """
        trainX,trainY,validX,validY = input_data
        
        print('\nTraining {0} models'.format(member))
        print('Training data shape {0}'.format(np.shape(trainX)))
        print('Training label data shape {0}\n'.format(np.shape(trainY)))
        print('Validation data shape {0}'.format(np.shape(validX)))
        print('Validation label data shape {0}\n'.format(np.shape(validY)))
        
        
        model_file = self.model_path + f'/{member}_{self.model_args}_CNN_model.h5'
        print(model_file)
        if not os.path.exists(model_file):
            # Clear graphs
            tf.keras.backend.clear_session()
            
            #Initiliaze Convolutional Neural Net (CNN)
            model = models.Sequential()
            input_shape = np.shape(trainX[0])
            
            #First layer: input shape (y,x,# variables) 
            #Add noise
            model.add(layers.GaussianNoise(0.01, input_shape=(input_shape)))
            for filters in [32,64,128]:
                model.add(layers.Conv2D(filters, (3,3),padding='same'))
                model.add(layers.Conv2D(filters, (3,3),padding='same'))
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU(alpha=0.3))
                model.add(layers.MaxPooling2D())
            
            #Flatten the last convolutional layer 
            model.add(layers.Flatten())
            model.add(layers.Dense(256))
            model.add(layers.LeakyReLU(alpha=0.3))
            model.add(layers.Dense(4,activation='softmax'))
            #Compile neural net
            model.compile(optimizer='adam',loss='categorical_crossentropy',
                metrics=[tf.keras.metrics.AUC()])
            print(model.summary())
            #fit neural net
            n_epochs = 10
            bs = 256

            #augment data
            aug = imagedatagenerator(
                rotation_range=10,zoom_range=0.15,
                width_shift_range=0.2,height_shift_range=0.2,
                fill_mode="nearest")
            
            train_generator = aug.flow(trainx,trainy,batch_size=bs)
            conv_hist = model.fit(
                train_generator,steps_per_epoch=len(trainx) // bs,
                epochs=n_epochs,verbose=1,class_weight=self.class_percentages)
            #save trained model
            model.save(model_file)
            print(f'Writing out {model_file}')
        else:
            model = tf.keras.models.load_model(model_file)
            print(f'\nOpening {model_file}\n')

        del trainY,trainX
        
        threshold_file = self.model_path + f'/{member}_{self.model_args}_CNN_model_threshold.h5'
        if os.path.exists(threshold_file): 
            del validX,validY
            return
        
        self.validate_CNN(model,validX,validY,threshold_file)
        return 

    def validate_CNN(self,model,validX,validY,threshold_file): 
        print()
        #Predict on validation data
        cnn_preds = model.predict(validX)
        sev_hail = cnn_preds[:,2]
        sig_hail = cnn_preds[:,3]
        #combine the severe hail and sig severe hail classes
        sev_prob_preds = sev_hail+sig_hail
        print('Max probability',np.nanmax(sev_prob_preds))
        #classify labels as severe hail or no hail
        true_preds = np.where(validY >= 2, 1, 0)
        del validX, validY
        
        df_best_score = pd.DataFrame(np.zeros((1,1)),columns=['Size Threshold'])
        #Find threshold with the highest validation AUC score 
        auc_score = []
        thresholds = np.arange(0.1,1.01,0.02)
        for t in thresholds:
            threshold_preds = np.where(sev_prob_preds >= t,1,0)
            auc_score.append(roc_auc_score(true_preds, threshold_preds))
        
        print(auc_score)
        #output threshold with highest AUC 
        df_best_score['Size Threshold'] = thresholds[np.argmax(auc_score)]
        print(df_best_score)
        df_best_score.to_csv(threshold_file)
        print(f'Writing out {threshold_file}')
        return 
                
    
    def predict_model(self,member,patch_map_conversion_indices,
        total_map_shape,subset_map_shape,date,patch_radius,forecast_grid_path,#):
        lon_grid,lat_grid):
        """
        Function that opens a pre-trained convolutional neural net (cnn). 
        and predicts hail probability forecasts for a single ensemble member.
    
        Args:
        Right now only includes severe hail prediction, not sig-severe
        """
        
        ################## 
        # Load in any saved DL model files
        ################## 
         
        #Clear any saved DL graphs
        tf.keras.backend.clear_session()
        
        #Load DL model
        model_file = self.model_path + f'/{member}_{self.model_args}_UNET.h5'
        DL_model = tf.keras.models.load_model(model_file,compile=False) 
        
        ################## 
        #Extract forecast data (#hours, #patches, nx, ny, #variables)
        ################## 
        forecast_data = self.dldataeng.read_files('forecast',member,date,[None],[None])

        if forecast_data is None: 
            print('No forecast data found')
            return
        
        ################## 
        # Standardize hourly data
        ################## 
        
        standard_forecast_data = np.array([self.dldataeng.standardize_data(member,forecast_data[hour]) 
            for hour in np.arange(forecast_data.shape[0])])
        del forecast_data
        ################## 
        # Produce gridded hourly hail forecast 
        ################## 
        
        total_grid = np.empty( (standard_forecast_data.shape[0],
            total_map_shape[0]*total_map_shape[1]) )*np.nan
        print(total_grid.shape)
        for hour in np.arange(standard_forecast_data.shape[0]):
            sliced_DL_prediction = np.array(DL_model.predict(standard_forecast_data[hour]))
            print(sliced_DL_prediction.shape)
            if self.model_type == 'UNET':
                for patch in np.arange(standard_forecast_data.shape[1]):
                    patch_indices = patch_map_conversion_indices[patch]
                    #Gets rid of overlapping edges
                    overlap_pt = 4
                    # If unet3+ then the last output tensor is the correct one
                    hourly_patch_data = sliced_DL_prediction[patch,overlap_pt:-overlap_pt,
                        overlap_pt:-overlap_pt,0].ravel()
                    total_grid[hour,patch_indices] = hourly_patch_data
         
        del sliced_DL_prediction
        del standard_forecast_data
        output_data=total_grid.reshape((total_grid.shape[0],)+total_map_shape)
        print(output_data.shape)
        print(np.nanmax(output_data))

        date_outpath = forecast_grid_path + f'{date[0][:-5]}/'
        #Output gridded forecasts
        if not os.path.exists(date_outpath): os.makedirs(date_outpath)
        gridded_out_file = date_outpath + f'{member}_{date[0]}_forecast_grid.h5'
        print(f'Writing out {gridded_out_file}')
        with h5py.File(gridded_out_file, 'w') as hf: 
            hf.create_dataset("data",data=output_data,
            compression='gzip',compression_opts=6)
        
        '''
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection = ccrs.LambertConformal())
        ax.add_feature(cf.COASTLINE)   
        ax.add_feature(cf.OCEAN)
        ax.add_feature(cf.BORDERS,linestyle='-')
        ax.add_feature(cf.STATES.with_scale('50m'),linestyle='-',edgecolor='black')
        plt.contourf(lon_grid,lat_grid,output_data[0,:,:],transform=ccrs.PlateCarree())
        plt.colorbar()
        plt.show()
        '''
        
        return

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / denominator

'''
From: https://idiotdeveloper.com/unet-segmentation-in-tensorflow/
''' 

def down_block(x, filters, kernel_size=(3, 3)):
    c = layers.Conv2D(filters, kernel_size, padding='same')(x)
    c = layers.LeakyReLU(alpha=0.2)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Conv2D(filters, kernel_size, padding='same')(c)
    c = layers.LeakyReLU(alpha=0.2)(c)
    c = layers.BatchNormalization()(c)
    p = layers.MaxPooling2D((2,2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3)):
    up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    concat = layers.Concatenate()([up, skip])
    c = layers.Conv2D(filters, kernel_size, padding='same')(concat)
    c = layers.LeakyReLU(alpha=0.2)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Conv2D(filters, kernel_size, padding='same')(c)
    c = layers.LeakyReLU(alpha=0.2)(c)
    c = layers.BatchNormalization()(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3)):
    c = layers.Conv2D(filters, kernel_size, padding='same')(x)
    c = layers.LeakyReLU(alpha=0.2)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Conv2D(filters, kernel_size, padding='same')(c)
    c = layers.LeakyReLU(alpha=0.2)(c)
    c = layers.BatchNormalization()(c)
    return c
