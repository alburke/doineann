from util.make_proj_grids import make_proj_grids, read_ncar_map_file
from os.path import exists
import pandas as pd
import numpy as np
import h5py
        
#Deep learning packages
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score,roc_auc_score


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
    
    
        self.model_args = '{0}_{1}_{2}'.format(
            self.start_dates,self.end_dates,self.num_examples)
        return

    def train_UNET(self,member,input_data):
        
        model_file = self.model_path + f'/{member}_{self.model_args}_UNET.h5'
        print(model_file)
        
        if exists(model_file):
            del input_data
            model = models.load_model(model_file)
            print(f'\nOpening {model_file}\n')
            #self.validate_UNET(model,validX,validY,threshold_file)
            return 


        trainX,trainY,validX,validY = input_data
        
        print('\nTraining {0} models'.format(member))
        print('Training data shape {0}'.format(np.shape(trainX)))
        print('Training label data shape {0}\n'.format(np.shape(trainY)))
        #print('Validation data shape {0}'.format(np.shape(validX)))
        #print('Validation label data shape {0}\n'.format(np.shape(validY)))
        
        input_shape = np.shape(trainX[0])
        filters = [16, 24, 32, 48, 64]

        #First layer: input shape (y,x,# variables) 
        inputs = layers.Input(input_shape)
        x = inputs
        # Add noise
        x = layers.GaussianNoise(0.01)(x)
        
        #Downsampling blocks 
        c1, p1 = down_block(x, filters[0]) #128 -> 64
        c2, p2 = down_block(p1, filters[1]) #64 -> 32
        c3, p3 = down_block(p2, filters[2]) #32 -> 16
        c4, p4 = down_block(p3, filters[3]) #16 -> 8

        #Bottleneck
        bn = bottleneck(p4, filters[4])

        #Upsampling blocks 
        u1 = up_block(bn, c4, filters[3]) #8 -> 16
        u2 = up_block(u1, c3, filters[2]) #16 -> 32
        u3 = up_block(u2, c2, filters[1]) #32 -> 64
        u4 = up_block(u3, c1, filters[0]) #64 -> 128
        
        #Output layer
        outputs = layers.Conv2D(1, (1, 1), padding="same", activation="relu")(u4)
        
        #Compile UNET
        unet = models.Model(inputs, outputs)
        unet.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse')
        print(unet.summary())
        
        #Augment data
        aug = ImageDataGenerator(
                rotation_range=10,zoom_range=0.15,
                width_shift_range=0.2,height_shift_range=0.2,
                fill_mode="nearest")
            
        #Fit UNET
        n_epochs = 10
        bs = 10
            
        train_generator = aug.flow(trainX,trainY,batch_size=bs)
        conv_hist = unet.fit(train_generator,epochs=n_epochs,verbose=1) 
        #Save trained model
        #model.save(model_file)
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
        if not exists(model_file):
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
            model = models.load_model(model_file)
            print(f'\nOpening {model_file}\n')

        del trainY,trainX
        
        threshold_file = self.model_path + f'/{member}_{self.model_args}_CNN_model_threshold.h5'
        if exists(threshold_file): 
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
        data = dldataeng.read_files('forecast',member,date,[None],[None])
        if data is None: 
            print('No forecast data found')
            return
        
        tf.keras.backend.clear_session()
        cnn_model_file = self.model_args+f'{member}_CNN_model.h5'
        cnn_model = models.load_model(model_file) 
        
        #Use minimum prob threshold chosen with validation data
        threshold_file = self.model_path + f'/{member}_{self.model_args}_CNN_model_threshold.h5'
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
