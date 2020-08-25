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
    def __init__(self,model_path,hf_path,start_dates,end_dates,
        num_examples,class_percentage,patch_radius,run_date_format,forecast_variables):
        
        self.model_path = model_path
        self.hf_path = hf_path
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.num_examples = num_examples
        self.class_percentage = class_percentage
        self.patch_radius = patch_radius
        self.run_date_format = run_date_format
        
        long_forecast_variables = []
        #Shorten variable names
        for variable in forecast_variables:
            if "_" in variable: 
                variable_name= ''.join([v[0].upper() for v in variable.split()]) + variable.split('_')[-1]
            elif " " in variable: 
                variable_name= ''.join([v[0].upper() for v in variable.split()])
            else:variable_name = variable
            long_forecast_variables.append(variable_name)
        self.forecast_variables = np.array(long_forecast_variables)

        return

    def preprocess_training_data(self,member):
        """
        Function that reads and extracts 2D sliced training data from 
        an ensemble member. 

        Args:
            member (str): Ensemble member data that trains a CNN
        Returns:
            Sliced ensemble member data and observations.
        """
        
        print()
        mode = 'train'

        #Find/create files containing information about training
        filename_args = (member,self.start_dates[mode].strftime('%Y%m%d'),
                        self.end_dates[mode].strftime('%Y%m%d'),self.num_examples)
        #Random patches for training
        training_filename=self.model_path+\
            '/{0}_{1}_{2}_{3}_training_examples.csv'.format(*filename_args)
        #Observations associated with random patches    
        obs_training_labels_filename=self.model_path+\
            '/{0}_{1}_{2}_{3}_training_obs_label.h5'.format(*filename_args)
        #Ensemble member data associated with random patches
        model_training_data_filename = self.model_path+\
            '/{0}_{1}_{2}_{3}_standard_training_patches.h5'.format(*filename_args)
       
        #Opening training data files
        if exists(obs_training_labels_filename) and exists(model_training_data_filename):
            #Opening training files
            print('Opening {0}'.format(obs_training_labels_filename))
            with h5py.File(obs_training_labels_filename, 'r') as ohf:
                member_obs_label = ohf['data'][()]
            
            print('Opening {0}'.format(model_training_data_filename))
            with h5py.File(model_training_data_filename, 'r') as mhf:
                standard_member_model_data = mhf['data'][()]
            '''
            print(np.shape(member_obs_label),np.shape(standard_member_model_data))
            c = standard_member_model_data.reshape(-1, standard_member_model_data.shape[-1])
            print(np.nanmax(c,axis=0), np.nanmin(c,axis=0))

            plt.figure(figsize=(30,30))
            plt.boxplot(c,labels=self.forecast_variables)
            plt.show()
            '''
        else:
            if exists(training_filename): 
                #Opening training examples data file
                print('Opening {0}'.format(training_filename))
                patches = pd.read_csv(training_filename,index_col=0)
            else:
                #Selecting random patches for training
                patches = self.training_data_selection(member,training_filename)
            
            #Extracting obs labels
            member_obs_label = patches['Obs Label'].values.astype('int64')
            with h5py.File(obs_training_labels_filename, 'w') as hf:
                hf.create_dataset("data",data=member_obs_label)
            print('Writing out {0}'.format(obs_training_labels_filename))
            
            #Extracting model patch data
            member_model_data = self.read_files(mode,member,patches['Random Date'], 
                patches['Random Hour'],patches['Random Patch'],patches['Data Augmentation'])
            
            if member_model_data is None: print('No training data found')
            else:
                #Augment data by adding small amounts of variance
                augment_inds = np.where(patches['Data Augmentation'] >0.5)[0]
                if len(augment_inds) >= 1:
                    print('Augmenting data')
                    print(len(augment_inds))
                    for var in np.arange(len(self.forecast_variables)):
                        total_var = np.nanvar(member_model_data[augment_inds,:,:,var])
                        for example in np.arange(len(augment_inds)):
                            member_model_data[example,:,:,var] =\
                            member_model_data[example,:,:,var]+total_var*np.random.choice(np.arange(-0.3,0.31,0.01))
                #Standardize data
                #standard_member_model_data = member_model_data
                standard_member_model_data = self.standardize_data(member,member_model_data,mode='train') 
                with h5py.File(model_training_data_filename, 'w') as hf:
                    hf.create_dataset("data",data=standard_member_model_data)
                print('Writing out {0}'.format(model_training_data_filename)) 
           
        return standard_member_model_data, member_obs_label

    def preprocess_validation_data(self,member):
        """
        Function that reads and extracts 2D sliced training data from 
        an ensemble member. 

        Args:
            member (str): Ensemble member data that trains a CNN
        Returns:
            Sliced ensemble member data and observations.
        """
        
        validation_filename_args = (member,self.start_dates['valid'].strftime('%Y%m%d'),
                        self.end_dates['valid'].strftime('%Y%m%d'))
        #Observations associated with validation patches    
        obs_validation_labels_filename=self.model_path+\
            '/{0}_{1}_{2}_validation_obs_label.h5'.format(*validation_filename_args)
        #Ensemble member data associated with validation patches
        model_validation_data_filename = self.model_path+\
            '/{0}_{1}_{2}_standard_validation_patches.h5'.format(*validation_filename_args)
        
        if exists(obs_validation_labels_filename) and exists(model_validation_data_filename): 
            #Opening validation files
            print('Opening {0}'.format(model_validation_data_filename))
            with h5py.File(model_validation_data_filename, 'r') as mhf:
                standard_validation_data = mhf['data'][()]
            
            print('Opening {0}'.format(obs_validation_labels_filename))
            with h5py.File(obs_validation_labels_filename, 'r') as ohf:
                reshaped_validation_obs = ohf['data'][()]
            
        else:
            total_validation_dates = pd.date_range(start=self.start_dates['valid'],
                    end=self.end_dates['valid'],freq='1D').strftime('%Y%m%d').values
            random.shuffle(total_validation_dates)
            total_validation_obs = []
            total_validation_data = []
            count=0
            print()
            for v_date in total_validation_dates:
                if count == 5: break
                obs_file = glob(self.hf_path + '/obs/*obs*{0}*'.format(v_date))
                model_file = glob(self.hf_path + '/{0}/*{1}*'.format(member,v_date))
                if len(obs_file) < 1: continue 
                if len(model_file) < len(self.forecast_variables): continue
                #Open obs file
                with h5py.File(obs_file[0], 'r') as hf:
                    obs = hf['data'][()]
                if np.nanmax(obs) < 3: continue
                no_hail_hours,no_hail_patches = np.where(obs == 0)
                validation_data = self.read_files('forecast',member,v_date)
                if validation_data is None: continue
                print('Adding {0} to validation set'.format(v_date))
                total_validation_obs.extend(obs[no_hail_hours,no_hail_patches].ravel())
                total_validation_data.extend(validation_data[no_hail_hours,no_hail_patches,:,:,:])
                count +=1
            reshaped_validation_data = np.array(total_validation_data).reshape(-1,
                *np.shape(total_validation_data)[-3:])
            reshaped_validation_obs = np.array(total_validation_obs).ravel()
            print(np.count_nonzero(reshaped_validation_obs))
            print(len(reshaped_validation_obs))
            #Standardize validation data 
            standard_validation_data = self.standardize_data(member,reshaped_validation_data)
            #Output data
            with h5py.File(obs_validation_labels_filename, 'w') as hf:
                hf.create_dataset("data",data=reshaped_validation_obs) 
            print('Writing out {0}'.format(obs_validation_labels_filename))
            
            with h5py.File(model_validation_data_filename, 'w') as hf:
                hf.create_dataset("data",data=standard_validation_data)
            print('Writing out {0}'.format(model_validation_data_filename)) 
            print(np.shape(standard_validation_data),np.shape(reshaped_validation_obs)) 
        return standard_validation_data,reshaped_validation_obs


    def training_data_selection(self,member,training_filename):
        """
        Function to select random patches to train a CNN. Observation files
        are loaded and read to create a balanced dataset with 
        multiple class examples.

        Args:
            member (str): Ensemble member data that trains a CNN
            training_filename (str): Filename and path to save the random
                training patches.
        Returns:
            Pandas dataframe with random patch information
        """ 
        
        string_dates = pd.date_range(start=self.start_dates['train'],
                    end=self.end_dates['train'],freq='1D').strftime(self.run_date_format)
        
        #Place all obs data into respective category
        cols = ['Random Date','Random Hour','Random Patch','Obs Label','Data Augmentation'] 
        obs_categories_examples = pd.DataFrame(columns=cols)

        print('Selecting {0} random {1} training samples'.format(
        self.num_examples,member))
        try:
            #Loop through each category:
            for c,category in enumerate(self.class_percentage.keys()):
                #Loop through each date
                all_days_obs_data = []
                var_file = self.hf_path + '/{0}/*{1}*{2}*.h5'
                for str_date in string_dates:
                    #If there are model or obs files, continue to next date
                    
                    #imodel_file = [glob(var_file.format(member,variable,str_date))[0]
                    #    for variable in self.forecast_variables
                    #    if len(glob(var_file.format(member,variable,str_date))) == 1]
                
                    model_file = glob(self.hf_path + '/{0}/*{1}*'.format(member,str_date))
                    obs_file = glob(self.hf_path + '/obs/*obs*{0}*'.format(str_date))
                    if len(model_file) < len(self.forecast_variables): continue
                    if len(obs_file) < 1: continue
                    #Open obs file
                    with h5py.File(obs_file[0], 'r') as hf:
                        data = hf['data'][()]
                    if data.shape[0] < 1:continue 
                    for hour in np.arange(data.shape[0]):
                        inds = np.where(data[hour] == category)[0]
                        if len(inds) >1:
                            for i in inds:
                                all_days_obs_data.append((str_date,hour,i))
                df_all_days_obs_data = pd.DataFrame(all_days_obs_data,columns=cols[:3])
                #Find the number of desired examples per category
                subset_class_examples = int(self.num_examples*self.class_percentage[category])
                print(category, len(df_all_days_obs_data),subset_class_examples, len(df_all_days_obs_data)/subset_class_examples)
                if len(df_all_days_obs_data) < subset_class_examples:
                    n_aug = subset_class_examples-len(df_all_days_obs_data)
                    randomly_sampled_patches_augment = df_all_days_obs_data.sample(
                            n=n_aug,replace=True,random_state=42)
                    df_all_days_obs_data['Data Augmentation'] = 0
                    randomly_sampled_patches_augment['Data Augmentation'] = 1
                    randomly_sampled_patches = pd.concat([randomly_sampled_patches_augment, 
                        df_all_days_obs_data], ignore_index=True) 
                else:
                    randomly_sampled_patches = df_all_days_obs_data.sample(
                        n=subset_class_examples,replace=False,random_state=42) 
                    randomly_sampled_patches['Data Augmentation'] = 0    
                randomly_sampled_patches['Obs Label'] = category
                obs_categories_examples = obs_categories_examples.append(randomly_sampled_patches,ignore_index=True,sort=True)
            obs_categories_examples = obs_categories_examples.sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)  
            print(obs_categories_examples)
            print('Writing out {0}'.format(training_filename))
            obs_categories_examples.to_csv(training_filename)
            return obs_categories_examples
        except:
            print('No training data found')
            raise
        
    def read_files(self,mode,member,dates,hour=None,patch=None,data_augment=None): 
        """
        Function to read pre-processed model data for each individual predictor
        variable.

        Args:
            mode (str): Read training or forecasting files
            member (str): Ensemble member 
            dates (list of strings): 
            hour (str): Random training patch hour
            patch (str): Random training patch
            data_augment (str): Augmentation of random training patch, binary. 
        """
        total_patch_data = []
        
        #Start up parallelization
        pool = Pool(mp.cpu_count())
        print('Reading {0} unique {1} date file(s)'.format(len(np.unique(dates)), member))
        var_file = self.hf_path + '/{0}/{1}*{2}*.h5'
        for d,date in enumerate(np.unique(dates)):
            #Find all model variable files for a given date
            var_files = [glob(var_file.format(member,variable,date))[0]
                        for variable in self.forecast_variables
                        if len(glob(var_file.format(member,variable,date))) == 1]
            #If at least one variable file is missing, go to next date
            if len(var_files) < len(self.forecast_variables): continue
            #Extract training data 
            if mode == 'train':
                date_inds = np.where(dates == date)[0]    
                if d%10 == 0:print(d,date)
                args = (mode,var_files,hour[date_inds].values,
                    patch[date_inds].values,data_augment[date_inds].values)
                total_patch_data.append(pool.apply_async(self.extract_data,args))
            #Extract forecast data
            elif mode =='forecast': total_patch_data.append(pool.apply_async(self.extract_data,(mode,var_files)))
        pool.close()
        pool.join()

        #If there are no data, return None
        if len(total_patch_data) <1: return None
        if len(total_patch_data) >= 1:
            total_unique_patch_data  = [data for pool_file in total_patch_data for data in pool_file.get()]
            return np.array(total_unique_patch_data)
        else:
            return None
    
    def extract_data(self,mode,files,hours=None,patches=None,data_augment=None):
        """
        Function to extract training and forecast patch data

        Args:
            mode (str): Training or forecasting
            files (list of str): List of model files
            hour (str): Random training patch hour
            patch (str): Random training patch
            data_augment (str): Augmentation of random training patch, binary. 

        Returns: 
            Extracted data 
        """
        if mode =='train':
            #Training patch data (ny,nx,#variables) 
            patch_data = np.zeros( (len(hours), self.patch_radius,
                    self.patch_radius,len(self.forecast_variables)) ) 
        else: 
            #Forecast patch data (hours,#patches,ny,nx,#variables) 
            data_shape = h5py.File(files[0],'r')['data'].shape+(len(self.forecast_variables),)
            patch_data = np.empty( data_shape )*np.nan
        for v,variable_file in enumerate(files):
            with h5py.File(variable_file, 'r') as hf:
                compressed_patch_data = hf['data']
                if mode =='train':
                    for i in np.arange(len(hours)):
                        patch_data[i,:,:,v] = compressed_patch_data[hours[i],patches[i],:,:]
                else: patch_data[:,:,:,:,v] = compressed_patch_data[()]
        return patch_data
    
    def standardize_data(self,member,model_data,mode=None):
        """
        Function to standardize data and output the training 
        mean and standard deviation to apply to testing data.

        Args:
            member (str): Ensemble member 
            model_data (ndarray): Data to standardize
        Returns:
            Standardized data
        """
        if mode is not None:print("Standardizing data") 
        scaling_file = self.model_path+'/{0}_{1}_{2}_{3}_training_scaling_values.csv'.format(
            member,self.start_dates['train'].strftime('%Y%m%d'),
            self.end_dates['train'].strftime('%Y%m%d'),self.num_examples)

        if exists(scaling_file):
            if mode is not None: print('Opening {0}'.format(scaling_file))
            scaling_values = pd.read_csv(scaling_file,index_col=0)
        
        else:
            scaling_values = pd.DataFrame(np.zeros((len(self.forecast_variables), 2), 
                dtype=np.float32),columns=['max','min'])
                #dtype=np.float32),columns=['mean','std'])
        
        #Standardize data
        standard_model_data = np.empty( np.shape(model_data) )*np.nan
        for n in np.arange(model_data.shape[-1]):
            three_std = 3*np.nanstd(model_data[:,:,:,n]) + np.nanmean(model_data[:,:,:,n])
            data = np.where(abs(model_data[:,:,:,n]) >= three_std, np.nanmean(model_data[:,:,:,n]),model_data[:,:,:,n])
            print(np.shape(data))
            print(np.nanmax(data),np.nanmin(data))
            
            if not exists(scaling_file):
                #scaling_values.loc[n, ['mean','std']] = [np.nanmean(model_data[:,:,:,n]),np.nanstd(model_data[:,:,:,n])]
                scaling_values.loc[n, ['max','min']] = [np.nanmax(data),np.nanmin(data)]

            #standard_model_data[:,:,:,n] = (model_data[:,:,:,n]-scaling_values.loc[n,'mean'])/(scaling_values.loc[n,'std'])
            standard_model_data[:,:,:,n] = (data-scaling_values.loc[n,'min'])/(scaling_values.loc[n,'max'] - scaling_values.loc[n,'min'])
        
        if not exists(scaling_file):
            #Output training scaling values
            print('Writing out {0}'.format(scaling_file))
            scaling_values.to_csv(scaling_file)
        del model_data,scaling_values
        return standard_model_data

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
        print('Validation data shape {0}'.format(np.shape(valid_data)))
        print('Validation label data shape {0}\n'.format(np.shape(valid_labels)))
        
        model_file = self.model_path+'/{0}_{1}_{2}_{3}_CNN_model.h5'.format(
            member,self.start_dates['train'].strftime('%Y%m%d'),
            self.end_dates['train'].strftime('%Y%m%d'),self.num_examples)
        if not exists(model_file):
            #Initiliaze Convolutional Neural Net (CNN)
            model = models.Sequential()
            #First layer, input shape (y,x,# variables) 
            model.add(layers.Conv2D(32, (3,3), 
                input_shape=(np.shape(model_data[0])),
                padding='same'))
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
            model.add(layers.Dropout(0.10))
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
                class_weight=self.class_percentage)
            #Save trained model
            model.save(model_file)
            print('Writing out {0}'.format(model_file))
        else:
            print('Opening {0}'.format(model_file))
            model = models.load_model(model_file)

        del model_labels,model_data
        
        #Find the threshold with the highest AUC score 
        #on validation data
        threshold_model_file = self.model_path+'/{0}_{1}_{2}_{3}_CNN_model_threshold.h5'.format(
            member,self.start_dates['train'].strftime('%Y%m%d'),
            self.end_dates['train'].strftime('%Y%m%d'),self.num_examples)
        
        if exists(threshold_model_file): 
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
        print('Writing out {0}'.format(threshold_model_file))
        df_best_score.to_csv(threshold_model_file)
        
        return 
    
    def gridded_forecasts(self,forecasts,inds,mapping_data,subset_data):
        """
        Function that opens map files over the total data domain
        and the subset domain (CONUS). Projects the patch data on 
        the subset domain to the large domain.
        
        Args:
            forecasts (ndarray): Patch probability predictions from CNN
            map_file (str): Filename and path to total domain map file
        """
        
        #Fill in total grid with predicted probabilities
        gridded_predictions = np.zeros((forecasts.shape[0],forecasts.shape[1],) +\
            mapping_data['lat'].ravel().shape )
        #Patch size to fill probability predictions
        subset_data_shape = np.array(subset_data.shape)[-2:]
        print(subset_data.shape)
        #Grid to project subset data onto
        total_grid = np.zeros_like( mapping_data['lat'].ravel() )
        #Output subset data onto full grid using indices from cKDtree
        for hail_size in np.arange(np.shape(forecasts)[0]):
            hail_size_pred = forecasts[hail_size]
            for hour in np.arange(hail_size_pred.shape[0]):
                prob_on_patches = []
                for patch in np.arange(hail_size_pred.shape[1]):
                    if hail_size_pred[hour,patch] > 0:
                        print(patch)
                        return
                    #prob_on_patches.append(np.full(subset_data_shape, hail_size_pred[hour,patch]))
        '''
                total_grid[inds] = np.array(prob_on_patches).ravel()
                gridded_predictions[hail_size,hour] = total_grid
        final_grid_shape = (forecasts.shape[0],forecasts.shape[1],)+mapping_data['lat'].shape
        return gridded_predictions.reshape(final_grid_shape)
        '''
