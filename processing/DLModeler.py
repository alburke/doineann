from util.make_proj_grids import make_proj_grids, read_ncar_map_file
from scipy.spatial import cKDTree
from sklearn.utils import shuffle
from os.path import exists
from glob import glob
import pandas as pd
import numpy as np
import random
import h5py
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#Parallelization packages
from multiprocessing import Pool
import multiprocessing as mp
        
#Deep learning packages
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

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
        self.forecast_variables = forecast_variables
        return

    def pre_process_training_data(self,member):
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
            '/{0}_{1}_{2}_{3}_training_patches.h5'.format(*filename_args)
        
        #Opening training examples file
        if exists(obs_training_labels_filename) and exists(model_training_data_filename):
            #Opening obs file
            print('Opening {0}'.format(obs_training_labels_filename))
            with h5py.File(obs_training_labels_filename, 'r') as ohf:
                member_obs_label = ohf['data'][()]
            
            #Opening selected patch data file
            print('Opening {0}'.format(model_training_data_filename))
            with h5py.File(model_training_data_filename, 'r') as mhf:
                member_model_data = mhf['data'][()]
            print(np.shape(member_obs_label),np.shape(member_model_data))
        else:
            if exists(training_filename): 
                #Opening training examples data file
                print('Opening {0}'.format(training_filename))
                patches = pd.read_csv(training_filename,index_col=0)
            else:
                #Selecting random patches for training
                patches = self.training_data_selection(member,training_filename)
            
            onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
            member_obs_label = onehot_encoder.fit_transform(patches['Obs Label'].values.reshape(-1, 1))
            
            with h5py.File(obs_training_labels_filename, 'w') as hf:
                hf.create_dataset("data",data=member_obs_label)
                print('Writing out {0}'.format(obs_training_labels_filename))
                
            #Extracting model patch data
            member_model_data = self.read_files(mode,member,patches['Random Date'], 
                patches['Random Hour'],patches['Random Patch'],patches['Data Augmentation'])
            
            if member_model_data is None: print('No training data found')
            else:
                with h5py.File(model_training_data_filename, 'w') as hf:
                    hf.create_dataset("data",data=member_model_data)
                print('Writing out {0}'.format(model_training_data_filename)) 
        return member_model_data,member_obs_label
    


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
                for str_date in string_dates:
                    #If there are model or obs files, continue to next date
                    model_file = glob(self.hf_path + '/{0}/*{1}*'.format(member,str_date))
                    obs_file = glob(self.hf_path + '/*obs*{0}*'.format(str_date))
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
                print(subset_class_examples)
                if len(df_all_days_obs_data) < subset_class_examples:
                    data_augment = 1
                    randomly_sampled_patches = df_all_days_obs_data.sample(
                    n=subset_class_examples,replace=True) 
                else:
                    data_augment = 0
                    randomly_sampled_patches = df_all_days_obs_data.sample(
                    n=subset_class_examples,replace=False) 
                randomly_sampled_patches['Obs Label'] = category
                randomly_sampled_patches['Data Augmentation'] = data_augment    
                obs_categories_examples = obs_categories_examples.append(randomly_sampled_patches,ignore_index=True)
            obs_categories_examples = obs_categories_examples.sort_values(by='Random Date')
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
        print('\nReading {0} unique member file(s)\n'.format(len(np.unique(dates))))
        var_file = self.hf_path + '/{0}/*{1}*{2}*.h5'
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
            elif mode =='forecast': total_patch_data.append(self.extract_data(mode,var_files)) 
        pool.close()
        pool.join()

        #If there are no data, return None
        if len(total_patch_data) <1: return None
        elif mode=='train':
            total_unique_patch_data  = [data for pool_file in total_patch_data for data in pool_file.get()]
            return np.array(total_unique_patch_data)
        else: 
            if len(total_patch_data) >= 1:
                return np.array(total_patch_data)[0]
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
                        if data_augment[i] > 0.5:
                            #Augment data by adding small amounts of variance
                            var_data = compressed_patch_data[hours[i],patches[i],:,:].ravel()
                            noise = np.nanvar(var_data)*np.random.choice(np.arange(-0.1,0.11,0.01))
                            patch_data[i,:,:,v] = (var_data + noise).reshape((self.patch_radius,self.patch_radius))
                        else: patch_data[i,:,:,v] = compressed_patch_data[hours[i],patches[i],:,:]
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
        if mode is None:print("Standardizing data") 
        scaling_file = self.model_path+'/{0}_{1}_{2}_{3}_training_scaling_values.csv'.format(
            member,self.start_dates['train'].strftime('%Y%m%d'),
            self.end_dates['train'].strftime('%Y%m%d'),self.num_examples)

        standard_model_data = np.ones(np.shape(model_data))*np.nan
        #If scaling values are already saved, input data
        if not exists(scaling_file):
            scaling_values = pd.DataFrame(np.zeros((len(self.forecast_variables), 2), 
                dtype=np.float32),columns=['mean','std'])
        else: 
            if mode is None: print('Opening {0}'.format(scaling_file))
            scaling_values = pd.read_csv(scaling_file,index_col=0)

        #Standardize data
        for n in np.arange(model_data.shape[-1]):
            if exists(scaling_file):
                standard_model_data[:,:,:,n] = (model_data[:,:,:,n]-scaling_values.loc[n,'mean'])/scaling_values.loc[n,'std']
                continue
            #Save mean and standard deviation values
            scaling_values.loc[n, ["mean", "std"]] = [np.nanmean(model_data[:,:,:,n]),np.nanstd(model_data[:,:,:,n])]
            standard_model_data[:,:,:,n] = (model_data[:,:,:,n]-scaling_values.loc[n,'mean'])/scaling_values.loc[n,'std']
        
        #Output training scaling values
        if not exists(scaling_file):
            print('Writing out {0}'.format(scaling_file))
            scaling_values.to_csv(scaling_file)
        del model_data,scaling_values
        return standard_model_data

    def train_CNN(self,member,model_data,model_labels):
        """
        Function to train a convolutional neural net (CNN) for random 
        training data and associated labels.

        Args:
            member (str): Ensemble member 
            model_data (ndarray): ensemble member data
            model_labels (ndarray): observation labels

        """
        print('\nTraining {0} models'.format(member))
        print('Model data shape {0}'.format(np.shape(model_data)))
        print('Label data shape {0}\n'.format(np.shape(model_labels)))
        
        #Initiliaze Convolutional Neural Net (CNN)
        model = models.Sequential()
        #First layer, input shape (y,x,# variables) 
        model.add(layers.Conv2D(32, (5,5), activation='relu', 
                input_shape=(np.shape(model_data[0]))
                ))
        model.add(layers.AveragePooling2D())
        model.add(layers.Dropout(0.3))
        #Second layer
        model.add(layers.Conv2D(64, (3,3),activation='relu'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Dropout(0.3))
        #Third layer
        model.add(layers.Conv2D(128, (3,3),activation='relu'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Dropout(0.3))
        #Flatten the last convolutional layer into a long feature vector
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))
        #Compile neural net
        model.compile(optimizer='adam',loss='categorical_crossentropy',
                metrics=[tf.keras.metrics.AUC()])
        print(model.summary())
        #Fit neural net
        X_train, X_test, y_train, y_test = train_test_split(model_data,model_labels,
            test_size=0.2,shuffle=True)
        
        es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20)
        n_epochs = 80
        conv_hist = model.fit(X_train,y_train,
                epochs=n_epochs,batch_size=1000,verbose=1,
                validation_data=(X_test,y_test),callbacks=[es])  
        #Save trained model
        model_file = self.model_path+'/{0}_{1}_{2}_{3}_CNN_model.h5'.format(
            member,self.start_dates['train'].strftime('%Y%m%d'),
            self.end_dates['train'].strftime('%Y%m%d'),self.num_examples)
        model.save(model_file)
        print('Writing out {0}'.format(model_file))
        del model_labels,model_data
        return 
    
    def gridded_forecasts(self,forecasts,map_file):
        """
        Function that opens map files over the total data domain
        and the subset domain (CONUS). Projects the patch data on 
        the subset domain to the large domain.
        
        Args:
            forecasts (ndarray): Patch probability predictions from CNN
            map_file (str): Filename and path to total domain map file
        """
        
        #Open mapfile over total domain
        proj_dict, grid_dict = read_ncar_map_file(map_file)
        mapping_data = make_proj_grids(proj_dict, grid_dict)
        #center_grid_point = int(np.ceil(self.patch_radius/2.0))
        
        #Create cKDtree to convert patch grid points to the total domain
        tree = cKDTree(np.c_[mapping_data['lon'].ravel(),mapping_data['lat'].ravel()])

        #Open mapfile over subset domain (CONUS)
        subset_map_file = glob(self.hf_path+'/*map*.h5')[0] 
        subset_data = h5py.File(subset_map_file, 'r')['data']
        
        #Convert subset grid points to total grid using cKDtree
        _,inds = tree.query(np.c_[subset_data[0].ravel(),
                        subset_data[1].ravel()])
        #Fill in total grid with predicted probabilities
        gridded_predictions = np.zeros( (forecasts.shape[0],forecasts.shape[1],) +\
            mapping_data['lat'].ravel().shape )
        #Patch size to fill probability predictions
        subset_data_shape = np.array(subset_data.shape)[-2:]
        #Grid to project subset data onto
        total_grid = np.zeros_like( mapping_data['lat'].ravel() )
        for size_pred in np.arange(forecasts.shape[0]):
            for hour in np.arange(forecasts.shape[1]):
                prob_on_patches = []
                for patch in np.arange(forecasts.shape[2]):
                    patch_data = forecasts[size_pred,hour,patch]
                    if size_pred == 0:
                        if patch_data > 0.45: patch_data=1.0
                        else:patch_data=0.0
                    if size_pred == 1:
                        if patch_data > 0.30: patch_data=1.0
                        else:patch_data=0.0
                    prob_on_patches.append(np.full(subset_data_shape, patch_data))
                #Output subset data onto full grid using indices from cKDtree
                total_grid[inds] = np.array(prob_on_patches).ravel()
                gridded_predictions[size_pred,hour,:] = total_grid
        final_grid_shape = (forecasts.shape[0],forecasts.shape[1],)+mapping_data['lat'].shape
        return gridded_predictions.reshape(final_grid_shape)
        
