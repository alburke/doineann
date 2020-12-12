import matplotlib.pyplot as plt
from os.path import exists
from glob import glob
import pandas as pd
import numpy as np
import h5py
import random 

#Parallelization packages
from multiprocessing import Pool
import multiprocessing as mp

#from imblearn.combine import SMOTEENN, SMOTETomek

class DLDataEngineering(object):
    def __init__(self,model_path,hf_path,start_dates,end_dates,
        num_examples,class_category,patch_radius,run_date_format,
        forecast_variables):
        
    
        self.model_path = model_path
        self.hf_path = hf_path
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.num_examples = num_examples
        self.class_category = class_category
        self.patch_radius = patch_radius
        self.run_date_format = run_date_format
        self.forecast_variables = forecast_variables 
   
        long_forecast_variables = []
        #Shorten variable names
        for variable in forecast_variables:
            if "_" in variable: 
                variable_name= variable.split('_')[0].upper() + variable.split('_')[-1]
            elif " " in variable: 
                variable_name= ''.join([v[0].upper() for v in variable.split()])
            else:variable_name = variable
            long_forecast_variables.append(variable_name)
         
        self.forecast_variables = np.array(long_forecast_variables)

        return
    
    ################################################
    # Training data 
    ################################################

    def extract_training_data(self,member):
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

        #Training arguments
        filename_args = self.model_path+\
            '/{0}_{1}_{2}_{3}_'.format(member,self.start_dates[mode].strftime('%Y%m%d'),
            self.end_dates[mode].strftime('%Y%m%d'),self.num_examples)

        #Random patches for training
        train_cases_file=filename_args+'training_cases.csv'
        #Observations associated with random patches    
        train_labels_file=filename_args+'training_labels.h5'
        #Standard ensemble member data associated with random patches
        train_features_file = filename_args+'standard_training_features.h5'
       
        #Opening training data files
        if exists(train_labels_file) and exists(train_features_file):
            #Opening training files
            with h5py.File(train_labels_file, 'r') as ohf: 
                member_label = ohf['data'][()]
            print(f'Opening {train_labels_file}')
            
            with h5py.File(train_features_file, 'r') as mhf: 
                standard_member_feature = mhf['data'][()]
            print(f'Opening {train_features_file}')
            
            print(np.shape(member_label))
            print(np.shape(standard_member_feature))
            return 
            
        else:
            if exists(train_cases_file): 
                print(f'Opening {train_cases_file}')
                patches = pd.read_csv(train_cases_file,index_col=0)
            else:
                #Selecting random patches for training
                patches = self.training_data_selection(member,train_cases_file)
             
            #Extracting obs labels
            member_label = patches['Obs Label'].values.astype('int64')
            
            #Extracting model patches
            member_data = self.read_files(mode,member,patches['Random Date'], 
                patches['Random Hour'],patches['Random Patch']) #,patches['Data Augmentation'])
            if member_data is None: 
                print('No training data found')
                return None,None
            
            print(np.shape(member_data))
            return 
            #Resample data for balanced classes
            #resampling_model = SMOTEENN(sampling_strategy=self.class_category,random_state=42)
            #member_smote_data = member_data.reshape(member_data.shape[0],*member_data.shape[1:].ravel())
            #print(member_smote_data.shape)

            #member_resampled_data, member_resampled_label
            
            
            #Standardize data
            standard_member_feature = self.standardize_data(member,member_data,mode='train') 

            '''
            #Output standard training data
            with h5py.File(train_features_file, 'w') as mhf: 
                mhf.create_dataset("data",data=standard_member_feature)
            print(f'Writing out {train_features_file}')
            
            with h5py.File(train_labels_file, 'w') as ohf: 
                ohf.create_dataset("data",data=member_label)
            print(f'Writing out {train_labels_file}')
            '''
        return standard_member_feature, member_label
    

    def training_data_selection(self,member,train_cases_file):
        """
        Function to select random patches to train a CNN. Observation files
        are loaded and read to create a balanced dataset with 
        multiple class examples.

        Args:
            member (str): Ensemble member data that trains a CNN
            train_cases_file (str): Filename and path to save the random
                training patches.
        Returns:
            Pandas dataframe with random patch information
        """ 
        
        string_dates = pd.date_range(start=self.start_dates['train'],
            end=self.end_dates['train'],freq='1D').strftime(self.run_date_format)
        
        #Place all obs data into respective category
        cols = ['Random Date','Random Hour','Random Patch','Obs Label'] 
        print(f'Selecting {self.num_examples} random {member} training samples')
        
        daily_obs = []
        #Loop through each date
        for str_date in string_dates: 
            print(str_date)
            #Loop through each size category:
            #If there are model or obs files, continue to next date
            obs_file = glob(self.hf_path+f'/obs/*obs*{str_date}*')
            # continue if daily obs file not found
            if len(obs_file) < 1: continue
            # continue if all daily feature files not found
            try:
                member_feature_files = [glob(self.hf_path+f'/{member}/{var}*{str_date}*')[0]
                for var in self.forecast_variables]
            except: continue
            #Open obs file
            with h5py.File(obs_file[0],'r') as ohf: obs_data = ohf['data'][()]
            for hour in np.arange(obs_data.shape[0]):
                for patch in np.arange(obs_data.shape[1]):
                    max_obs_value = np.nanmax(obs_data[hour,patch])
                    if max_obs_value > 50.: 
                        daily_obs.append((str_date,hour,patch,3))
                    elif max_obs_value > 25.: 
                        daily_obs.append((str_date,hour,patch,2))
                    elif max_obs_value > 5.: 
                        daily_obs.append((str_date,hour,patch,1))
                    else: 
                        daily_obs.append((str_date,hour,patch,0))
            del obs_data
        if len(daily_obs) < 1:
            print('No training data found')
            return None
        daily_obs_df = pd.DataFrame(daily_obs,columns=cols)
        #Find the number of desired examples per category
        total_obs_df = pd.DataFrame(columns=cols)
        for c,category in enumerate(self.class_category.keys()):
            category_examples_num = int(self.class_category[category])
            print(category, len(daily_obs_df), category_examples_num, 
                len(daily_obs_df)/category_examples_num)
            category_data = daily_obs_df[daily_obs_df.loc[:,'Obs Label'] == category]
            #If not enough obs per category, sample with replacement
            # Otherwise sample without replacement
            if len(category_data) < category_examples_num: category_obs = category_data
            else: 
                category_obs = category_data.sample(n=category_examples_num,replace=False)
            total_obs_df = total_obs_df.append(category_obs,ignore_index=True,sort=True)
        total_obs_df = total_obs_df.sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)  
        print(total_obs_df)

        print('Writing out {0}'.format(train_cases_file))
        total_obs_df.to_csv(train_cases_file)
        return total_obs_df

    ################################################
    # Validation data 
    ################################################
    
    def extract_validation_data(self,member):
        """
        Function that reads and extracts 2D sliced training data from 
        an ensemble member. 

        Args:
            member (str): Ensemble member data that trains a CNN
        Returns:
            Sliced ensemble member data and observations.
        """
        print()
        mode = 'valid'

        filename_args = self.model_path+'/{0}_{1}_{2}_'.format(
            member,self.start_dates['valid'].strftime('%Y%m%d'),
            self.end_dates['valid'].strftime('%Y%m%d'))
        
        #Random days for validation
        valid_cases_file=filename_args+'validation_cases.csv'
        #Validation Observations 
        valid_labels_file=filename_args+'validation_labels.h5'
        #Ensemble member validation patches
        valid_features_file=filename_args+'standard_validation_features.h5'
        
        if exists(valid_labels_file) and exists(valid_features_file): 
            #Opening validation files
            with h5py.File(valid_labels_file, 'r') as ohf:
                valid_member_label = ohf['data'][()]
            print(f'Opening {valid_labels_file}')
            
            with h5py.File(valid_features_file, 'r') as mhf:
                standard_valid_feature = mhf['data'][()]
            print(f'Opening {valid_features_file}')
            return standard_valid_feature,valid_member_label
        
        if exists(valid_cases_file): 
            patches = pd.read_csv(valid_cases_file,index_col=0)
            print(f'Opening {valid_cases_file}')
        else:
            #Selecting random patches for training
            patches = self.validation_data_selection(member,valid_cases_file)
            
        #Extracting obs labels
        valid_member_label = patches['Obs Label'].values.astype('int64')
        with h5py.File(valid_labels_file, 'w') as hf: 
            hf.create_dataset("data",data=valid_member_label)
        print(f'Writing out {valid_labels_file}')
            
        #Extracting model patches
        valid_member_data=self.read_files(mode,member,patches['Random Date'],patches['Hour']) 
        if valid_member_data is None: 
            print('No validation data found')
            return None,None
            
        #Standardize data
        standard_valid_feature = self.standardize_data(member,valid_member_data,mode='valid') 
            
        #Output standard training data
        with h5py.File(valid_features_file, 'w') as hf: 
            hf.create_dataset("data",data=standard_valid_feature)
        print(f'Writing out {valid_features_file}')
        print(f'\nValidation features: {np.shape(standard_valid_feature)}'+\
            f', labels: {np.shape(valid_member_label)}') 
        
        return standard_valid_feature,valid_member_label
    
    def validation_data_selection(self,member,valid_cases_file):
        
        valid_string_dates = pd.date_range(start=self.start_dates['valid'],
            end=self.end_dates['valid'],freq='1D').strftime('%Y%m%d').values
        random.shuffle(valid_string_dates)
        
        cols = ['Random Date','Hour','Obs Label'] 
        obs_categories = pd.DataFrame(columns=cols)
        
        valid_obs_data = []
        print()

        count=0
        for v_date in valid_string_dates:
            #only save 10 random days
            if count == len(valid_string_dates)//3.: break
            # continue if daily obs file not found
            obs_file = glob(self.hf_path+f'/obs/*obs*{v_date}*')
            if len(obs_file) < 1: continue 
            # continue if all daily feature files not found
            try:
                member_feature_files = [glob(self.hf_path+\
                f'/{member}/{var}*{v_date}*')[0] for var in self.forecast_variables]
            except: continue
            #Open obs file
            with h5py.File(obs_file[0], 'r') as hf: obs_data = hf['data'][()]
            #Only keeps days/hours with severe hail
            random_hours = np.random.choice( 
                np.arange(obs_data.shape[0]),size=2,replace=False)
            #if np.nanmax(obs_data) < 2: continue
            #for hour in np.arange(obs_data.shape[0]):
            for hour in random_hours:
                for label in obs_data[hour,:]: valid_obs_data.append((v_date,hour,label))
            count += 1
            
        obs_categories['Random Date'] = np.array(valid_obs_data)[:,0]
        obs_categories['Hour']  = np.array(valid_obs_data)[:,1]
        obs_categories['Obs Label'] = np.array(valid_obs_data)[:,2]
            
        if len(obs_categories.index.values) < 1:
            print('No training data found')
            return None
        print(obs_categories)
        print(f'Writing out {valid_cases_file}')
        obs_categories.to_csv(valid_cases_file)
        return obs_categories
        
    ################################################
    # Functions to read and standardize data 
    ################################################
        
    def read_files(self,mode,member,dates,hour=None,patch=None): 
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
        patch_data = []
        #Start up parallelization
        pool = Pool(mp.cpu_count())

        print(f'Reading {len(np.unique(dates))} unique {member} date file(s)')
        for d,date in enumerate(np.unique(dates)):
            #Find all model variable files for a given date
            #If at least one variable file is missing, go to next date
            try:
                member_feature_files = [glob(self.hf_path+f'/{member}/{var}*{date}*')[0]
                    for var in self.forecast_variables]
            except: continue
            #Extract forecast data
            if mode =='forecast': patch_data.append(
                pool.apply_async(self.extract_data,(mode,member_feature_files)))
            #Extract training data 
            date_inds = np.where(dates == date)[0]    
            if mode == 'train':
                if d%20 == 0:print(d,date)
                args = (mode,member_feature_files,hour[date_inds].values,
                    patch[date_inds].values)
                patch_data.append(pool.apply_async(self.extract_data,args))
            #Extract validation data 
            elif mode == 'valid': 
                if d%5 == 0:print(d,date)
                args = (mode,member_feature_files,np.unique(hour[date_inds].values),None)
                patch_data.append(pool.apply_async(self.extract_data,args))
        
        pool.close()
        pool.join()
        #If there are no data, return None
        if len(patch_data) <1: return None
        all_patch_data = np.concatenate([pool_file.get() for pool_file in patch_data],axis=0)
        del patch_data
        return all_patch_data
    
    def extract_data(self,mode,files,hours=None,patches=None):
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
        patch_data = []
        for v,variable_file in enumerate(files):
            h5 = h5py.File(variable_file,'r')
            compressed_data = h5['data']
            if hours is None and patches is None: patch_data.append(compressed_data[()])
            elif patches is None:
                hourly_data=np.array([compressed_data[int(hours[i])] 
                    for i in np.arange(len(hours))])
            else:
                hourly_data=np.array([compressed_data[hours[i],
                    patches[i],:,:] for i in np.arange(len(hours))])
            patch_data.append(np.array(hourly_data).reshape(-1, 
                *np.shape(hourly_data)[-2:]))
            del compressed_data
            h5.close()
        # Move variables to last dimension 
        reshaped_patch_data = np.empty( (np.shape(patch_data)[1:]+(np.shape(patch_data)[0],)) )*np.nan
        for v in np.arange(np.shape(patch_data)[0]): reshaped_patch_data[:,:,:,v] = patch_data[v]
        del patch_data
        return reshaped_patch_data

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
        scaling_file = self.model_path+\
            f'/{member}_{self.start_dates["train"].strftime("%Y%m%d")}'+\
            f'_{self.end_dates["train"].strftime("%Y%m%d")}'+\
            f'_{self.num_examples}_training_scaling_values.csv'

        if exists(scaling_file):
            if mode is not None: print(f'Opening {scaling_file}')
            scaling_values = pd.read_csv(scaling_file,index_col=0)
        else:
            scaling_values = pd.DataFrame( np.zeros((len(self.forecast_variables), 2), 
                dtype=np.float32),columns=['max','min']) 
            for n in np.arange(model_data.shape[-1]):
                scaling_values.loc[n,['max','min']] = [ 
                    np.nanmax(model_data[:,:,:,n]), np.nanmin(model_data[:,:,:,n])] 
            print(f'Writing out {scaling_file}')
            scaling_values.to_csv(scaling_file)
            
        min_max_values = (scaling_values['max'] - scaling_values['min']).values
        standard_model_data = (model_data - scaling_values['min'].values)/min_max_values
        for n in np.arange(model_data.shape[-1]):
            print(self.forecast_variables[n],'  Max',
                np.nanmax(standard_model_data[:,:,:,n]), 
                '   Min',np.nanmin(standard_model_data[:,:,:,n]))
        del scaling_values, model_data
        return standard_model_data
    
