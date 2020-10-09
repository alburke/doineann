from dataclasses import dataclass
from os.path import exists
from glob import glob
import pandas as pd
import numpy as np
import h5py

#Parallelization packages
from multiprocessing import Pool
import multiprocessing as mp


import matplotlib.pyplot as plt

@dataclass
class DLDataEngineering(object):
    #def __init__(self,model_path,hf_path,start_dates,end_dates,
    #    num_examples,class_percentage,patch_radius,run_date_format,
    #    forecast_variables):
        
        model_path
        hf_path
        start_dates
        end_dates
        num_examples
        class_percentage
        patch_radius
        run_date_format
        
        long_forecast_variables = []
        #Shorten variable names
        for variable in forecast_variables:
            if "_" in variable: 
                variable_name= ''.join([v[0].upper() for v in variable.split()]) + variable.split('_')[-1]
            elif " " in variable: 
                variable_name= ''.join([v[0].upper() for v in variable.split()])
            else:variable_name = variable
            long_forecast_variables.append(variable_name)
        
        forecast_variables = np.array(long_forecast_variables)

        return

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

        #Find/create files containing information about training
        filename_args = self.model_path+\
            '/{0}_{1}_{2}_{3}_'.format(member,self.start_dates[mode].strftime('%Y%m%d'),
            self.end_dates[mode].strftime('%Y%m%d'),self.num_examples)

        #Random patches for training
        train_cases_file=filename_args+'training_cases.csv'

        #Observations associated with random patches    
        train_labels_file=filename_args+'training_labels.csv'
        
        #Ensemble member data associated with random patches
        train_features_file = filename_args+'standard_training_features.csv'
       
        #Opening training data files
        if exists(train_labels_file) and exists(train_features_file):
            #Opening training files
            print(f'Opening {train_labels_file}')
            with h5py.File(train_labels_file, 'r') as ohf: member_label = ohf['data'][()]
            
            print(f'Opening {train_features_file}')
            with h5py.File(train_features_file, 'r') as mhf: standard_member_feature = mhf['data'][()]
            '''
            print(np.shape(member_label),np.shape(standard_member_feature))
            c = standard_member_feature.reshape(-1, standard_member_feature.shape[-1])
            print(np.nanmax(c,axis=0), np.nanmin(c,axis=0))

            plt.figure(figsize=(30,30))
            plt.boxplot(c,labels=self.forecast_variables)
            plt.show()
            '''
        else:
            if exists(train_cases_file): 
                #Opening training examples data file
                print(f'Opening {train_cases_file}')
                patches = pd.read_csv(train_cases_file,index_col=0)
            else:
                #Selecting random patches for training
                patches = self.training_data_selection(member,train_cases_file)
            
            #Extracting obs labels
            member_label = patches['Obs Label'].values.astype('int64')
            with h5py.File(train_labels_file, 'w') as hf: hf.create_dataset("data",data=member_label)
            print(f'Writing out {train_labels_file}')
            
            #Extracting model patch data
            member_data = self.read_files(mode,member,patches['Random Date'], 
                patches['Random Hour'],patches['Random Patch']) #,patches['Data Augmentation'])
            
            if member_data is None: 
                print('No training data found')
                return None,None
            
            #Standardize data
            standard_member_feature = self.standardize_data(member,member_data,mode='train') 
            with h5py.File(train_features_file, 'w') as hf: 
                hf.create_dataset("data",data=standard_member_feature)
            print(f'Writing out {train_features_file}')
           
        return standard_member_feature, member_label

    def extract_validation_data(self,member):
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
        cols = ['Random Date','Random Hour','Random Patch','Obs Label'] #,'Data Augmentation'] 
        obs_categories = pd.DataFrame(columns=cols)

        print(f'Selecting {self.num_examples} random {member} training samples'.format(
        
        #try:
        #Loop through each size category:
        for c,category in enumerate(self.class_percentage.keys()):
            #Loop through each date
            daily_obs = []
            #var_file = self.hf_path + '/{0}/*{1}*{2}*.h5'
            for str_date in string_dates:
                #If there are model or obs files, continue to next date
                    
                #imodel_file = [glob(var_file.format(member,variable,str_date))[0]
                #    for variable in self.forecast_variables
                #    if len(glob(var_file.format(member,variable,str_date))) == 1]
                
                obs_file = glob(self.hf_path+f'/obs/*obs*{str_date}*')
                # continue if daily obs file not found
                if len(obs_file) < 1: continue
                    
                # continue if all daily feature files not found
                try:
                    member_feature_files = [glob(self.hf_path+\
                    f'/{member}/{var}*{str_data}*')[0] for var in self.forecast_variables]
                except: continue
                    
                #if len(member_feature_files) < len(self.forecast_variables): continue
                #Open obs file
                with h5py.File(obs_file[0],'r') as ohf: obs_data = ohf['data'][()]
                #if data.shape[0] < 1:continue 
                for hour in np.arange(data.shape[0]):
                    #Find where categorical hail occurs every hour
                    hourly_obs_inds = np.where(data[hour] == category)[0]
                    if len(hourly_obs_inds) < 1: continue
                    for i in inds: daily_obs.append((str_date,hour,i))
                
            daily_obs_df = pd.DataFrame(daily_obs,columns=cols) #[:3])
            
            #Find the number of desired examples per category
            cat_examples = int(self.num_examples*self.class_percentage[category])
                
            print(category, len(daily_obs_df), cat_examples, len(daily_obs_df)/cat_examples)
            if len(daily_obs_df) < cat_examples:
                n_aug = cat_examples-len(daily_obs_df)
                random_cat_examples = daily_obs_df.sample(n=n_aug,replace=True)
                #,random_state=42)
                #daily_obs_df['Data Augmentation'] = 0
                #random_cat_examples['Data Augmentation'] = 1
                random_obs_df = pd.concat([random_cat_examples,daily_obs_df],ignore_index=True) 
            else: random_obs_df = daily_obs_df.sample(n=cat_examples,replace=False)
                #,random_state=42) 
            #    random_obs_df['Data Augmentation'] = 0    
            random_obs_df['Obs Label'] = category
            obs_categories = obs_categories.append(random_obs_df,ignore_index=True,sort=True)
        
        if len(obs_categories.index) < 1:
            print('No training data found')
            return None

        obs_categories = obs_categories.sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)  
        print(obs_categories)
        print('Writing out {0}'.format(train_cases_file))
        obs_categories.to_csv(train_cases_file)
        return obs_categories
        
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
        print(f'Reading {len(np.unique(dates)} unique {member} date file(s)')
        #var_file = self.hf_path + '/{0}/{1}*{2}*.h5'
        for d,date in enumerate(np.unique(dates)):
            #Find all model variable files for a given date
            #If at least one variable file is missing, go to next date
            try:
                member_feature_files = [glob(self.hf_path+\
                f'/{member}/{var}*{str_data}*')[0] for var in self.forecast_variables]
            except: continue
            
            #var_files = [glob(var_file.format(member,variable,date))[0]
            #            for variable in self.forecast_variables
            #            if len(glob(var_file.format(member,variable,date))) == 1]
            #if len(var_files) < len(self.forecast_variables): continue
            
            #Extract training data 
            if mode == 'train':
                date_inds = np.where(dates == date)[0]    
                if d%10 == 0:print(d,date)
                args = (mode,member_feature_files,hour[date_inds].values,
                    patch[date_inds].values,data_augment[date_inds].values)
                total_patch_data.append(pool.apply_async(self.extract_data,args))
            #Extract forecast data
            elif mode =='forecast': total_patch_data.append(
                pool.apply_async(self.extract_data,(mode,member_feature_files)))
        pool.close()
        pool.join()

        #If there are no data, return None
        if len(total_patch_data) <1: return None
        total_unique_patch_data  = [data for pool_file in 
            total_patch_data for data in pool_file.get()]
        return np.array(total_unique_patch_data)
    
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
            if mode is not None: print(f'Opening {scaling_file}')
            scaling_values = pd.read_csv(scaling_file,index_col=0)
        
        else:
            scaling_values = pd.DataFrame(np.zeros((len(self.forecast_variables), 2), 
                dtype=np.float32),columns=['max','min'])
        
        #Standardize data
        standard_model_data = np.empty( np.shape(model_data) )*np.nan
        for n in np.arange(model_data.shape[-1]):
            three_std = 3*np.nanstd(model_data[:,:,:,n]) + np.nanmean(model_data[:,:,:,n])
            data = np.where(abs(model_data[:,:,:,n]) >= three_std, np.nanmean(model_data[:,:,:,n]),model_data[:,:,:,n])
            print(np.shape(data))
            print(np.nanmax(data),np.nanmin(data))
            
            if not exists(scaling_file): scaling_values.loc[n,['max','min']] = [np.nanmax(data),np.nanmin(data)]
            standard_model_data[:,:,:,n] = (data-scaling_values.loc[n,'min'])/(
                scaling_values.loc[n,'max'] - scaling_values.loc[n,'min'])
        
        if not exists(scaling_file):
            #Output training scaling values
            print(f'Writing out {scaling_file}')
            scaling_values.to_csv(scaling_file)
        del model_data,scaling_values
        return standard_model_data
