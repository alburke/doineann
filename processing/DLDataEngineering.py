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

class DLDataEngineering(object):
    def __init__(self,model_path,hf_path,
        num_examples,class_category,predictors,
        model_args):
        
    
        self.model_path = model_path
        self.hf_path = hf_path
        self.num_examples = num_examples
        self.class_category = class_category
        self.model_args = model_args

        long_predictors = []
        #Shorten predictor names
        for predictor in predictors:
            if "_" in predictor: 
                predictor_name= predictor.split('_')[0].upper() + predictor.split('_')[-1]
            elif " " in predictor: 
                predictor_name= ''.join([v[0].upper() for v in predictor.split()])
            else:predictor_name = predictor
            long_predictors.append(predictor_name)
         
        self.predictors = np.array(long_predictors)
        return
    
    ################################################
    # Training data 
    ################################################

    def extract_training_data(self,member,train_dates,model_type):
        """
        Function that reads and extracts 2D sliced training data from 
        an ensemble member. 

        Args:
            member (str): Ensemble member data that trains a CNN
        Returns:
            Sliced ensemble member data and observations.
        """
        print()

        filename_args = self.model_path+f'/{member}_{self.model_args}_'
        
        #Random patches for training
        train_cases_file = filename_args+'training_cases.csv'
        
        #Observations associated with random patches    
        if model_type == 'CNN':
            train_obs_file = filename_args+'training_obs.h5'
        elif model_type == 'UNET':
            train_obs_file = self.model_path+\
            f'/{member}_{self.model_args}_training_obs.h5'
        
        #Standard ensemble member data associated with random patches
        train_predictor_file = filename_args+'standard_training_predictors.h5'
       
        #Opening training data files
        if exists(train_obs_file) and exists(train_predictor_file):
            #Opening training files
            with h5py.File(train_obs_file,'r') as ohf: obs_data=ohf['data'][()]
            print(f'Opening {train_obs_file}')
            
            with h5py.File(train_predictor_file, 'r') as mhf: standard_data=mhf['data'][()]
            print(f'Opening {train_predictor_file}')
            
        else:
            if exists(train_cases_file): 
                print(f'Opening {train_cases_file}')
                patches = pd.read_csv(train_cases_file,index_col=0)
            else:
                #Selecting random patches for training
                patches = self.training_data_selection(member,train_dates,
                train_cases_file,model_type)

            #Extracting obs labels
            if model_type == 'CNN': 
                obs_data = patches['Obs Label'].values.astype('int')
            elif model_type == 'UNET':
                obs_data = self.read_files('obs','Observation',patches['Random Date'], 
                    patches['Random Hour'],patches['Random Patch']) 

            #Extracting model patches
            member_data = self.read_files('train',member,patches['Random Date'], 
                patches['Random Hour'],patches['Random Patch']) 
            if member_data is None: 
                print('No training data found')
                return None,None
            
            #Standardize data
            standard_data = self.standardize_data(
                member,member_data,train_dates,'train') 

            #Output standard training data
            with h5py.File(train_predictor_file, 'w') as mhf: 
                mhf.create_dataset("data",data=standard_data)
            print(f'Writing out {train_predictor_file}')
            
            with h5py.File(train_obs_file, 'w') as ohf: 
                ohf.create_dataset("data",data=obs_data)
            print(f'Writing out {train_obs_file}')
        return standard_data, obs_data
    

    def training_data_selection(self,member,train_dates,train_cases_file,
        model_type):
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
        
        #Place all obs data into respective category
        cols = ['Random Date','Random Hour','Random Patch','Obs Label'] 
        print(f'Selecting {self.num_examples} random {member} training samples')
        
        daily_obs = []
        #Loop through each date
        for str_date in train_dates: 
            print(str_date)
            #If there are model or obs files, continue to next date
            obs_file = glob(self.hf_path+f'/obs/*obs*{str_date}*')
            # continue if daily obs file not found
            if len(obs_file) < 1: continue
            # continue if all daily feature files not found
            data_file = glob(self.hf_path+f'/{member}/*{str_date}*')
            if len(data_file) < 1: continue
            with h5py.File(data_file[0],'r') as dhf:
                data_keys = dhf.keys()
                if len(data_keys) < len(self.predictors): continue
            #print(str_date)
            #Open obs file
            with h5py.File(obs_file[0],'r') as ohf: obs_data = ohf['data'][()]
            #Get obs labels
            for hour in np.arange(obs_data.shape[0]):
                for patch in np.arange(obs_data.shape[1]):
                    if model_type == 'CNN': 
                        daily_obs.append((str_date,hour,patch,obs_data[hour,patch]))
                    else:
                        max_obs_value = np.nanmax(obs_data[hour,patch])
                        if max_obs_value >= 50.: 
                            daily_obs.append((str_date,hour,patch,3))
                        #elif max_obs_value >= 25.: 
                        #    daily_obs.append((str_date,hour,patch,2))
                        #elif max_obs_value >= 12.5: 
                        #    daily_obs.append((str_date,hour,patch,1))
                        #else: 
                        #    daily_obs.append((str_date,hour,patch,0))
            
            del obs_data
        if len(daily_obs) < 1:
            print('No training data found')
            return None
        daily_obs_df = pd.DataFrame(daily_obs,columns=cols)
        #Find the number of desired examples per category
        total_obs_df = pd.DataFrame(columns=cols)
        for c,category in enumerate(self.class_category.keys()):
            category_examples_num = int(self.class_category[category])
            category_data = daily_obs_df[daily_obs_df.loc[:,'Obs Label'] == category]
            print(f'Category: {category},' 
                f'Desired num examples: {category_examples_num},'
                f'Actual num examples: {len(category_data)},'
                f'Ratio: {len(category_data)/category_examples_num}')
            #If not enough obs per category, sample with replacement
            # Otherwise sample without replacement
            if len(category_data) < category_examples_num: category_obs = category_data
            else: category_obs = category_data.sample(n=category_examples_num,replace=False)
            total_obs_df = total_obs_df.append(category_obs,ignore_index=True,sort=True)
        total_obs_df = total_obs_df.sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)  
        print(total_obs_df)

        print('Writing out {0}'.format(train_cases_file))
        total_obs_df.to_csv(train_cases_file)
        return total_obs_df


    '''
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
        standard_valid_feature = self.standardize_data(
            member,valid_member_data,mode='valid') 
            
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
                f'/{member}/{var}*{v_date}*')[0] for var in self.predictors]
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
    '''    
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
        pool = Pool(30) #mp.cpu_count())

        print(f'Reading {len(np.unique(dates))} unique {member} date file(s)')
        for d,date in enumerate(np.unique(dates)):
            #Find all model variable files for a given date
            if mode == 'obs':
                try: extraction_files = glob(self.hf_path+f'/obs/*{date}*.h5')[0]
                except: continue
            else: 
                try: 
                    extraction_files = glob(self.hf_path+f'/{member}/*{date}*')[0]
                except: continue
            if d%20 == 0:print(d,date)
            if len(dates) == 1: 
                pool.close()
                pool.join()
                return self.extract_data(extraction_files,np.array(hour)[0],np.array(patch)[0])
            else: 
                date_inds = np.where(dates == date)[0]
                args = (extraction_files,np.array(hour)[date_inds],np.array(patch)[date_inds])
                patch_data.append(pool.apply_async(self.extract_data,args))
        pool.close()
        pool.join()
        #If there are no data, return None
        if len(patch_data) <1: return None
        all_patch_data = [pool_file.get() for pool_file in patch_data]
        del patch_data
        return np.concatenate(all_patch_data,axis=0)

    
    def extract_data(self,filename,hours=None,patches=None):
        """
        Function to extract training and forecast patch data

        Args:
            files (list of str): List of model files
            hour (str): Random training patch hour
            patch (str): Random training patch
            data_augment (str): Augmentation of random training patch, binary. 

        Returns: 
            Extracted data 
        """
        h5 = h5py.File(filename,'r')
        compressed_data = np.array([h5[variable] for variable in h5.keys()])
        h5.close()
        if hours is None and patches is None:
            reshaped_data = np.moveaxis(compressed_data,0,-1)
            return reshaped_data
        if np.ndim(compressed_data) < 5:  
            hourly_data = np.array([
                compressed_data[hours[example],patches[example],:,:] 
                for example in np.arange(len(hours))])
            reshaped_data = hourly_data
        else:  
            hourly_data = np.array([
            compressed_data[:,hours[example],patches[example],:,:] 
                for example in np.arange(len(hours))])
            reshaped_data = np.moveaxis(hourly_data, 1, -1)
        del hourly_data
        del compressed_data
        return reshaped_data

    def standardize_data(self,member,model_data,
        train_dates=None,mode=None):
        """
        Function to standardize data and output the training 
        mean and standard deviation to apply to testing data.

        Args:
            member (str): Ensemble member 
            model_data (ndarray): Data to standardize
        Returns:
            Standardized data
        """
        scaling_file = self.model_path+f'/{member}_{self.model_args}_training_scaling_values.csv'

        if exists(scaling_file):
            if mode is not None: print(f'Opening {scaling_file}')
            scaling_values = pd.read_csv(scaling_file,index_col=0)
        else:
            scaling_values = pd.DataFrame( np.zeros((len(self.predictors), 2), 
                dtype=np.float32),columns=['max','min']) 
            for n in np.arange(model_data.shape[-1]):
                scaling_values.loc[n,['max','min']] = [ 
                    np.nanmax(model_data[:,:,:,n]), np.nanmin(model_data[:,:,:,n])] 
            print(f'Writing out {scaling_file}')
            scaling_values.to_csv(scaling_file)
            
        min_max_values = (scaling_values['max'] - scaling_values['min']).values 
        standard_model_data = (model_data - scaling_values['min'].values)/min_max_values
        del model_data
        print('Data Standardized')
        print(np.nanmax(standard_model_data), np.nanmin(standard_model_data))
        return np.array(standard_model_data)
    
