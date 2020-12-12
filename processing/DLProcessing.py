from util.make_proj_grids import make_proj_grids, read_ncar_map_file
from util.GridOutput import *
from datetime import timedelta
import pandas as pd
import numpy as np
import traceback
import h5py
import os

class DLPreprocessing(object):
    def __init__(self,train,ensemble_name,model_path,
        hf_path,patch_radius,run_date_format,
        forecast_variables,storm_variables,
        potential_variables,start_hour,end_hour,
        mask=None):
        
        self.train = train
        self.ensemble_name = ensemble_name
        self.model_path = model_path
        self.hf_path = hf_path
        self.patch_radius = patch_radius
        self.run_date_format = run_date_format
        self.forecast_variables = forecast_variables
        self.storm_variables = storm_variables
        self.potential_variables = potential_variables
        self.start_hour= start_hour
        self.end_hour=end_hour
        self.mask = mask

        return

    def process_map_data(self,map_file):
        lon_lat_file = f'{self.hf_path}/{self.ensemble_name}_map_data.h5'
        if not os.path.exists(lon_lat_file):
            proj_dict, grid_dict = read_ncar_map_file(map_file)
            mapping_data = make_proj_grids(proj_dict, grid_dict)
            if self.mask is not None:
                mapping_lat_data = mapping_data['lat']*self.mask
                mapping_lon_data = mapping_data['lon']*self.mask
            else:
                mapping_lat_data = mapping_data['lat']
                mapping_lon_data = mapping_data['lon']
            #lon_slices = self.cnn_slice_into_patches(mapping_lon_data,self.patch_radius,self.patch_radius)
            #lat_slices = self.cnn_slice_into_patches(mapping_lat_data,self.patch_radius,self.patch_radius)
            
            lon_slices = self.unet_slice_into_patches(mapping_lon_data,self.patch_radius,self.patch_radius)
            lat_slices = self.unet_slice_into_patches(mapping_lat_data,self.patch_radius,self.patch_radius)
            lon_lat_data = np.array((lon_slices,lat_slices))
            print(f'\nWriting map file: {lon_lat_file}\n')
            with h5py.File(lon_lat_file, 'w') as hf:
                hf.create_dataset("data",data=lon_lat_data,
                compression='gzip',compression_opts=6)
        return 

    def process_observational_data(self,run_date,mrms_variable,mrms_path):
        """
        Process observational data by both slicing the data and labeling
        MESH values at different thresholds for classification modeling. 
    
        The observational data is in the format (# of hours,x,y)

        Args:
            run_date(datetime): datetime object containing date of mrms data
            config(obj): Config object containing member parameters
        """
        obs_patch_labels = []
        
        print("Starting obs process", run_date)
        #Find start and end date given the start and end hours 
        start_date = run_date + timedelta(hours=self.start_hour)
        end_date = run_date + timedelta(hours=self.end_hour)
        #Create gridded mrms object 
        gridded_obj = GridOutput(run_date,start_date,end_date)
        gridded_obs_data = gridded_obj.load_obs_data(mrms_variable,mrms_path)

        if gridded_obs_data is None: 
            print('No observations on {0}'.format(start_date))
            return
        for hour in range(len(gridded_obs_data[1:])): 
            #Slice mrms data 
            if self.mask is not None: hourly_obs_data = gridded_obs_data[hour]*self.mask
            else: hourly_obs_data = gridded_obs_data[hour]
            hourly_obs_patches = self.unet_slice_into_patches(hourly_obs_data,self.patch_radius,self.patch_radius)
            #Label cnn mrms data
            #labels = self.unet_label_obs_patches(hourly_obs_patches)
            obs_patch_labels.append(hourly_obs_patches)

        if np.nanmax(obs_patch_labels) <= 0: return 
        obs_filename=f'{self.hf_path}/obs/obs_{run_date.strftime(self.run_date_format)}.h5'
        print(f'Writing obs file: {obs_filename}')
        #Write file out using Hierarchical Data Format 5 (HDF5) format. 
        with h5py.File(obs_filename, 'w') as hf:
            hf.create_dataset("data",data=obs_patch_labels,
            compression='gzip',compression_opts=6)
        return 

    def process_ensemble_member(self,run_date,member,member_path,map_file):
        """
        Slice ensemble data in the format (# of hours,x,y)
        Args:
            run_date(datetime): datetime object containing date of mrms data
            member (str): name of the ensemble member
            member_path(str): path to the member patch files 
            lon_lat_file (str): path to the member map file
            config(obj): Config object containing member parameters
        """
        #Find start and end date given the start and end hours 
        start_date = run_date + timedelta(hours=self.start_hour)
        end_date = run_date + timedelta(hours=self.end_hour)
        #Create gridded variable object 
        gridded_obj = GridOutput(run_date,start_date,end_date,member)
        print("Starting ens processing", member, run_date)
        #Slice each member variable separately over each hour
        gridded_variable_data = gridded_obj.load_model_data(self.model_path,forecast_variables)
        if gridded_variable_data is None: return
        for v,variable in enumerate(self.forecast_variables):
            hourly_patches = [] 
            #Slice hourly data
            for hour in np.arange(1,gridded_variable_data.shape[0]):
                #Storm variables are sliced at the current forecast hour
                #if variable in self.storm_variables:var_hour = hour
                #Potential (environmental) variables are sliced at the previous forecast hour
                #elif variable in self.potential_variables: var_hour = hour-1
                if self.mask is not None: masked_gridded_variable = gridded_variable_data[hour,v,:,:]*self.mask
                else: masked_gridded_variable = gridded_variable_data[hour,v,:,:]
                #patches = self.cnn_slice_into_patches(masked_gridded_variable,self.patch_radius,self.patch_radius)
                patches = self.unet_slice_into_patches(masked_gridded_variable,self.patch_radius,self.patch_radius)
                hourly_patches.append(patches)
                #del masked_gridded_variable
            #Shorten variable names
            if "_" in variable: 
                variable_name=variable.split('_')[0].upper() + variable.split('_')[-1]
            elif " " in variable: 
                variable_name= ''.join([v[0].upper() for v in variable.split()])
            else: variable_name = variable
            var_filename = '{0}/{2}/{1}_{2}_{3}.h5'.format(member_path,
                variable_name,member,run_date.strftime(self.run_date_format)) 
            print(f'Writing model file: {var_filename}')
            #Write file out using Hierarchical Data Format 5 (HDF5) format. 
            with h5py.File(var_filename, 'w') as hf:
                hf.create_dataset("data",data=np.array(hourly_patches),
                compression='gzip',compression_opts=6)
        return

    def cnn_slice_into_patches(self,data2d, patch_ny, patch_nx):
        '''
        A function to slice a 2-dimensional [ny, nx] array into rectangular patches and return 
        the sliced data in an array of shape [npatches, nx_patch, ny_patch].
      
        If the array does not divide evenly into patches, excess points from the northern and 
        eastern edges of the array will be trimmed away (incomplete patches are not included
        in the array returned by this function).

        Input variables:   
                    data2d -- the data you want sliced.  Must be a 2D (nx, ny) array
                    ny_patch -- the number of points in the patch (y-dimension)
                    nx_patch -- the number of points in the patch (x-dimension)
        '''

        #Determine the number of patches in each dimension
        x_patches = int(data2d.shape[0]/patch_nx)
        y_patches = int(data2d.shape[1]/patch_ny) 
        npatches = y_patches * x_patches #Total number of patches
    
        #Define array to store sliced data and populate it from data2d
        sliced_data = [] 
        
        for i in np.arange(0,data2d.shape[0],patch_nx): 
            next_i = i+patch_nx
            if next_i > data2d.shape[0]:
                break 
            for j in  np.arange(0,data2d.shape[1],patch_ny):
                next_j = j+patch_ny
                if next_j > data2d.shape[1]:
                    break
                data = data2d[i:next_i,j:next_j]
                if any(np.isnan(data.flatten())) == True:
                    continue
                sliced_data.append(data)
        return np.array(sliced_data)
    
    def unet_slice_into_patches(self,data2d,patch_ny,patch_nx,overlap_points=16):
        '''
        A function to slice a 2-dimensional [ny, nx] array into rectangular patches and return 
        the sliced data in an array of shape [npatches, nx_patch, ny_patch].
      
        If the array does not divide evenly into patches, excess points from the northern and 
        eastern edges of the array will be trimmed away (incomplete patches are not included
        in the array returned by this function).

        Input variables:   
                    data2d -- the data you want sliced.  Must be a 2D (nx, ny) array
                    ny_patch -- the number of points in the patch (y-dimension)
                    nx_patch -- the number of points in the patch (x-dimension)
        '''
        #Determine the step needed for overlaping tiles
        nstep_y = patch_ny-overlap_points
        nstep_x = patch_nx-overlap_points
        
        #Define array to store sliced data 
        sliced_data = [] 
        
        for i in np.arange(0,data2d.shape[0],nstep_x): 
            next_i = i+patch_nx
            if next_i > data2d.shape[0]:break 
            for j in np.arange(0,data2d.shape[1],nstep_y):
                next_j = j+patch_ny
                if next_j > data2d.shape[1]: break
                data = data2d[i:next_i,j:next_j]
                if any(np.isnan(data.flatten())) == True: continue
                sliced_data.append(data)
        return np.array(sliced_data)


    def cnn_label_obs_patches(self,obs_patches,label_thresholds=[5,25,50]):
        '''
        A function to generate labels for MESH patch data.  Labels can be defined by passing in a list of
        thresholds on which to divide the categories.  If not provided, default label thresholds of 5, 25, 
        and 50 mm will be used.  The default label thresholds will result in MESH data being labelled as
        follows:
                    Label       Meaning
        No Hail:      0         No pixel exceeding MESH = 5.0 mm in patch 
        Non-severe:   1         5.0 mm < Highest pixel value of MESH in patch < 25.0 mm
        Severe:       2         25.0 mm < Highest pixel value of MESH in patch < 50.0 mm
        Sig. Severe:  3         Highest pixel value of MESH > 50.0 mm

        The input data (obs_patches) must be a list of patches of dimensions [npatches, ny_patch, nx_patch]
        The function returns a list of labels of shape [npatches].

        NOTE:  This function assumes MESH is provided in mm.  If the units of MESH in the input data you
            are using are not mm, either convert them to mm before using this function, or specify 
            appropriate label thresholds using the "label_thresholds" input variable.
        '''
        obs_labels = []
        for k in np.arange(0, obs_patches.shape[0], 1):
            if (np.nanmax(obs_patches[k]) > 50.0):
                label = 3
            elif (np.nanmax(obs_patches[k]) > 25.0):
                label = 2
            elif (np.nanmax(obs_patches[k]) > 5.0):
                label = 1
            else:
                label = 0
            obs_labels.append(label)
        return obs_labels
    
    """
    def unet_label_obs_patches(self,obs_patches):
        '''
        See cnn_label_obs_patches documentation
        '''
        for o,obs_patch in enumerate(obs_patches):
            obs_patches[o][(obs_patch >= 0) & (obs_patch < 5)] = 0
            obs_patches[o][(obs_patch >= 5) & (obs_patch < 25)] = 1
            obs_patches[o][(obs_patch >= 25) & (obs_patch < 50)] = 2
            obs_patches[o][(obs_patch >= 50)] = 3
        return obs_patches
    """

    def select_training_data(self,member,training_filename):
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
