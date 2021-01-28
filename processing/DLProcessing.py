from util.make_proj_grids import make_proj_grids, read_ncar_map_file
from util.GridOutput import *
from datetime import timedelta
import pandas as pd
import numpy as np
import h5py
import os

class DLPreprocessing(object):
    def __init__(self,train,ensemble_name,model_path,
        hf_path,patch_radius,overlap_size,run_date_format,
        predictors,start_hour,end_hour,model_type,mask=None):
        
        self.train = train
        self.ensemble_name = ensemble_name
        self.model_path = model_path
        self.hf_path = hf_path
        self.patch_radius = patch_radius
        self.overlap_size = overlap_size
        self.run_date_format = run_date_format
        self.predictors = predictors
        self.start_hour= start_hour
        self.end_hour=end_hour
        self.mask = mask
        self.model_type = model_type
        return

    def process_map_data(self,map_file):
        lon_lat_file = f'{self.hf_path}/{self.ensemble_name}_map_data.h5'
        if len(glob(lon_lat_file)) < 1: 
            proj_dict, grid_dict = read_ncar_map_file(map_file)
            mapping_data = make_proj_grids(proj_dict, grid_dict)
            mapping_lat_data = mapping_data['lat']
            mapping_lon_data = mapping_data['lon']
            if self.model_type == 'CNN':
                lon_slices = self.cnn_slice_into_patches(
                    mapping_lon_data,self.patch_radius,self.patch_radius)
                lat_slices = self.cnn_slice_into_patches(
                   mapping_lat_data,self.patch_radius,self.patch_radius)
            elif self.model_type == 'UNET':
                lon_slices = self.unet_slice_into_patches(
                    mapping_lon_data,self.patch_radius,self.patch_radius)
                lat_slices = self.unet_slice_into_patches(
                   mapping_lat_data,self.patch_radius,self.patch_radius)
            
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
            print(f'No {run_date} obs data') 
            return
        for hour in np.arange(gridded_obs_data.shape[0]): 
            #Slice mrms data 
            if self.model_type == 'CNN':
                labels = self.cnn_slice_into_patches(gridded_obs_data[hour])
            #Label cnn mrms data
            elif self.model_type == 'UNET':
                labels = self.unet_slice_into_patches(
                    gridded_obs_data[hour],self.patch_radius,self.patch_radius)
            obs_patch_labels.append(labels)
        if np.nanmax(obs_patch_labels) <= 0: return 
        obs_filename=f'{self.hf_path}/obs/obs_{run_date.strftime(self.run_date_format)}.h5'
        print(f'Writing obs file: {obs_filename}')
        #Write file out using Hierarchical Data Format 5 (HDF5) format. 
        with h5py.File(obs_filename, 'w') as hf:
            hf.create_dataset("data",data=obs_patch_labels,
            compression='gzip',compression_opts=6)
        del gridded_obs_data

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
        gridded_variable_data = gridded_obj.load_model_data(self.model_path,self.predictors)
        if gridded_variable_data is None: 
            print(f'No {run_date} {member} data') 
            return
        for v,variable in enumerate(self.predictors):
            hourly_patches = [] 
            #Slice hourly data
            for hour in np.arange(gridded_variable_data.shape[0]):
                if self.model_type == 'CNN':
                    patches = self.cnn_slice_into_patches(
                        gridded_variable_data[hour,v,:,:],self.patch_radius,self.patch_radius)
                elif self.model_type == 'UNET':
                    patches = self.unet_slice_into_patches(
                        gridded_variable_data[hour,v,:,:],self.patch_radius,self.patch_radius)
                    if patches is None: print(variable,start_date)
                if np.shape(patches)[0] < 1: continue
                hourly_patches.append(patches)
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
                hf.create_dataset("data",data=hourly_patches,
                compression='gzip',compression_opts=6)
        del gridded_variable_data
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
            if next_i > data2d.shape[0]: break 
            for j in  np.arange(0,data2d.shape[1],patch_ny):
                next_j = j+patch_ny
                if next_j > data2d.shape[1]: break
                mask_values = self.mask[i:next_i,j:next_j]
                if all(np.isnan(mask_values.flatten())) == True: continue
                patch_data = data2d[i:next_i,j:next_j]
                sliced_data.append(patch_data)
        del data2d
        return np.array(sliced_data)
    
    def unet_slice_into_patches(self,data2d,patch_ny,patch_nx):
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
        nstep_y = patch_ny-self.overlap_size
        nstep_x = patch_nx-self.overlap_size
        
        #Define array to store sliced data 
        sliced_data = [] 
        for i in np.arange(0,data2d.shape[0],nstep_x): 
            next_i = i+patch_nx
            if next_i > data2d.shape[0]:break 
            for j in np.arange(0,data2d.shape[1],nstep_y):
                next_j = j+patch_ny
                if next_j > data2d.shape[1]: break
                mask_values = self.mask[i:next_i,j:next_j]
                if all(np.isnan(mask_values.flatten())) == True: continue
                patch_data = data2d[i:next_i,j:next_j]
                if any(np.isnan(patch_data.flatten())) == True: return None
                sliced_data.append(patch_data)
        del data2d
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
