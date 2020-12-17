#!/usr/bin/env python
from datetime import timedelta 
from glob import glob
import pandas as pd
import xarray as xr
import numpy as np
import pygrib

'''
Gagne II, D. J., A. McGovern, N. Snook, R. Sobash, J. Labriola, J. K. Williams, S. E. Haupt, and M. Xue, 2016: 
Hagelslag: Scalable object-based severe weather analysis and forecasting. Proceedings of the Sixth Symposium on 
Advances in Modeling and Analysis Using Python, New Orleans, LA, Amer. Meteor. Soc., 447.
'''

class GridOutput(object):
    def __init__(self,run_date,start_date,end_date,member=None):
        self.run_date = pd.to_datetime(str(run_date))
        self.member = member
        self.unknown_names = {3: "LCDC", 4: "MCDC", 5: "HCDC", 6: "Convective available potential energy", 7: "Convective inhibition", 
                            197: "RETOP", 198: "MAXREF", 199: "MXUPHL", 200: "MNUPHL", 220: "MAXUVV", 
                            221: "MAXDVV", 222: "MAXUW", 223: "MAXVW"}
        self.unknown_units = {3: "%", 4: "%", 5: "%", 6: "J kg-1", 7: "J kg-1", 197: "m", 198: "dB", 
                            199: "m**2 s**-2", 200: "m**2 s**-2", 220: "m s**-1", 
                            221: "m s**-1", 222: "m s**-1", 223: "m s**-1"}
        self.forecast_hours = np.arange((start_date-run_date).total_seconds() / 3600,
                                        (end_date-run_date).total_seconds() / 3600 , dtype=int)      
    
    def load_obs_data(self,obs_variable,obs_path):
        """
        Loads data files and stores the output in the data attribute.
        """
        data = []
        day_after_date = (self.run_date+timedelta(days=1)).strftime("%Y%m%d") 
        #We are forecasting for 1200 UTC to 1200 UTC, hence we need the 
        # valid date and also the next day     
        first_half_obs_file = glob(f'{obs_path}/{obs_variable}/*{self.run_date.strftime("%Y%m%d")}*.nc')
        second_half_obs_file = glob(f"{obs_path}/{obs_variable}/*{day_after_date}*.nc")
        
        if len(first_half_obs_file) < 1 or len(second_half_obs_file) < 1: 
            print(f'No obs data on {self.run_date.strftime("%Y%m%d")}')
            return None
        first_half_obs_data = xr.open_dataset(first_half_obs_file[0])[obs_variable][12:].values
        second_half_obs_data = xr.open_dataset(second_half_obs_file[0])[obs_variable][:12].values
        
        obs_data = np.concatenate((first_half_obs_data, second_half_obs_data))
        obs_data[obs_data < 0] = 0
        obs_data[obs_data > 150] = 150
        if obs_data.shape[0] < len(self.forecast_hours): return None
        else: return obs_data

    def find_data_files(self,model_path):
        filenames = []
        day_before_date = (self.run_date-timedelta(days=1)).strftime("%Y%m%d") 
        member_name = str(self.member.split("_")[0])
        if '00' in self.member:
            inilization='00'
            hours = self.forecast_hours
            date = self.run_date.strftime("%Y%m%d")
        elif '12' in self.member:
            inilization='12'
            hours = self.forecast_hours+12
            date = day_before_date
        for forecast_hr in hours:
            if 'nam' in self.member:
                files = glob('{0}/{1}/nam*conusnest*{2}f*{3}*'.format(model_path,
                        date,inilization,forecast_hr))
                if not files:
                    files = glob('{0}/{1}/nam*t{2}z*{3}*'.format(model_path,
                            date,inilization,forecast_hr))
            else:
                files = glob('{0}/{1}/*hiresw*conus{2}*{3}f*{4}*'.format(model_path,
                        date,member_name,inilization,forecast_hr))
            if len(files) >=1:
                filenames.append(files[0])
        return filenames
    
    def load_model_data(self,model_path,predictors):
        """
        Loads data from grib2 file objects or list of grib2 file objects. 
        Handles specific grib2 variable names and grib2 message numbers.
            
        Returns:
            Array of data loaded from files in (time, y, x) dimensions, Units
        """
        
        data=None 
        u_v_variables = ['U component of wind','V component of wind','10 metre U wind component',
                        '10 metre V wind component','u','v','10u','10v']

        filenames = self.find_data_files(model_path)
        #Open each file for reading.
        if len(filenames) < len(self.forecast_hours): 
            print("Less than 24 hours of {0} model runs on {1}".format(self.member,self.run_date))
            units = None
            return data

        for f, g_file in enumerate(filenames):
            grib = pygrib.open(g_file)
            message_keys = np.array([[message.name,message.shortName,
                        message.level,message.typeOfLevel] for message in grib])
            for p,predictor in enumerate(predictors):    
                if type(predictor) is int:
                    data_values = grib[predictor].values
                    if grib[predictor].units == 'unknown':
                        Id = grib[predictor].parameterNumber
                elif type(predictor) is str:
                    if '_' in predictor:
                        #Multiple levels
                        variable = predictor.split('_')[0]
                        level = predictor.split('_')[1]
                    else:
                        #Only single level 
                        variable=predictor
                        level=None
                        
                    ##################################
                    # U/V wind string variables
                    ##################################
                
                    if variable in u_v_variables:
                        u_v_ind = np.where(
                            (message_keys[:,0] == variable) | (message_keys[:,1] == variable) &
                            (message_keys[:,2] == level) | (message_keys[:,3] == level))[0]
                        #Grib messages begin at one
                        grib_u_v_ind = int(u_v_ind[0]+1)
                        data_values = grib[grib_u_v_ind].values
                        continue
                    try:
                        
                        ##################################
                        # Unknown string variables
                        ##################################
                        if variable in self.unknown_names.values(): 
                            Id, units = self.format_grib_name(variable)
                            var_data = pygrib.index(g_file,'parameterNumber')(parameterNumber=Id)
                    
                        ##################################
                        # Known string variables
                        ##################################
                        
                        elif variable in message_keys[:,0]:
                            var_data = pygrib.index(g_file,'name')(name=variable)
                    
                        elif variable in message_keys[:,1]:
                            var_data = pygrib.index(g_file,'shortName')(shortName=variable)
                    except: 
                        print('No {0} {1} grib message found for {2} {3}'.format(
                        self.run_date,self.member,variable,level))
                        continue
                    
                    if level is None: 
                        if len(var_data) > 1: 
                            raise NameError(
                            'Multiple {0} {1} {2} records found. Rename with {2}_level.'.format(
                            self.run_date,self.member,predictor))
                        data_values = var_data[0].values
                        continue
                    for v in np.arange(len(var_data)): 
                        if var_data[v].level == int(level): 
                            data_values = var_data[v].values
                            break
                        elif var_data[v].typeOfLevel == level: 
                            data_values = var_data[v].values
                            break
                if data is None:
                    data = np.empty( (len(self.forecast_hours),len(predictors),
                        np.shape(data_values)[0], np.shape(data_values)[1]), dtype=float )*np.nan
                data[f,p,:,:]=data_values
                del data_values
            grib.close()
        return data
        
    def format_grib_name(self,selected_variable):
        """
        Assigns name to grib2 message number with name 'unknown'. Names based on NOAA grib2 abbreviations.
        
        Names:
            3: LCDC: Low Cloud Cover
            4: MCDC: Medium Cloud Cover
            5: HCDC: High Cloud Cover
            6: Convective available potential energy (CAPE)
            7: Convective Inhibition (CIN)
            197: RETOP: Echo Top
            198: MAXREF: Hourly Maximum of Simulated Reflectivity at 1 km AGL
            199: MXUPHL: Hourly Maximum of Updraft Helicity over Layer 2km to 5 km AGL, and 0km to 3km AGL
                    examples:' MXUPHL_5000' or 'MXUPHL_3000'
            200: MNUPHL: Hourly Minimum of Updraft Helicity at same levels of MXUPHL
                     examples:' MNUPHL_5000' or 'MNUPHL_3000'
            220: MAXUVV: Hourly Maximum of Upward Vertical Velocity in the lowest 400hPa
            221: MAXDVV: Hourly Maximum of Downward Vertical Velocity in the lowest 400hPa
            222: MAXUW: U Component of Hourly Maximum 10m Wind Speed
            223: MAXVW: V Component of Hourly Maximum 10m Wind Speed
        
        Args:
            selected_variable(str): Name of selected variable for loading
        Returns:
            Given an uknown string name of a variable, returns the grib2 message Id
            and units of the variable, based on the self.unknown_name and
            self.unknown_units dictonaries above. Allows access of
            data values of unknown variable name, given the ID.
        """
        names = self.unknown_names
        units = self.unknown_units
        for key, value in names.items():
            if selected_variable == value:
                Id = key
                u = units[key]
        return Id, u
    
