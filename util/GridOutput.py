#!/usr/bin/env python
from datetime import timedelta 
from netCDF4 import Dataset
from glob import glob
import pandas as pd
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
        self.valid_dates = pd.date_range(
            start=start_date,end=end_date,freq='1H')
        self.unknown_names = {3: "LCDC", 4: "MCDC", 5: "HCDC", 6: "Convective available potential energy", 7: "Convective inhibition", 
                            197: "RETOP", 198: "MAXREF", 199: "MXUPHL", 200: "MNUPHL", 220: "MAXUVV", 
                            221: "MAXDVV", 222: "MAXUW", 223: "MAXVW"}
        self.unknown_units = {3: "%", 4: "%", 5: "%", 6: "J kg-1", 7: "J kg-1", 197: "m", 198: "dB", 
                            199: "m**2 s**-2", 200: "m**2 s**-2", 220: "m s**-1", 
                            221: "m s**-1", 222: "m s**-1", 223: "m s**-1"}
        self.forecast_hours = np.arange((start_date-run_date).total_seconds() / 3600,
                                        (end_date-run_date).total_seconds() / 3600 + 1, dtype=int)


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
    
    def load_model_data(self,model_variable,model_path,data=None):
        """
        Loads data from grib2 file objects or list of grib2 file objects. 
        Handles specific grib2 variable names and grib2 message numbers.
            
        Returns:
            Array of data loaded from files in (time, y, x) dimensions, Units
        """
        
        
        u_v_variables = ['U component of wind','V component of wind','10 metre U wind component',
                        '10 metre V wind component','u','v','10u','10v']

        filenames = self.find_data_files(model_path)
        #Open each file for reading.
        if len(filenames) <1: 
            print("No {0} model runs on {1}".format(self.member,self.run_date))
            units = None
            return data
        
        for f, g_file in enumerate(filenames):
            
            grib = pygrib.open(g_file)
            
            if type(model_variable) is int:
                data_values = grib[model_variable].values
                print(grib[model_variable])
                if grib[model_variable].units == 'unknown':
                    Id = grib[model_variable].parameterNumber
            
            elif type(model_variable) is str:
                if '_' in model_variable:
                    #Multiple levels
                    variable = model_variable.split('_')[0]
                    level = model_variable.split('_')[1]
                else:
                    #Only single level 
                    variable=model_variable
                    level=None
                
                message_keys = np.array([[message.name,message.shortName,
                    message.level,message.typeOfLevel] for message in grib])
                
                ##################################
                # U/V wind string variables
                ##################################
                
                if variable in u_v_variables:
                    u_v_ind = np.where(
                        (message_keys[:,0] == variable) | (message_keys[:,1] == variable) &
                        (message_keys[:,2] == level) | (message_keys[:,3] == level))[0]
                    #Grib messages begin at one
                    grib_u_v_ind = int(u_v_ind[0]+1)
                    u_v_grib_data = grib[grib_u_v_ind].values
                
                ##################################
                # Unknown string variables
                ##################################

                elif variable in self.unknown_names.values(): 
                    Id, units = self.format_grib_name(variable)
                    if level is None: grib_data = pygrib.index(g_file,'parameterNumber')(parameterNumber=Id)
                    elif level in message_keys[:,2]: grib_data = pygrib.index(g_file,
                        'parameterNumber','level')(parameterNumber=Id,level=level)
                    elif level in message_keys[:,3]: grib_data = pygrib.index(g_file,
                        'parameterNumber','typeofLevel')(parameterNumber=Id,typeOfLevel=level)
                    else: 
                        print('No {0} {1} grib message found for {2} {3}'.format(
                        self.run_date,self.member,variable,level))
                        continue
                
                ##################################
                # Known string variables
                ##################################

                elif variable in message_keys[:,0]:
                    if level is None: grib_data = pygrib.index(g_file,'name')(name=variable)
                    elif level in message_keys[:,2]: grib_data = pygrib.index(g_file,
                        'name','level')(name=variable,level=level)
                    elif level in message_keys[:,3]: grib_data = pygrib.index(g_file,
                        'name','typeOfLevel')(name=variable, typeOfLevel=level)
                    else: 
                        print('No {0} {1} grib message found for {2} {3}'.format(
                        self.run_date,self.member,variable,level))
                        continue
                    
                elif variable in message_keys[:,1]:
                    if level is None: grib_data = pygrib.index(g_file,'shortName')(shortName=variable)
                    elif level in message_keys[:,2]: grib_data = pygrib.index(g_file,
                        'shortName','level')(shortName=variable,level=level)
                    elif level in message_keys[:,3]: grib_data = pygrib.index(g_file,
                        'shortName','typeOfLevel')(shortName=variable,typeOfLevel=level)
                    else: 
                        print('No {0} {1} grib message found for {2} {3}'.format(
                        self.run_date,self.member,variable,level))
                        continue
                
            if variable in u_v_variables: data_values = u_v_grib_data
            elif len(grib_data) > 1: raise NameError(
                "Multiple '{0}' records found for {1} {2}.\n Please rename with more description'".format(
                model_variable,self.run_date,self.member))
            else: data_values = grib_data[0].values
        
            grib.close()
            if data is None:
                data = np.empty((len(self.valid_dates), data_values.shape[0], data_values.shape[1]), dtype=float)
                data[f]=data_values[:]
            else:
                data[f]=data_values[:]
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
    
    def load_obs_data(self,obs_variable,obs_path):
        """
        Loads data files and stores the output in the data attribute.
        """
        data = []
        next_date = (self.run_date+timedelta(days=1)).strftime("%Y%m%d") 
        filename_args = "{0}/{1}/*{2}*.nc"
        obs_file = [glob(filename_args.format(obs_path,obs_variable,date))[0] 
                    for date in [self.run_date.strftime("%Y%m%d"),next_date]
                    if len(glob(filename_args.format(obs_path,obs_variable,date)))>=1]  
        if len(obs_file) < 2: return None
        for h,fore_hour in enumerate(self.forecast_hours):
            if fore_hour <= 24:
                obs_data = Dataset(obs_file[0])
                data.append(obs_data.variables[obs_variable][h])
            else:
                obs_data = Dataset(obs_file[1])
                data.append(obs_data.variables[obs_variable][h%13])
        data = np.array(data)
        data[data < 0] = 0
        data[data > 150] = 150
        if len(data) < 1: return None
        else: return data
