# %% Importing Packages

import xarray as xr
import geopandas as gpd

base_path = '/home/jez/'
repo_path = f'{base_path}Bias_Correction_Application/'
external_datapath = f'{base_path}DSNE_ice_sheets/Jez/Slides/'
tutorial_datapath = f'{repo_path}walkthrough_tutorial/tutorial_data/'
shapefiles_path = f'/home/jez/DSNE_ice_sheets/Jez/AWS_Observations/Shp/'

# %% Loading Data

ds_aws_stacked = xr.open_dataset(f'{external_datapath}ds_aws_stacked.nc')
ds_climate = xr.open_dataset(f'{external_datapath}ds_climate.nc')

icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
gdf_icesheet = gpd.read_file(icehsheet_shapefile)

# %% Climate Model Data Adjustments
# Dropping unnecessary coordinates and renaming variables
ds_climate = ds_climate.drop_vars(['Model','Mean Temperature','Mean June Temperature','latitude','longitude'])
ds_climate = ds_climate.rename({'LSM':'LandSeaMask',
                                'Time':'time',
                                'Elevation':'elevation',
                                'Temperature':'temperature',
                                'Latitude':'latitude',})

# Coarsening the data
ds_climate_coarse = ds_climate.coarsen(grid_latitude=4,grid_longitude=4).mean()
ds_climate_coarse = ds_climate_coarse.drop_indexes(['time','grid_latitude','grid_longitude'])

# Applying LandSeaMask
ds_climate_coarse = ds_climate_coarse.where(ds_climate_coarse.LandSeaMask>0.8)

# Dropping LandSeaMask
ds_climate_coarse = ds_climate_coarse.drop_vars('LandSeaMask')

# %% Weather Station Data Adjustments
# Dropping unnecessary coordinates and renaming variables
ds_aws_stacked = ds_aws_stacked.drop_indexes(['Station'])
ds_aws_stacked = ds_aws_stacked.drop_vars(['June Temperature Records',
                                   'Mean Temperature',
                                   'June Mean Temperature',
                                   'Temperature Records',
                                   'Lon(℃)',                        
                                   ])
ds_aws_stacked = ds_aws_stacked.rename({'Lat(℃)':'latitude',
                                        'Month':'month',
                                        'Year':'year',
                                        'Station':'station',
                                        'Elevation(m)':'elevation',
                                        'Temperature':'temperature',
                                        'X':'t'})

ds_aws_stacked = ds_aws_stacked.set_coords(("grid_latitude","grid_longitude"))

# %% Creating consistent index
def consistent_index(years,months):
    index = (years-1980)*12 + months 
    return index

ds_aws_stacked = ds_aws_stacked.reindex({"t": consistent_index(ds_aws_stacked.year,ds_aws_stacked.month)})
ds_climate_coarse = ds_climate_coarse.assign_coords(t=("time", consistent_index(ds_climate_coarse.year,ds_climate_coarse.month).data))

# %% Saving the data
ds_climate_coarse.to_netcdf(f'{tutorial_datapath}ds_climate.nc')
ds_aws_stacked.to_netcdf(f'{tutorial_datapath}ds_aws.nc')
gdf_icesheet.to_file(f"{tutorial_datapath}icesheet_shapefile/icesheet.shp")

