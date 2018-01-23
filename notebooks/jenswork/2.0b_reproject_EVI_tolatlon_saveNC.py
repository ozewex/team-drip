
# coding: utf-8

# In[1]:


import xarray as xr
import collections, math
import numpy as np
import pandas as pd
import os
from osgeo import gdal, gdal_array, osr


# ## Read in MODIS reflectance (all bands), rename bands, calculate EVI

# In[2]:


ds_all=xr.open_mfdataset(
    '/g/data/oe9/project/team-drip/modis_h30v12_reflectance/h30v12_reflectance_????.nc',
    chunks=dict(time=12, x=1200), #chunks should be in multiples of saved chunk size
)


# In[3]:


# rename vars for simple calcs
blue = ds_all.blue_450_520
green = ds_all.green_530_610
red = ds_all.red_630_690
nir1 = ds_all.nir1_780_900
nir2 = ds_all.nir2_1230_1250

# Using the MODIS-EVI coefficients
L, C1, C2, G = 1, 6, 7.5, 2.5
evi = G * ((nir1 - red) / (nir1 + C1 * red - C2 * blue + L))
evi.rename('evi')


# ##  Function for Reproject EVI to lat lon

# In[4]:


#function for reprojecting modis sinusoidal to lat/lon
AffineGeoTransform = collections.namedtuple(
    'GeoTransform', ['origin_x', 'pixel_width', 'x_rot',
                     'origin_y', 'y_rot', 'pixel_height'])


def get_geot(ds):
    """Take an Xarray object with x and y coords; return geotransform."""
    return AffineGeoTransform(*map(float, (
        # Affine matrix - start/step/rotation, start/rotation/step - in 1D
        ds.x[0], (ds.x[-1] - ds.x[0]) / ds.x.size, 0,
        ds.y[0], 0, (ds.y[-1] - ds.y[0]) / ds.y.size
    )))


## set bounding box around data (can be larger than data area
class aus:
    start_lat = -30
    stop_lat = -37
    start_lon = 138
    stop_lon = 153

out_res_degrees = 0.005

ll_geot = AffineGeoTransform(
    origin_x=aus.start_lon, pixel_width=out_res_degrees, x_rot=0,
    origin_y=aus.start_lat, y_rot=0, pixel_height=-out_res_degrees
)

new_shape = (
    math.ceil((aus.start_lat - aus.stop_lat) / out_res_degrees),
    math.ceil((aus.stop_lon - aus.start_lon) / out_res_degrees),
)

ll_coords = dict(
    latitude=np.arange(new_shape[0]) * ll_geot.pixel_height + ll_geot.origin_y,
    longitude=np.arange(new_shape[1]) * ll_geot.pixel_width + ll_geot.origin_x,
    )

# MAGIC - describes the MODIS projection
wkt_str = (
    'PROJCS["Sinusoidal",GEOGCS["GCS_Undefined",DATUM["Undefined",'
    'SPHEROID["User_Defined_Spheroid",6371007.181,0.0]],PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]],PROJECTION["Sinusoidal"],'
    'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",0.0],UNIT["Meter",1.0]]"'
)


# Next, define some generically useful functions:


def project_array(array, geot=None):
    """Reproject a tile from Modis Sinusoidal to WGS84 Lat/Lon coordinates.
    Metadata is handled by the calling function.
    """
    # Takes around seven seconds per layer for in-memory Australia mosaics
    if geot is None:
        geot = get_geot(array)
    assert isinstance(geot, AffineGeoTransform)

    def array_to_raster(array, geot):
        ysize, xsize = array.shape  # unintuitive order, but correct!
        dataset = gdal.GetDriverByName('MEM').Create(
            '', xsize, ysize,
            eType=gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype))
        dataset.SetGeoTransform(geot)
        dataset.SetProjection(wkt_str)
        dataset.GetRasterBand(1).WriteArray(array)
        return dataset

    if isinstance(array, xr.DataArray):
        array = array.values
    input_data = array_to_raster(array, geot)

    # Set up the reference systems and transformation
    from_sr = osr.SpatialReference()
    from_sr.ImportFromWkt(wkt_str)
    to_sr = osr.SpatialReference()
    to_sr.SetWellKnownGeogCS("WGS84")

    # Get new geotransform and create destination raster
    dest_arr = np.empty(new_shape)
    dest_arr[:] = np.nan
    dest = array_to_raster(dest_arr, ll_geot)

    # Perform the projection/resampling
    gdal.ReprojectImage(
        input_data, dest,
        wkt_str, to_sr.ExportToWkt(),
        gdal.GRA_NearestNeighbour)
    
    return xr.DataArray(
        dest.GetRasterBand(1).ReadAsArray(),
        dims=('latitude', 'longitude'),
        coords=ll_coords)


# ## Reproject EVI for each year and Generate NetCDF
# 

# In[5]:


for year in range(2001,2018):
    some_evi=evi.sel(time=str(year))
         
    fname = '/g/data/oe9/project/team-drip/h30v12_evi_ll_{}.nc'.format(year)
    if os.path.isfile(fname):
        print('already done', fname)
        continue        
  
    out = xr.concat(
    [project_array(some_evi.sel(time=step)) for step in some_evi.time], 
    dim=some_evi.time)
    
   
    try:
        out.to_netcdf(fname)
    except Exception as e:
        print('Year {} failed with {}'.format(year, type(e)))
        print(e)

