import warnings
import pandas as pd
import time
import xarray as xr
import sys
import rioxarray as rxr
from rasterio import features
import numpy as np
from affine import Affine
import geopandas as gpd
import matplotlib.pyplot as plt
import sklearn.metrics


def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes, coords, fill=np.nan, **kwargs):
    """
    Rasterize a list of (geometry, fill_value) tuples onto the given
    xarray coordinates. This only works for 1d latitude and longitude
    arrays.
    """
    transform = transform_from_latlon(coords['latitude'], coords['longitude'])
    out_shape = (len(coords['latitude']), len(coords['longitude']))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    return xr.DataArray(raster, coords=coords, dims=('latitude', 'longitude'))


def make_mask():
    # Input data:
    path = ('/rds/general/user/tk22/projects/leverhulme_wildfires_theo_keeping/'+
            'live/start_inputs/genesis_counts_FPAFOD_20020101_20181231.nc')
    ds = xr.open_dataset(path)
    path = '/rds/general/user/tk22/home/start_model_3/plots/ecoregions/us_eco_l3.shp'
    ecoregions_lvl3 = gpd.read_file(path)
    ecoregions_lvl3 = ecoregions_lvl3.to_crs(epsg = 4326)
    ecoregions_lvl3_ids = list(np.unique(ecoregions_lvl3.US_L3NAME))
    # Building mask:
    shapes = [(shape, n) for n, shape in enumerate(ecoregions_lvl3.geometry)]
    ids = np.array([(ID, n) for n, ID in enumerate(ecoregions_lvl3.US_L3NAME)])
    mask = xr.Dataset(coords={'longitude': np.array(ds.lon),
                              'latitude': np.array(ds.lat)})
    mask['ecoregion_codes'] = rasterize(shapes, mask.coords).astype(float)
    mask['ecoregion_names'] = mask['ecoregion_codes'].astype(str)
    # Naming regions:
    for region in np.unique(ecoregions_lvl3.US_L3NAME):
        indices = np.array(ids[ids[:,0] == region][:,1], dtype = float)
        mask['ecoregion_names'].values[np.isin(mask['ecoregion_codes'].values,
                                               indices)] = region
    # Reapplying codes:
    code = 0
    for region in np.unique(mask['ecoregion_names']):
        code += 1
        mask['ecoregion_codes'].values[np.array(mask['ecoregion_names'] == region)] = code
    mask['ecoregion_codes'].values[np.array(mask['ecoregion_names'] == 'nan')] = np.nan
    return mask, list(np.unique(mask.ecoregion_names))


def make_data(ds1, var = 'run_N_e_counts', tag = 'tag'):
    mask, ecoregion_ids = make_mask()
    path = ('/rds/general/user/tk22/projects/leverhulme_wildfires_theo_keeping/'+
            'live/start_inputs/genesis_counts_FPAFOD_20020101_20181231.nc')
    ds2 = xr.open_dataset(path)
    months = np.array(ds1.date.dt.month, dtype = float)

    count = 0
    t0 = time.time()

    month_codes = []
    region_codes = []
    mean_counts = []
    mean_ecounts = []
    sterr_counts = []
    sterr_ecounts = []
    auc = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category = RuntimeWarning)
        for month in np.unique(months):
            for region in np.unique(ecoregion_ids):
                if count % 10 == 0:
                    print(f'Step {int(count/10)} of '+
                          f'{len(np.unique(months)) * len(np.unique(ecoregion_ids))/10}. '+
                          f'{(time.time() - t0)/60:.1f} minutes.')
                    sys.stdout.flush()

                count += 1
                # Mask:
                temp_mask = (mask.ecoregion_names == region)
                temp_date = (months == month)
                temp_mask = np.einsum('ij,k->ijk', temp_mask, temp_date)

                # Mean Stats:
                mean_counts.append(np.nanmean(ds2.counts.values[temp_mask]))
                mean_ecounts.append(np.nanmean(ds1[var].values[temp_mask]))
                # Std. Error Stats:
                sterr_counts.append((np.nanstd(ds2.counts.values[temp_mask]) / 
                                     np.sqrt(len(ds2.counts.values[temp_mask]))))
                sterr_ecounts.append((np.nanstd(ds1[var].values[temp_mask]) / 
                                      np.sqrt(len(ds1[var].values[temp_mask]))))
                # AUC:
                if len(ds1[var].values[temp_mask]) > 0:
                    x, y = ds2.counts.values[temp_mask], ds1[var].values[temp_mask]
                    df = pd.DataFrame(list(zip(x,y)), columns = ['counts', 'e_counts'])
                    df = df.sort_values(by = 'e_counts')
                    df.counts = (df.counts >= 1)
                    df = df.dropna()
                    try:
                        auc.append(sklearn.metrics.roc_auc_score(df.counts, df.e_counts))
                    except:
                        auc.append(np.nan)
                else:
                    auc.append(np.nan)
                # Labels:
                month_codes.append(month)
                region_codes.append(region)

            df = pd.DataFrame(list(zip(region_codes, month_codes,
                                       mean_counts, mean_ecounts,
                                       sterr_counts, sterr_ecounts, auc)),
                              columns =['region', 'month',
                                        'mean_counts','mean_ecounts',
                                        'sterr_counts','sterr_ecounts', 'AUC'])
            out_path = (f'/rds/general/user/tk22/home/start_model_3/plots/'+
                        f'ecoregion_lvl3_plot_data_by_month_{tag}.csv')
            df.to_csv(out_path, index = False)
    return


if __name__ == '__main__':
    tag = sys.argv[1]
    path = ('/rds/general/user/tk22/home/egu_23/'+
            f'prob_{tag}.nc')
    ds1 = xr.open_dataset(path)
    var = 'p'
    make_data(ds1, var = var, tag = tag)