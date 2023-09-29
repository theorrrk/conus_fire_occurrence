import numpy as np
import xarray as xr


def seasonal_concentration_monthly(x, ds):
    theta = 2 * np.pi * (ds.date.dt.month - 1) / 12
    lx = np.sum(x * np.cos(theta))
    ly = np.sum(x * np.sin(theta))
    c = np.sqrt(lx**2 + ly**2) / np.sum(x)
    p = np.arctan(lx / ly)
    return c, p


def grid_run_monthly(arr, ds):
    cs = np.zeros((len(ds.lat), len(ds.lon)))
    ps = np.zeros((len(ds.lat), len(ds.lon)))

    for i in range(len(ds.lat)):
        for j in range(len(ds.lon)):
            try:
                c,p = seasonal_concentration_monthly(arr[:,i,j], ds)
            except:
                c,p = seasonal_concentration_monthly(arr[i,j,:], ds)
            cs[i,j] = c
            ps[i,j] = p


    output = xr.Dataset(data_vars =  {'c': (['lat', 'lon'] , cs),
                                      'p': (['lat', 'lon'] , ps)},
                        coords =  {'lat': ds.lat.data,
                                   'lon': ds.lon.data})
    return output


def nme(cell_area, arr_obs, arr_mod):
    nme_out = (np.nansum(cell_area * np.abs(arr_obs - arr_mod)) / 
               np.nansum(cell_area * np.abs(arr_obs - np.nanmean(arr_obs))))
    return nme_out


def mpd(cell_area, mod_season, obs_season):
    pd = cell_area * np.arccos( np.cos(mod_season.p.to_numpy() - obs_season.p.to_numpy()))
    mpd = (1 / np.pi) * np.nansum(pd) / np.nansum(cell_area)
    return mpd


def geospatial_nme(ds):
    # Making relevant data:
    arr_mod = ds.p.mean(dim = 'date').to_numpy()
    arr_obs = ds.fire.mean(dim = 'date').to_numpy()
    cell_area = ds.cell_area[0,:,:].to_numpy()
    # Finding nme
    nme_geo = nme(cell_area, arr_obs, arr_mod)
    return nme_geo


def seasonal_concentration_nme(mod_season, obs_season, ds):
    # Making relevant data:
    arr_mod = mod_season.c.to_numpy()
    arr_obs = obs_season.c.to_numpy()
    cell_area = ds.cell_area[0,:,:].to_numpy()
    # Finding nme:
    nme_seas_c = nme(cell_area, arr_obs, arr_mod)
    return nme_seas_c


def interannual_nme(mod_fires, obs_fires):
    pseudo_areas = np.ones_like(mod_fires)
    nme_inter = nme(pseudo_areas, obs_fires.to_numpy(), mod_fires.to_numpy())
    return nme_inter


def benchmark_stats(ds1, ds2):
    # Seasonal stats:
    mod_season = grid_run_monthly(ds1.p.to_numpy(), ds1)
    obs_season = grid_run_monthly(ds1.fire.to_numpy(), ds1)
    # Geospatial:
    nme_geospatial = geospatial_nme(ds1)
    print(f'Geospatial NME   = \t{nme_geospatial:.5f}')
    # Seasonal phase:
    cell_area = ds1.cell_area[0,:,:].to_numpy()
    mpd_seasonal = mpd(cell_area, mod_season, obs_season)
    print(f'Season Phase MPD = \t{mpd_seasonal:.5f}')
    # Seasonal concentration:
    nme_seasonal = seasonal_concentration_nme(mod_season, obs_season, ds1)
    print(f'Season Conc. NME = \t{nme_seasonal:.5f}')
    # Interannual:
    nme_interannual = interannual_nme(ds2.p, ds2.fire)
    print(f'Interannual NME  = \t{nme_interannual:.5f}')
    return nme_geospatial, mpd_seasonal, nme_seasonal, nme_interannual


