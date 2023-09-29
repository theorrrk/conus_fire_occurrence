from matplotlib import pyplot as plt
import pandas as pd
import xarray as xr
import time
from cartopy import crs as ccrs
import sys
import os
import scipy
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import statsmodels as sm
from sklearn.metrics import roc_auc_score
import warnings
from matplotlib import colors


def fit_model(thresh_row, N = 10**8):
    variables = [x[3:] for x in list(thresh_row.index) if x[:3] == 'lo_']
    ds_list = []
    for var in variables + ['fire', 'cell_area']:
        ds_list.append(xr.open_mfdataset('/rds/general/user/tk22/'+
                                         'projects/leverhulme_wil'+
                                         'dfires_theo_keeping/liv'+
                                         'e/start_inputs/genesis_'+
                                         f'{var}*20020101_20181231.nc'))
    ds = xr.merge(ds_list, compat = 'override')

    temp = ds['DTR'].to_dataframe()
    mask = np.logical_not(np.isnan(np.array(temp['DTR'])))
    true_inds = np.argwhere(mask)[:,0]

    sample_inds = np.random.choice(true_inds, size = int(1.25 * N), replace = False)
    df = temp.iloc[sample_inds]
    df = df.drop(columns = ['DTR'])

    whitelist = variables + ['fire', 'cell_area']
    print('Adding variables to test-train:')
    for i, var in enumerate(whitelist):
        if (i+1) % np.floor(len(whitelist)/5) == 0:
            print(f'\t{20 * (i+1) / (len(whitelist)/5):.1f}% built')
        df[var] = np.array(ds[var].to_dataframe())[sample_inds]

    df = df.dropna()
    train, test = train_test_split(df, test_size = 0.20)


    formula = 'fire ~ ' + ' + '.join(variables)

    model = smf.glm(formula,
                    data = train,
                    offset = np.log(train['cell_area']),
                    family = sm.genmod.families.family.Binomial(
                    sm.genmod.families.links.Logit())).fit(disp = 0) 
    
    return model


def t_values(indices, labels, colours, model):
    df = pd.DataFrame(model.tvalues)
    df = df.reindex(indices)

    plt.figure(figsize = (5,4))
    plt.title('Predictor Effect (t-values)')
    plt.barh(labels,
             df.values[:,0],
             color = colours)

    path = '/rds/general/user/tk22/home/fire_genesis/final_paper_plots/t_values.png'
    plt.savefig(path, bbox_inches = 'tight', facecolor = 'white', dpi = 600)
    plt.show()
    return


def timescales(thresh_row):
    variables = [x[3:] for x in list(thresh_row.index) if x[:3] == 'lo_']
    # Dataset:
    ds_list = []
    for var in variables:
        temp = xr.open_mfdataset('/rds/general/user/tk22/'+
                                 'projects/leverhulme_wil'+
                                 'dfires_theo_keeping/liv'+
                                 'e/start_inputs/genesis_'+
                                 f'{var}*20020101_20181231.nc')
        temp[var] = temp[var].clip(min = thresh_row['lo_' + var],
                                   max = thresh_row['hi_' + var])
        ds_list.append(temp)
        del temp
    ds = xr.merge(ds_list, compat = 'override')
    del ds_list
    ds = ds.load()
    # Timesteps:
    days = np.array(np.unique(np.round(1.5**np.linspace(0,15,34), 0))[:-1], dtype = int)
    # Timescale Dictionary:
    timescale_dict = {}
    for var in list(set(variables) - set(['fire', 'cell_area'])):
        print(var)
        base_arr = ds[var].to_numpy()
        metric = []
        if var == 'T':
            base_arr = base_arr + 273.15
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            count = 0
            for timescale in days:
                count+=1

                time_dim = int(timescale * np.floor(base_arr.shape[2]/timescale))
                arr = base_arr[:,:,:time_dim]
                arr = arr.reshape(251, 581, int(time_dim/timescale), timescale).mean(axis = 3)

                mean = np.nanmean(arr, axis = 2)

                temp = np.abs(arr[:,:,:-1] - arr[:,:,1:])
                temp = np.divide(temp, mean[:,:,np.newaxis])

                metric.append(np.nanmean(temp))

            timescale_dict[var] = metric
    # PLots:
    fig, axs = plt.subplots(4,4, figsize = (16,8))
    fig.suptitle('Variable Sensitivity Across Timescales:\t\t'+
                 '$\overline{ \dfrac{\| X_{i,j,k} - X_{i,j,k+1} \|}{\mu_{i,j}} }$',
                 fontsize = 14)

    for i,ax in enumerate(axs.reshape(-1)):
        try:
            var = variables[i]
            ax.set_title(var)
            ax.plot(days, timescale_dict[var], c = 'r')
            ax.set_ylabel('Sensitivity')
            ax.set_xlabel('Timescale (days)')
        except:
            ax.remove()

    plt.tight_layout()
    path = '/rds/general/user/tk22/home/fire_genesis/final_paper_plots/timescales.png'
    plt.savefig(path, bbox_inches = 'tight', facecolor = 'white', dpi = 600)
    plt.show()
    return


def geospatial_map(ds1, ds2):
    fig = plt.figure(figsize = (6,6))

    ax1 = fig.add_subplot(211, projection=ccrs.InterruptedGoodeHomolosine())
    ax1.set_global()
    ax1.coastlines()
    ax1.set_xlim(ds1.lon.min(), ds1.lon.max())
    ax1.set_ylim(ds1.lat.min(), ds1.lat.max())
    data1 = ds1.p.mean(axis = 2).to_numpy()
    im1 = ax1.pcolormesh(
        ds1.lon, ds1.lat, data1,
        norm = colors.LogNorm(vmin = 10**-4, vmax = 10**-1, clip = False),
        transform = ccrs.InterruptedGoodeHomolosine(),
        cmap = 'jet')
    ax1.set_title('Model Mean', fontsize = 14)

    ax2 = fig.add_subplot(212, projection=ccrs.InterruptedGoodeHomolosine())
    ax2.set_global()
    ax2.coastlines()
    ax2.set_xlim(ds1.lon.min(), ds1.lon.max())
    ax2.set_ylim(ds1.lat.min(), ds1.lat.max())
    data2 = ds2.fire.mean(axis = 2).to_numpy() * (data1*0+1)
    data2[data2 < 10**-10] = 10**-10
    im2 = ax2.pcolormesh(
        ds2.lon, ds2.lat, data2,
        norm = colors.LogNorm(vmin = 10**-4, vmax = 10**-1, clip = False),
        transform = ccrs.InterruptedGoodeHomolosine(),
        cmap = 'jet')
    ax2.set_title('Observational Mean', fontsize = 14)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.935, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im2, cax = cbar_ax, extend = 'both')
    cbar.set_label('Rate of Wildfire Occurrence', fontsize = 12)
    fontsetter = [t.set_fontsize(12) for t in cbar_ax.get_yticklabels()]

    plt.savefig(('/rds/general/user/tk22/home/fire_genesis/'+
                 f'final_paper_plots/geospatial_map.png'), 
                bbox_inches = 'tight', facecolor = 'white', dpi = 600)
    plt.show()
    return


def seasonal_concentration_monthly(x, ds):
    theta = 2 * np.pi * (ds.date.dt.month - 1) / 12
    lx = np.sum(x * np.cos(theta))
    ly = np.sum(x * np.sin(theta))
    c = np.sqrt(lx**2 + ly**2) / np.sum(x)
    p = np.arctan(lx / ly)
    return c, p

def seasonal_concentration_daily(x, ds):
    theta = 2 * np.pi * ((ds.date.dt.dayofyear / 365.25) % 1)
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


def seasonal_plot(data, metric = 'p', model = True, cmap = 'cividis'):
    fig = plt.figure(figsize=(6, 2.6))
    ax = plt.axes(projection = ccrs.InterruptedGoodeHomolosine())
    ax.set_global()
    ax.coastlines()
    ax.set_xlim(data.lon.min(), data.lon.max())
    ax.set_ylim(data.lat.min(), data.lat.max())
    
    if metric == 'p':
        metric_label = 'Phase'
        metric_tag = 'phase'
        ticks = np.linspace(-np.pi/2, np.pi/2, 13)[::3]
        labels = ['JUN', 'SEP', 'DEC', 'MAR', 'JUN']
        im = data[metric].plot.pcolormesh(
            transform = ccrs.InterruptedGoodeHomolosine(),
            cmap = cmap,
            cbar_kwargs = {'label': 'Seasonal Phase'}
        )
        
        cb = fig.axes[-1]
        cb.yaxis.set_ticks(ticks, labels=labels)
        cb.minorticks_off()
        
    if metric == 'c':
        metric_label = 'Concentration'
        metric_tag = 'concentration'
        im = data[metric].plot.pcolormesh(
            transform = ccrs.InterruptedGoodeHomolosine(),
            cmap = cmap,
            cbar_kwargs = {'label': 'Seasonal Concentration'}
        )
    if model == True:
        type_label = 'Model'
        type_tag = 'mod'
    if model == False:
        type_label = 'Observation'
        type_tag = 'obs'
        
    ax.set_title(f'{type_label} Seasonal {metric_label}', 
                 fontsize = 10)

    plt.savefig(('/rds/general/user/tk22/home/fire_genesis/'+
                 f'final_paper_plots/seasonal_{metric_tag}_{type_tag}.png'), 
                bbox_inches = 'tight', facecolor = 'white', dpi = 600)
    plt.show()
    return


def iav_plot(ds1, ds2):
    mod_counts = ds1.p.resample(date = '1Y').sum().sum(axis = (0,1))
    act_counts = ds2.fire.resample(date = '1Y').sum().sum(axis = (0,1))

    plt.figure(figsize = (6,4))
    plt.title('Interannual Variability')
    plt.bar(act_counts.date.dt.year, act_counts, alpha = 0.5)
    plt.bar(mod_counts.date.dt.year, mod_counts, alpha = 0.5)
    plt.legend(['FPA FOD Record', 'Model Output'])
    plt.xlabel('Year')
    plt.ylabel('Number of Wildfire Occurrences')
    plt.xticks([2002, 2006, 2010, 2014, 2018])
    plt.yticks(np.arange(0, 10**5, 20000))

    plt.savefig(('/rds/general/user/tk22/home/fire_genesis/'+
                 'final_paper_plots/iav_bar_chart.png'), 
                bbox_inches = 'tight', facecolor = 'white', dpi = 600)
    plt.show()
    return


def phase_mask(obs_season, season = 'summer'):
    if season == 'summer':
        mask = np.logical_and(
            (obs_season.p.to_numpy() > 
             np.linspace(-np.pi/2, np.pi/2, 5)[0]),
            (obs_season.p.to_numpy() < 
             np.linspace(-np.pi/2, np.pi/2, 5)[1])
        )
    if season == 'spring':
        mask = np.logical_and(
            (obs_season.p.to_numpy() > 
              np.linspace(-np.pi/2, np.pi/2, 5)[1]),
            (obs_season.p.to_numpy() < 
              np.linspace(-np.pi/2, np.pi/2, 5)[2])
        )
    if season == 'winter':
        mask = np.logical_and(
            (obs_season.p.to_numpy() > 
              np.linspace(-np.pi/2, np.pi/2, 5)[2]),
            (obs_season.p.to_numpy() < 
              np.linspace(-np.pi/2, np.pi/2, 5)[3])
        )
    if season == 'autumn':
        mask = np.logical_and(
            (obs_season.p.to_numpy() > 
              np.linspace(-np.pi/2, np.pi/2, 5)[3]),
            (obs_season.p.to_numpy() < 
              np.linspace(-np.pi/2, np.pi/2, 5)[4])
        )
    return mask


def concentration_mask(obs_season, steps = 10, mode = 1):
    # Mode from 1-10:
    mask = np.logical_and((obs_season.c.to_numpy() > (mode - 1)/10),
                          (obs_season.c.to_numpy() < (mode/10)))
    return mask

def seasonal_trend_data(ds, obs_season, mode = 1, steps = 10, season = 'spring'):
    mode_mask = concentration_mask(obs_season, mode = mode, steps = steps)
    season_mask = phase_mask(obs_season, season = season)
    mask = np.logical_and(mode_mask, season_mask)

    mod_data = ds.p.to_numpy()[mask]

    obs_data = (ds.fire.to_numpy() * (ds.p.to_numpy()*0+1))[mask]

    doys = ds.date.dt.dayofyear.to_numpy()

    mod_trend = []
    obs_trend = []

    for doy in range(1,366):
        mod_trend.append(mod_data[:,doys == doy])
        obs_trend.append(obs_data[:,doys == doy])

    mod_trend = np.nanmean(np.array(mod_trend), axis = (1,2))
    obs_trend = np.nanmean(np.array(obs_trend), axis = (1,2))
    return mod_trend, obs_trend

def iav_scatter(ds1, ds2):
    mod_counts = ds1.p.resample(date = '1Y').sum().sum(axis = (0,1))
    act_counts = ds2.fire.resample(date = '1Y').sum().sum(axis = (0,1))

    df = pd.DataFrame({'model': mod_counts.data,
                       'observation': act_counts.data})
    df = df.sort_values(by = 'observation')

    plt.figure(figsize = (4,4))
    plt.title('Observed and Modelled\nAnnual Number of Fires')
    plt.scatter(df.observation, df.model, c = 'k', s = 7)

    plt.xlabel('Annual Number of Observed Fires')
    plt.ylabel('Annual Number of Modelled Fires')

    minv = np.floor(df.min().min()/10**4) * 10**4
    maxv = np.ceil(df.max().max()/10**4) * 10**4

    plt.xticks(np.arange(minv, maxv, 2*10**4))
    plt.yticks(np.arange(minv, maxv, 2*10**4))
    plt.xlim((df.min().min()*0.95, df.max().max()*1.05))
    plt.ylim((df.min().min()*0.95, df.max().max()*1.05))

    r = scipy.stats.pearsonr(mod_counts.data, act_counts.data)[0]
    p = scipy.stats.pearsonr(mod_counts.data, act_counts.data)[1]

    plt.text(1.05*df.min().min(), 0.95*df.max().max(), f'r = {r:.3f}, with\np = {p:.2e}',
             ha="left",
             size=10,
             bbox=dict(boxstyle='round', fc="w", ec="k"))


    plt.savefig(('/rds/general/user/tk22/home/fire_genesis/'+
                 'final_paper_plots/obs_vs_mod_corr_coef.png'), 
                bbox_inches = 'tight', facecolor = 'white', dpi = 600)
    return


def plot_daily_maps(ds1):
    for i in range(0,6209,7):
        plt.figure(figsize=(6, 2.6))
        ax = plt.axes(projection = ccrs.InterruptedGoodeHomolosine())
        ax.set_global()
        ax.coastlines()

        ax.set_xlim(ds1.lon.min(), ds1.lon.max())
        ax.set_ylim(ds1.lat.min(), ds1.lat.max())

        im = ds1.p[:,:,i].plot.pcolormesh(
            transform = ccrs.InterruptedGoodeHomolosine(),
            norm = colors.LogNorm(vmin = 10**-4, vmax = 10**-1),
            cmap = 'jet'
        )
        title_date = ds1.date.dt.strftime("%B %d, %Y").data[i]
        ax.set_title('Modelled Probability of Wildfire Occurrence:\n'+
                     f'{title_date}', 
                     fontsize = 10)
        save_date = ds1.date.dt.strftime('%Y%m%d').data[i]
        plt.tight_layout()
        plt.savefig(('/rds/general/user/tk22/home/fire_genesis/final_paper_plots/'+
                     f'daily/model_prob_{save_date}.png'), 
                    bbox_inches = 'tight', facecolor = 'white', dpi = 600)
        plt.show()
    return