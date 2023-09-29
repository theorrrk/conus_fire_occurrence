from build_dataset import *
from stretch_dataset import *
from dataset_benchmarks import *

def main(index):
    # 0: Get variables and thresholds:
    directory = '/rds/general/user/tk22/home/fire_genesis/step_3/final_thresholds/'
    path = directory + f'threshold_summary_{index}.csv'
    thresh_row = pd.read_csv(path, index_col = 0)
    variables = [x[3:] for x in list(thresh_row.index) if x[:3] == 'lo_']
    
    # 1: Build dataset based on step 2:
    ds, aic, auc, max_vif = build_dataset(variables, thresh_row)
    ds = ds.to_dataset()
    
    # 2: Find best scaling of dataset via RSS: (~8 hours)
    path = ('/rds/general/user/tk22/projects/leverhulme_wildfires_theo_keeping/live/'+
            'start_inputs/genesis_fire_FPAFOD_20020101_20181231.nc')
    ds['fire'] = xr.open_dataset(path).fire
    df = stretch_dataset(ds, decimals = 4)
    a = float(df[df.rss == df.rss.min()]['a'].iloc[0])
    b = float(df[df.rss == df.rss.min()]['b'].iloc[0])
    
    # 3: Build stretched dataset:
    ds['p'] = a * ds.p ** b
    
    # 4: Benchmarks of stretched dataset:
    path = ('/rds/general/user/tk22/projects/leverhulme_wildfires_theo_keeping/'+
            'live/start_inputs/genesis_cell_area_20020101_20181231.nc')
    ds['cell_area'] = xr.open_dataset(path).cell_area
    # Building data (monthly, 0.5-degree interpolation to match FireMIP):
    ds1 = ds.copy(deep = True)
    ds1 = ds1.coarsen({'lat':5,'lon':5}, boundary = 'trim').mean()
    ds1 = ds1.resample(date = '1M').mean()
    # Annual data:
    ds2 = ds.copy(deep = True).resample(date = '1Y').sum()
    # Getting stats:
    nme_geospatial, mpd_seasonal, nme_seasonal, nme_interannual = benchmark_stats(ds1, ds2)

    # 5: Save key variables (thresholds, stretch coef, benchmarks) to a dataframe:
    output = thresh_row.drop(['AIC', 'AUC'])
    output = output.rename(columns = {'threshold': 'value'})
    stats = pd.DataFrame({'index': ['a', 'b', 'AUC', 'AIC', 'MaxVIF', 'NME_geospatial',
                                    'NME_interannual', 'NME_seasonal', 'MPD_seasonal'],
                          'value': [a, b, auc, aic, max_vif, nme_geospatial,
                                    nme_interannual, nme_seasonal, mpd_seasonal]})
    stats = stats.set_index('index')
    output = pd.concat([stats, output])
    directory = '/rds/general/user/tk22/home/fire_genesis/step_3/output/'
    path = directory + f'output_summary_{index}.csv'
    output.to_csv(path)
    return


if __name__ == '__main__':
    index = int(os.getenv('PBS_ARRAY_INDEX'))
    main(index)