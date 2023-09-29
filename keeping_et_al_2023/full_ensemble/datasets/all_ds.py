from os import listdir
import pandas as pd
import numpy as np
from paretoset import paretoset
from quick_build import *
import time
import os


def main(index):

    direc = '/rds/general/user/tk22/home/fire_genesis/step_3/output/'
    paths = listdir(direc)

    key_stats = ['AUC', 'NME_geospatial', 'NME_interannual', 'NME_seasonal', 'MPD_seasonal']

    data_list = []
    for path in paths:
        if path[-4:] == '.csv':
            temp = pd.read_csv(direc + path)
            temp = temp.set_index(temp.columns[0])
            data_list.append([int(path.split('_')[-1].split('.')[0])] + list(temp.loc[key_stats]['value']))
    df = pd.DataFrame(data_list, columns = ['code']+key_stats)

    metrics = df[['NME_geospatial', 'NME_interannual', 'NME_seasonal', 'MPD_seasonal']]
    mask = paretoset(metrics, sense=['min','min','min','min'])
    superior_df = df[mask]
    best_code = int(superior_df[superior_df.AUC == superior_df.AUC.max()].code)
    superior_codes = list(superior_df.code)
    all_codes = list(df.code)

    paths = [direc + f'output_summary_{code}.csv' for code in all_codes]
    paths = paths[(index-1)*20:index*20]
    
    thresh_row_list = []
    variables_list = []
    for path in paths:
        df = pd.read_csv(path)
        thresh_row = df.set_index(df.columns[0]).T
        thresh_row_list.append(thresh_row)
        variables_list.append([x[3:] for x in thresh_row.columns if x[:3] == 'lo_'])

    all_variables = list(np.unique([item for sublist in variables_list for item in sublist]))

    # Building dataset:
    ds_list = []
    for var in all_variables + ['fire', 'cell_area']:
        ds_list.append(xr.open_mfdataset('/rds/general/user/tk22/'+
                                         'projects/leverhulme_wil'+
                                         'dfires_theo_keeping/liv'+
                                         'e/start_inputs/genesis_'+
                                         f'{var}*20020101_20181231.nc'))
    ds = xr.merge(ds_list, compat = 'override')
    del ds_list
    # Building test-train:
    test, train = build_inputs(ds, all_variables + ['fire', 'cell_area'],
                               N = 10 ** 7, seed = np.random.randint(1,10**7))

    t0 = time.time()
    for i in range(len(variables_list)):
        print(f'Step {i} of {len(variables_list)}: {(time.time() - t0)/60:.1f} minutes')
        thresh_row = thresh_row_list[i]
        variables = variables_list[i]

        a = float(thresh_row.a)
        b = float(thresh_row.b)
        
        # Applying thresholds:
        temp_test, temp_train = apply_thresh(test[variables_list[i]],
                                             train[variables_list[i]],
                                             thresh_row, variables)
        temp_test['cell_area'] = test['cell_area']
        temp_test['fire'] = test['fire']
        temp_train['cell_area'] = train['cell_area']
        temp_train['fire'] = train['fire']

        formula = 'fire ~ ' + ' + '.join(variables)
        da, params = build_ds(temp_test, temp_train, formula, variables, thresh_row, ds)
        if i == 0:
            prob = (a * da ** b).to_numpy()
        else:
            prob += (a * da ** b).to_numpy()
    prob = prob 
    np.save(f'/rds/general/user/tk22/ephemeral/all_ds_mean_set_{index}.npy', prob)
    return

if __name__ == '__main__':
    index = int(os.getenv('PBS_ARRAY_INDEX'))
    main(index)