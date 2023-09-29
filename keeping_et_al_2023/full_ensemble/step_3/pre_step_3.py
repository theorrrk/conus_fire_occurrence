import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    directory = '/rds/general/user/tk22/home/fire_genesis/step_2/threshold_summary/'
    thresh_list = []
    for path in Path(directory).rglob('*.csv'):
        #print(path)
        series = pd.read_csv(path, index_col = 0).iloc[-1]
        #if series not in thresh_list:
        new = True
        for thresh in thresh_list:
            if series.equals(thresh):
                print(f'{path}\n\tAlready in list.')
                new = False
        if new == True:
            thresh_list.append(series)

    directory = '/rds/general/user/tk22/home/fire_genesis/step_3/final_thresholds/'
    for i in range(len(thresh_list)):
        thresh_list[i].to_csv(directory + f'threshold_summary_{i+1}.csv',
                              index = True, header = ['threshold'])