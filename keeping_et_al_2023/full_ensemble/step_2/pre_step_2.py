import json
import sys
import numpy as np


def main(n_vars = 12):
    for i in range(1,101):
        path = ('/rds/general/user/tk22/home/fire_genesis/'+
                f'step_1/summary_{i}th_run.txt')
        with open(path, 'r') as file:
            data = file.read().split('\n')
            data = [sorted([y[1:-1] for y in x[17:-1].split(', ')])
                    for x in data[3::13]]
        try:
            critical_list = [datum for datum in data 
                             if len(datum) == n_vars][1]
        except IndexError:
            try:
                critical_list = [datum for datum in data 
                                 if len(datum) == n_vars][0]
            except IndexError:
                print([len(d) for d in data])
                print(f'{i}th selection does not have {n_vars} predictors')
                break

        list_path = ('/rds/general/user/tk22/home/fire_genesis/'+
                     f'step_2/selected_vars_by_n/{n_vars}_'+
                     f'vars/predictor_list_{n_vars}_{i}.txt')
        with open(list_path, "w") as file:
            json.dump(critical_list, file)

    variable_lists = []

    for i in range(1,101):
        path = (f'/rds/general/user/tk22/home/fire_genesis/step_2/selected_vars'+
                f'_by_n/{n_vars}_vars/predictor_list_{n_vars}_{i}.txt')
        with open(path, 'r') as file:
            variable_list = json.load(file)
        variable_lists.append(variable_list)

    unique_lists = [list(x) for x in set(tuple(x) for x in variable_lists)]
    for i,u in enumerate(unique_lists):
        path = (f'/rds/general/user/tk22/home/fire_genesis/step_2/'+
                f'selected_vars/predictor_list_{i+1}.txt')
        with open(path, "w") as file:
            json.dump(u, file)
    return


if __name__ == '__main__':
    main()