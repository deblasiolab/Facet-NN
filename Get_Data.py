import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv

def Clean_and_save_data(fold_k):
    fea = pd.read_csv('/mnt/disk001/dfdeblasio/realign/features.tsv', sep=r'[,//\t\n]', engine='python', dtype=str, header=None)
    acc = pd.read_csv('/mnt/disk001/dfdeblasio/realign/accuracy.out', sep=r'[,//\t\n]', engine='python', dtype=str, header=None)

    test = pd.read_csv(f"/mnt/disk001/dfdeblasio/ParamAdvising/1028_paramadvisor_data_transfer/test_VTML200.45.11.42.40-12-{fold_k}", sep=r'[\t]', engine='python', dtype=str, header=None)
    train = pd.read_csv(f"/mnt/disk001/dfdeblasio/ParamAdvising/1028_paramadvisor_data_transfer/train_VTML200.45.11.42.40-12-{fold_k}", sep=r'[\t]', engine='python', dtype=str, header=None)

    final_fea_train = pd.DataFrame()
    final_acc_train = pd.DataFrame()
    for i in tqdm(range(len(train)), desc="Loading Data"):
        temp_fea = fea.loc[fea[1].str[:] == train[0][i]]
        temp_acc = acc.loc[acc[1].str[:] == train[0][i]]
        final_fea_train = pd.concat([final_fea_train, temp_fea])
        final_acc_train = pd.concat([final_acc_train, temp_acc])

    final_fea_test = pd.DataFrame()
    final_acc_test = pd.DataFrame()
    for i in tqdm(range(len(test)), desc="Loading Data"):
        temp_fea = fea.loc[fea[1].str[:] == test[0][i]]
        temp_acc = acc.loc[acc[1].str[:] == test[0][i]]
        final_fea_test = pd.concat([final_fea_test, temp_fea])
        final_acc_test = pd.concat([final_acc_test, temp_acc])

    final_acc_train = final_acc_train.rename(columns={1: 'bench', 2:'ac', 3:'ex'})
    final_acc_test = final_acc_test.rename(columns={1: 'bench', 2:'ac', 3:'ex'})

    train = train[0]
    test = test[0]

    data_train = pd.DataFrame()
    for z in tqdm(train, desc="Loading Data"):
        temp_fea = final_fea_train.loc[final_fea_train[1].str[:] == z]
        temp_acc = final_acc_train.loc[final_acc_train['bench'].str[:] == z]
        final = pd.merge(temp_fea, temp_acc)
        data_train = pd.concat([data_train, final])

    try: data_train.to_csv(f"Data/Train/data_train_{fold_k}.csv", index=False)
    except: print('no memory')

    data_test = pd.DataFrame()
    for z in tqdm(test, desc="Loading Data"):
        temp_fea = final_fea_test.loc[final_fea_test[1].str[:] == z]
        temp_acc = final_acc_test.loc[final_acc_test['bench'].str[:] == z]
        final = pd.merge(temp_fea, temp_acc)
        data_test = pd.concat([data_test, final])

    try: data_test.to_csv(f"Data/Test/data_test_{fold_k}.csv", index=False)
    except: print('no memory')

if __name__ == "__main__":
    
    fold_k = 0

    print('')
    print(f'Saving Data Fold {fold_k}...')
    Clean_and_save_data(fold_k)

