import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def Plot_Estimator(y, y_hat, network, train_test, fold_k, r2):
    print(f'Plotting Facet {network} {train_test} {fold_k}')
    plt.rc('xtick.major', size=8)
    plt.rc('ytick.major', size=8)
    plt.rc('axes', linewidth=5)
    plt.figure(figsize=(12,10))
    plt.rc('font', weight='bold')
    plt.tick_params(axis='both', which='both', labelsize=18, width=5, pad=10, direction='out',top=False,right=False)
    axis_label_properties = {
            'family' : 'sans-serif',
                'weight' : 'bold',
                    'size'   : 24}
    markers = ['o','D','s','x','']
    colors = ['k','r','b','g','#ff6600','k']
    plt.title(f"Facet {network} {train_test} Fold: {fold_k} R2: {r2}", axis_label_properties)
    plt.scatter(y, y_hat, s=80, facecolors='none', edgecolors='blue')
    plt.plot(y, y, linewidth=5, color='black')
    plt.ylabel("Estimated", axis_label_properties)
    plt.xlabel("Accuracy", axis_label_properties)
    plt.tight_layout()
    plt.savefig(f'Facet_{network}_{train_test}_Fold_{fold_k}.png', dpi=1400)
    plt.clf()
    plt.close()

def Get_Results(accuracy, network, fold_k):
    estimations = pd.read_csv(f'/mnt/disk023/lcedillo/My_Home/New_test/Results/{network}/{network}_{fold_k}.out' , sep=r'[,//\t\n]', engine='python', dtype=str, header=None).rename(columns={2: 'estimation'})
    test = pd.read_csv(f"/mnt/disk001/dfdeblasio/ParamAdvising/1028_paramadvisor_data_transfer/test_VTML200.45.11.42.40-12-{fold_k}", sep=r'[\t]', engine='python', dtype=str, header=None).drop([1], axis=1).rename(columns={0: 1})

    complete = pd.merge(accuracy, estimations, how='inner', on=[0, 1])
    test_split = pd.merge(test, complete, how='inner', on=[1])

    y, y_hat = np.array(complete['accuracy'], dtype=np.float64), np.array(complete['estimation'], dtype=np.float64) 
    test_y, test_y_hat = np.array(test_split['accuracy'], dtype=np.float64), np.array(test_split['estimation'], dtype=np.float64) 

    r2, test_r2 = r2_score(y, y_hat), r2_score(test_y, test_y_hat)
    r2, test_r2 = str(r2)[:5], str(test_r2)[:5]

    return y, test_y, y_hat, test_y_hat, r2, test_r2


if __name__ == "__main__":
    network = 'LinearRegression'
    for fold_k in range(12):
        accuracy = pd.read_csv('/mnt/disk001/dfdeblasio/realign/accuracy.out', sep=r'[,//\t\n]', engine='python', dtype=str, header=None).drop([3], axis=1).rename(columns={2: 'accuracy'})
        y, test_y, y_hat, test_y_hat, r2, test_r2 = Get_Results(accuracy, network, fold_k)
        Plot_Estimator(test_y, test_y_hat, network, 'test', fold_k, test_r2)
        Plot_Estimator(y, y_hat, network, 'complete', fold_k, r2)