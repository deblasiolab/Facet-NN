import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def Plot_results(Facet, New_Facet, network, adv_set_type, train_test, fold_k):
    print('')
    if fold_k == -1: print(f'Plotting Facet {network} {adv_set_type} {train_test} Complete')
    else: print(f'Plotting Facet {network} {adv_set_type} {train_test} {fold_k}')
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
    x = np.arange(2,26)
    y_min, y_max = min(np.min(Facet), np.min(New_Facet)), max(np.max(Facet), np.max(New_Facet))
    mid = np.full((24,), (y_max+y_min)/2)
    plt.plot(x, mid, linewidth=3, color='black')
    if fold_k == -1: plt.title(f"Facet {network} {adv_set_type} {train_test}", axis_label_properties)
    else: plt.title(f"Facet {network} {adv_set_type} {train_test} Fold: {fold_k}", axis_label_properties)
    plt.plot(x, Facet, label=f'Facet {adv_set_type} {train_test}', linestyle = '-', linewidth=5)
    plt.plot(x, New_Facet, label=f'Facet {network} {adv_set_type} {train_test}', linestyle = '--', linewidth=5)
    plt.ylabel("Accuracy",axis_label_properties)
    plt.xlabel("Advisor Set Sizes", axis_label_properties)
    plt.grid(True)
    plt.legend(loc="lower center", edgecolor='black', bbox_to_anchor=(0.5, -0.2), fontsize=17, ncol=2, borderpad=0.7)
    plt.tight_layout()
    if fold_k == -1: plt.savefig(f'Advisor/{adv_set_type}/Facet_{network}/Plot/Facet_{network}_{adv_set_type}_{train_test}.png', dpi=1400)
    else: plt.savefig(f'Advisor/{adv_set_type}/Facet_{network}/Folds/Plot/Facet_{network}_{adv_set_type}_{train_test}_{fold_k}.png', dpi=1400)
    plt.clf()
    plt.close()

def Get_results_per_fold(Facet_Results_path, New_Results_path, fold_k):
    Facet = pd.read_csv(Facet_Results_path, sep=r'[//\t= ]', engine='python', dtype=str, header=None).drop([2,3,4,5,6,7], axis=1)
    Facet = np.array(Facet.loc[Facet[0] == str(fold_k)].drop([0], axis=1), dtype=np.float64)
    Facet = Facet[Facet[:, 0].argsort()]

    New_Facet = pd.read_csv(New_Results_path, sep=r'[//\t= ]', engine='python', dtype=str, header=None).drop([2,3,4,5,6,7], axis=1)
    New_Facet = np.array(New_Facet.loc[New_Facet[0] == str(fold_k)].drop([0], axis=1), dtype=np.float64)
    New_Facet = New_Facet[New_Facet[:, 0].argsort()]
    
    return Facet[:,1], New_Facet[:,1]

def Plot_results_for_all(Facet_Results_path, New_Results_path, plot, network, adv_set_type, train_test):
    Old_Avg, New_Avg = [], []
    for fold_k in range(12):
        Facet, New_Facet = Get_results_per_fold(Facet_Results_path, New_Results_path, fold_k)
        if plot: Plot_results(Facet, New_Facet, network, adv_set_type, train_test, fold_k)
        Old_Avg.append(Facet)
        New_Avg.append(New_Facet)
    Old_Avg, New_Avg = np.mean(Old_Avg, axis=0), np.mean(New_Avg, axis=0)
    
    Plot_results(Old_Avg, New_Avg, network, adv_set_type, train_test, -1)

# Run
if __name__ == "__main__":
    Facet_Results_path = ''
    network = ['NN'] # ['NN', 'ResNet']
    adv_set_type = ['Greedy_Clean'] #['Greedy', 'Oracle']
    train_test = ['test'] # ['train', 'test'] 
    all_folds = False
    fold_k = -1

    for net in network:
        for adv_type in adv_set_type:
            for t in train_test:
                if adv_type == 'Greedy': Facet_Results_path = f'/mnt/disk001/dfdeblasio/OpalStructureAdvising/old_results_greedy_{t}'
                elif adv_type == 'Greedy_Clean': Facet_Results_path = f'/mnt/disk001/dfdeblasio/Greedy_Algo/results/advising/advising_facet_small_{t}'
                elif adv_type == 'Oracle': Facet_Results_path = f'/mnt/disk001/dfdeblasio/OpalStructureAdvising/old_results_oracle_{t}'
                if fold_k != -1:
                    New_Results_path = f'Advisor/{adv_type}/Facet_{net}/Folds/Out/Facet_{net}_{adv_type}_{t}_{fold_k}.out'
                    Facet, New_Facet = Get_results_per_fold(Facet_Results_path, New_Results_path, fold_k)
                    Plot_results(Facet, New_Facet, net, adv_type, t, fold_k)
                else:
                    if adv_type == 'Greedy_Clean': 
                        New_Results_path = f'/mnt/disk001/dfdeblasio/Greedy_Algo/results/advising/advising_facet-nn_small_{t}'
                        Plot_results_for_all(Facet_Results_path, New_Results_path, all_folds, net, adv_type, t)
                    else:
                        New_Results_path = f'Advisor/{adv_type}/Facet_{net}/Out/Facet_{net}_{adv_type}_{t}.out'
                        Plot_results_for_all(Facet_Results_path, New_Results_path, all_folds, net, adv_type, t)