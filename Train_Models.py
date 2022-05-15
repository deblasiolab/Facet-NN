import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def Train_Model(model_alg, k):
    train, test = pd.read_csv('./Folds/Data/train_data_{}.csv'.format(k)), pd.read_csv('./Folds/Data/test_data_{}.csv'.format(k))
    complete = pd.concat([train, test]).reset_index().drop(['index'], axis=1)
    features = np.array(complete[['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']], dtype=np.float64)
    accuracy = np.array(complete['19'], dtype=np.float64).reshape(-1, 1)

    train_features = np.array(train[['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']], dtype=np.float64)
    train_accuracy = np.array(train['19'], dtype=np.float64).reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(train_features, train_accuracy, test_size=0.4, random_state=0)

    model = model_alg.fit(x_train, y_train.ravel())

    complete_predictions = model.predict(features).reshape(-1, 1)
    predictions = model.predict(x_test).reshape(-1, 1)

    param_bench = complete[['0','1']]
    param_bench['out'] = pd.DataFrame(complete_predictions)

    final_accuracy_file = []
    for i in tqdm(range(len(param_bench))):
        final_accuracy_file.append(str(param_bench['0'][i]) + '/' + str(param_bench['1'][i]) + '\t' + str(param_bench['out'][i]))

    with open(f"New_test/Results/RandomForestRegressor/RandomForestRegressor_{k}.out", "w") as txt_file:
        for line in final_accuracy_file:
            txt_file.write("".join(line) + "\n")

    r2_complete = r2_score(accuracy, complete_predictions)
    r2_test = r2_score(y_test, predictions)

    return accuracy, complete_predictions, y_test, predictions, r2_complete, r2_test


def Plot_Estimator(y, y_hat, model, train_test, fold_k, r2):
    print(f'Plotting Facet {model} {train_test} {fold_k}')
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
    plt.title(f"{model} {train_test} Fold: {fold_k} R2: {r2}", axis_label_properties)
    plt.scatter(y, y_hat, s=80, facecolors='none', edgecolors='blue')
    plt.plot(y, y, linewidth=5, color='black')
    plt.ylabel("Estimated", axis_label_properties)
    plt.xlabel("Accuracy", axis_label_properties)
    plt.tight_layout()
    plt.savefig(f'New_test/Viz/{model}/{model}_{train_test}_Fold_{fold_k}.png', dpi=1400)
    plt.clf()
    plt.close()


def run(m_alg, model_alg, k):
    accuracy, complete_predictions, y_test, predictions, r2_complete, r2_test = Train_Model(model_alg, k)
    print('Fold', k)
    print(m_alg)
    print('Complete:', r2_complete)
    print('Test:', r2_test)
    Plot_Estimator(accuracy, complete_predictions, m_alg, 'Complete', k, r2_complete)
    Plot_Estimator(y_test, predictions, m_alg, 'Test', k, r2_test)
    print('Done')

for k in range(12):
    m, m_s = LinearRegression(), 'LinearRegression'
    run(m_s, m, k)

for k in range(12):
    m, m_s = DecisionTreeRegressor(), 'DecisionTreeRegressor'
    run(m_s, m, k)

for k in range(12):
    m, m_s = RandomForestRegressor(), 'RandomForestRegressor'
    run(m_s, m, k)