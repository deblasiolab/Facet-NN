import csv
from tqdm import tqdm

def Run_Advisor(adv_accuracy, adv_scoring, network, adv_set_type, train_test, start_fold, end_fold):

    singleton = False
    if start_fold == end_fold: singleton = True

    results_adv = []
    for fold_k in range(start_fold, end_fold+1):
        print(f'Advising Fold {fold_k}...')
        for cardinality in tqdm(range(2, 26)):
        
            # Accuracy
            accuracy = {}
            acc_f = open(adv_accuracy)
            lines = csv.reader(acc_f, delimiter="\t")
            for row in lines:
                if(adv_scoring == "q"): accuracy[row[0]] = float(row[1])
                if(adv_scoring == "tc"): accuracy[row[0]] = float(row[2])

            # Estimator
            adv_estimator = f'Estimator/Facet_{network}_Estimator/Facet_{network}_Fold_{fold_k}.out' 
            estimator = {}
            est_f = open(adv_estimator)
            lines = csv.reader(est_f, delimiter="\t")
            for row in lines: estimator[row[0]] = float(row[1])

            # Benchmarks
            adv_benchmarks = f'/mnt/disk001/dfdeblasio/ParamAdvising/1028_paramadvisor_data_transfer/{train_test}_VTML200.45.11.42.40-12-{fold_k}'              
            benchmarks = {}
            benchmarks_f = open(adv_benchmarks)
            lines = csv.reader(benchmarks_f, delimiter="\t")
            for row in lines: benchmarks[row[0]] = float(row[1])

            # Advisor set
            adv_set = None
            if adv_set_type == 'Greedy': adv_set = [f'/mnt/disk001/dfdeblasio/ParamAdvising/FindParameters/results/sets/facet/greedy_empty/CONFIG.25.NOEST.train_VTML200.45.11.42.40-12-{fold_k}_{cardinality}.noBinSingle.delta0.diffFit.greedy.average.startEmpty.set']
            elif adv_set_type == 'Oracle': adv_set = [f'/mnt/disk001/dfdeblasio/ParamAdvising/FindParameters/results/sets/oracle/CONFIG.25.NOEST.train_VTML200.45.11.42.40-12-{fold_k}_{cardinality}.set']
            for setfname in adv_set:
                set = []
                set_f = open(setfname)
                for row in set_f: set.append(row.strip())
                acc_total = 0
                acc_count = 0
                for benchmark in benchmarks.keys():
                    max_est, max_acc, max_acc_count = -1, -1, 1
                    for param in set:
                        if estimator[param+"/"+benchmark] > max_est:
                            max_est = estimator[param+"/"+benchmark]
                            max_acc = accuracy[param+"/"+benchmark]
                            max_acc_count = 1
                        if estimator[param+"/"+benchmark] == max_est:
                            max_acc += accuracy[param+"/"+benchmark]
                            max_acc_count += 1
                    if max_acc > -1:
                        acc_total += (1.0/benchmarks[benchmark])*(max_acc/max_acc_count)
                        acc_count += (1.0/benchmarks[benchmark])

                # Record results
                results_adv.append(str(fold_k) + "\t" + str(cardinality) + "\t" + str(acc_total) + " / " + str(acc_count) + " = " + str(acc_total/acc_count))
    
    # Record Complete
    if singleton == False:
        with open(f'Advisor/{adv_set_type}/Facet_{network}/Out/Facet_{network}_{adv_set_type}_{train_test}.out', 'w') as txt_file:
            for line in results_adv: txt_file.write("".join(line) + "\n")
        print('Saved complete')
    else:
        with open(f'Advisor/{adv_set_type}/Facet_{network}/Folds/Out/Facet_{network}_{adv_set_type}_{train_test}_{fold_k}.out', 'w') as txt_file:
            for line in results_adv: txt_file.write("".join(line) + "\n")
        print('Saved complete')

# Run
if __name__ == "__main__":
    adv_accuracy = '/mnt/disk001/dfdeblasio/realign/accuracy.out' 
    adv_scoring = 'q'
    
    network = ['NN'] # ['NN', 'ResNet']
    adv_set_type = ['Greedy', 'Oracle']
    train_test = ['test'] # ['train', 'test']
    start_fold, end_fold = 0, 0

    for net in network:
        for adv_type in adv_set_type:
            for t in train_test:
                print('')
                print(f'Saving Facet {net} {adv_type} {t} from {start_fold} to {end_fold}...')
                Run_Advisor(adv_accuracy, adv_scoring, net, adv_type, t, start_fold, end_fold)
