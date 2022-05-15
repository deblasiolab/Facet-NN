#!/usr/bin/python3
import argparse
import csv
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser(description='Create an oracle advisor set from accuracies.')
parser.add_argument('accuracy', type=str, help='Accuracy file, as TSV')
parser.add_argument('scatter', type=str, help='Output File Name')
parser.add_argument('--benchest', action='append', nargs=2, metavar=('benchmarks','estimator'))
parser.set_defaults(scoring="q")
parser.add_argument('--tc', help='Set to use Total Column rather than Sum-of-Pairs scoring', action='store_const', const="tc", dest="scoring")
parser.add_argument('--q',  help='Set to use Sum-of-Pairs rather than Total Column scoring', action='store_const', const="q", dest="scoring", default="q")
parser.set_defaults(accuracy_ftype="tsv")
parser.add_argument('--accuracy_pickle',  help='Accuracy file is in Pickle format not TSV', action='store_const', const="pickle", dest="accuracy_ftype")
parser.set_defaults(estimator_ftype="tsv")
parser.add_argument('--estimator_pickle',  help='Estimator file is in Pickle format not TSV', action='store_const', const="pickle", dest="estimator_ftype")
parser.add_argument('--color', default="b",help="Scatter Line Color in standard matplotlib format")
#parser.add_argument('--tltext', default="",help="Text to put in the top left to label the figure")
parser.add_argument('--axisname', default="Estimated",help="Text to put in the top left to label the figure")
args = parser.parse_args()

plt.rc('xtick.major', size=8)
plt.rc('ytick.major', size=8)
plt.rc('axes', linewidth=5)
plt.figure(figsize=(12,10))
plt.rc('font', weight='bold',size=28)
plt.tick_params(axis='both', which='both', labelsize=18, width=5, pad=10, direction='out',top=False,right=False)

axis_label_properties = {
            'family' : 'sans-serif',
                'weight' : 'bold',
                    'size'   : 32}
markers = ['o','D','s','x','']
colors = ['k','r','b','g','#ff6600','k']

# Read the accuracy file, should be a TSV with 3 columns: alignment name, Q score, TC score
accuracy = {}
accuracy_var = {}
if(args.accuracy_ftype == "tsv"):
    acc_f = open(args.accuracy)
    lines = csv.reader(acc_f, delimiter="\t")
    for row in lines:
        if(args.scoring == "q"):
            accuracy[row[0]] = float(row[1])
        if(args.scoring == "tc"):
            accuracy[row[0]] = float(row[2])
elif(args.accuracy_ftype == "pickle"):
    acc_f = open(args.accuracy,'rb')
    accuracy = pickle.loads(acc_f.read())
else:
    print("Accuracy file type not Pickle or TSV.")
    exit(20)

x = []
y = []

for (benchFname, estFname) in args.benchest:
    print("Working on ", benchFname , ",", estFname)
    benchmarks = {}
    benchmarks_f = open(benchFname)
    bench_key = ""
    lines = csv.reader(benchmarks_f, delimiter="\t")
    for row in lines:
        benchmarks[row[0]] = float(row[1])
        bench_key = row[0]

    estimator = {}  
    if(args.estimator_ftype == "tsv"):
        est_f = open(estFname)
        lines = csv.reader(est_f, delimiter="\t")
        for row in lines:
            estimator[row[0]] = float(row[-1])
    elif(args.accuracy_ftype == "pickle"):
        est_f = open(estFname,'rb')
        estimator = pickle.loads(est_f.read())
    else:
        print("Estimator file type not Pickle or TSV.")
        exit(20)
        
    for k in estimator.keys():
        aln_name = k.split("/");
        if(aln_name[1] in benchmarks.keys() and k in accuracy.keys()):
            y.append(estimator[k])
            x.append(accuracy[k])


#ax = plt.gca()
#vals = ax.get_yticks()
#vals = ["{:1.1f}".format(xp) for xp in vals]
#ax.set_yticklabels(vals)
# vals = ax.get_xticks()
# vals = ["{:3.0f}%".format(xp*100) for xp in vals]
# ax.set_xticklabels(vals)

plt.plot(x,y,'o',color=args.color,markersize=4,fillstyle="none",linewidth=1)

xnp = np.array(x)
ynp = np.array(y)
(m, b, r_value, p_value, std_err) = stats.linregress(xnp, ynp)
#plt.figtext(0.15,.88,args.tltext,fontdict={'size' : 28})
plt.figtext(0.2,.9,f"R-squared: {r_value**2:.3f}",fontdict={'size' : 28})
print(f"R-squared: {r_value**2:.6f}")
plt.plot([np.min(xnp),np.max(xnp)], [m*np.min(xnp)+b,m*np.max(xnp)+b], color="black", linewidth=5)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.ylabel(args.axisname,axis_label_properties)
plt.xlabel("Accuracy",axis_label_properties)
plt.tight_layout()
plt.savefig(args.scatter)
