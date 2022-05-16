import matplotlib.pyplot as plt
import csv
import numpy as np
from os.path import join, isdir
from os import listdir, mkdir
import seaborn as sns
import argparse
palette = sns.color_palette('muted')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', default='CUB')
parser.add_argument('--generalized', '-g', default=False, action='store_true')
args = parser.parse_args()

dataset = args.dataset
generalized = args.generalized

gind = 0
gkey = 'ZSL'
if generalized:
	gind = 3
	gkey = 'GZSL'

idir = '../accuracies_fewshot'
odir = 'fewshot'

files = listdir(idir)

modes = ['acc_'+dataset+'__False_']
names = ['Traditional ZSL']
colors = [palette[0], palette[1], palette[2], palette[3], palette[4], palette[5], palette[6]]

new_modes = [
	'acc_'+dataset+'__Falserandom+',
	'acc_'+dataset+'__Falseinteractive_siblings+',
	'acc_'+dataset+'__Falseinterlatent+',
]
new_names = [
	'ZSL-Interactive w/ Random',
	'ZSL-Interactive w/ Sibling-variance',
	'ZSL-Interactive w/ Representation-change',
]

new_colors = palette[1:len(new_names)+1]

new_modes.append('acc_'+dataset+'__Falseoneshotactive+')
new_names.append('ZSL-Interactive w/ Image-based')
names = ['OSL+'+tmp for tmp in names]
new_names = ['OSL+'+tmp for tmp in new_names]
new_colors.append(palette[len(new_names)+1])

if dataset == 'SUN':
	costs = [(0, 102)]*7
elif dataset == 'CUB':
	costs = [(0, 28)]*7
elif dataset == 'APY':
	costs = [(0, 64)]*7
elif dataset == 'AWA2' or dataset == 'AWA3':
	costs = [(0, 85)]*7
new_costs = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]

data = [{} for tmp in range(len(modes))]
datan = [{} for tmp in range(len(new_modes))]

# Unsupervised ZSL methods are evaluated separately. Inferred values are used here
unsup_acc = {
	'AWA2': {'ZSL': {'W2V': 42.34, 'CAAP': 53.46, 'ZSLNS': 32.15, 'ZSLNS+CADA-VAE': 55.23, 'GAZSL': None, 'GAZSL+CADA-VAE': None}, 'GZSL': {'W2V': 46.13, 'CAAP': 55.23, 'ZSLNS': 43.94, 'ZSLNS+CADA-VAE': 58.86, 'GAZSL': None, 'GAZSL+CADA-VAE': None}},
	'CUB': {'ZSL': {'W2V': 30.01, 'CAAP': 29.24, 'ZSLNS': 18.97, 'ZSLNS+CADA-VAE': 25.78, 'GAZSL': 24.32, 'GAZSL+CADA-VAE': 30.22}, 'GZSL': {'W2V': 32.64, 'CAAP': 28.74, 'ZSLNS': 20.12, 'ZSLNS+CADA-VAE': 25.67, 'GAZSL': 24.08, 'GAZSL+CADA-VAE': 32.54}},
	'SUN': {'ZSL': {'W2V': 37.01, 'CAAP': 32.77, 'ZSLNS': None, 'ZSLNS+CADA-VAE': None, 'GAZSL': None, 'GAZSL+CADA-VAE': None}, 'GZSL': {'W2V': 27.09, 'CAAP': 23.89, 'ZSLNS': None, 'ZSLNS+CADA-VAE': None, 'GAZSL': None, 'GAZSL+CADA-VAE': None}}
}


for file in files:
	ffound = False
	nmode = False
	for mind, mode in enumerate(modes):
		if mode in file:
			ffound = True
			break
	for mind2, mode2 in enumerate(new_modes):
		if mode2 in file:
			ffound = True
			nmode = True
			break
	if not ffound:
		continue
	with open(join(idir, file)) as ifd:
		if nmode:
			newkey = str(int(file[:-4].split('_')[-3].split('+')[-1]))
			if newkey not in datan[mind2]:
				datan[mind2][newkey] = []
			reader = csv.reader(ifd, delimiter=',')
			for row in reader:
				datan[mind2][newkey].append(float(row[gind])*100)
		else:
			newkey = file[:-4].split('_')[-1]
			if newkey not in data[mind]:
				data[mind][newkey] = []
			reader = csv.reader(ifd, delimiter=',')
			for row in reader:
				data[mind][newkey].append(float(row[gind])*100)

for datum in data:
	keys = list(datum.keys())
	for key in keys:
		datum[float(key)] = datum[key]
	for key in keys:
		del datum[key]
	print([len(datum[key]) for key in sorted(list(datum.keys()))])
	for key in sorted(list(datum.keys())):
		datum[key] = (np.mean(datum[key]), np.std(datum[key])/np.sqrt(len(datum[key])))

for datum in datan:
	keys = list(datum.keys())
	for key in keys:
		datum[float(key)] = datum[key]
	for key in keys:
		del datum[key]
	print([len(datum[key]) for key in sorted(list(datum.keys()))])
	for key in sorted(list(datum.keys())):
		datum[key] = (np.mean(datum[key]), np.std(datum[key])/np.sqrt(len(datum[key])))


for dat in data:
	print(dat)
for name, dat in zip(new_names, datan):
	print(name, dat)

markersize = 20
oneshot = True

with plt.style.context('seaborn'):
	# print(matplotlib.rcParams["font.family"])
	# print(matplotlib.rcParams["font.sans-serif"])
	# matplotlib.rcParams["font.family"] = ['Times New Roman']
	plt.figure(figsize=(10, 10))
	plt.tight_layout()
	for j in range(1, 2):
		ax = plt.subplot(1, 1, j+1-1)
		for i in range(len(names)):
			if i==0:
				keys = sorted(data[i].keys())
				values = [data[i][tmp][0] for tmp in keys]
				stde = np.array([data[i][tmp][1] for tmp in keys])
				keys = np.array([np.ceil(costs[i][1]*tmp+costs[i][0]) for j, tmp in enumerate(keys)])
				ax.errorbar(keys, values, stde, ecolor=colors[i], fmt='^ ', label=names[i], lw=3, markersize=markersize, c=colors[i])
			else:
				keys = [0]
				values = [unsup_acc[dataset][gkey][names[i]]]
				ax.plot(keys, values, '^ ', label=names[i], lw=3, markersize=markersize, c=colors[i])

		for i in range(len(new_modes)):
			keys = sorted(datan[i].keys())
			values = np.array([datan[i][tmp][0] for tmp in keys])
			stde = np.array([datan[i][tmp][1] for tmp in keys])

			keys = np.array([new_costs[i][1]*tmp+new_costs[i][0] for j, tmp in enumerate(keys)])
			ax.plot(keys, values, 'o--', label=new_names[i], lw=3, markersize=markersize, c=new_colors[i])
			ax.fill_between(keys, values-stde, values+stde, alpha=0.5, color=new_colors[i])
	ax.set_xlabel('# Attribute Annotations per class', fontsize=25)
	ax.set_ylabel('Top-1 Accuracy', fontsize=25)

	handles, labels = ax.get_legend_handles_labels()
	handles = handles[:-2]+handles[-1:]+handles[-2:-1]
	labels = labels[:-2]+labels[-1:]+labels[-2:-1]
	ax.legend(handles, labels, prop={'size': 20})

	ax.tick_params(labelsize=20)

	if generalized:
		ax.set_title(dataset+": Harmonic generalized accuracy", fontsize=30)
	else:
		ax.set_title(dataset+": Unseen accuracy", fontsize=30)

	if oneshot and dataset=="SUN":
		if generalized:
			ax.set_ylim(35, 40)
		else:
			ax.set_ylim(50, 60)

	if not isdir(odir):
		mkdir(odir)
	plt.savefig(join(odir, 'acc_'+dataset+'_'+str(generalized)+'.png'))
