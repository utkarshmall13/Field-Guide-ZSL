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
if generalized:
	gind = 3

idir = '../accuracies'
odir = 'notaxo'

files = listdir(idir)

modes = ['acc_'+dataset+'__False_']
names = ['Traditional ZSL']
colors = [palette[0]]

new_modes = [
	'acc_'+dataset+'__Falseinteractive_siblings+',
	'acc_'+dataset+'__Falseinteractive_all+',
]

new_names = [
	'Sibling-variance',
	'Sibling-variance w/o taxonomy',
]
new_colors = [palette[2], palette[8]]


if dataset == 'SUN':
	costs = [(0, 102)]
elif dataset == 'CUB':
	costs = [(0, 28)]
elif dataset == 'APY':
	costs = [(0, 64)]
elif dataset == 'AWA2' or dataset == 'AWA3':
	costs = [(0, 85)]
new_costs = [(1, 1), (1, 1), (1, 1), (1, 1), (0, 1), (0, 1), (0, 1)]

data = [{} for tmp in range(len(modes))]
datan = [{} for tmp in range(len(new_modes))]


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

print(data)
print(datan)
markersize = 20
with plt.style.context('seaborn'):
	plt.figure(figsize=(10, 10))
	plt.tight_layout()
	for j in range(1, 2):
		plt.subplot(1, 1, j+1-1)
		for i in range(len(modes)):
			keys = sorted(data[i].keys())
			values = [data[i][tmp][0] for tmp in keys]
			stde = np.array([data[i][tmp][1] for tmp in keys])
			keys = np.array([np.ceil(costs[i][1]*tmp+costs[i][0]) for j, tmp in enumerate(keys)])
			plt.errorbar(keys, values, stde, ecolor=colors[i], fmt='^ ', label=names[i], lw=3, markersize=markersize, c=colors[i])

		for i in range(len(new_modes)):
			keys = sorted(datan[i].keys())
			values = [datan[i][tmp][0] for tmp in keys]
			stde = np.array([datan[i][tmp][1] for tmp in keys])
			keys = np.array([new_costs[i][1]*tmp+new_costs[i][0] for j, tmp in enumerate(keys)])
			plt.plot(keys, values, 'o--', label=new_names[i], lw=3, markersize=markersize, c=new_colors[i])
			plt.fill_between(keys, values-stde, values+stde, alpha=0.5, color=new_colors[i])

	plt.xlabel('# Attribute Annotations per class', fontsize=25)
	plt.ylabel('Top-1 Accuracy', fontsize=25)
	plt.legend(prop={'size': 20})
	plt.tick_params(labelsize=20)

	if generalized:
		plt.title(dataset+": Harmonic generalized accuracy", fontsize=30)
	else:
		plt.title(dataset+": Unseen accuracy", fontsize=30)

	if not isdir(odir):
		mkdir(odir)
	plt.savefig(join(odir, 'acc_'+dataset+'_'+str(generalized)+'.png'))
