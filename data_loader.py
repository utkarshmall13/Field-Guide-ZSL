# Modified code from https://github.com/edgarschnfld/CADA-VAE-PyTorch

import numpy as np
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import copy
from sklearn.metrics import pairwise_distances
from numpy.linalg import norm
from os.path import join


################################################################################
# Pickle functions
def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo)
    return dic


def inpickle(diction, file):
    with open(file, 'wb') as fo:
        pickle.dump(diction, fo)
################################################################################


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label


class DATA_LOADER(object):
    def __init__(self, dataset, aux_datasource, device='cuda', discrete=False, mode='', run=0, fraction=1.0, test_mode='standard'):
        folder = str(Path(os.getcwd()))
        if folder[-5:] == 'model':
            project_directory = Path(os.getcwd()).parent
        else:
            project_directory = folder

        print('Project Directory:', project_directory)
        data_path = join(str(project_directory), 'data')
        print('Data Path:', data_path)
        sys.path.append(data_path)
        self.mode = mode
        self.run = run
        self.fraction = fraction
        self.data_path = data_path
        self.device = device
        self.dataset = dataset
        self.auxiliary_data_source = aux_datasource
        self.discrete = discrete
        self.test_mode = test_mode

        self.all_data_sources = ['resnet_features'] + [self.auxiliary_data_source]

        if self.dataset == 'CUB':
            self.datadir = join(self.data_path, 'CUB')
        elif self.dataset == 'SUN':
            self.datadir = join(self.data_path, 'SUN')
        elif self.dataset == 'AWA2':
            self.datadir = join(self.data_path, 'AWA2')

        self.read_matdataset()
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        #####################################################################
        # gets batch from train_feature
        #####################################################################
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.data['train_seen']['resnet_features'][idx]
        batch_label = self.data['train_seen']['labels'][idx]
        batch_att = self.aux_data[batch_label]
        return batch_label, [batch_feature, batch_att]

    def get_attribute_grouping(self, attribute_names):
        #####################################################################
        # gets starting and ending index of groups of attributes and the names of the groups.
        # If lets say attributes are grouped as a, a, a, b, b, c
        # It returns [0, 3, 5, 6] and [a, b, c]
        #####################################################################
        attgrpcum = [0]
        attnames = attribute_names
        curr = attnames[0].split('::')[0]
        grpatt = [curr]
        for i, attname in enumerate(attnames):
            if attname.split('::')[0]!=curr:
                attgrpcum.append(i)
                curr = attname.split('::')[0]
                grpatt.append(curr)
        grpatt = np.array(grpatt)
        attgrpcum.append(len(attnames))
        return attgrpcum, grpatt

    def read_matdataset(self):
        path = join(self.datadir, self.dataset+'_r101.pkl')
        print('Data file path: ', path)
        matcontent = unpickle(path)

        self.class_names = matcontent['class_names']
        feature = matcontent['features']
        label = matcontent['labels']

        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc']
        # unused so commented
        # train_loc = matcontent['train_loc']
        # val_unseen_loc = matcontent['val_loc']
        test_seen_loc = matcontent['test_seen_loc']
        test_unseen_loc = matcontent['test_unseen_loc']

        # top improve randomization over different runs with shuffle attributes in a randomly.
        # the order of shuffle is is deterministically random for reproducibility (loaded from order..npy files)
        attgrpcum, grpatt = self.get_attribute_grouping(matcontent['attribute_names'])
        order = np.load(join('att_orders', self.dataset, 'order'+str(self.run)+'.npy'))
        self.new_attgrpcum = [0]
        for ord in order:
            self.new_attgrpcum.append(self.new_attgrpcum[-1]+attgrpcum[ord+1]-attgrpcum[ord])
        self.grpatt = grpatt[order]

        if self.mode=='glovar':
            # When reducing the number of attributes, use the attributes with most variance
            new_original_attributes = []
            for ind in range(len(order)):
                new_original_attributes.append(matcontent['original_attributes'][:, attgrpcum[ind]:attgrpcum[ind+1]].T)
            new_original_attributes = np.concatenate(new_original_attributes, axis=0).T

            std = np.std(new_original_attributes, axis=0)
            mean_std = []
            for j in range(len(order)):
                mean_std.append(np.max(std[self.new_attgrpcum[j]:self.new_attgrpcum[j+1]]))
            mean_std = np.array(mean_std)
            diff = np.argsort(mean_std)[::-1]
            diff = diff[:int(np.ceil(len(diff)*self.fraction))]

            new_original_attributes = []
            for ind in diff:
                new_original_attributes.append(matcontent['original_attributes'][:, attgrpcum[ind]:attgrpcum[ind+1]].T)
            new_original_attributes = np.concatenate(new_original_attributes, axis=0).T
            self.new_original_attributes = new_original_attributes

        else:
            new_original_attributes = []
            for ind in order[:int(np.ceil(len(order)*self.fraction))]:
                new_original_attributes.append(matcontent['original_attributes'][:, attgrpcum[ind]:attgrpcum[ind+1]].T)
            new_original_attributes = np.concatenate(new_original_attributes, axis=0).T
            self.new_original_attributes = new_original_attributes

        if self.discrete:
            dis = preprocessing.normalize(((new_original_attributes/100.0)>0.5).astype(float))
        else:
            dis = preprocessing.normalize(new_original_attributes)

        # Option for different acquisition functions
        if self.test_mode=='standard':
            self.aux_data = torch.from_numpy(dis).float().to(self.device)

        elif 'expert' in self.test_mode:
            new_dis = copy.deepcopy(dis)
            numchanges = int(self.test_mode.split('+')[1])
            nns = np.load('experts/CUB/nn_'+str(self.run)+'.npy')
            diffs = np.load('experts/CUB/diff_'+str(self.run)+'.npy')

            novel_cinds = np.unique(label[test_unseen_loc])
            for i, ind in enumerate(novel_cinds):
                new_dis[ind, :] = new_dis[nns[i], :]
                for j, jnd in enumerate(diffs[i]):
                    if j==numchanges:
                        break
                    ll = self.new_attgrpcum[jnd]
                    r = self.new_attgrpcum[jnd+1]
                    new_dis[ind, ll:r] = dis[ind, ll:r]
            self.aux_data = torch.from_numpy(preprocessing.normalize(new_dis)).float().to(self.device)

        elif 'random' in self.test_mode:
            new_dis = copy.deepcopy(dis)
            numchanges = int(self.test_mode.split('+')[1])
            nns = np.load('data/'+self.dataset+'/'+self.dataset+'_nns.npy')
            novel_cinds = np.unique(label[test_unseen_loc])
            for i, ind in enumerate(novel_cinds):
                new_dis[ind, :] = new_dis[nns[i], :]
                ranchoice = np.random.choice(len(self.new_attgrpcum)-1, numchanges)
                rancrhoice = np.random.choice(len(self.new_attgrpcum)-1, numchanges, replace=False)
                frac = numchanges/(len(self.new_attgrpcum)-1)
                if frac < 0.5:
                    frac = 0
                ranchoice = np.concatenate((rancrhoice[:int(len(ranchoice)*frac)], ranchoice[int(len(ranchoice)*frac):]))

                for j in ranchoice:
                    ll = self.new_attgrpcum[j]
                    r = self.new_attgrpcum[j+1]
                    new_dis[ind, ll:r] = dis[ind, ll:r]
            self.aux_data = torch.from_numpy(preprocessing.normalize(new_dis)).float().to(self.device)

        elif 'interactive' in self.test_mode:
            # interactive_siblings is the sibling-variance method from the paper
            self.parent_class_names = np.load('data/'+self.dataset+'/'+self.dataset+'_parent.npy')

            new_dis = copy.deepcopy(dis)
            numchanges = int(self.test_mode.split('+')[1])
            nns = np.load('data/'+self.dataset+'/'+self.dataset+'_nns.npy')

            novel_cinds = np.unique(label[test_unseen_loc])
            for i, ind in enumerate(novel_cinds):
                parent_ind = self.parent_class_names[nns[i]]
                siblings = np.argwhere(self.parent_class_names==parent_ind)[:, 0]
                notsiblings = np.argwhere(self.parent_class_names>-1)[:, 0]

                attributes = dis[siblings]
                std = np.std(attributes, axis=0)

                notattributes = dis[notsiblings]
                notstd = np.std(notattributes, axis=0)
                mean_std = []
                mean_notstd = []
                for j in range(len(self.new_attgrpcum)-1):
                    mean_std.append(np.max(std[self.new_attgrpcum[j]:self.new_attgrpcum[j+1]]))
                    mean_notstd.append(np.max(notstd[self.new_attgrpcum[j]:self.new_attgrpcum[j+1]]))
                mean_std = np.array(mean_std)
                mean_notstd = np.array(mean_notstd)
                if '_siblings' in self.test_mode:
                    diff = np.argsort(mean_std)[::-1]
                elif '_all' in self.test_mode:
                    diff = np.argsort(mean_notstd)[::-1]
                elif '_relative' in self.test_mode:
                    diff = np.argsort(mean_std/mean_notstd)[::-1]
                new_dis[ind, :] = new_dis[nns[i], :]
                for j, jnd in enumerate(diff):
                    if j==numchanges:
                        break
                    ll = self.new_attgrpcum[jnd]
                    r = self.new_attgrpcum[jnd+1]
                    new_dis[ind, ll:r] = dis[ind, ll:r]
            self.aux_data = torch.from_numpy(preprocessing.normalize(new_dis)).float().to(self.device)

        elif 'interlatent' in self.test_mode or 'overlatent' in self.test_mode or 'oneshotactive' in self.test_mode or 'interalllatent' in self.test_mode or 'overlatent' in self.test_mode:
            self.aux_data = torch.from_numpy(preprocessing.normalize(dis)).float().to(self.device)
            self.novel_cinds = np.unique(label[test_unseen_loc])
            self.parent_class_names = np.load('data/'+self.dataset+'/'+self.dataset+'_parent.npy')

        scaler = preprocessing.MinMaxScaler()

        train_feature = scaler.fit_transform(feature[trainval_loc])
        test_seen_feature = scaler.transform(feature[test_seen_loc])
        test_unseen_feature = scaler.transform(feature[test_unseen_loc])

        train_feature = torch.from_numpy(train_feature).float().to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature).float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float().to(self.device)

        train_label = torch.from_numpy(label[trainval_loc]).long().to(self.device)
        self.train_label = train_label
        test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long().to(self.device)
        self.test_unseen_label = test_unseen_label
        test_seen_label = torch.from_numpy(label[test_seen_loc]).long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.novelclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.novelclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]

        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label

        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]

    def interlatent_transfer(self, model, device):
        if 'interlatent' in self.test_mode or 'interalllatent' in self.test_mode:
            dis = self.aux_data.cpu().numpy()
            new_dis = copy.deepcopy(dis)
            numchanges = int(self.test_mode.split('+')[1])
            nns = np.load('data/'+self.dataset+'/'+self.dataset+'_nns.npy')

            number_of_inds = int(self.test_mode.split('+')[1])

            novel_cinds = self.novel_cinds
            for i, ind in enumerate(novel_cinds):
                parent_ind = self.parent_class_names[nns[i]]
                if 'interlatent' in self.test_mode:
                    siblings = np.argwhere(self.parent_class_names==parent_ind)[:, 0]
                if 'interalllatent' in self.test_mode:
                    siblings = np.argwhere(self.parent_class_names>-1)[:, 0]

                attributes = dis[siblings]
                std = np.std(attributes, axis=0)

                useful_inds = []
                for num in range(number_of_inds):
                    x = copy.deepcopy(new_dis[nns[i]:nns[i]+1, :])
                    for jnd in useful_inds:
                        ll = self.new_attgrpcum[jnd]
                        r = self.new_attgrpcum[jnd+1]
                        x[0, ll:r] = dis[ind, ll:r]
                    x = preprocessing.normalize(x)
                    x = torch.tensor(x, requires_grad=True).to(device)

                    changes = []
                    for dim in range(64):
                        changes.append(torch.autograd.grad(model(x)[0][0, dim], x)[0].cpu().numpy()[0])
                    changes = np.array(changes)
                    changes = norm(changes, axis=0)

                    changes = changes*std

                    mean_change = []
                    for j in range(len(self.new_attgrpcum)-1):
                        mean_change.append(np.max(changes[self.new_attgrpcum[j]:self.new_attgrpcum[j+1]]))
                    mean_change = np.array(mean_change)
                    diffs = np.argsort(mean_change)[::-1]
                    firstnotalreadyin = [tmp for tmp in diffs if tmp not in useful_inds]
                    useful_inds.append(firstnotalreadyin[0])

                new_dis[ind, :] = new_dis[nns[i], :]
                for jnd in useful_inds:
                    ll = self.new_attgrpcum[jnd]
                    r = self.new_attgrpcum[jnd+1]
                    new_dis[ind, ll:r] = dis[ind, ll:r]
            self.aux_data = torch.from_numpy(preprocessing.normalize(new_dis)).float().to(self.device)
        elif 'overlatent' in self.test_mode or 'overalllatent' in self.test_mode:
            dis = self.aux_data.cpu().numpy()
            new_dis = copy.deepcopy(dis)
            numchanges = int(self.test_mode.split('+')[1])
            nns = np.load('data/'+self.dataset+'/'+self.dataset+'_nns.npy')

            number_of_inds = int(self.test_mode.split('+')[1])

            novel_cinds = self.novel_cinds
            for i, ind in enumerate(novel_cinds):
                parent_ind = self.parent_class_names[nns[i]]
                if 'overlatent' in self.test_mode:
                    siblings = np.argwhere(self.parent_class_names==parent_ind)[:, 0]
                if 'overalllatent' in self.test_mode:
                    siblings = np.argwhere(self.parent_class_names>-1)[:, 0]
                attributes = dis[siblings]

                x = copy.deepcopy(new_dis[nns[i]:nns[i]+1, :])
                x = np.repeat(x, len(siblings), axis=0)

                total_var = []
                for j in range(len(self.new_attgrpcum)-1):
                    ll = self.new_attgrpcum[j]
                    r = self.new_attgrpcum[j+1]

                    tempx = copy.deepcopy(x)
                    tempx[:, ll:r] = attributes[:, ll:r]
                    tempx = torch.from_numpy(preprocessing.normalize(tempx)).float().to(self.device)
                    tempx = model(tempx)[0].detach().cpu().numpy()
                    total_var.append(np.mean(np.power(pairwise_distances(tempx, np.mean(tempx, axis=0, keepdims=True)), 2)))
                useful_inds = np.argsort(total_var)[::-1][:numchanges]

                new_dis[ind, :] = new_dis[nns[i], :]
                for jnd in useful_inds:
                    ll = self.new_attgrpcum[jnd]
                    r = self.new_attgrpcum[jnd+1]
                    new_dis[ind, ll:r] = dis[ind, ll:r]
            self.aux_data = torch.from_numpy(preprocessing.normalize(new_dis)).float().to(self.device)

        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[self.train_label]
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[self.test_unseen_label]
        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]

    def oneshot_transfer(self, model, oneshot_encoded_proto, oneshot_decoded_proto, device):
        if 'oneshotactive' in self.test_mode:
            dis = self.aux_data.cpu().numpy()
            new_dis = copy.deepcopy(dis)
            numchanges = int(self.test_mode.split('+')[1])
            nns = np.load('data/'+self.dataset+'/'+self.dataset+'_nns.npy')
            oneshot_decoded_proto = preprocessing.normalize(oneshot_decoded_proto)

            novel_cinds = self.novel_cinds
            for i, ind in enumerate(novel_cinds):
                useful_inds = []
                for num in range(numchanges):
                    x = copy.deepcopy(new_dis[nns[i]:nns[i]+1, :])
                    for jnd in useful_inds:
                        ll = self.new_attgrpcum[jnd]
                        r = self.new_attgrpcum[jnd+1]
                        x[0, ll:r] = dis[ind, ll:r]
                    x = preprocessing.normalize(x)

                    x = np.repeat(x, len(self.new_attgrpcum)-1, axis=0)
                    for j in range(len(self.new_attgrpcum)-1):
                        ll = self.new_attgrpcum[j]
                        r = self.new_attgrpcum[j+1]
                        x[j, ll:r] = oneshot_decoded_proto[i, ll:r]
                    x = torch.from_numpy(preprocessing.normalize(x)).float().to(self.device)
                    x = model(x)[0].detach().cpu().numpy()

                    # same = copy.deepcopy(x[len(x)-1:, :])
                    # x = copy.deepcopy(x[:len(x)-1, :])
                    # print(x.shape)

                    dists = pairwise_distances(x, oneshot_encoded_proto[i:i+1])
                    diffs = np.argsort(dists[:, 0])[:numchanges]

                    # dists = pairwise_distances(x, same)
                    # diffs = np.argsort(dists[:, 0])[:numchanges]
                    firstnotalreadyin = [tmp for tmp in diffs if tmp not in useful_inds]
                    useful_inds.append(firstnotalreadyin[0])

                new_dis[ind, :] = new_dis[nns[i], :]
                for jnd in useful_inds:
                    ll = self.new_attgrpcum[jnd]
                    r = self.new_attgrpcum[jnd+1]
                    new_dis[ind, ll:r] = dis[ind, ll:r]
            self.aux_data = torch.from_numpy(preprocessing.normalize(new_dis)).float().to(self.device)

        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[self.train_label]
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[self.test_unseen_label]
        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]

    def transfer_features(self, n, num_queries='num_features'):
        print('size before')
        print(self.data['test_unseen']['resnet_features'].size())
        print(self.data['train_seen']['resnet_features'].size())
        print(self.data['test_unseen'].keys())
        for i, s in enumerate(self.novelclasses):

            features_of_that_class = self.data['test_unseen']['resnet_features'][self.data['test_unseen']['labels']==s, :]

            if 'attributes' == self.auxiliary_data_source:
                attributes_of_that_class = self.data['test_unseen']['attributes'][self.data['test_unseen']['labels']==s, :]
                use_att = True
            else:
                use_att = False

            num_features = features_of_that_class.size(0)
            # indices = torch.randperm(num_features)
            indices = np.array(list(range(num_features)))

            if num_queries!='num_features':
                indices = indices[:n+num_queries]

            if i==0:
                new_train_unseen = features_of_that_class[indices[:n], :]
                if use_att:
                    new_train_unseen_att = attributes_of_that_class[indices[:n], :]
                new_train_unseen_label = s.repeat(n)
                new_test_unseen = features_of_that_class[indices[n:], :]
                new_test_unseen_label = s.repeat(len(indices[n:]))
            else:
                new_train_unseen = torch.cat((new_train_unseen, features_of_that_class[indices[:n], :]), dim=0)
                new_train_unseen_label = torch.cat((new_train_unseen_label, s.repeat(n)), dim=0)
                new_test_unseen = torch.cat((new_test_unseen, features_of_that_class[indices[n:], :]), dim=0)
                new_test_unseen_label = torch.cat((new_test_unseen_label, s.repeat(len(indices[n:]))), dim=0)

                if use_att:
                    new_train_unseen_att = torch.cat((new_train_unseen_att, attributes_of_that_class[indices[:n], :]), dim=0)

        print('new_test_unseen.size(): ', new_test_unseen.size())
        print('new_test_unseen_label.size(): ', new_test_unseen_label.size())
        print('new_train_unseen.size(): ', new_train_unseen.size())
        print('new_train_unseen_label.size(): ', new_train_unseen_label.size())
        print('>> num novel classes: ' + str(len(self.novelclasses)))

        #######
        ##
        #######

        self.data['test_unseen']['resnet_features'] = copy.deepcopy(new_test_unseen)

        self.data['test_unseen']['labels'] = copy.deepcopy(new_test_unseen_label)

        self.data['train_unseen']['resnet_features'] = copy.deepcopy(new_train_unseen)
        self.data['train_unseen']['labels'] = copy.deepcopy(new_train_unseen_label)
        self.ntrain_unseen = self.data['train_unseen']['resnet_features'].size(0)

        if use_att:
            self.data['train_unseen']['attributes'] = copy.deepcopy(new_train_unseen_att)
        ####
        self.data['train_seen_unseen_mixed'] = {}
        self.data['train_seen_unseen_mixed']['resnet_features'] = torch.cat((self.data['train_seen']['resnet_features'], self.data['train_unseen']['resnet_features']), dim=0)
        self.data['train_seen_unseen_mixed']['labels'] = torch.cat((self.data['train_seen']['labels'], self.data['train_unseen']['labels']), dim=0)

        self.ntrain_mixed = self.data['train_seen_unseen_mixed']['resnet_features'].size(0)

        if use_att:
            self.data['train_seen_unseen_mixed']['attributes'] = torch.cat((self.data['train_seen']['attributes'], self.data['train_unseen']['attributes']), dim=0)
