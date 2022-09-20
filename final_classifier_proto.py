import torch
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label


class CLASSIFIER:
    def __init__(self, _train_X, _train_Y, _test_seen_X, _test_seen_Y, _test_novel_X, _test_novel_Y, seenclasses, novelclasses, _nclass, class_names):
        self.test_seen_feature = _test_seen_X.cpu().numpy()
        self.test_seen_label = _test_seen_Y.cpu().numpy()
        self.test_novel_feature = _test_novel_X.cpu().numpy()
        self.test_novel_label = _test_novel_Y.cpu().numpy()

        self.class_names = class_names

        self.seenclasses = seenclasses.cpu().numpy()
        self.novelclasses = novelclasses.cpu().numpy()

        self.nclass = _nclass

        self.train_X = normalize(_train_X)
        self.train_Y = _train_Y
        self.test_seen_feature = normalize(self.test_seen_feature)
        self.test_novel_feature = normalize(self.test_novel_feature)

        self.new_novelclass = self.novelclasses
        self.new_allclass = np.concatenate([self.seenclasses, self.novelclasses])

    def compute_per_class_acc(self):
        seen_dist = pairwise_distances(self.test_seen_feature, self.train_X)
        unseen_dist = pairwise_distances(self.test_novel_feature, self.train_X)
        data = []
        for factor in range(50, 200, 4):
            # unseen accuracy
            pred = self.new_novelclass[np.argmin(unseen_dist[:, self.new_novelclass], axis=1)]
            true = self.test_novel_label
            unseen_mat = np.zeros((self.nclass, self.nclass))
            for i, j in zip(true, pred):
                unseen_mat[i, j]+=1
            if factor==50:
                with np.errstate(divide='ignore', invalid='ignore'):
                    pcacc = np.diag(unseen_mat)/np.sum(unseen_mat, axis=1)
                valid_inds = np.argwhere(np.logical_not(np.isnan(pcacc)))[:, 0]
                classes = [self.class_names[ind] for ind in valid_inds]
                newinds = np.argsort(classes)
                valid_inds = valid_inds[newinds]

            with np.errstate(divide='ignore', invalid='ignore'):
                unseen_acc = np.nanmean(np.diag(unseen_mat)/np.sum(unseen_mat, axis=1))

            # unseen gen accuracy
            mask = np.ones((1, self.nclass))
            for i in self.novelclasses.tolist():
                mask[0, i] = factor/100

            unseen_dist_tmp = unseen_dist*mask
            pred = self.new_allclass[np.argmin(unseen_dist_tmp[:, self.new_allclass], axis=1)]
            true = self.test_novel_label
            unseengen_mat = np.zeros((self.nclass, self.nclass))
            for i, j in zip(true, pred):
                unseengen_mat[i, j]+=1
            with np.errstate(divide='ignore', invalid='ignore'):
                unseengen_acc = np.nanmean(np.diag(unseengen_mat)/np.sum(unseengen_mat, axis=1))

            seen_dist_tmp = seen_dist*mask
            pred = self.new_allclass[np.argmin(seen_dist_tmp[:, self.new_allclass], axis=1)]
            true = self.test_seen_label
            seengen_mat = np.zeros((self.nclass, self.nclass))
            for i, j in zip(true, pred):
                seengen_mat[i, j]+=1
            with np.errstate(divide='ignore', invalid='ignore'):
                seengen_acc = np.nanmean(np.diag(seengen_mat)/np.sum(seengen_mat, axis=1))

            hmacc = 2*unseengen_acc*seengen_acc/(unseengen_acc+seengen_acc)
            # print(unseen_acc, unseengen_acc, seengen_acc, hmacc)
            data.append([unseen_acc, unseengen_acc, seengen_acc, hmacc])
        data = np.array(data)
        return data[np.argmax(data[:, 3])]
