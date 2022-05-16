# vaemodel
# Modified code from https://github.com/edgarschnfld/CADA-VAE-PyTorch
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import DATA_LOADER as dataloader
import final_classifier_proto as classifier_proto
import models
import numpy as np
from numpy.linalg import norm
from os.path import isdir, join
from os import mkdir


def normalize(vec):
    return vec/norm(vec)


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction = nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


class Model(nn.Module):
    def __init__(self, hyperparameters):
        super(Model, self).__init__()

        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources = ['resnet_features', self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.fraction = hyperparameters['fraction']
        if 'generalized' in hyperparameters:
            self.generalized = hyperparameters['generalized']
        else:
            self.generalized = False
        self.classifier_batch_size = 32
        self.img_seen_samples = hyperparameters['samples_per_class'][self.DATASET][0]
        self.att_seen_samples = hyperparameters['samples_per_class'][self.DATASET][1]
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
        self.reco_loss_function = hyperparameters['loss']
        self.mode = hyperparameters['mode']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']

        self.test_mode = hyperparameters['test_mode']
        self.train_mode = hyperparameters['train_mode']
        self.dataset = dataloader(self.DATASET, copy.deepcopy(self.auxiliary_data_source), device=self.device, discrete=hyperparameters["discrete"], mode=hyperparameters["mode"], run=hyperparameters['run'], fraction=self.fraction, test_mode=self.test_mode)
        self.discrete = hyperparameters['discrete']
        self.run = hyperparameters['run']

        if self.DATASET=='CUB':
            self.num_classes=200
            self.num_novel_classes = 50
        elif self.DATASET=='SUN':
            self.num_classes=717
            self.num_novel_classes = 72
        elif self.DATASET=='AWA1' or self.DATASET=='AWA2':
            self.num_classes=50
            self.num_novel_classes = 10
        elif self.DATASET=='APY' or self.DATASET=='APY':
            self.num_classes=32
            self.num_novel_classes = 12

        feature_dimensions = [2048, self.dataset.aux_data.size(1)]

        # Here, the encoders and decoders for all modalities are created and put into dict
        self.encoder = {}

        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.encoder[datatype] = models.encoder_template(dim, self.latent_size, self.hidden_size_rule[datatype], self.device)
            print("Dimension of "+str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size, dim, self.hidden_size_rule[datatype], self.device)

        # An optimizer for all encoders and decoders is defined here
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize += list(self.encoder[datatype].parameters())
            parameters_to_optimize += list(self.decoder[datatype].parameters())

        self.optimizer = optim.Adam(parameters_to_optimize, lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        if self.reco_loss_function=='l2':
            self.reconstruction_criterion = nn.MSELoss(reduction='sum')

        elif self.reco_loss_function=='l1':
            self.reconstruction_criterion = nn.L1Loss(reduction='sum')

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self, label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label==classes[i]] = i

        return mapped_label

    def trainstep(self, img, att):
        if self.train_mode=='missingatts':
            zeros = np.random.choice(len(self.dataset.new_attgrpcum)-1, (len(self.dataset.new_attgrpcum)-1)//2, replace=False)
            for zero in zeros:
                att[:, self.dataset.new_attgrpcum[zero]:self.dataset.new_attgrpcum[zero+1]] = 0.0
        ##############################################
        # Encode image features and additional
        # features
        ##############################################
        mu_img, logvar_img = self.encoder['resnet_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        ##############################################
        # Reconstruct inputs
        ##############################################

        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)
        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) + self.reconstruction_criterion(att_from_att, att)

        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)

        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) + self.reconstruction_criterion(att_from_img, att)

        ##############################################
        # KL-Divergence
        ##############################################

        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

        ##############################################
        # Distribution Alignment
        ##############################################
        distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))
        distance = distance.sum()

        ##############################################
        # scale the loss terms according to the warmup
        # schedule
        ##############################################

        f1 = 1.0*(self.current_epoch - self.warmup['cross_reconstruction']['start_epoch'])/(1.0*(self.warmup['cross_reconstruction']['end_epoch']- self.warmup['cross_reconstruction']['start_epoch']))
        f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
        cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1, 0), self.warmup['cross_reconstruction']['factor'])])

        f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / (1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])

        f3 = 1.0*(self.current_epoch - self.warmup['distance']['start_epoch'])/(1.0*(self.warmup['distance']['end_epoch'] - self.warmup['distance']['start_epoch']))
        f3 = f3*(1.0*self.warmup['distance']['factor'])
        distance_factor = torch.cuda.FloatTensor([min(max(f3, 0), self.warmup['distance']['factor'])])

        ##############################################
        # Put the loss together and call the optimizer
        ##############################################

        self.optimizer.zero_grad()
        loss = reconstruction_loss - beta * KLD

        if cross_reconstruction_loss>0:
            loss += cross_reconstruction_factor*cross_reconstruction_loss
        if distance_factor >0:
            loss += distance_factor*distance

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_vae(self):
        losses = []

        self.dataset.novelclasses =self.dataset.novelclasses.long().cuda()
        self.dataset.seenclasses =self.dataset.seenclasses.long().cuda()
        # leave both statements
        self.train()
        self.reparameterize_with_noise = True

        print('train for reconstruction')
        for epoch in range(0, self.nepoch):
            self.current_epoch = epoch
            i=-1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i+=1

                label, data_from_modalities = self.dataset.next_batch(self.batch_size)

                label= label.long().to(self.device)
                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(self.device)
                    data_from_modalities[j].requires_grad = False

                loss = self.trainstep(data_from_modalities[0], data_from_modalities[1])
                if i%50==0:
                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+ ' | loss ' + str(loss)[:5])
                if i%50==0 and i>0:
                    losses.append(loss)

        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()

        return losses

    def train_proto_classifier(self, show_plots=False):
        if self.num_shots > 0:
            self.dataset.transfer_features(self.num_shots, num_queries='num_features')

        if 'interlatent' in self.test_mode or 'overlatent' in self.test_mode or 'interalllatent' in self.test_mode or 'overalllatent' in self.test_mode:
            self.dataset.interlatent_transfer(self.encoder[self.auxiliary_data_source], self.device)
        if 'oneshotactive' in self.test_mode:
            feats = self.dataset.data['train_unseen']['resnet_features']
            oneshot_encoded_proto = self.encoder['resnet_features'](feats)[0].detach()
            oneshot_decoded_proto = self.decoder[self.auxiliary_data_source](oneshot_encoded_proto).detach().cpu().numpy()
            oneshot_encoded_proto = oneshot_encoded_proto.cpu().numpy()
            self.dataset.oneshot_transfer(self.encoder[self.auxiliary_data_source], oneshot_encoded_proto, oneshot_decoded_proto, self.device)

        cls_seenclasses = self.dataset.seenclasses
        cls_novelclasses = self.dataset.novelclasses

        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels']

        novelclass_aux_data = self.dataset.novelclass_aux_data
        seenclass_aux_data = self.dataset.seenclass_aux_data

        novel_corresponding_labels = self.dataset.novelclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)

        # The resnet_features for testing the classifier are loaded here
        novel_test_feat = self.dataset.data['test_unseen']['resnet_features']  # self.dataset.test_novel_feature.to(self.device)
        seen_test_feat = self.dataset.data['test_seen']['resnet_features']  # self.dataset.test_seen_feature.to(self.device)
        test_seen_label = self.dataset.data['test_seen']['labels']  # self.dataset.test_seen_label.to(self.device)
        test_novel_label = self.dataset.data['test_unseen']['labels']  # self.dataset.test_novel_label.to(self.device)

        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']

        # in ZSL mode:
        if not self.generalized:
            novel_corresponding_labels = self.map_label(novel_corresponding_labels, novel_corresponding_labels)

            if self.num_shots > 0:
                train_unseen_label = self.map_label(train_unseen_label, cls_novelclasses)
            test_novel_label = self.map_label(test_novel_label, cls_novelclasses)
            cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)

        if self.generalized:
            print('mode: gzsl')
        else:
            print('mode: zsl')

        with torch.no_grad():
            self.reparameterize_with_noise = False

            mu1, var1 = self.encoder['resnet_features'](novel_test_feat)
            test_novel_X = mu1
            test_novel_Y = test_novel_label.to(self.device)

            mu2, var2 = self.encoder['resnet_features'](seen_test_feat)
            test_seen_X = mu2
            test_seen_Y = test_seen_label.to(self.device)

            self.reparameterize_with_noise = True

            img_seen_feat, img_seen_label = train_seen_feat, train_seen_label
            img_unseen_feat, img_unseen_label = train_unseen_feat, train_unseen_label
            att_unseen_feat, att_unseen_label = novelclass_aux_data, novel_corresponding_labels
            att_seen_feat, att_seen_label = seenclass_aux_data, seen_corresponding_labels

            def convert_datapoints_to_z(features, encoder):
                if features is not None and features.size(0) != 0:
                    mu_, logvar_ = encoder(features)
                    return mu_
                else:
                    return torch.cuda.FloatTensor([])

            z_seen_img = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
            z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])
            # print(z_unseen_img, img_unseen_label)

            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])

            # train_Z = [z_seen_img, z_unseen_img, z_seen_att, z_unseen_att]
            # train_L = [img_seen_label, img_unseen_label, att_seen_label, att_unseen_label]

            seen_img_protos = []
            seen_att_protos = []
            unseen_img_protos = []
            unseen_att_protos = []

            for i in range(self.num_classes):
                if img_seen_label is not None and len(np.argwhere(img_seen_label.cpu().numpy()==i)[:, 0])>0:
                    seen_img_protos.append(torch.mean(z_seen_img[np.argwhere(img_seen_label.cpu().numpy()==i)[:, 0]], dim=0).cpu().numpy())
                else:
                    seen_img_protos.append(None)

                if att_seen_label is not None and len(np.argwhere(att_seen_label.cpu().numpy()==i)[:, 0])>0:
                    seen_att_protos.append(torch.mean(z_seen_att[np.argwhere(att_seen_label.cpu().numpy()==i)[:, 0]], dim=0).cpu().numpy())
                else:
                    seen_att_protos.append(None)

                if img_unseen_label is not None and len(np.argwhere(img_unseen_label.cpu().numpy()==i)[:, 0])>0:
                    unseen_img_protos.append(torch.mean(z_unseen_img[np.argwhere(img_unseen_label.cpu().numpy()==i)[:, 0]], dim=0).cpu().numpy())
                else:
                    unseen_img_protos.append(None)

                if att_unseen_label is not None and len(np.argwhere(att_unseen_label.cpu().numpy()==i)[:, 0])>0:
                    unseen_att_protos.append(torch.mean(z_unseen_att[np.argwhere(att_unseen_label.cpu().numpy()==i)[:, 0]], dim=0).cpu().numpy())
                else:
                    unseen_att_protos.append(None)

            protos = np.array([np.mean([normalize(tmp) for tmp in cls if tmp is not None], axis=0) for cls in zip(seen_img_protos, seen_att_protos, unseen_img_protos, unseen_att_protos)])
            train_X = protos
            train_Y = np.array([i for i in range(self.num_classes)])
            # print(train_X.shape)

        cls = classifier_proto.CLASSIFIER(train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X, test_novel_Y, cls_seenclasses, cls_novelclasses, self.num_classes, self.dataset.class_names)
        accs = cls.compute_per_class_acc()
        print('Accuracies:', accs)

        if self.num_shots==0:
            odir = 'accuracies'
        else:
            odir = 'accuracies_fewshot'
        if not isdir(odir):
            mkdir(odir)
        if self.test_mode=='standard' and self.train_mode=='standard':
            np.savetxt(join(odir, 'acc_'+self.DATASET+'_'+self.mode+'_'+str(self.discrete)+'_'+str(self.run)+'_'+str(self.fraction)+'.txt'), [accs], delimiter=',')
        elif self.train_mode=='standard':
            np.savetxt(join(odir, 'acc_'+self.DATASET+'_'+self.mode+'_'+str(self.discrete)+self.test_mode+'_'+str(self.run)+'_'+str(self.fraction)+'.txt'), [accs], delimiter=',')
        else:
            np.savetxt(join(odir, 'acc_'+self.DATASET+'_'+self.mode+'_'+str(self.discrete)+self.train_mode+'_'+self.test_mode+'_'+str(self.run)+'_'+str(self.fraction)+'.txt'), [accs], delimiter=',')
