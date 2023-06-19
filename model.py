# File that defines all the models explored
import torch
from torch import nn
from layers import *

# BENDR model: convolutional + transformer
class BENDRClassification(nn.Module):

    def __init__(self, targets=2, samples=769, channels=20, 
                 encoder_h=512,
                 contextualizer_hidden=3076,
                 projection_head=False,
                 new_projection_layers=0, 
                 dropout=0., 
                 trial_embeddings=None, 
                 layer_drop=0, 
                 keep_layers=None,
                 mask_p_t=0.01, 
                 mask_p_c=0.005, 
                 mask_t_span=0.1, 
                 mask_c_span=0.1,
                 return_features=True, 
                 multi_gpu=False):

        super().__init__()
        self.samples = samples
        self.channels = channels
        self.return_features = return_features
        self.targets = targets
        self.num_features_for_classification = encoder_h
        self.make_new_classification_layer() # To add/remove the final linear layer with softmax activation
        self._init_state = self.state_dict()
        self.encoder_h = encoder_h
        self.contextualizer_hidden = contextualizer_hidden

        encoder = ConvEncoderBENDR(in_features=channels, encoder_h=encoder_h, dropout=dropout, projection_head=projection_head)
        encoded_samples = encoder.downsampling_factor(samples)

        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)
        contextualizer = BENDRContextualizer(encoder_h, hidden_feedforward=contextualizer_hidden, finetuning=True,
                                             mask_p_t=mask_p_t, mask_p_c=mask_p_c, layer_drop=layer_drop,
                                             mask_c_span=mask_c_span, dropout=dropout,
                                             mask_t_span=mask_t_span)

        self.encoder = nn.DataParallel(encoder, device_ids=1) if multi_gpu else encoder
        self.contextualizer = nn.DataParallel(contextualizer, device_ids=1) if multi_gpu else contextualizer

        self.trial_embeddings = nn.Embedding(trial_embeddings, encoder_h, scale_grad_by_freq=True) \
            if trial_embeddings is not None else trial_embeddings

    # @property
    def num_features_for_classification(self):
        return self.encoder_h

    def features_forward(self, *x):
        encoded = self.encoder(x[0])

        if self.trial_embeddings is not None and len(x) > 1:
            embeddings = self.trial_embeddings(x[-1])
            encoded += embeddings.unsqueeze(-1).expand_as(encoded)

        context = self.contextualizer(encoded)
        return context[:, :, -1]

    def classifier_forward(self, features):
        # return torch.sigmoid(self.classifier(features))
        return self.classifier(features)


    def forward(self, *x):
        features = self.features_forward(*x)
        if self.return_features:
            return self.classifier_forward(features), features
        else:
            return self.classifier_forward(features)

    def reset(self):
        self.load_state_dict(self._init_state)

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(unfreeze=not freeze)

    def load_contextualizer(self, contextualizer_file, freeze=False, strict=True):
        self.contextualizer.load(contextualizer_file, strict=strict)
        self.contextualizer.freeze_features(unfreeze=not freeze)

    def load_pretrained_modules(self, encoder_file, contextualizer_file, freeze_encoder=False,
                                freeze_contextualizer=False, freeze_position_conv=False,
                                freeze_mask_replacement=True, strict=False):
        self.load_encoder(encoder_file, freeze=freeze_encoder, strict=strict)
        self.load_contextualizer(contextualizer_file, freeze=freeze_contextualizer, strict=strict)
        self.contextualizer.mask_replacement.requires_grad = freeze_mask_replacement
        if freeze_position_conv:
            for p in self.contextualizer.relative_position.parameters():
                p.requires_grad = False

    def make_new_classification_layer(self):
        # Fnunction to distinct between the classification layer(s) and the rest of the network
        # This method is for implementing the classification side, so that methods like :py:meth:`freeze_features` works as intended.
        classifier = nn.Linear(self.num_features_for_classification, self.targets)
        nn.init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)
        """
        classifier = nn.Linear(self.num_features_for_classification, 1)
        nn.init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)
        """
        
    def freeze_features(self, unfreeze=False, freeze_classifier=False):
        # This method freezes (or un-freezes) all but the `classifier` layer. So that any further training does not (or
        # does if unfreeze=True) affect these weights.
        for param in self.parameters():
            param.requires_grad = unfreeze

        if isinstance(self.classifier, nn.Module) and not freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def load(self, filename, include_classifier=False, freeze_features=True):
        state_dict = torch.load(filename)
        if not include_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)
        if freeze_features:
            self.freeze_features()

    def save(self, filename, ignore_classifier=False):
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        print("Saving to {} ...".format(filename))
        torch.save(state_dict, filename)

# MAEEG model: convolutional + transformer + (linear+conv)
class MAEEG(nn.Module):

    def __init__(self, targets=2, samples=769, channels=20, 
                 encoder_h=512,
                 contextualizer_hidden=3076,
                 projection_head=False,
                 new_projection_layers=0, 
                 dropout=0., 
                 trial_embeddings=None, 
                 layer_drop=0, 
                 keep_layers=None,
                 mask_p_t=0.01, 
                 mask_p_c=0.005, 
                 mask_t_span=0.1, 
                 mask_c_span=0.1,
                 return_features=True, 
                 multi_gpu=False):

        super().__init__()
        self.samples = samples
        self.channels = channels
        self.return_features = return_features
        self.targets = targets
        self.num_features_for_classification = encoder_h
        self.make_new_classification_layer() # To add/remove the final linear layer with softmax activation
        self._init_state = self.state_dict()
        self.encoder_h = encoder_h
        self.contextualizer_hidden = contextualizer_hidden

        encoder = ConvEncoderBENDR(in_features=channels, encoder_h=encoder_h, dropout=dropout, projection_head=projection_head)
        encoded_samples = encoder.downsampling_factor(samples)

        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)
        contextualizer = BENDRContextualizer(encoder_h, hidden_feedforward=contextualizer_hidden, finetuning=True,
                                             mask_p_t=mask_p_t, mask_p_c=mask_p_c, layer_drop=layer_drop,
                                             mask_c_span=mask_c_span, dropout=dropout,
                                             mask_t_span=mask_t_span)

        self.encoder = nn.DataParallel(encoder, device_ids=1) if multi_gpu else encoder
        self.contextualizer = nn.DataParallel(contextualizer, device_ids=1) if multi_gpu else contextualizer

        self.trial_embeddings = nn.Embedding(trial_embeddings, encoder_h, scale_grad_by_freq=True) \
            if trial_embeddings is not None else trial_embeddings
        
        self.maeeg1 = nn.Linear(self.num_features_for_classification,self.samples)
        self.maeeg2 = nn.Conv1d(1, self.channels, 3, padding=1)

    # @property
    def num_features_for_classification(self):
        return self.encoder_h

    def features_forward(self, *x):
        encoded = self.encoder(x[0])

        if self.trial_embeddings is not None and len(x) > 1:
            embeddings = self.trial_embeddings(x[-1])
            encoded += embeddings.unsqueeze(-1).expand_as(encoded)

        context = self.contextualizer(encoded)
        return context[:, :, -1]

    def classifier_forward(self, features):
        # return torch.sigmoid(self.classifier(features))
        return self.classifier(features)

    def maeeg_forward(self, features): 
        maeeg1 = self.maeeg1(features)
        maeeg1 = torch.reshape(maeeg1, (maeeg1.shape[0], 1, maeeg1.shape[1]))
        maeeg2 = self.maeeg2(maeeg1)
        return maeeg2

# LINEAR model: convolutional + linear layer
class LinearHeadBENDR(nn.Module):

    # @property
    def num_features_for_classification(self):
        return self.encoder_h * self.pool_length

    def features_forward(self, x):
        x = self.encoder(x)
        x = self.enc_augment(x)
        x = self.summarizer(x)
        return self.extended_classifier(x)

    def classifier_forward(self, features):
        # return torch.sigmoid(self.classifier(features))
        return self.classifier(features)

    def __init__(self, targets, samples, channels, 
                 encoder_h=512, 
                 projection_head=False,
                 enc_do=0.1, 
                 feat_do=0.4, 
                 pool_length=4, 
                 mask_p_t=0.01, 
                 mask_p_c=0.005, 
                 mask_t_span=0.05,
                 mask_c_span=0.1, 
                 classifier_layers=1, 
                 return_features=False,
                 multi_gpu=False):
        
        super().__init__()
        self.samples = samples
        self.channels = channels
        self.return_features = return_features
        self.targets = targets
        self.num_features_for_classification = encoder_h * pool_length
        self.make_new_classification_layer()
        self._init_state = self.state_dict()

        if classifier_layers < 1:
            self.pool_length = pool_length
            self.encoder_h = 3 * encoder_h
        else:
            self.pool_length = pool_length // classifier_layers
            self.encoder_h = encoder_h
        
        encoder = ConvEncoderBENDR(channels, encoder_h=encoder_h, projection_head=projection_head, dropout=enc_do)
        encoded_samples = encoder.downsampling_factor(samples)

        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        mask_t_span = 0 if encoded_samples < 2 else mask_t_span
        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)

        enc_augment = EncodingAugment(encoder_h, mask_p_t, mask_p_c, mask_c_span=mask_c_span,
                                           mask_t_span=mask_t_span)
        summarizer = nn.AdaptiveAvgPool1d(pool_length)

        classifier_layers = [self.encoder_h * self.pool_length for i in range(classifier_layers)] if \
            not isinstance(classifier_layers, (tuple, list)) else classifier_layers
        classifier_layers.insert(0, 3 * encoder_h * pool_length)
        extended_classifier = nn.Sequential(Flatten())
        for i in range(1, len(classifier_layers)):
            extended_classifier.add_module("ext-classifier-{}".format(i), nn.Sequential(
                nn.Linear(classifier_layers[i - 1], classifier_layers[i]),
                nn.Dropout(feat_do),
                nn.ReLU(),
                nn.BatchNorm1d(classifier_layers[i]),
            ))

        # Data Parallelization if multi_gpu == True
        self.encoder = nn.DataParallel(encoder, device_ids=[0,1,2,3]) if multi_gpu else encoder
        self.enc_augment = nn.DataParallel(enc_augment, device_ids=[0,1,2,3]) if multi_gpu else enc_augment
        self.summarizer = nn.DataParallel(summarizer, device_ids=[0,1,2,3]) if multi_gpu else summarizer
        self.extended_classifier = nn.DataParallel(extended_classifier, device_ids=[0,1,2,3]) if multi_gpu else extended_classifier

    def reset(self):
        self.load_state_dict(self._init_state)

    def forward(self, *x):
        features = self.features_forward(*x)
        if self.return_features:    
            return self.classifier_forward(features), features
        else:
            return self.classifier_forward(features)

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(not freeze)
        # print("Loaded {}".format(encoder_file))

    def load_pretrained_modules(self, encoder_file, contextualizer_file, strict=False, freeze_encoder=True):
        self.load_encoder(encoder_file, strict=strict, freeze=freeze_encoder)
        self.enc_augment.init_from_contextualizer(contextualizer_file)

    def make_new_classification_layer(self):
        # Fnunction to distinct between the classification layer(s) and the rest of the network
        # This method is for implementing the classification side, so that methods like :py:meth:`freeze_features` works as intended.
        classifier = nn.Linear(self.num_features_for_classification, self.targets)
        nn.init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)
        """
        classifier = nn.Linear(self.num_features_for_classification, 1)
        nn.init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)
        """

    def freeze_features(self, unfreeze=False, freeze_classifier=False):
        # This method freezes (or un-freezes) all but the `classifier` layer. So that any further training does not (or
        # does if unfreeze=True) affect these weights.
        for param in self.parameters():
            param.requires_grad = unfreeze

        if isinstance(self.classifier, nn.Module) and not freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def load(self, filename, include_classifier=False, freeze_features=True):
        state_dict = torch.load(filename)
        if not include_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)
        if freeze_features:
            self.freeze_features()

    def save(self, filename, ignore_classifier=False):
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        print("Saving to {} ...".format(filename))
        torch.save(state_dict, filename)

# CNN model
class CNN_9k(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1,4), stride = (1,1), padding = 'same')
        self.bn1 = torch.nn.BatchNorm2d(4)
        # self.conv1.weight.data.normal_(0, 1)
        # self.conv1.bias.data.normal_(0, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,8), stride=(1,8))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(1,16), stride = (1,1), padding = 'same')
        # self.conv2.weight.data.normal_(0, 1)
        # self.conv2.bias.data.normal_(0, 1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,8), stride = (1,1), padding = 'same')
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(16,1), stride = (1,1), padding = 'same')
        # self.conv4.weight.data.normal_(0, 1)
        # self.conv4.bias.data.normal_(0, 1)
        self.bn4 = torch.nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=(4,1), stride=(4,1))
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(8,1), stride = (1,1), padding = 'same')
        # self.conv5.weight.data.normal_(0, 1)
        # self.conv5.bias.data.normal_(0, 1)
        self.bn5 = torch.nn.BatchNorm2d(16)
        # self.pool5 = nn.MaxPool2d(kernel_size=(4,1), stride=(4,1))
        self.pool6 = nn.AdaptiveAvgPool2d((1,1))    
        self.flatten = nn.Flatten()
        # self.fcn = nn.Linear(16, 2) # CrossEntropy
        self.fcn = nn.Linear(16, 1) # BCE loss


    
    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x))) #
        x= self.pool1(x)
        x=F.relu(self.bn2(self.conv2(x)))  #self.bn2
        x= self.pool2(x)
        x=F.relu(self.bn3(self.conv3(x)))
        x= self.pool3(x)
        x=F.relu(self.bn4(self.conv4(x)))  #self.bn4
        x= self.pool4(x)
        x=F.relu(self.bn5(self.conv5(x)))  #self.bn5
        x= self.pool6(x)
        x = self.flatten(x)
        x= self.fcn(x)
        return x