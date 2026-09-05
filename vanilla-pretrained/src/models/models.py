# Copyright 2024 Enrico Castrechini
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description: [Brief description of what the file or class does]

import torchvision.models as models
import torch.nn as nn
import torch
import timm
import os

class BuildModel:
    def __init__(self, **kwargs):
        self.fr_layer = kwargs['fr_layer'] if 'fr_layer' in kwargs else 0
        self.out_neurons = kwargs['out_neurons'] if 'out_neurons' in kwargs else 1
        self.model_type = kwargs['model_name'] if 'model_name' in kwargs else 'vgg16'
        self.pretrained = kwargs['pretrained'] if 'pretrained' in kwargs else False

    # Count number of layers in a model
    def len_layers(self, model):
        count = 0
        for param in model.parameters():
            count += 1
        return count

    # Count number of training parameters
    def count_tr_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count number of non-trainable parameters
    def count_ntr_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    def print_model_layers_names(self, model):
        layer_num = 1
        for name, layer in model.named_modules():
            print(layer_num, name)
    
    # Get number of last layers to freeze
    def num_freeze_layers(self, model, block):
        count = 0
        flg = 0
        for layer_name, param in reversed(list(model.named_parameters())):

            if flg == 1:
                if block in layer_name:
                    count += 1
            else:
                count += 1
                if block in layer_name:
                    flg = 1
        return count

    def get_model(self):
        lay = self.fr_layer # Number of last layers to enabler training
        
        if self.model_type.lower() == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
            filters = model.fc.in_features
            model.fc = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'resnet34':
            model = models.resnet34(pretrained=self.pretrained)
            filters = model.fc.in_features
            model.fc = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
            filters = model.fc.in_features
            model.fc = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'inceptionv3':
            model = models.inception_v3(pretrained=self.pretrained)
            filters = model.fc.in_features
            model.fc = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'efficientnetv2s':
            model = models.efficientnet_v2_s(pretrained=self.pretrained)
            # Assuming the classifier is a Sequential module
            classifier_modules = list(model.classifier.children())
            # Find the Linear layer in the classifier
            for module in classifier_modules:
                if isinstance(module, nn.Linear):
                    filters = module.in_features
                    break
            else:
                raise ValueError("No Linear layer found in the classifier of the model.")
            
            model.classifier = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'efficientnetv2m':
            model = models.efficientnet_v2_m(pretrained=self.pretrained)
            # Assuming the classifier is a Sequential module
            classifier_modules = list(model.classifier.children())
            # Find the Linear layer in the classifier
            for module in classifier_modules:
                if isinstance(module, nn.Linear):
                    filters = module.in_features
                    break
            else:
                raise ValueError("No Linear layer found in the classifier of the model.")
            
            model.classifier = nn.Linear(filters, self.out_neurons)
        
        elif self.model_type.lower() == 'efficientnetv2l':
            model = models.efficientnet_v2_l(pretrained=self.pretrained)
            # Assuming the classifier is a Sequential module
            classifier_modules = list(model.classifier.children())
            # Find the Linear layer in the classifier
            for module in classifier_modules:
                if isinstance(module, nn.Linear):
                    filters = module.in_features
                    break
            else:
                raise ValueError("No Linear layer found in the classifier of the model.")
            
            model.classifier = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'convnextbase':
            model = models.convnext_base(pretrained=self.pretrained)
            last_layer_index = len(model.classifier) - 1
            filters = model.classifier[last_layer_index].in_features
            model.classifier[last_layer_index] = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'convnextsmall':
            model = models.convnext_small(pretrained=self.pretrained)
            last_layer_index = len(model.classifier) - 1
            filters = model.classifier[last_layer_index].in_features
            model.classifier[last_layer_index] = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'convnextsmall_pseudo':
            weight_path = os.environ.get('BAC_PSEUDO_WEIGHTS')
            if not weight_path:
                raise ValueError('Set BAC_PSEUDO_WEIGHTS to use convnextsmall_pseudo.')
            state_dict = torch.load(weight_path, map_location='cpu')
            model = models.convnext_small()
            model_dict = model.state_dict()
            # Find mismatched keys
            mismatched_keys = {}
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape != v.shape:
                    mismatched_keys[k] = (model_dict[k].shape, v.shape)

            # Print mismatched keys and their shapes
            print("Mismatched keys and their shapes:")
            for k, (model_shape, pretrained_shape) in mismatched_keys.items():
                print(f"{k}: model shape {model_shape}, pretrained shape {pretrained_shape}")

            # Filter out weights that don't match the model's state_dict
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

            # Update the model's state_dict
            model_dict.update(pretrained_dict)
            msg = model.load_state_dict(model_dict, strict=False)

            print(format(msg))


            last_layer_index = len(model.classifier) - 1
            filters = model.classifier[last_layer_index].in_features
            model.classifier[last_layer_index] = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'convnextv2tiny':
            # Load model directly
            model = timm.create_model('convnext_tiny.fb_in1k', pretrained=self.pretrained)
            filters = model.head.fc.in_features
            model.head.fc = nn.Linear(filters, self.out_neurons)
        
        elif self.model_type.lower() == 'convnextv2base':
            # Load model directly
            model = timm.create_model('convnext_base.fb_in1k', pretrained=self.pretrained)
            filters = model.head.fc.in_features
            model.head.fc = nn.Linear(filters, self.out_neurons)
           

        
        elif self.model_type.lower() == 'convnextsmall_spark':
            # Load your custom state dictionary
            weight_path = os.environ.get('BAC_SPARK_WEIGHTS')
            if not weight_path:
                raise ValueError('Set BAC_SPARK_WEIGHTS to use convnextsmall_spark.')
            state_dict = torch.load(weight_path, map_location='cpu')
            state_dict = state_dict["module"]

            # Replace keys to match the convnext_small model
            state_dict = {k.replace("downsample_layers.", "features."): v for k, v in state_dict.items()}
            state_dict = {k.replace("stages.", "features."): v for k, v in state_dict.items()}

            # Initialize the PyTorch convnext_small model
            model = models.convnext_small()

            # Get the state dictionary of the model
            model_dict = model.state_dict()

            # Find mismatched keys
            mismatched_keys = {}
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape != v.shape:
                    mismatched_keys[k] = (model_dict[k].shape, v.shape)

            # Print mismatched keys and their shapes
            print("Mismatched keys and their shapes:")
            for k, (model_shape, pretrained_shape) in mismatched_keys.items():
                print(f"{k}: model shape {model_shape}, pretrained shape {pretrained_shape}")

            # Filter out weights that don't match the model's state_dict
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

            # Update the model's state_dict
            model_dict.update(pretrained_dict)
            msg = model.load_state_dict(model_dict, strict=False)

            # Print the loading result to verify
            print(format(msg))

            last_layer_index = len(model.classifier) - 1
            filters = model.classifier[last_layer_index].in_features
            model.classifier[last_layer_index] = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'convnextsmall_student_2':
            # Load your custom state dictionary
            weight_path = os.environ.get('BAC_STUDENT_WEIGHTS')
            if not weight_path:
                raise ValueError('Set BAC_STUDENT_WEIGHTS to use convnextsmall_student_2.')
            state_dict = torch.load(weight_path, map_location='cpu')
            state_dict = state_dict["model"]

            # Replace keys to match the convnext_small model
            # state_dict = {k.replace("downsample_layers.", "features."): v for k, v in state_dict.items()}
            # state_dict = {k.replace("stages.", "features."): v for k, v in state_dict.items()}

            # Initialize the PyTorch convnext_small model
            model = models.convnext_small()

            # Get the state dictionary of the model
            model_dict = model.state_dict()

            # Find mismatched keys
            mismatched_keys = {}
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape != v.shape:
                    mismatched_keys[k] = (model_dict[k].shape, v.shape)

            # Print mismatched keys and their shapes
            print("Mismatched keys and their shapes:")
            for k, (model_shape, pretrained_shape) in mismatched_keys.items():
                print(f"{k}: model shape {model_shape}, pretrained shape {pretrained_shape}")

            # Filter out weights that don't match the model's state_dict
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

            # Update the model's state_dict
            model_dict.update(pretrained_dict)
            msg = model.load_state_dict(model_dict, strict=False)

            # Print the loading result to verify
            print(format(msg))

            last_layer_index = len(model.classifier) - 1
            filters = model.classifier[last_layer_index].in_features
            model.classifier[last_layer_index] = nn.Linear(filters, self.out_neurons)



        elif self.model_type.lower() == 'vgg16':
            model = models.vgg16(pretrained=self.pretrained)
            last_layer_index = len(model.classifier) - 1
            filters = model.classifier[last_layer_index].in_features
            model.classifier[last_layer_index] = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'vgg16bn':
            model = models.vgg16_bn(pretrained=self.pretrained)
            last_layer_index = len(model.classifier) - 1
            filters = model.classifier[last_layer_index].in_features
            model.classifier[last_layer_index] = nn.Linear(filters, self.out_neurons)
        
        elif self.model_type.lower() == 'vgg16mod':
            model = models.vgg16(pretrained=self.pretrained)
            filters = model.classifier[-7].in_features
            model.classifier[-7] = nn.Linear(filters, 256)
            model.classifier[-4] = nn.Linear(256, 256)
            model.classifier[-1] = nn.Linear(256, self.out_neurons)

            # Modify the activations in the features module
            for i in range(len(model.features)):
                if isinstance(model.features[i], nn.ReLU):
                    # Replace ReLU with leaky ReLU
                    model.features[i] = nn.LeakyReLU(negative_slope=0.3)
    
            # Modify the activations in the classifier module
            for i in range(len(model.classifier)):
                if isinstance(model.classifier[i], nn.ReLU):
                    # Replace ReLU with leaky ReLU
                    model.classifier[i] = nn.LeakyReLU(negative_slope=0.3)
                elif isinstance(model.classifier[i], nn.Dropout):
                    # Change dropout rate to 0.3
                    model.classifier[i] = nn.Dropout(p=0.3)

        elif self.model_type.lower() == 'vgg16mod2':
            model = models.vgg16(pretrained=self.pretrained)
            filters = model.classifier[-7].in_features
            model.classifier[-7] = nn.Linear(filters, 256)
            model.classifier[-4] = nn.Linear(256, 256)
            model.classifier[-1] = nn.Linear(256, self.out_neurons)

            for i in range(len(model.classifier)):
                if isinstance(model.classifier[i], nn.Dropout):
                    # Change dropout rate to 0.3
                    model.classifier[i] = nn.Dropout(p=0.3)

        elif self.model_type.lower() == 'vgg16mod3':
            model = models.vgg16(pretrained=self.pretrained)
            filters = model.classifier[-7].in_features
            model.classifier[-7] = nn.Linear(filters, 256)
            model.classifier[-4] = nn.Linear(256, 256)
            model.classifier[-1] = nn.Linear(256, self.out_neurons)

        elif self.model_type.lower() == 'vgg16mod4':
            model = models.vgg16(pretrained=self.pretrained)
            filters = model.classifier[-7].in_features
            model.classifier[-7] = nn.Linear(filters, 2048)
            model.classifier[-4] = nn.Linear(2048, 2048)
            model.classifier[-1] = nn.Linear(2048, self.out_neurons)

        else:
            return 'Model option not available'
        
        if not self.pretrained:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
        if isinstance(lay, str):
            lay = self.num_freeze_layers(model, lay)

        if lay != 0:
            layers = self.len_layers(model)
            print(layers)
            for param in model.parameters():
                if layers == lay:
                    break
                param.requires_grad = False
                layers -= 1

        model = nn.DataParallel(model)

        print('Model {}.........'.format(self.model_type.lower()))
        print('Number of training parameters: {}'.format(str(self.count_tr_parameters(model))))
        print('Number of non training parameters: {}'.format(str(self.count_ntr_parameters(model))))
        
        return model