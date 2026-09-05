# from __future__ import absolute_import

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models.convnext import convnext_tiny, convnext_small, convnext_base, convnext_large

# __all__ = ['convnext']

# class ConvNeXt(nn.Module):
#     def __init__(self, model_name='convnext_tiny', num_classes=10):
#         super(ConvNeXt, self).__init__()
        
#         if model_name == 'convnext_tiny':
#             self.model = convnext_tiny(pretrained=False)
#         elif model_name == 'convnext_small':
#             self.model = convnext_small(pretrained=False)
#         elif model_name == 'convnext_base':
#             self.model = convnext_base(pretrained=False)
#         elif model_name == 'convnext_large':
#             self.model = convnext_large(pretrained=False)
#         else:
#             raise ValueError('model_name should be convnext_tiny, convnext_small, convnext_base, or convnext_large')

#         # Modify the classifier to match the number of classes
#         self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)
        
#         # Get the feature extraction modules
#         self.features = nn.ModuleList([
#             self.model.features[0],   # Stem
#             self.model.features[1],   # Stage 1
#             self.model.features[2],   # Stage 2
#             self.model.features[3],   # Stage 3
#             self.model.features[4]    # Stage 4
#         ])
        
#     def get_feat_modules(self):
#         return self.features
    
#     def get_bn_before_relu(self):
#         bns = []
#         for module in self.features:
#             if isinstance(module, nn.Sequential):
#                 for layer in module:
#                     if isinstance(layer, nn.BatchNorm2d):
#                         bns.append(layer)
#         return bns

#     def forward(self, x, is_feat=False, preact=False):
#         feats = []
#         preacts = []
        
#         for module in self.features:
#             x = module(x)
#             if isinstance(module, nn.Sequential):
#                 for layer in module:
#                     if isinstance(layer, nn.ReLU):
#                         preacts.append(x)
#             feats.append(x)
        
#         x = self.model.avgpool(x)
#         x = torch.flatten(x, 1)
#         feats.append(x)
#         x = self.model.classifier(x)
        
#         if is_feat:
#             if preact:
#                 return preacts, x
#             else:
#                 return feats, x
#         else:
#             return x


# def convnext_tiny_model(**kwargs):
#     return ConvNeXt(model_name='convnext_tiny', **kwargs)


# def convnext_small_model(**kwargs):
#     return ConvNeXt(model_name='convnext_small', **kwargs)


# def convnext_base_model(**kwargs):
#     return ConvNeXt(model_name='convnext_base', **kwargs)


# def convnext_large_model(**kwargs):
#     return ConvNeXt(model_name='convnext_large', **kwargs)


# if __name__ == '__main__':
#     x = torch.randn(2, 3, 224, 224)
#     net = convnext_tiny_model(num_classes=20)
#     feats, logit = net(x, is_feat=True, preact=True)

#     for f in feats:
#         print(f.shape, f.min().item())
#     print(logit.shape)

#     for m in net.get_bn_before_relu():
#         if isinstance(m, nn.BatchNorm2d):
#             print('pass')
#         else:
#             print('warning')


# import torch.nn as nn
# import torch.nn.functional as F
# import math

# __all__ = [
#     'ConvNeXt', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'
# ]

# model_urls = {
#     'convnext_tiny': 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth',
#     'convnext_small': 'https://download.pytorch.org/models/convnext_small-0c510722.pth',
#     'convnext_base': 'https://download.pytorch.org/models/convnext_base-5a3e7c1f.pth',
#     'convnext_large': 'https://download.pytorch.org/models/convnext_large-8e6c12d8.pth',
# }

# class ConvNeXtBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvNeXtBlock, self).__init__()
#         self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
#         self.norm = nn.LayerNorm(in_channels, eps=1e-6)
#         self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)
#         self.gelu = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * in_channels, out_channels)
    
#     def forward(self, x):
#         residual = x
#         x = self.dwconv(x)
#         # NHWC format for LayerNorm
#         x = x.permute(0, 2, 3, 1)  # NCHW to NHWC
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.gelu(x)
#         x = self.pwconv2(x)
#         x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
#         x += residual
#         return x

# class ConvNeXt(nn.Module):
#     def __init__(self, cfg, num_classes=1000):
#         super(ConvNeXt, self).__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(3, cfg[0], kernel_size=4, stride=4),
#             nn.LayerNorm(cfg[0], eps=1e-6)
#         )
        
#         self.blocks = nn.ModuleList()
#         for i in range(len(cfg) - 1):
#             self.blocks.append(self._make_layer(cfg[i], cfg[i+1]))
        
#         self.norm = nn.LayerNorm(cfg[-1], eps=1e-6)
#         self.head = nn.Linear(cfg[-1], num_classes)
        
#         self._initialize_weights()
    
#     def _make_layer(self, in_channels, out_channels, num_blocks=3):
#         layers = []
#         for _ in range(num_blocks):
#             layers.append(ConvNeXtBlock(in_channels, out_channels))
#         return nn.Sequential(*layers)
    
#     def forward(self, x, is_feat=False, preact=False):
#         h = x.shape[2]
#         x = self.stem(x)
#         f0 = x
#         feats = [f0]
#         for block in self.blocks:
#             x = block(x)
#             feats.append(x)
        
#         x = x.mean([2, 3])  # Global average pooling
#         f_last = x
#         x = self.norm(x)
#         x = self.head(x)
        
#         if is_feat:
#             if preact:
#                 return feats, x
#             else:
#                 return feats, x
#         else:
#             return x
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1)
#             elif isinstance(m, nn.Linear):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.trunc_normal_(m.weight, std=0.02)

# cfg = {
#     'tiny': [96, 192, 384, 768],
#     'small': [96, 192, 384, 768],
#     'base': [128, 256, 512, 1024],
#     'large': [192, 384, 768, 1536]
# }

# def convnext_tiny(**kwargs):
#     model = ConvNeXt(cfg['tiny'], **kwargs)
#     return model

# def convnext_small(**kwargs):
#     model = ConvNeXt(cfg['small'], **kwargs)
#     return model

# def convnext_base(**kwargs):
#     model = ConvNeXt(cfg['base'], **kwargs)
#     return model

# def convnext_large(**kwargs):
#     model = ConvNeXt(cfg['large'], **kwargs)
#     return model

# if __name__ == '__main__':
#     import torch

#     x = torch.randn(2, 3, 224, 224)
#     net = convnext_tiny(num_classes=100)
#     feats, logit = net(x, is_feat=True, preact=True)

#     for f in feats:
#         print(f.shape, f.min().item())
#     print(logit.shape)

#     # Checking LayerNorm in ConvNeXt blocks
#     for m in net.modules():
#         if isinstance(m, nn.LayerNorm):
#             print('LayerNorm found')
#         else:
#             print('Not a LayerNorm')

