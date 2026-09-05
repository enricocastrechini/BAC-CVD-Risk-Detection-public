# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime

import os

import time
from collections import defaultdict, deque
from typing import Iterator

import numpy as np

import torch


def save_checkpoint(save_to, args, epoch, performance_desc, model_without_ddp_state, optimizer_state):
    checkpoint_path = os.path.join(args.save_folder, save_to)
    to_save = {
        'args': str(args),
        'epoch': epoch,
        'performance_desc': performance_desc,
        'module': model_without_ddp_state,
        'optimizer': optimizer_state,
        'is_pretrain': True,
    }
    torch.save(to_save, checkpoint_path)


def load_checkpoint(resume_from, model_without_ddp, optimizer):
    if len(resume_from) == 0:
        return 0, '[no performance_desc]'
    print(f'[try to resume from file `{resume_from}`]')
    checkpoint = torch.load(resume_from, map_location='cpu')
    
    ep_start, performance_desc = checkpoint.get('epoch', -1) + 1, checkpoint.get('performance_desc', '[no performance_desc]')
    missing, unexpected = model_without_ddp.load_state_dict(checkpoint['module'], strict=False)
    print(f'[load_checkpoint] missing_keys={missing}')
    print(f'[load_checkpoint] unexpected_keys={unexpected}')
    print(f'[load_checkpoint] ep_start={ep_start}, performance_desc={performance_desc}')
    
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return ep_start, performance_desc