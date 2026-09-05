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

import numpy as np
import torch
import random


def reproducibility(seed: int):
    """
    Set seed for experiment reproducibility

    :param seed: seed
    """

    # for torch
    torch.manual_seed(seed)

    # for numpy
    np.random.seed(seed)

    # for random
    random.seed(seed)

    # for cuda
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
