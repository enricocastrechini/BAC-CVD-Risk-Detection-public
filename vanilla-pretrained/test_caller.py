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

import argparse
from omegaconf import OmegaConf
from src.utils.test import TestModel
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',  '-c',
                        dest="config",
                        metavar='Config file',
                        help =  'config file path',
                        default= 'src/config/cfg_test.yaml')
    
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Set the parameters
    test_params = cfg.test
    model_params = cfg.models
    vis_params = cfg.visualize
    data_params = cfg.data
    
    # Testing
    exp = TestModel(test_params,model_params,vis_params,data_params.transform)

    exp.test()
    

    