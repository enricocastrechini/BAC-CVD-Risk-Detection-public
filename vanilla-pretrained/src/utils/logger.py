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

import datetime
import os

class GenerateLogs:
    def __init__(self, path, mode='w', flag = True):
        self.path = path
        self.mode = mode
        file_path = path[:path.rfind('/')]
        self.checkncreate_dir(file_path)
        file = open(self.path, self.mode)
        if flag:
            self.print_log('Initiating the logs --------------------')
        file.close()
    
    def checkncreate_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def print_log(self, in_str):
        file = open(self.path, self.mode)
        timestamp = datetime.datetime.now()
        
        print(timestamp, end = ' - ', file=file)
        print(str(in_str), file=file)

        file.close()

        print(timestamp, end = ' - ')
        print(str(in_str))