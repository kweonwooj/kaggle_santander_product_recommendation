# -*- coding: utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: clean directory
"""

import os

def _clean_dir(path, suffix):
  files = sorted(os.listdir(path))
  for file in files:
    if os.path.isdir(os.path.join(path,file)):
      _clean_dir(os.path.join(path,file), suffix)
    elif file.endswith(suffix):
      os.remove(os.path.join(path,file))

# clearn .pyc and .out
_clean_dir(os.path.abspath('.'), suffix='pyc')
_clean_dir(os.path.abspath('.'), suffix='out')

# clear memory
cmd = "sudo echo 3 | sudo tee /proc/sys/vm/drop_caches"
os.system(cmd)
