# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import urllib.request
from argparse import Namespace
from os.path import exists as opexists
import torch
"\nDue to the fair-esm repository being archived,\nit can no longer be updated to support newer versions of PyTorch.\nStarting from PyTorch 2.6, the default value of the weights_only argument\nin torch.load has been changed from False to True,\nwhich enhances security but causes loading ESM models to fail\nwith the following error:\n\n_pickle.UnpicklingError: Weights only load failed. This file can still be loaded...\nThis error occurs because the model file contains argparse.Namespace,\nwhich is not allowed by default in the secure unpickling process of PyTorch 2.6+.\n\n✅ Solution (Patch)\nSince we cannot modify the fair-esm source code,\nwe can apply a patch before calling load_model_and_alphabet_local\nby manually adding argparse.Namespace to PyTorch's safe globals list.\n"
torch.serialization.add_safe_globals([Namespace])

def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
    """Callback for tracking download progress."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    bar_length = 30
    filled_length = int(bar_length * percent // 100)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    status = f'\r[{bar}] {percent:.1f}%'
    print(status, end='', flush=True)
    if downloaded >= total_size:
        print()

def download_from_url(tos_url: str, checkpoint_path: str, check_weight: bool=True) -> None:
    """Internal helper to download from URL and verify weight files."""
    urllib.request.urlretrieve(tos_url, checkpoint_path, reporthook=progress_callback)
    if check_weight:
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            del ckpt
        except Exception as e:
            if opexists(checkpoint_path):
                os.remove(checkpoint_path)
            raise RuntimeError(f'Download model checkpoint failed: {e}. Please download manually with: wget {tos_url} -O {checkpoint_path}') from e
