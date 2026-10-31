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

from contextlib import nullcontext
import numpy as np
import torch

def autocasting_disable_decorator(disable_casting):

    def func_wrapper(func):

        def new_func(*args, **kwargs):
            _amp_context = torch.autocast(device_type='cuda', enabled=False) if disable_casting else nullcontext()

            def conditioned_cast(tensor):
                if disable_casting and isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
                    return tensor.to(dtype=torch.float32)
                return tensor
            with _amp_context:
                return func(*(conditioned_cast(v) for v in args), **{k: conditioned_cast(v) for k, v in kwargs.items()})
        return new_func
    return func_wrapper

def dict_to_tensor(feature_dict):
    for k, v in feature_dict.items():
        if not isinstance(v, torch.Tensor):
            dtype = feature_dict[k].dtype
            feature_dict[k] = torch.tensor(v)
            if dtype in [np.int64, np.int32]:
                feature_dict[k] = feature_dict[k].to(torch.int64)
            elif dtype in [np.float32, np.float64]:
                feature_dict[k] = feature_dict[k].to(torch.float32)
    return feature_dict
