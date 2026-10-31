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

# pylint: disable=C0114,C0301
# 推理发行版：按符号摘取；来源哈希见runtime_sources.json。
import os
from pathlib import Path
root = Path(os.environ.get('PROTENIX_ROOT_DIR', Path(__file__).resolve().parents[1] / 'assets'))
data_configs = dict(ccd_components_file=str(root / 'common/components.cif'), ccd_components_rdkit_mol_file=str(root / 'common/components.cif.rdkit_mol.pkl'))
