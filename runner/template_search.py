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
import pathlib
import shutil
import time
from pathlib import Path
from typing import Optional
from protenix.data.tools.search import HmmsearchConfig, run_hmmsearch_with_a3m
from protenix.utils.logger import get_logger
logger = get_logger(__name__)
TEMPLATE_SEARCH_DATABASE_URL = 'https://protenix.tos-cn-beijing.volces.com/search_database/pdb_seqres_2022_09_28.fasta'

def ensure_ends_with_newline(s: str) -> str:
    """
    Ensure the string ends with a newline character.

    Args:
        s: The input string.

    Returns:
        The string with a trailing newline if it wasn't empty.
    """
    if not s.endswith('\n') and len(s) > 0:
        s += '\n'
    return s

def run_template_search(msa_for_template_search_dir: Optional[str]=None, msa_for_template_search_name: Optional[str]=None, hmmsearch_binary_path: Optional[str]=None, hmmbuild_binary_path: Optional[str]=None, seqres_database_path: Optional[str]=None) -> None:
    """
    Run template search using hmmsearch with a3m files.

    Args:
        msa_for_template_search_dir: Directory containing MSA files.
            Templates will be saved in the same directory.
        msa_for_template_search_name: Comma-separated names of MSA files to search.
        hmmsearch_binary_path: Path to hmmsearch binary.
        hmmbuild_binary_path: Path to hmmbuild binary.
        seqres_database_path: Path to sequence database.
    """
    assert msa_for_template_search_dir is not None, 'input msa dir should not be None'
    assert msa_for_template_search_name is not None, 'input msa name should not be None'
    if hmmsearch_binary_path is None:
        hmmsearch_binary_path = shutil.which('hmmsearch')
        if hmmsearch_binary_path is None:
            raise AssertionError('hmmsearch binary path should not be None. You can install hmmer using: apt install hmmer or conda install -c bioconda hmmer')
    elif not os.path.exists(hmmsearch_binary_path):
        raise AssertionError(f'hmmsearch binary path {hmmsearch_binary_path} does not exist')
    if hmmbuild_binary_path is None:
        hmmbuild_binary_path = shutil.which('hmmbuild')
        if hmmbuild_binary_path is None:
            raise AssertionError('hmmbuild binary path should not be None. You can install hmmer using: apt install hmmer or conda install -c bioconda hmmer')
    elif not os.path.exists(hmmbuild_binary_path):
        raise AssertionError(f'hmmbuild binary path {hmmbuild_binary_path} does not exist')
    if seqres_database_path is None:
        _HOME_DIR = pathlib.Path(os.environ.get('PROTENIX_ROOT_DIR', str(Path.home())))
        _SEQRES_DATABASE_PATH = _HOME_DIR / 'search_database' / 'pdb_seqres_2022_09_28.fasta'
        seqres_database_path = _SEQRES_DATABASE_PATH.as_posix()
    if not os.path.exists(seqres_database_path):
        from runner.download import download_from_url
        os.makedirs(os.path.dirname(seqres_database_path), exist_ok=True)
        logger.info(f'Downloading template search database from {TEMPLATE_SEARCH_DATABASE_URL} to {seqres_database_path}')
        download_from_url(TEMPLATE_SEARCH_DATABASE_URL, seqres_database_path)
    logger.info('Template search start!')
    template_start_time = time.time()
    hmmsearch_config = HmmsearchConfig(hmmsearch_binary_path=hmmsearch_binary_path, hmmbuild_binary_path=hmmbuild_binary_path, filter_f1=0.1, filter_f2=0.1, filter_f3=0.1, e_value=100, inc_e=100, dom_e=100, incdom_e=100, alphabet='amino')
    max_a3m_query_sequences = 300
    msa_search_list = msa_for_template_search_name.split(',')
    msa_a3m = ''
    for unpaired_msa in msa_search_list:
        unpaired_msa_path = f'{msa_for_template_search_dir}/{unpaired_msa}.a3m'
        logger.info(f'msa path: {unpaired_msa_path}')
        if os.path.exists(unpaired_msa_path):
            with open(unpaired_msa_path, 'r') as f:
                unpaired_msa_a3m = f.read()
        else:
            unpaired_msa_a3m = ''
        unpaired_msa_a3m = ensure_ends_with_newline(unpaired_msa_a3m)
        msa_a3m = msa_a3m + unpaired_msa_a3m
    msa_a3m = ensure_ends_with_newline(msa_a3m)
    hmmsearch_a3m = run_hmmsearch_with_a3m(database_path=seqres_database_path, hmmsearch_config=hmmsearch_config, max_a3m_query_sequences=max_a3m_query_sequences, a3m=msa_a3m)
    with open(f'{msa_for_template_search_dir}/hmmsearch.a3m', 'w') as f:
        f.write(hmmsearch_a3m)
    template_end_time = time.time()
    logger.info(f'Template search done!, using {template_end_time - template_start_time}')
    logger.info(f'Template result is saved at: {msa_for_template_search_dir}/hmmsearch.a3m')
