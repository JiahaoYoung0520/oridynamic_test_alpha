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

import dataclasses
import os
import pathlib
import re
import subprocess
import tempfile
import time
from typing import Any, List, Optional, Union
from protenix.data.tools.common import convert_a3m_to_stockholm, convert_stockholm_to_a3m, lazy_fasta_parse
from protenix.utils.logger import get_logger
logger = get_logger(__name__)

def run_shell(cmd: List[str], name: str, **kwargs: Any) -> subprocess.CompletedProcess:
    """
    Run a shell command and log its execution and results.

    Args:
        cmd: List of command arguments.
        name: Logical name for the command (for logging).
        **kwargs: Additional keyword arguments for subprocess.run.

    Returns:
        CompletedProcess object.

    Raises:
        RuntimeError: If the command fails.
    """
    logger.info(f"Executing {name}: {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, **kwargs)
        logger.info(f'{name} finished in {time.time() - t0:.3f}s')
        return proc
    except subprocess.CalledProcessError as e:
        logger.error(f'{name} failed. Stderr: {e.stderr}')
        raise RuntimeError(f'{name} failed\nStdout: {e.stdout}\nStderr: {e.stderr}') from e

class BinaryWrapper:
    """
    Base class for wrapping external bioinformatics binaries.

    Args:
        binary_path: Path to the executable.
        name: Name of the tool.

    Raises:
        RuntimeError: If the binary path does not exist.
    """

    def __init__(self, binary_path: str, name: str) -> None:
        if not os.path.exists(binary_path):
            raise RuntimeError(f'{name} not found at {binary_path}')
        self.path = binary_path

class Hmmbuild(BinaryWrapper):
    """
    Wrapper for HMMER hmmbuild.

    Args:
        binary_path: Path to the hmmbuild executable.
        singlemx: Whether to use --singlemx flag.
        alphabet: Optional alphabet specification (e.g., 'amino', 'rna', 'dna').
    """

    def __init__(self, binary_path: str, singlemx: bool=False, alphabet: Optional[str]=None) -> None:
        super().__init__(binary_path, 'hmmbuild')
        self.opts = (['--singlemx'] if singlemx else []) + ([f'--{alphabet}'] if alphabet else [])

    def build(self, msa: str, informat: str, model: str='fast') -> str:
        """
        Build a HMM profile from a Multiple Sequence Alignment.

        Args:
            msa: Input MSA string.
            informat: Format of the input MSA (e.g., 'afa', 'stockholm').
            model: Model construction strategy ('fast' or 'hand').

        Returns:
            The HMM profile string.
        """
        with tempfile.TemporaryDirectory() as tmp:
            in_p, out_p = (f'{tmp}/i', f'{tmp}/o')
            pathlib.Path(in_p).write_text(msa)
            cmd = [self.path, '--informat', informat] + self.opts
            if model == 'hand':
                cmd.append('--hand')
            run_shell(cmd + [out_p, in_p], 'hmmbuild')
            return pathlib.Path(out_p).read_text()

    def build_profile_from_a3m(self, a3m: str) -> str:
        """
        Build HMM from A3M format by removing insertions.

        Args:
            a3m: Input A3M string.

        Returns:
            HMM profile string.
        """
        clean = ''.join((f">{d}\n{re.sub('[a-z]+', '', s)}\n" for s, d in lazy_fasta_parse(a3m)))
        return self.build(clean, 'afa')

    def build_profile_from_sto(self, sto: str, model: str='fast') -> str:
        """
        Build HMM from Stockholm format.

        Args:
            sto: Input Stockholm string.
            model: Model construction strategy.

        Returns:
            HMM profile string.
        """
        return self.build(sto, 'stockholm', model)

class Hmmsearch(BinaryWrapper):
    """
    Wrapper for HMMER hmmsearch.

    Args:
        binary_path: Path to hmmsearch executable.
        hmmbuild_binary_path: Path to hmmbuild executable.
        database_path: Path to the sequence database to search.
        alphabet: Alphabet type.
        **filters: Search filters (e.g., e_value, inc_e, filter_f1, etc.).
    """

    def __init__(self, binary_path: str, hmmbuild_binary_path: str, database_path: str, alphabet: str='amino', **filters: Any) -> None:
        super().__init__(binary_path, 'hmmsearch')
        self.builder = Hmmbuild(binary_path=hmmbuild_binary_path, alphabet=alphabet)
        self.db = database_path
        self.flags = ['--max'] if filters.get('filter_max') else []
        if not self.flags:
            map_f = {'filter_f1': '--F1', 'filter_f2': '--F2', 'filter_f3': '--F3', 'e_value': '-E', 'inc_e': '--incE', 'dom_e': '--domE', 'incdom_e': '--incdomE'}
            for k, v in filters.items():
                if k in map_f and v is not None:
                    self.flags.extend([map_f[k], str(v)])

    def query_with_hmm(self, hmm: str) -> str:
        """
        Search a database using an HMM profile.

        Args:
            hmm: HMM profile string.

        Returns:
            Search result in A3M format.
        """
        with tempfile.TemporaryDirectory() as tmp:
            hmm_p, sto_p = (f'{tmp}/q.hmm', f'{tmp}/o.sto')
            pathlib.Path(hmm_p).write_text(hmm)
            cmd = [self.path, '--noali', '--cpu', '8'] + self.flags + ['-A', sto_p, hmm_p, self.db]
            run_shell(cmd, 'hmmsearch')
            with open(sto_p) as f:
                return convert_stockholm_to_a3m(f, remove_gaps=False, linewidth=60)

    def query_with_sto(self, sto: str, model: str='fast') -> str:
        """
        Build an HMM from a Stockholm MSA and search the database.

        Args:
            sto: Input Stockholm MSA string.
            model: Model construction strategy.

        Returns:
            Search result in A3M format.
        """
        return self.query_with_hmm(self.builder.build_profile_from_sto(sto, model))

@dataclasses.dataclass(frozen=True, slots=True)
class HmmsearchConfig:
    """Config for an hmmsearch run."""
    hmmsearch_binary_path: str
    hmmbuild_binary_path: str
    e_value: float = 100
    inc_e: float = 100
    dom_e: float = 100
    incdom_e: float = 100
    alphabet: str = 'amino'
    filter_f1: Optional[float] = 0.1
    filter_f2: Optional[float] = 0.1
    filter_f3: Optional[float] = 0.1
    filter_max: bool = False

def run_hmmsearch_with_a3m(database_path: Union[str, os.PathLike], hmmsearch_config: HmmsearchConfig, max_a3m_query_sequences: Optional[int], a3m: Optional[str]) -> str:
    """
    Search database using hmmsearch with an input A3M string.

    Args:
        database_path: Path to search database.
        hmmsearch_config: HmmsearchConfig object.
        max_a3m_query_sequences: Max sequences to use for building the HMM.
        a3m: Input A3M format string.

    Returns:
        Search results in A3M format.
    """
    cfg = dataclasses.asdict(hmmsearch_config)
    search_bin = cfg.pop('hmmsearch_binary_path')
    build_bin = cfg.pop('hmmbuild_binary_path')
    searcher = Hmmsearch(binary_path=search_bin, hmmbuild_binary_path=build_bin, database_path=database_path, **cfg)
    sto = convert_a3m_to_stockholm(a3m, max_a3m_query_sequences)
    return searcher.query_with_sto(sto, model='hand')
