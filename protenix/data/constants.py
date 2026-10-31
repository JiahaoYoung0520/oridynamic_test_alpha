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

from collections.abc import Set
from typing import Final
import numpy as np
from rdkit.Chem import GetPeriodicTable
mmcif_restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'B': 'ASX', 'Z': 'GLX'}
mmcif_restype_3to1 = {v: k for k, v in mmcif_restype_1to3.items()}
'\nvdw table from rdkit; indices match those of the ligand atom_types.\nhttps://github.com/rdkit/rdkit/blob/master/Code/GraphMol/atomic_data.cpp#L46\n'
rdkit_vdws = [1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54, 2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88, 2.8, 2.4, 2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1, 2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3, 2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25, 2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.45, 2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3, 2.28, 2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05, 2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4, 2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
'\natom37 vdw table. Orders match atom37 indices. Note: the vdw\'s for N and O are different from rdkit_van_der_waals in this file.\nWe used the rdkit values for consistency.\nRef to https://github.com/aqlaboratory/openfold/blob/80c85b54e1a81d9a66df3f1b6c257ff97f10acd3/openfold/utils/loss.py#L1208C5-L1211C6\nrdkit_van_der_waals_radius = {\n    "C": 1.7,\n    "N": 1.6,\n    "O": 1.55,\n    "S": 1.8,\n}\n\natom37_vdw = [\n        rdkit_van_der_waals_radius[name[0]]\n        for name in residue_constants.atom_types\n    ]\n\n'
PRO_STD_RESIDUES = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'UNK': 20}
RNA_STD_RESIDUES = {'A': 21, 'G': 22, 'C': 23, 'U': 24, 'N': 25}
DNA_STD_RESIDUES = {'DA': 26, 'DG': 27, 'DC': 28, 'DT': 29, 'DN': 30}
RNA_START_INDEX = 21
DNA_START_INDEX = 26
GAP = {'-': 31}
STD_RESIDUES = PRO_STD_RESIDUES | RNA_STD_RESIDUES | DNA_STD_RESIDUES
STD_RESIDUES_WITH_GAP = STD_RESIDUES | GAP
STD_RESIDUES_WITH_GAP_ID_TO_NAME = {idx: res_type for res_type, idx in STD_RESIDUES_WITH_GAP.items()}
MSA_PROTEIN_SEQ_TO_ID = {'A': 0, 'B': 3, 'C': 4, 'D': 3, 'E': 6, 'F': 13, 'G': 7, 'H': 8, 'I': 9, 'J': 20, 'K': 11, 'L': 10, 'M': 12, 'N': 2, 'O': 20, 'P': 14, 'Q': 5, 'R': 1, 'S': 15, 'T': 16, 'U': 4, 'V': 19, 'W': 17, 'X': 20, 'Y': 18, 'Z': 6, '-': 31}
MSA_RNA_SEQ_TO_ID = {**{chr(i): 25 for i in range(ord('A'), ord('Z') + 1)}, '-': 31, 'A': 21, 'G': 22, 'C': 23, 'U': 24}
MSA_DNA_SEQ_TO_ID = {**{chr(i): 30 for i in range(ord('A'), ord('Z') + 1)}, '-': 31, 'A': 26, 'G': 27, 'C': 28, 'T': 29}
TEMPLATE_PROTEIN_SEQ_TO_ID = MSA_PROTEIN_SEQ_TO_ID
TEMPLATE_RNA_SEQ_TO_ID = MSA_RNA_SEQ_TO_ID
TEMPLATE_DNA_SEQ_TO_ID = MSA_DNA_SEQ_TO_ID
PROTEIN_CHAIN: Final[str] = 'polypeptide(L)'
RNA_CHAIN: Final[str] = 'polyribonucleotide'
DNA_CHAIN: Final[str] = 'polydeoxyribonucleotide'
BRANCHED_CHAIN: Final[str] = 'branched'
MACROLIDE_CHAIN: Final[str] = 'macrolide'
NON_POLYMER_CHAIN: Final[str] = 'non-polymer'
LIGAND_CHAIN_TYPES: Final[Set[str]] = {BRANCHED_CHAIN, MACROLIDE_CHAIN, NON_POLYMER_CHAIN}
STANDARD_POLYMER_CHAIN_TYPES: Final[Set[str]] = {PROTEIN_CHAIN, DNA_CHAIN, RNA_CHAIN}

def get_all_elems():
    """
    Retrieve a list of all element symbols from the RDKit periodic table up to a specified cutoff.

    Returns:
        list: A list of element symbols strings.
    """
    elem_list = []
    pt = GetPeriodicTable()
    for i in range(1, 119):
        elem_list.append(pt.GetElementSymbol(i).upper())
    elem_list += [f'UNK_ELEM_{i}' for i in range(119, 129)]
    return elem_list
ELEMS = dict([(i, len(STD_RESIDUES) + idx) for idx, i in enumerate(get_all_elems())])
RES_ATOMS_DICT = {'ALA': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OXT': 5}, 'ARG': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'NE': 7, 'CZ': 8, 'NH1': 9, 'NH2': 10, 'OXT': 11}, 'ASN': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'OD1': 6, 'ND2': 7, 'OXT': 8}, 'ASP': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'OD1': 6, 'OD2': 7, 'OXT': 8}, 'CYS': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'SG': 5, 'OXT': 6}, 'GLN': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'OE1': 7, 'NE2': 8, 'OXT': 9}, 'GLU': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'OE1': 7, 'OE2': 8, 'OXT': 9}, 'GLY': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'OXT': 4}, 'HIS': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'ND1': 6, 'CD2': 7, 'CE1': 8, 'NE2': 9, 'OXT': 10}, 'ILE': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6, 'CD1': 7, 'OXT': 8}, 'LEU': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7, 'OXT': 8}, 'LYS': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'CE': 7, 'NZ': 8, 'OXT': 9}, 'MET': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'SD': 6, 'CE': 7, 'OXT': 8}, 'PHE': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7, 'CE1': 8, 'CE2': 9, 'CZ': 10, 'OXT': 11}, 'PRO': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'OXT': 7}, 'SER': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OG': 5, 'OXT': 6}, 'THR': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OG1': 5, 'CG2': 6, 'OXT': 7}, 'TRP': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7, 'NE1': 8, 'CE2': 9, 'CE3': 10, 'CZ2': 11, 'CZ3': 12, 'CH2': 13, 'OXT': 14}, 'TYR': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7, 'CE1': 8, 'CE2': 9, 'CZ': 10, 'OH': 11, 'OXT': 12}, 'VAL': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6, 'OXT': 7}, 'UNK': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'OXT': 6}, 'DA': {'OP3': 0, 'P': 1, 'OP1': 2, 'OP2': 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9, "C2'": 10, "C1'": 11, 'N9': 12, 'C8': 13, 'N7': 14, 'C5': 15, 'C6': 16, 'N6': 17, 'N1': 18, 'C2': 19, 'N3': 20, 'C4': 21}, 'DC': {'OP3': 0, 'P': 1, 'OP1': 2, 'OP2': 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9, "C2'": 10, "C1'": 11, 'N1': 12, 'C2': 13, 'O2': 14, 'N3': 15, 'C4': 16, 'N4': 17, 'C5': 18, 'C6': 19}, 'DG': {'OP3': 0, 'P': 1, 'OP1': 2, 'OP2': 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9, "C2'": 10, "C1'": 11, 'N9': 12, 'C8': 13, 'N7': 14, 'C5': 15, 'C6': 16, 'O6': 17, 'N1': 18, 'C2': 19, 'N2': 20, 'N3': 21, 'C4': 22}, 'DT': {'OP3': 0, 'P': 1, 'OP1': 2, 'OP2': 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9, "C2'": 10, "C1'": 11, 'N1': 12, 'C2': 13, 'O2': 14, 'N3': 15, 'C4': 16, 'O4': 17, 'C5': 18, 'C7': 19, 'C6': 20}, 'DN': {'OP3': 0, 'P': 1, 'OP1': 2, 'OP2': 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9, "C2'": 10, "C1'": 11}, 'A': {'OP3': 0, 'P': 1, 'OP1': 2, 'OP2': 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9, "C2'": 10, "O2'": 11, "C1'": 12, 'N9': 13, 'C8': 14, 'N7': 15, 'C5': 16, 'C6': 17, 'N6': 18, 'N1': 19, 'C2': 20, 'N3': 21, 'C4': 22}, 'C': {'OP3': 0, 'P': 1, 'OP1': 2, 'OP2': 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9, "C2'": 10, "O2'": 11, "C1'": 12, 'N1': 13, 'C2': 14, 'O2': 15, 'N3': 16, 'C4': 17, 'N4': 18, 'C5': 19, 'C6': 20}, 'G': {'OP3': 0, 'P': 1, 'OP1': 2, 'OP2': 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9, "C2'": 10, "O2'": 11, "C1'": 12, 'N9': 13, 'C8': 14, 'N7': 15, 'C5': 16, 'C6': 17, 'O6': 18, 'N1': 19, 'C2': 20, 'N2': 21, 'N3': 22, 'C4': 23}, 'U': {'OP3': 0, 'P': 1, 'OP1': 2, 'OP2': 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9, "C2'": 10, "O2'": 11, "C1'": 12, 'N1': 13, 'C2': 14, 'O2': 15, 'N3': 16, 'C4': 17, 'O4': 18, 'C5': 19, 'C6': 20}, 'N': {'OP3': 0, 'P': 1, 'OP1': 2, 'OP2': 3, "O5'": 4, "C5'": 5, "C4'": 6, "O4'": 7, "C3'": 8, "O3'": 9, "C2'": 10, "O2'": 11, "C1'": 12}}
PROT_STD_RESIDUES_ONE_TO_THREE = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'X': 'UNK', '-': '-'}
PROTEIN_TYPES_ONE_LETTER = ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')
PROTEIN_COMMON_ONE_TO_THREE = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
RNA_TYPES = ('A', 'G', 'C', 'U')
DNA_TYPES = ('DA', 'DG', 'DC', 'DT')
ATOM14 = {'ALA': ('N', 'CA', 'C', 'O', 'CB'), 'ARG': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'), 'ASN': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'), 'ASP': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'), 'CYS': ('N', 'CA', 'C', 'O', 'CB', 'SG'), 'GLN': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'), 'GLU': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'), 'GLY': ('N', 'CA', 'C', 'O'), 'HIS': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'), 'ILE': ('N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'), 'LEU': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'), 'LYS': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'), 'MET': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'), 'PHE': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'), 'PRO': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'), 'SER': ('N', 'CA', 'C', 'O', 'CB', 'OG'), 'THR': ('N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'), 'TRP': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'), 'TYR': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'), 'VAL': ('N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'), 'UNK': ()}
DENSE_ATOM = {**ATOM14, 'A': ('OP3', 'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'), 'C': ('OP3', 'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'), 'G': ('OP3', 'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'), 'U': ('OP3', 'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6'), 'DA': ('OP3', 'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'), 'DC': ('OP3', 'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'), 'DG': ('OP3', 'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'), 'DT': ('OP3', 'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6')}
ATOM14_PADDED = {k: list(v) + [''] * (14 - len(v)) for k, v in ATOM14.items()}
ATOM37 = ('N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT')
ATOM37_ORDER = {name: i for i, name in enumerate(ATOM37)}
ATOM37_NUM = len(ATOM37)
_CHI_ANGLES_MASK = ((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 0.0), (1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
_CHI_ANGLES_ATOMS = {'ALA': [], 'ARG': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'), ('CB', 'CG', 'CD', 'NE'), ('CG', 'CD', 'NE', 'CZ')], 'ASN': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'OD1')], 'ASP': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'OD1')], 'CYS': [('N', 'CA', 'CB', 'SG')], 'GLN': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'), ('CB', 'CG', 'CD', 'OE1')], 'GLU': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'), ('CB', 'CG', 'CD', 'OE1')], 'GLY': [], 'HIS': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'ND1')], 'ILE': [('N', 'CA', 'CB', 'CG1'), ('CA', 'CB', 'CG1', 'CD1')], 'LEU': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')], 'LYS': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'), ('CB', 'CG', 'CD', 'CE'), ('CG', 'CD', 'CE', 'NZ')], 'MET': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'SD'), ('CB', 'CG', 'SD', 'CE')], 'PHE': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')], 'PRO': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD')], 'SER': [('N', 'CA', 'CB', 'OG')], 'THR': [('N', 'CA', 'CB', 'OG1')], 'TRP': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')], 'TYR': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')], 'VAL': [('N', 'CA', 'CB', 'CG1')]}

def _make_restype_rigidgroup_dense_atom_idx():
    """Create Mapping from rigid_groups to dense_atom indices."""
    num_restypes_with_unk_and_gap = len(STD_RESIDUES_WITH_GAP)
    base_atom_indices = np.zeros((num_restypes_with_unk_and_gap, 8, 3), dtype=np.int32)
    for restype, restype_letter in enumerate(PROTEIN_TYPES_ONE_LETTER):
        resname = PROTEIN_COMMON_ONE_TO_THREE[restype_letter]
        dense_atom_names = ATOM14[resname]
        base_atom_indices[restype, 0, :] = [dense_atom_names.index(atom) for atom in ['C', 'CA', 'N']]
        base_atom_indices[restype, 3, :] = [dense_atom_names.index(atom) for atom in ['CA', 'C', 'O']]
        for chi_idx in range(4):
            if _CHI_ANGLES_MASK[restype][chi_idx]:
                atom_names = _CHI_ANGLES_ATOMS[resname][chi_idx]
                base_atom_indices[restype, chi_idx + 4, :] = [dense_atom_names.index(atom) for atom in atom_names[1:]]
    dense_atom_names = DENSE_ATOM['A']
    nucleic_rigid_atoms = [dense_atom_names.index(atom) for atom in ["C1'", "C3'", "C4'"]]
    for nanum, _ in enumerate(RNA_TYPES):
        resnum = nanum + RNA_START_INDEX
        base_atom_indices[resnum, 0, :] = nucleic_rigid_atoms
    for nanum, _ in enumerate(DNA_TYPES):
        resnum = nanum + DNA_START_INDEX
        base_atom_indices[resnum, 0, :] = nucleic_rigid_atoms
    return base_atom_indices
RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX = _make_restype_rigidgroup_dense_atom_idx()

def _make_restype_pseudobeta_idx():
    """Returns indices of residue's pseudo-beta."""
    num_restypes_with_unk_and_gap = len(STD_RESIDUES_WITH_GAP)
    restype_pseudobeta_index = np.zeros((num_restypes_with_unk_and_gap,), dtype=np.int32)
    for restype, restype_letter in enumerate(PROTEIN_TYPES_ONE_LETTER):
        restype_name = PROTEIN_COMMON_ONE_TO_THREE[restype_letter]
        atom_names = list(ATOM14[restype_name])
        if restype_name in {'GLY'}:
            restype_pseudobeta_index[restype] = atom_names.index('CA')
        else:
            restype_pseudobeta_index[restype] = atom_names.index('CB')
    for nanum, resname in enumerate(RNA_TYPES):
        atom_names = list(DENSE_ATOM[resname])
        restype = nanum + RNA_START_INDEX
        if resname in {'A', 'G'}:
            restype_pseudobeta_index[restype] = atom_names.index('C4')
        else:
            restype_pseudobeta_index[restype] = atom_names.index('C2')
    for nanum, resname in enumerate(DNA_TYPES):
        atom_names = list(DENSE_ATOM[resname])
        restype = nanum + DNA_START_INDEX
        if resname in {'DA', 'DG'}:
            restype_pseudobeta_index[restype] = atom_names.index('C4')
        else:
            restype_pseudobeta_index[restype] = atom_names.index('C2')
    return restype_pseudobeta_index
RESTYPE_PSEUDOBETA_INDEX = _make_restype_pseudobeta_idx()

def _make_aatype_dense_atom_to_atom37():
    """Map from dense_atom to atom37 per residue type."""
    num_dense = max((len(v) for v in DENSE_ATOM.values()))
    restype_dense_atom_to_atom37 = []
    for rt in PROTEIN_TYPES_ONE_LETTER:
        atom_names = ATOM14_PADDED[PROTEIN_COMMON_ONE_TO_THREE[rt]]
        extended_atom_names = list(atom_names) + [''] * (num_dense - len(atom_names))
        restype_dense_atom_to_atom37.append([ATOM37_ORDER[name] if name else 0 for name in extended_atom_names])
    for _ in range(2 + len(RNA_STD_RESIDUES) + len(DNA_STD_RESIDUES)):
        restype_dense_atom_to_atom37.append([0] * num_dense)
    restype_dense_atom_to_atom37 = np.array(restype_dense_atom_to_atom37, dtype=np.int32)
    return restype_dense_atom_to_atom37
PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37 = _make_aatype_dense_atom_to_atom37()
