"""公开PTX JSON输入的薄适配；不构造化学图、不读取轨迹或缓存特征。"""
import copy
import json
from pathlib import Path
import re

from oridynamic.local_msa import validate_a3m
from oridynamic.prepare import AMINO_ACIDS, file_info


def proteins(target):
    """单体旧接口保持；复合物按蛋白entity准备MSA，同源拷贝共享一份。"""
    return target.get('protein_inputs', [target])


def native_sample(target):
    if 'native_input' not in target:
        protein = dict(sequence=target['sequence'], count=1)
        if target.get('msa_a3m'):
            protein['unpairedMsaPath'] = target['msa_a3m']
        return dict(name=target['name'], sequences=[dict(proteinChain=protein)])
    sample = copy.deepcopy(target['native_input'])
    for row in proteins(target):
        entity = sample['sequences'][row['entity_index']]['proteinChain']
        for native_key, key in (('unpairedMsaPath', 'msa_a3m'), ('pairedMsaPath', 'paired_msa_a3m')):
            if row.get(key):
                entity[native_key] = row[key]
    return sample


def resolved_file(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError('文件路径须为非空字符串')
    p = Path(value)
    return (p if p.is_absolute() else base / p).resolve(strict=True)


def load_native_targets(path, args):
    """使用PTX的name/sequences/proteinChain/ligand格式，拒绝未接入字段。"""
    from oridynamic.batch_predict import validate_name
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('--input-json须为非空PTX任务数组')
    result, names = [], set()
    base = Path(path).resolve().parent
    for original in rows:
        if not isinstance(original, dict):
            raise ValueError('每个任务须为JSON对象')
        unknown = set(original) - {'name', 'sequences', 'initial_structure', 'structure_chains'}
        if unknown:
            raise ValueError(f'尚未接入的任务字段：{sorted(unknown)}；不静默丢弃约束/共价键')
        name = original.get('name'); validate_name(name)
        if name in names:
            raise ValueError(f'重复目标名：{name}')
        names.add(name)
        entities = copy.deepcopy(original.get('sequences'))
        if not isinstance(entities, list) or not entities:
            raise ValueError(f'{name}: sequences必须为非空列表')
        target = dict(name=name, protein_inputs=[], native_input=dict(name=name, sequences=entities))
        ids, protein_count, ligand_count = set(), 0, 0
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict) or len(entity) != 1 or next(iter(entity)) not in ('proteinChain', 'ligand'):
                raise ValueError('本入口接受proteinChain和ligand；RNA/DNA/离子/共价修饰另验，不静默转换')
            kind, spec = next(iter(entity.items()))
            if not isinstance(spec, dict):
                raise ValueError('entity内容须为对象')
            allowed = {'id', 'count', 'sequence', 'unpairedMsaPath', 'pairedMsaPath'} if kind == 'proteinChain' else {'id', 'count', 'ligand'}
            if set(spec) - allowed:
                raise ValueError(f'{name}/{kind}: 未接入字段 {sorted(set(spec)-allowed)}')
            count = spec.setdefault('count', 1)
            chain_ids = spec.get('id')
            if type(count) is not int or count < 1 or not isinstance(chain_ids, list) or len(chain_ids) != count:
                raise ValueError('每个entity须显式id列表，且长度与正整数count一致（默认count=1）')
            for chain in chain_ids:
                if not isinstance(chain, str) or not re.fullmatch(r'[A-Za-z0-9]{1,8}', chain) or chain in ids:
                    raise ValueError('链ID须为1–8位字母数字，并在整个复合物中唯一')
                ids.add(chain)
            if kind == 'proteinChain':
                seq = spec.get('sequence')
                if not isinstance(seq, str) or not seq or set(seq) - (AMINO_ACIDS-{'X'}):
                    raise ValueError('蛋白链须为明确的20种标准氨基酸大写序列')
                row = dict(name=chain_ids[0], sequence=seq, entity_index=index)
                for native_key, key in (('unpairedMsaPath', 'msa_a3m'), ('pairedMsaPath', 'paired_msa_a3m')):
                    if spec.get(native_key):
                        p = resolved_file(spec[native_key], base)
                        _, n = validate_a3m(p, seq)
                        row[key], row[key+'_identity'] = str(p), file_info(p)
                        row[key+'_records'] = n
                        spec[native_key] = str(p)
                if row.get('msa_a3m'):
                    row['msa_mode'] = 'provided'
                elif args.msa_local_db:
                    row['msa_mode'] = 'local'
                elif args.single_sequence:
                    row['msa_mode'] = 'single_sequence'
                else:
                    raise ValueError(f'{name}/{chain_ids[0]}缺少unpaired MSA：提供A3M或本地库；不会自动调用在线服务')
                target['protein_inputs'].append(row)
                protein_count += count
            else:
                value = spec.get('ligand')
                if not isinstance(value, str) or not value.strip():
                    raise ValueError('ligand须为CCD_CODE、SMILES或FILE_文件路径')
                if value.startswith('CCD_') and not re.fullmatch(r'CCD_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*', value):
                    raise ValueError('CCD标识格式错误')
                if value.startswith('FILE_'):
                    p = resolved_file(value[5:], base)
                    if p.suffix.lower() not in ('.mol', '.sdf', '.mol2', '.pdb'):
                        raise ValueError('配体FILE_仅支持PTX原生MOL/SDF/MOL2/PDB')
                    spec['ligand'] = 'FILE_' + str(p)
                    target.setdefault('ligand_files', []).append(file_info(p))
                ligand_count += count
        if protein_count == 0:
            raise ValueError('本入口至少需要一条蛋白链，不将孤立小分子当PL')
        target.update(modality='pl' if ligand_count else ('pp' if protein_count > 1 else 'monomer'),
                      protein_chains=protein_count, ligand_chains=ligand_count)
        modes = {p['msa_mode'] for p in proteins(target)}
        target['msa_mode'] = next(iter(modes)) if len(modes) == 1 else 'mixed'
        if original.get('initial_structure'):
            p = resolved_file(original['initial_structure'], base)
            if p.suffix.lower() not in ('.cif', '.mmcif', '.pdb'):
                raise ValueError('初始复合物结构须为PDB/mmCIF')
            target.update(initial_structure=str(p), initial_structure_identity=file_info(p))
        mapping = original.get('structure_chains')
        if mapping is not None:
            if not target.get('initial_structure') or not isinstance(mapping, dict) or set(mapping) != ids:
                raise ValueError('structure_chains须在给定anchor时覆盖全部输入链ID')
            if any(not isinstance(v, str) or not v for v in mapping.values()) or len(set(mapping.values())) != len(ids):
                raise ValueError('structure_chains必须一对一映射到anchor的author链ID')
            target['structure_chains'] = mapping
        result.append(target)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit必须为正')
        result = result[:args.limit]
    return result
