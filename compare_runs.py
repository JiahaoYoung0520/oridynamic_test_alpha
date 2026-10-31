#!/usr/bin/env python3
"""比较同anchor/同预算的两个推理产物；坐标差不是对GT的质量指标。"""
import argparse
import json
from pathlib import Path


def compare(a, b):
    import torch
    from inference_runtime import map_structure
    from biotite.structure.io import pdbx
    # 不运行网络；只读取完成的产物，地图不读取GT。
    left, right = [json.loads((p/'run_metadata.json').read_text()) for p in (a,b)]
    requests = [json.loads((p/'request.json').read_text())['msa'] for p in (a,b)]
    if requests[0]['mode'] != requests[1]['mode']:
        raise ValueError('两组MSA输入模式不同')
    if requests[0]['mode'] == 'provided' and requests[0]['input']['sha256'] != requests[1]['input']['sha256']:
        raise ValueError('两组提供的MSA内容不同')
    for key in ('status', 'checkpoint_sha256', 'n_atoms', 'n_tokens', 'seeds', 'samples_per_seed',
                'chunk_size', 'dtype', 'attention_backend', 'sampler', 'input_identity', 'effective_joint_config',
                'feature_seed', 'codec_cycles', 'online_cycles', 'tf32', 'code_sha256'):
        if left[key] != right[key] or left['status'] != 'inference_complete':
            raise ValueError(f'不能逐样本比较：{key}不一致/任务未完成')
    # 模型坐标轴相同才有直接RMS差的意义，不能将独立刚体变化当结构差。
    atoms = pdbx.get_structure(pdbx.CIFFile.read(a/'anchor.cif'), model=1, use_author_fields=True)
    x = map_structure(a/'anchor.cif', atoms)
    y = map_structure(b/'anchor.cif', atoms)
    if not torch.equal(x, y):
        raise ValueError('两个anchor坐标不完全相同；本工具不自动对齐或改变比较口径')
    rows=[]
    for seed in left['seeds']:
        packets=[torch.load(p/f'seed_{seed}.pt',map_location='cpu',weights_only=True) for p in (a,b)]
        if packets[0]['atom_identities'] != packets[1]['atom_identities']:
            raise ValueError('采样包原子身份不同')
        for key in ('latent','proposal','final'):
            u,v=[p[key].float() for p in packets]
            if u.shape != v.shape or not all(torch.isfinite(t).all() for t in (u,v)):
                raise ValueError(f'{seed}/{key}形状或有限性错误')
            delta=u-v
            row=dict(seed=seed,field=key,max_abs=float(delta.abs().max()),component_rms=float(delta.square().mean().sqrt()))
            if key != 'latent':
                rmsd=delta.square().sum(-1).mean(-1).sqrt()
                row.update(mean_paired_atom_rmsd_A=float(rmsd.mean()),max_paired_atom_rmsd_A=float(rmsd.max()))
            rows.append(row)
    paths=[p/'input_fingerprints.json' for p in (a,b)]
    trace={'status':'not_recorded_in_both_runs'}
    if all(p.is_file() for p in paths):
        f,g=[json.loads(p.read_text()) for p in paths]
        trace=dict(status='recorded',equal={k:f.get(k)==g.get(k) for k in sorted(set(f)|set(g))})
    return dict(anchor_exactly_equal=True, rows=rows, fingerprints=trace,
                note='同序号样本直接比较，不作Kabsch；不是GT重建或ensemble质量指标。无上下文指纹不能确认历史根因。')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('a',type=Path); p.add_argument('b',type=Path)
    a=p.parse_args()
    print(json.dumps(compare(a.a.resolve(),a.b.resolve()),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
