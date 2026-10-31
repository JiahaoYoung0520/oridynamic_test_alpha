"""只使用release+公共CCD，验收五例原生特征/完整anchor映射/CIF回读；无权重/训练/GPU。"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-root',type=Path,required=True)
    p.add_argument('--output-json',type=Path)
    a=p.parse_args()
    os.environ.update(PROTENIX_ROOT_DIR=str(a.assets_root.resolve()), LAYERNORM_TYPE='torch', PROTENIX_DISABLE_FUSED_LN='1')
    import numpy as np
    import torch
    from ml_collections import ConfigDict
    import batch_predict
    from complex_inputs import native_sample
    from inference_runtime import prepare_features, map_structure, write_coordinates, make_batch
    from inference_timing import TimingLog
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from runtime_utils import reset_template_probe_rng, build_reference_identities
    from predict import select_runtime
    select_runtime(ROOT)
    args=batch_predict.parser().parse_args(['--input-json',str(ROOT/'examples/five_cases.json'),
        '--single-sequence','--out','unused'])
    rows=batch_predict.load_targets(args);reports=[]
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='oridynamic-features-') as tmp:
        for row in rows:
            out=Path(tmp)/row['name'];out.mkdir()
            cfg=ConfigDict(json.loads((ROOT/'configs/inference_defaults.json').read_text()))
            cfg.use_template=False;cfg.use_msa=True;cfg.esm.enable=False
            cfg.input_json_path=str(out/'input.json');cfg.dump_dir=str(out/'native')
            sample=native_sample(row)
            print('[CPU原生特征]',row['name'],flush=True)
            f,atoms,ids=prepare_features(SimpleNamespace(out=out,single_sequence=True,msa_a3m=None,
                input_json=True,timing=TimingLog(out)),cfg,[sample])
            # 与直接调用原生SampleDictToFeatures逐值核对化学特征，而非自行生成另一份化学参考。
            reset_template_probe_rng(101)
            raw,raw_atoms,_=SampleDictToFeatures(copy.deepcopy(sample),extract_features_for_tfg=False).get_feature_dict()
            assert build_reference_identities(raw_atoms)==ids
            keys=('ref_pos','ref_mask','ref_element','ref_charge','ref_atom_name_chars','atom_to_token_idx')
            for key in keys:
                torch.testing.assert_close(f[key],torch.as_tensor(raw[key]).to(f[key].dtype),rtol=0,atol=0)
            xyz=torch.randn(len(atoms),3,generator=torch.Generator().manual_seed(8))
            n=len(f['token_index'])
            context=(torch.zeros(n,449),torch.zeros(n,384),torch.zeros(n,n,128))
            batch=make_batch(f,xyz,context)
            for key in keys:
                torch.testing.assert_close(batch['codec_context']['input_feature_dict'][key],f[key],rtol=0,atol=0)
            torch.testing.assert_close(batch['canonical_frame'].from_canonical(
                batch['codec_context']['reference_coordinate'],batch['atom_mask']),xyz,rtol=0,atol=1e-5)
            path=out/'roundtrip.cif'
            error=write_coordinates(path,atoms,ids,xyz[None])
            mapped=map_structure(path,atoms)
            torch.testing.assert_close(mapped,xyz,rtol=0,atol=5.1e-4)
            reports.append(dict(name=row['name'],modality=row['modality'],atoms=len(atoms),
                tokens=len(f['token_index']),chains=len(set(atoms.chain_id.tolist())),
                protein_atoms=int(f['is_protein'].sum()),ligand_atoms=int(f['is_ligand'].sum()),
                native_chemical_features_equal=True,canonical_roundtrip=True,cif_error_A=error))
    result=dict(status='cpu_features_pass',cases=reports,model_forward=False,gpu_run=False,
                runtime_root=str(ROOT),external_research_imports=False)
    if a.output_json:
        with a.output_json.open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
