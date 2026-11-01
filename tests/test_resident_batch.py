"""批量生命周期/输入路由/受限缓存测试；不运行GPU或真实MSA搜索。"""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from contextlib import redirect_stdout

import numpy as np
import torch
import biotite.structure as struc

from oridynamic import batch_predict as cli
from oridynamic.batch_runtime import ResidentRuntime, pack_atoms, unpack_atoms, write_prepared, read_prepared
from oridynamic.inference_timing import TimingLog
from oridynamic.runtime_utils import build_reference_identities


class ResidentBatchTests(unittest.TestCase):
    def args(self, root, **kwargs):
        fasta = root/'sequences.fasta'
        fasta.write_text('>one\nACDE\n>two\nACDF\n')
        args = cli.parser().parse_args(['--fasta',str(fasta),'--out',str(root/'out')])
        for k,v in kwargs.items():
            setattr(args,k,v)
        return args

    def test_missing_msa_does_not_fallback_online(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.args(Path(tmp))
            with self.assertRaisesRegex(ValueError,'online search is not automatic'):
                cli.load_targets(a)
            a.single_sequence=True
            self.assertEqual([r['msa_mode'] for r in cli.load_targets(a)], ['single_sequence']*2)

    def test_partial_msa_uses_local_only_for_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'one.a3m').write_text('>q\nACDE\n')
            a=self.args(root,msa_dir=root,msa_local_db=root/'db')
            rows=cli.load_targets(a)
            self.assertEqual([r['msa_mode'] for r in rows], ['provided','local'])
            self.assertEqual(rows[0]['msa_records'],1)

    def test_input_validation_and_optional_structures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.args(root,single_sequence=True,structure_dir=root)
            (root/'one.cif').write_text('mock_structure')
            rows=cli.load_targets(a)
            self.assertTrue(rows[0]['initial_structure'].endswith('one.cif'))
            self.assertNotIn('initial_structure',rows[1])
            (root/'one.pdb').write_text('mock_structure')
            with self.assertRaises(ValueError):cli.load_targets(a)
            a.structure_dir=None
            for text in ('>../escape\nACD\n', '>one\nACD\n>one\nACE\n', '>one\nACX\n'):
                a.fasta.write_text(text)
                with self.assertRaises(ValueError):cli.load_targets(a)

    def test_manifest_relative_and_old_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.args(root)
            (root/'one.a3m').write_text('>q\nACDE\n')
            a.fasta=None;a.manifest=root/'m.json'
            a.manifest.write_text(json.dumps(dict(schema='oridynamic_offline_batch_v1',targets=[
                dict(name='one',sequence='ACDE',msa_a3m='one.a3m',trajectory='not_read')])) )
            row=cli.load_targets(a)[0]
            self.assertNotIn('trajectory',row)
            self.assertEqual(row['msa_a3m'],str(root/'one.a3m'))

    def test_all_base_before_one_joint_load(self):
        engine=mock.Mock(); events=[]
        engine.load_base.side_effect=lambda:events.append('load_base')
        engine.prepare.side_effect=lambda t:events.append('prepare_'+t['name'])
        engine.load_joint.side_effect=lambda:events.append('load_joint')
        engine.sample.side_effect=lambda t:events.append('sample_'+t['name'])
        targets=[dict(name='one'),dict(name='two')]
        cli.execute_batch(engine,targets,lambda *a:None)
        self.assertEqual(events,['load_base','prepare_one','prepare_two','load_joint','sample_one','sample_two'])
        engine.load_base.assert_called_once();engine.load_joint.assert_called_once();engine.close.assert_called_once()

    def test_prepare_failure_never_mutates_base_to_joint(self):
        engine=mock.Mock();engine.prepare.side_effect=RuntimeError('prepare failed')
        with self.assertRaises(RuntimeError):cli.execute_batch(engine,[dict(name='x')],lambda *a:None)
        engine.load_joint.assert_not_called();engine.sample.assert_not_called();engine.close.assert_called_once()

    def atoms(self):
        atoms=struc.AtomArray(4)
        atoms.chain_id=['A']*4;atoms.res_id=[1]*4;atoms.res_name=['ALA']*4
        atoms.atom_name=['N','CA','C','O'];atoms.element=['N','C','C','O']
        atoms.coord=np.arange(12,dtype=np.float32).reshape(4,3)
        atoms.set_annotation('entity_id_int',np.zeros(4,dtype=int))
        atoms.set_annotation('sym_id_int',np.zeros(4,dtype=int))
        atoms.bonds=struc.BondList(4,np.asarray([[0,1,1],[1,2,1]],dtype=np.uint32))
        return atoms

    def test_atom_annotations_and_bonds_roundtrip(self):
        atoms=self.atoms();back=unpack_atoms(pack_atoms(atoms))
        np.testing.assert_array_equal(atoms.coord,back.coord)
        for name in atoms.get_annotation_categories():
            np.testing.assert_array_equal(atoms.get_annotation(name),back.get_annotation(name))
        np.testing.assert_array_equal(atoms.bonds.as_array(),back.bonds.as_array())

    def test_cache_weights_only_identity_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atoms=self.atoms();ids=build_reference_identities(atoms)
            features=dict(a=torch.ones(3).bfloat16(),scalar=torch.tensor(9))
            c0=(torch.rand(1,4),torch.rand(1,2),torch.rand(1,1,3))
            contract=dict(target='one',weights='hash',msa='hash',dtype='bf16')
            write_prepared(root,features,atoms,ids,torch.from_numpy(atoms.coord),c0,contract)
            payload,back=read_prepared(root,contract)
            torch.testing.assert_close(payload['features']['a'],features['a'])
            for got,want in zip(payload['c0'],c0):torch.testing.assert_close(got,want)
            self.assertEqual(build_reference_identities(back),ids)
            with self.assertRaises(ValueError):read_prepared(root,dict(contract,target='two'))
            with self.assertRaises(ValueError):write_prepared(root,features,atoms,ids,torch.zeros(4,3),c0,contract)
            with (root/'prepared.pt').open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):read_prepared(root,contract)

    def test_numpy_object_annotations_use_restricted_cache(self):
        # 原生特征中的object数组，tolist()后仍可能含np.str_。
        atoms=self.atoms()
        labels=np.empty(len(atoms),dtype=object)
        labels[:]=[np.str_('protein')]*len(atoms)
        atoms.set_annotation('chain_mol_type',labels)
        self.assertIsInstance(labels.tolist()[0],np.generic)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ids=build_reference_identities(atoms)
            features=dict(a=torch.ones(3).bfloat16())
            c0=(torch.rand(1,4),torch.rand(1,2),torch.rand(1,1,3))
            contract=dict(target='one',dtype='bf16')
            write_prepared(root,features,atoms,ids,torch.from_numpy(atoms.coord),c0,contract)
            # 必须无需新增allowlist即可读取，不能靠关闭weights_only绕过。
            raw=torch.load(root/'prepared.pt',map_location='cpu',weights_only=True)
            values=raw['atoms']['annotations']['chain_mol_type']['values']
            self.assertTrue(all(type(v) is str for v in values))
            payload,back=read_prepared(root,contract)
            torch.testing.assert_close(payload['features']['a'],features['a'],rtol=0,atol=0)
            for got,want in zip(payload['c0'],c0):
                torch.testing.assert_close(got,want,rtol=0,atol=0)
            np.testing.assert_array_equal(back.coord,atoms.coord)
            np.testing.assert_array_equal(back.bonds.as_array(),atoms.bonds.as_array())
            for name in atoms.get_annotation_categories():
                expected=atoms.get_annotation(name);actual=back.get_annotation(name)
                self.assertEqual(actual.dtype,expected.dtype)
                np.testing.assert_array_equal(actual,expected)
            self.assertEqual(build_reference_identities(back),ids)
            self.assertIsInstance(atoms.chain_mol_type[0],np.generic)

    def test_numpy_annotation_scalars_roundtrip(self):
        atoms=self.atoms()
        values=np.empty(len(atoms),dtype=object)
        values[:]=[np.str_('protein'),np.float64(0.75),np.int64(7),np.bool_(True)]
        atoms.set_annotation('mixed_metadata',values)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'atoms.pt'
            torch.save(pack_atoms(atoms),path)
            packed=torch.load(path,map_location='cpu',weights_only=True)
            actual=packed['annotations']['mixed_metadata']['values']
            self.assertEqual([type(v) for v in actual],[str,float,int,bool])
            back=unpack_atoms(packed)
            np.testing.assert_array_equal(back.mixed_metadata,values)
            self.assertEqual(back.mixed_metadata.dtype,values.dtype)

    def test_cache_normalizes_metadata_without_allowing_custom_objects(self):
        atoms=self.atoms();ids=build_reference_identities(atoms)
        features={np.str_('metadata'):(np.int64(4),[np.bool_(True),np.float64(0.5)])}
        contract=dict(target=np.str_('one'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_prepared(root,features,atoms,ids,torch.from_numpy(atoms.coord),(),contract)
            payload,_=read_prepared(root,contract)
            self.assertIs(type(next(iter(payload['features']))),str)
            metadata=payload['features']['metadata']
            self.assertIs(type(metadata),tuple)
            self.assertIs(type(metadata[0]),int)
            self.assertEqual([type(v) for v in metadata[1]],[bool,float])
            self.assertIs(type(payload['contract']['target']),str)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            values=np.empty(len(atoms),dtype=object)
            values[:]=[SimpleNamespace(value='unsupported')]*len(atoms)
            atoms.set_annotation('custom_metadata',values)
            with self.assertRaisesRegex(TypeError,'Unsupported cache object'):
                write_prepared(root,{},atoms,ids,torch.from_numpy(atoms.coord),(),contract)
            self.assertFalse((root/'prepared.pt').exists())

    def fake_resources(self, root):
        assets=root/'assets/common';assets.mkdir(parents=True)
        for name in ('components.cif','components.cif.rdkit_mol.pkl'):(assets/name).write_text('mock')
        (root/'weight.pt').write_text('mock')
        paths=root/'paths.json'
        paths.write_text(json.dumps(dict(checkpoints={k:'weight.pt' for k in ('native','codec','joint')})))
        return ['--paths',str(paths),'--assets-root',str(assets.parent)]

    def test_plan_has_no_writes_search_or_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.args(root)
            with mock.patch.object(cli,'run_search') as search, mock.patch.object(cli,'execute_batch') as execute, redirect_stdout(io.StringIO()):
                cli.main(['--fasta',str(a.fasta),'--out',str(a.out),'--single-sequence','--plan',*self.fake_resources(root)])
            search.assert_not_called();execute.assert_not_called();self.assertFalse(a.out.exists())

    def test_search_before_engine_and_query_not_replaced(self):
        # 正式main的输入/搜索/模型路由模拟；不使用网络、不加载真实大模型。
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.args(root);events=[]
            (root/'one.a3m').write_text('>q\nACDE\n')
            (root/'db').mkdir()
            def search(args,out):
                events.append('search_'+args.fasta.parent.name)
                out.mkdir();path=out/'result.a3m';path.write_text('>q\nACDF\n')
                return path,{}
            engine=mock.Mock(results=[],load_counts=dict(native=1,codec=1,joint=1))
            def build(*args):events.append('model_init');return engine
            cwd=Path.cwd()
            try:
                with mock.patch.dict(os.environ), mock.patch.object(cli,'run_search',side_effect=search), \
                     mock.patch.object(cli,'executable',side_effect=lambda s:s), \
                     mock.patch('oridynamic.batch_runtime.ResidentRuntime',side_effect=build), redirect_stdout(io.StringIO()):
                    cli.main(['--fasta',str(a.fasta),'--out',str(a.out),'--msa-dir',str(root),
                        '--msa-local-db',str(root/'db'),*self.fake_resources(root)])
            finally:
                os.chdir(cwd)
            self.assertEqual(events,['search_two','model_init'])
            self.assertTrue((a.out/'targets/one/provided.a3m').is_file())
            self.assertFalse((a.out/'targets/one/msa').exists())
            self.assertEqual(engine.load_base.call_count,1);self.assertEqual(engine.load_joint.call_count,1)

    def test_resident_sample_twice_real_tiny_codec_dit_diffusion(self):
        # 真正执行新sample路径和受限缓存；只替换trunk及CUDA管理，不伪造生成器输出。
        from fixture_inference import fixture
        model, old = fixture(checkpoint=False)
        model.native.relative_position_encoding=SimpleNamespace(generate_relp=lambda f:f)
        features=dict(old['features'], token_index=torch.arange(3), ref_space_uid=torch.zeros(6,dtype=torch.long))
        teacher=old['codec_context']
        c0=tuple(teacher[k] for k in ('s_inputs','s_trunk','z_trunk'))
        anchor=old['canonical_frame'].from_canonical(teacher['reference_coordinate'])
        atoms=struc.AtomArray(6)
        atoms.chain_id=['A']*6;atoms.res_id=[1,1,2,2,3,3];atoms.res_name=['GLY']*6
        atoms.atom_name=['N','CA']*3;atoms.element=['N','C']*3
        atoms.set_annotation('entity_id_int',np.zeros(6,dtype=int))
        atoms.set_annotation('sym_id_int',np.zeros(6,dtype=int))
        ids=build_reference_identities(atoms)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);a=self.args(root,single_sequence=True)
            a.out.mkdir();a.runtime_root=Path(__file__).resolve().parents[1]
            a.check=True;a.dtype='fp32';a.seeds='4';a.samples_per_seed=2
            a.timing=TimingLog(a.out);a.timing.configure_cuda(lambda:None)
            engine=ResidentRuntime.__new__(ResidentRuntime)
            engine.a=a;engine.timer=a.timing;engine.model=model;engine.device=torch.device('cpu')
            engine.contracts={};engine.results=[];engine.hashes={};engine.code={};engine.joint_step=0
            state={k:v.clone() for k,v in model.state_dict().items()}
            for name in ('one','two'):
                target=dict(name=name,msa_mode='single_sequence')
                folder=a.out/'targets'/name;folder.mkdir(parents=True)
                (folder/'anchor_phase').mkdir();(folder/'anchor_phase/timing.json').write_text('{"wall_seconds":0,"stages":[]}')
                contract=dict(target=name,anchor_selection=dict(source='user_structure'))
                engine.contracts[name]=contract
                write_prepared(folder,features,atoms,ids,anchor,c0,contract)
                with mock.patch('torch.cuda.get_device_name',return_value='CPU fixture'), \
                     mock.patch('torch.cuda.synchronize'), mock.patch('torch.cuda.empty_cache'), \
                     mock.patch('torch.cuda.get_rng_state',return_value=torch.get_rng_state()), \
                     mock.patch('torch.cuda.max_memory_allocated',return_value=0):
                    engine.sample(target)
            first=torch.load(a.out/'targets/one/seed_4.pt',weights_only=True)
            second=torch.load(a.out/'targets/two/seed_4.pt',weights_only=True)
            torch.testing.assert_close(first['final'],second['final'],atol=0,rtol=0)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
            self.assertEqual(len(engine.results),2)
            self.assertEqual(len([e for e in a.timing.data['stages'] if e['stage']=='ensemble_seed']),2)


if __name__=='__main__':unittest.main()
