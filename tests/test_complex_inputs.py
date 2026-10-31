"""PL/PP输入不丢链、不丢配体；不需要真实权重或CCD数据库的CPU回归。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import biotite.structure as struc

import batch_predict
from complex_inputs import native_sample, proteins
from inference_runtime import map_structure, validate_complex_features, write_coordinates
from runtime_utils import build_reference_identities


def tasks():
    protein = {'proteinChain': {'id':['A'], 'count':1, 'sequence':'ACDEFG'}}
    return [dict(name='pl', sequences=[protein, {'ligand':{'id':['L'],'count':1,'ligand':'CCD_ATP'}}]),
            dict(name='pp', sequences=[{'proteinChain':{'id':['A','B'],'count':2,'sequence':'ACDEFG'}}])]


class ComplexInputTests(unittest.TestCase):
    def load(self, root, rows, flags=('--single-sequence',)):
        path=root/'input.json'; path.write_text(json.dumps(rows))
        a=batch_predict.parser().parse_args(['--input-json',str(path),'--out',str(root/'out'),*flags])
        return batch_predict.load_targets(a)

    def test_native_entities_and_homomer_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=self.load(Path(tmp),tasks())
            self.assertEqual([r['modality'] for r in rows],['pl','pp'])
            self.assertEqual(native_sample(rows[0]),tasks()[0])
            self.assertEqual(native_sample(rows[1])['sequences'][0]['proteinChain']['count'],2)
            self.assertEqual(len(proteins(rows[1])),1)

    def test_per_chain_msa_and_file_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'a.a3m').write_text('>q\nACDEFG\n'); (root/'x.mol').write_text('not_parsed_yet')
            task=tasks()[0];p=task['sequences'][0]['proteinChain']
            p.update(unpairedMsaPath='a.a3m', pairedMsaPath='a.a3m')
            task['sequences'][1]['ligand']['ligand']='FILE_x.mol'
            row=self.load(root,[task],())[0]
            self.assertEqual(row['msa_mode'],'provided')
            native=native_sample(row)
            self.assertEqual(native['sequences'][1]['ligand']['ligand'],'FILE_'+str(root/'x.mol'))
            self.assertEqual(native['sequences'][0]['proteinChain']['pairedMsaPath'],str(root/'a.a3m'))

    def test_no_implicit_search_and_reject_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'不会自动调用在线'):self.load(root,tasks(),())
            for mutate in (
                lambda t:t.update(covalent_bonds=[]),
                lambda t:t['sequences'][1]['ligand'].update(id=['A']),
                lambda t:t['sequences'][0]['proteinChain'].update(sequence='ACX'),
                lambda t:t['sequences'][0]['proteinChain'].update(count=2),
                lambda t:t.update(structure_chains={'A':'X'}),
            ):
                task=tasks()[0];mutate(task)
                with self.assertRaises(ValueError):self.load(root,[task])

    def atoms(self):
        a=struc.AtomArray(8)
        a.chain_id=['A']*4+['L']*4;a.res_id=[1]*8
        a.res_name=['ALA']*4+['LIG']*4;a.atom_name=['N','CA','C','O','C1','C2','O1','N1']
        a.element=['N','C','C','O','C','C','O','N'];a.hetero=[False]*4+[True]*4
        a.coord=np.arange(24,dtype=np.float32).reshape(8,3)
        a.set_annotation('entity_id_int',np.array([0]*4+[1]*4))
        a.set_annotation('sym_id_int',np.zeros(8,dtype=int))
        return a

    def test_complete_complex_anchor_no_separate_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.atoms();ids=build_reference_identities(a);path=Path(tmp)/'a.cif'
            xyz=torch.from_numpy(a.coord.copy());xyz[4:]+=torch.tensor([10.,20.,30.])
            write_coordinates(path,a,ids,xyz[None])
            torch.testing.assert_close(map_structure(path,a),xyz)
            expected=a.copy();expected.chain_id=['X']*4+['Y']*4
            torch.testing.assert_close(map_structure(path,expected,chain_map={'X':'A','Y':'L'}),xyz)
            with self.assertRaises(ValueError):map_structure(path,expected)
            expected.atom_name[-1]='N9'
            with self.assertRaises(ValueError):map_structure(path,expected,chain_map={'X':'A','Y':'L'})

    def test_missing_ligand_or_duplicate_chain_map_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.atoms();path=Path(tmp)/'only_protein.cif';p=a[:4]
            write_coordinates(path,p,build_reference_identities(p),torch.from_numpy(p.coord)[None])
            with self.assertRaises(ValueError):map_structure(path,a)
            with self.assertRaises(ValueError):map_structure(path,a,chain_map={'A':'A','L':'A'})

    def test_feature_gate_preserves_ligand_and_chains(self):
        a=self.atoms(); f=dict(is_protein=torch.tensor([1]*4+[0]*4),is_ligand=torch.tensor([0]*4+[1]*4),
            asym_id=torch.tensor([0,1]), atom_to_token_idx=torch.tensor([0]*4+[1]*4),ref_pos=torch.from_numpy(a.coord))
        validate_complex_features(f,a,tasks()[0])
        with self.assertRaises(ValueError):validate_complex_features(f,a[:4],tasks()[0])
        bad=copy.deepcopy(f);bad['is_ligand'][:]=0
        with self.assertRaises(ValueError):validate_complex_features(bad,a,tasks()[0])


if __name__=='__main__': unittest.main()
