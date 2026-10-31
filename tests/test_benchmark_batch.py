"""离线批量输入、路径、预算与指纹的CPU回归，不运行GPU/搜索。"""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import io
from contextlib import redirect_stdout

import torch
from benchmark_batch import load_manifest, command, summarize, save_progress
from inference_runtime import fingerprints
import benchmark_batch


class BatchTests(unittest.TestCase):
    def fixture(self, root):
        (root/'msa.a3m').write_text('>query\nACDE\n>other\nACDE\n')
        path=root/'manifest.json'
        path.write_text(json.dumps(dict(schema='oridynamic_offline_batch_v1',
            targets=[dict(name='test_A',sequence='ACDE',msa_a3m='msa.a3m')])))
        return path

    def test_manifest_relative_and_msa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=load_manifest(self.fixture(root))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['msa_a3m'],str(root/'msa.a3m'))
            self.assertEqual(rows[0]['msa_records'],2)
            (root/'msa.a3m').write_text('>query\nAAAA\n')
            with self.assertRaises(ValueError):load_manifest(root/'manifest.json')

    def test_manifest_rejects_duplicate_and_unsafe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=self.fixture(root); doc=json.loads(path.read_text())
            for name in ('test_A','../escape'):
                doc['targets'].append(dict(doc['targets'][0],name=name));path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):load_manifest(path)
                doc['targets'].pop()

    def test_command_is_local_offline_and_anchor_optional(self):
        args=SimpleNamespace(out=Path('/output'),assets_root='/assets',paths='/weights.json',seeds='0,1',
            samples_per_seed=8,sample_chunk_size=1,anchor_candidates=4,anchor_seed=101,
            attention_chunk_size=32,attention_backend='torch',dtype='bf16',device='cuda:0',trace_inputs=False)
        row=dict(name='target',msa_a3m='/msa.a3m')
        cmd=command(args,row,Path('/output/targets/target'))
        self.assertEqual(Path(cmd[3]).resolve(),Path(benchmark_batch.__file__).resolve().parent/'predict.py')
        self.assertIn('--msa-a3m',cmd);self.assertNotIn('--single-sequence',cmd)
        self.assertNotIn('--msa-local-db',cmd);self.assertNotIn('--initial-structure',cmd)
        row.update(initial_structure='/anchor.cif',structure_chain='A');args.trace_inputs=True
        cmd=command(args,row,Path('/output/targets/target'))
        self.assertIn('--initial-structure',cmd);self.assertIn('--trace-inputs',cmd)

    def test_summary_rejects_search_and_wrong_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            done=dict(status='inference_complete',msa_mode='provided',total_samples=16,n_tokens=4,n_atoms=20)
            timing=dict(status='inference_complete',total_samples=16,wall_seconds=10,ensemble_seconds=4,
                ensemble_seconds_per_sample=.25,stages=[dict(stage='ensemble_seed',status='complete',seconds=4)])
            (root/'completion.json').write_text(json.dumps(done));(root/'timing.json').write_text(json.dumps(timing))
            row=summarize(root,dict(name='x',tokens=4),11,16)
            self.assertEqual(row['ensemble_seconds'],4)
            save_progress(root,[row,dict(name='y',status='failed',error='test')],2,20)
            self.assertEqual(json.loads((root/'summary.json').read_text())['completed'],1)
            with self.assertRaises(ValueError):summarize(root,dict(name='x',tokens=4),11,8)
            timing['stages'].append(dict(stage='msa_online',status='complete',seconds=1))
            (root/'timing.json').write_text(json.dumps(timing))
            with self.assertRaises(ValueError):summarize(root,dict(name='x',tokens=4),11,16)

    def test_fingerprints_do_not_change_rng_or_values(self):
        x=torch.randn(3,4).bfloat16(); before=x.clone(); rng=torch.get_rng_state().clone()
        value={'x':x,'scalar':torch.tensor(7),'empty':torch.empty(0),'nested':[3,None]}
        a=fingerprints(value);b=fingerprints(value)
        self.assertEqual(a,b);self.assertTrue(torch.equal(x,before));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertNotEqual(fingerprints(x),fingerprints(x.float()))

    def test_plan_does_not_launch_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=self.fixture(root); out=root/'out'
            with mock.patch('sys.argv',['benchmark_batch.py','--manifest',str(path),'--out',str(out),'--plan']), \
                 mock.patch.object(benchmark_batch.subprocess,'Popen') as launch, redirect_stdout(io.StringIO()):
                benchmark_batch.main()
            launch.assert_not_called();self.assertFalse(out.exists())

    def test_failed_target_does_not_stop_remaining_or_count_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=self.fixture(root); doc=json.loads(path.read_text());out=root/'out'
            doc['targets'].append(dict(doc['targets'][0],name='second'));path.write_text(json.dumps(doc))
            def launch(cmd, **kwargs):
                folder=Path(cmd[cmd.index('--out')+1]);folder.mkdir()
                if folder.name=='test_A':return SimpleNamespace(wait=lambda **kw:1)
                done=dict(status='inference_complete',msa_mode='provided',total_samples=16,n_tokens=4,n_atoms=20)
                timing=dict(status='inference_complete',total_samples=16,wall_seconds=10,ensemble_seconds=4,
                    ensemble_seconds_per_sample=.25,stages=[dict(stage='ensemble_seed',status='complete',seconds=4)])
                (folder/'completion.json').write_text(json.dumps(done));(folder/'timing.json').write_text(json.dumps(timing))
                self.assertNotIn('PYTHONPATH',kwargs['env'])
                return SimpleNamespace(wait=lambda **kw:0)
            with mock.patch('sys.argv',['benchmark_batch.py','--manifest',str(path),'--out',str(out)]), \
                 mock.patch.object(benchmark_batch.subprocess,'Popen',side_effect=launch) as run, redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as failed:benchmark_batch.main()
            self.assertEqual(failed.exception.code,1);self.assertEqual(run.call_count,2)
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual((summary['completed'],summary['failed']),(1,1))
            self.assertEqual(summary['totals']['samples'],16)


if __name__=='__main__':unittest.main()
