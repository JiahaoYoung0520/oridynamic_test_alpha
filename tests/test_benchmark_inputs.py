"""四组真实路由与计时汇总的CPU测试，不搜索或运行模型。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from benchmark_inputs import CASES, ORDER, command, main, parser, summarize, used_msa


class BenchmarkInputsTests(unittest.TestCase):
    def test_four_online_routes(self):
        a = parser().parse_args(['--fasta', 'query.fa', '--out', 'new', '--search-mode', 'online'])
        for case, (anchor, msa, _) in CASES.items():
            cmd = command(a, case, Path('anchor.cif'), Path('msa.a3m'))
            self.assertEqual('--initial-structure' in cmd, anchor)
            self.assertEqual('--msa-a3m' in cmd, msa)
            self.assertEqual('--msa-server-mode' in cmd, not msa)
            self.assertNotIn('--single-sequence', cmd)
            self.assertNotIn('--check', cmd)
        self.assertEqual(ORDER[0], 'neither')

    def test_four_local_routes(self):
        a = parser().parse_args(['--fasta', 'q.fa', '--out', 'new', '--search-mode', 'local',
                                '--msa-local-db', '/db', '--msa-threads', '32'])
        for case, (_, msa, _) in CASES.items():
            cmd = command(a, case, Path('anchor.cif'), Path('msa.a3m'))
            self.assertEqual('--msa-local-db' in cmd, not msa)
            self.assertNotIn('--msa-server-mode', cmd)
            if not msa:
                self.assertEqual(cmd[cmd.index('--msa-threads')+1], '32')

    def write_run(self, root, stages, status='inference_complete'):
        (root/'completion.json').write_text(json.dumps(dict(status=status)))
        (root/'timing.json').write_text(json.dumps(dict(status=status, wall_seconds=90,
            ensemble_seconds=20, total_samples=4, ensemble_seconds_per_sample=5,
            stages=[dict(stage=n, seconds=s, status='complete') for n,s in stages])))

    def test_summary_routes_and_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case, (anchor, msa, _) in CASES.items():
                stages = [('base_context', 3), ('online_context', 4), ('ensemble_seed', 20)]
                if not msa: stages.append(('msa_local', 30))
                if not anchor: stages.extend([('ptx_anchor', 5), ('ptx_anchor', 6)])
                self.write_run(root, stages)
                row = summarize(root, case, 95)
                self.assertEqual(row['process_seconds'], 95)
                self.assertEqual(row['context_seconds'], 7)
                self.assertEqual(row['anchor_predict_seconds'], 0 if anchor else 11)
                self.assertEqual(row['msa_search_seconds'], 0 if msa else 30)

    def test_reject_missing_search_and_cpu_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root, [('ptx_anchor', 3)])
            with self.assertRaises(ValueError): summarize(root, 'neither', 100)
            self.write_run(root, [], status='cpu_check_complete')
            with self.assertRaises(ValueError): summarize(root, 'anchor_msa', 100)

    def test_use_actual_resolved_msa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            msa = root/'search_output.a3m'
            msa.write_text('>query\nACD\n>hit\nAC-\n')
            inp = root/'actual.json'
            inp.write_text(json.dumps([dict(sequences=[dict(proteinChain=dict(unpairedMsaPath=str(msa)))])]))
            (root/'resolved_config.json').write_text(json.dumps(dict(input_json_path=str(inp))))
            path, count = used_msa(root, 'ACD')
            self.assertEqual(path, msa)
            self.assertEqual(count, 2)
            with self.assertRaises(ValueError): used_msa(root, 'AAA')

    def test_complete_matrix_with_fake_inference(self):
        # 仅模拟子进程产物，核验调度/路径/汇总；不当成真实推理速度。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root/'input.fa'
            fasta.write_text('>q\nACD\n')
            out = root/'out'
            calls = []
            def fake_run(cmd, **kwargs):
                folder = Path(cmd[cmd.index('--out')+1])
                folder.mkdir()
                case = folder.name
                calls.append(case)
                anchor, msa, _ = CASES[case]
                if msa:
                    path = Path(cmd[cmd.index('--msa-a3m')+1])
                else:
                    path = folder/'actual.a3m'
                    path.write_text('>q\nACD\n>homolog\nA-D\n')
                if anchor:
                    self.assertTrue(Path(cmd[cmd.index('--initial-structure')+1]).is_file())
                (folder/'anchor.cif').write_text('fake_cpu_test_not_real_structure')
                inp = folder/'input.json'
                inp.write_text(json.dumps([dict(sequences=[dict(proteinChain=dict(unpairedMsaPath=str(path)))])]))
                (folder/'resolved_config.json').write_text(json.dumps(dict(input_json_path=str(inp))))
                stages = [('ensemble_seed', 20)]
                if not msa: stages.append(('msa_online', 30))
                if not anchor: stages.append(('ptx_anchor', 5))
                self.write_run(folder, stages)
                self.assertNotIn('PYTHONPATH', kwargs['env'])
            argv = ['benchmark_inputs.py', '--fasta', str(fasta), '--out', str(out), '--search-mode', 'online']
            with patch('sys.argv', argv), patch('benchmark_inputs.subprocess.run', side_effect=fake_run):
                main()
            self.assertEqual(calls, list(ORDER))
            rows = json.loads((out/'summary.json').read_text())
            self.assertEqual([r['case'] for r in rows], list(CASES))
            self.assertTrue(all(r['msa_matches_first_run'] for r in rows))
            self.assertFalse((out/'failure.json').exists())

    def test_plan_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root/'input.fa'
            fasta.write_text('>q\nACD\n')
            argv = ['benchmark_inputs.py', '--fasta', str(fasta), '--out', str(root/'plan'),
                    '--search-mode', 'online', '--plan']
            with patch('sys.argv', argv), patch('benchmark_inputs.subprocess.run') as run, patch('builtins.print'):
                main()
            run.assert_not_called()
            self.assertFalse((root/'plan').exists())


if __name__ == '__main__':
    unittest.main()
