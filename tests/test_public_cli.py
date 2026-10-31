"""公开CLI路由、资源检查与单目标快捷输入；不运行模型。"""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import batch_predict
from oridynamic.cli import main
from oridynamic.doctor import inspect


class PublicCliTests(unittest.TestCase):
    def test_help_and_version_without_torch(self):
        for command in (['--help'], ['--version'], ['sample', '--help'], ['msa', '--help']):
            code = "from oridynamic.cli import main; import sys\ntry: main(" + repr(command) + ")\nexcept SystemExit as e: assert e.code == 0\nassert 'torch' not in sys.modules"
            subprocess.run([sys.executable, '-B', '-c', code], check=True, capture_output=True)

    def test_sample_alias_and_environment_precedence(self):
        with patch.dict(os.environ, {'ORIDYNAMIC_PATHS': '/weights.json', 'ORIDYNAMIC_ASSETS': '/assets'}), \
                patch('batch_predict.main') as sample:
            main(['sample', '--fasta', 'a.fasta', '--out', 'run'])
            self.assertEqual(sample.call_args.args[0][-4:], ['--paths', '/weights.json', '--assets-root', '/assets'])
            main(['--fasta', 'a.fasta', '--paths=custom.json', '--out', 'run'])
            self.assertNotIn('/weights.json', sample.call_args.args[0])

    def test_single_files_route_to_original_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / 'one.fasta'; fasta.write_text('>one\nACDE\n')
            msa = root / 'one.a3m'; msa.write_text('>query\nACDE\n')
            anchor = root / 'anchor.cif'; anchor.write_text('not_a_real_structure')
            a = batch_predict.parser().parse_args(['--fasta', str(fasta), '--msa-a3m', str(msa),
                '--initial-structure', str(anchor), '--structure-chain', 'A', '--out', str(root / 'out')])
            rows = batch_predict.load_targets(a)
            self.assertEqual(rows[0]['msa_mode'], 'provided')
            self.assertEqual(rows[0]['initial_structure'], str(anchor))
            self.assertEqual(rows[0]['structure_chain'], 'A')
            fasta.write_text('>one\nACDE\n>two\nACDF\n')
            with self.assertRaises(ValueError): batch_predict.load_targets(a)

    def test_doctor_does_not_load_weights_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('importlib.metadata.version', return_value='test-version'):
                report = inspect(root / 'missing.json', root / 'missing_assets')
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(list(root.iterdir()), [])

    def test_unknown_command_rejected(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit): main(['train'])


if __name__ == '__main__': unittest.main()
