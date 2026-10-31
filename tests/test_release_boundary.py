"""只检查发行目录：数学关键路径未改、无训练入口、无外部运行代码。"""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def normalized_ast(value):
    if value is Ellipsis:
        return {'constant': 'Ellipsis'}
    if isinstance(value, ast.AST):
        return {'node': type(value).__name__, **{k: normalized_ast(v) for k, v in ast.iter_fields(value)
            if k != 'type_params' and v is not None and v != []}}
    if isinstance(value, list):
        return [normalized_ast(v) for v in value]
    return value


class ReleaseBoundaryTests(unittest.TestCase):
    def test_sampling_math_matches_recorded_source(self):
        expected = json.loads((ROOT / 'tests/sampling_ast.json').read_text())
        tree = ast.parse((ROOT / 'protenix/model/modules/joint_ensemble.py').read_text())
        for name, digest in expected.items():
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            node.decorator_list = []
            signature = json.dumps(normalized_ast(node), sort_keys=True)
            self.assertEqual(hashlib.sha256(signature.encode()).hexdigest(), digest, name)

    def test_no_training_entry_or_objective(self):
        self.assertFalse(list((ROOT / 'runner').glob('train*.py')))
        for top in ('protenix', 'runner'):
            for path in (ROOT / top).rglob('*.py'):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        self.assertNotIn(node.name, ('main_train_loop', 'native_training_loss',
                            'ensemble_statistics', 'geometry_losses', 'legacy_replay', 'save_checkpoint'))
                    if isinstance(node, ast.Call):
                        text = ast.unparse(node.func)
                        self.assertFalse(text.startswith('torch.optim.'), str(path))
        data = json.loads((ROOT / 'configs/inference_defaults.json').read_text())
        self.assertNotIn('loss', data)
        self.assertNotIn('train_sets', data['data'])
        self.assertNotIn('/storage9920/', json.dumps(data))


if __name__ == '__main__':
    unittest.main()
