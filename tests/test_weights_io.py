"""推理包受限加载与拒绝训练状态；不需要真实权重。"""
from pathlib import Path
import ast
import hashlib
from unittest.mock import patch
import tempfile
import unittest
import torch
from oridynamic.weights_io import load_weights, validate_payload, SCHEMAS, META_KEYS, JOINT_KEYS, PROFILE


class WeightTests(unittest.TestCase):
    def test_native_exact_roundtrip(self):
        value = dict(schema_version=SCHEMAS['native'], source_sha256='0'*64,
                     model={'weight': torch.arange(12).reshape(3, 4)})
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'weights.pt'
            torch.save(value, path)
            got = load_weights(path, 'native')
            torch.testing.assert_close(got['model']['weight'], value['model']['weight'], rtol=0, atol=0)

    def test_training_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            validate_payload(dict(model={'x': torch.ones(1)}, optimizer={}), 'native')
        good = dict(schema_version=SCHEMAS['native'], source_sha256='0'*64, model={'x': torch.ones(1)})
        with self.assertRaises(ValueError):
            validate_payload(dict(good, optimizer={}), 'native')
        with self.assertRaises(ValueError):
            validate_payload(dict(good, model={'x': [1]}), 'native')

    def test_joint_metadata_is_closed(self):
        value = dict(schema_version=SCHEMAS['joint'], source_sha256='0'*64, step=1,
            trainable_state={'latent_mean': torch.zeros(2)},
            metadata={k: {} for k in META_KEYS}, joint_config={k: 1 for k in JOINT_KEYS})
        value['metadata']['native_inference_profile'] = PROFILE
        value['joint_config']['activation_checkpoint'] = False
        validate_payload(value, 'joint')
        value['metadata']['paths'] = {'source_manifest': '/private/data'}
        with self.assertRaises(ValueError):
            validate_payload(value, 'joint')

    def test_official_native_requires_exact_file_hash(self):
        from oridynamic.weights_io import native_compatibility_hash, LEGACY_NATIVE_SHA256
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'official.pt'
            state = {'weight': torch.arange(6).reshape(2, 3)}
            torch.save({'model': state}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            # An arbitrary model-only checkpoint must not bypass the export contract.
            with self.assertRaises(ValueError):
                load_weights(path, 'native')
            with patch('oridynamic.weights_io.OFFICIAL_NATIVE_SHA256', digest):
                got = load_weights(path, 'native')
                self.assertEqual(got['source_sha256'], digest)
                torch.testing.assert_close(got['model']['weight'], state['weight'], rtol=0, atol=0)
                self.assertEqual(native_compatibility_hash(digest), LEGACY_NATIVE_SHA256)
                torch.save({'model': {'weight': torch.zeros(2, 3)}}, path)
                with self.assertRaises(ValueError):
                    load_weights(path, 'native')
            self.assertEqual(native_compatibility_hash('unknown'), 'unknown')

    def test_all_direct_torch_load_calls_are_restricted(self):
        root = Path(__file__).resolve().parents[1]
        count = 0
        for directory in ('oridynamic', 'protenix', 'runner'):
            for path in (root / directory).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Call) and ast.unparse(node.func) in ('torch.load', 'torch.serialization.load'):
                        value = next((k.value for k in node.keywords if k.arg == 'weights_only'), None)
                        self.assertIsInstance(value, ast.Constant, str(path))
                        self.assertIs(value.value, True, str(path))
                        count += 1
        self.assertGreaterEqual(count, 3)


if __name__ == '__main__':
    unittest.main()
