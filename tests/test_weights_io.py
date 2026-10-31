"""推理包受限加载与拒绝训练状态；不需要真实权重。"""
from pathlib import Path
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

    def test_no_unrestricted_load_in_release(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('weights_io.py', 'inference_runtime.py', 'runtime_utils.py'):
            self.assertNotIn('weights_only=False', (root / 'oridynamic' / name).read_text())


if __name__ == '__main__':
    unittest.main()
