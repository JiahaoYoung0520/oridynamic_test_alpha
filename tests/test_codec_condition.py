"""真实小模块验证两种codec条件、非原地路由及跨调用cache隔离。"""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).parent))
from fixture_inference import fixture
from oridynamic.codec_condition import FIXED_CONTEXT, ONLINE_CONTEXT, joint_model_class
from oridynamic.inference_runtime import load_parent
from protenix.model.modules.joint_ensemble import JointEnsembleModel


class CodecConditionTests(unittest.TestCase):
    def test_online_condition_matches_explicit_routing_and_keeps_input(self):
        torch.set_num_threads(1)
        model, batch = fixture(checkpoint=False)
        original = batch['codec_context']
        original_tensors = {k: original[k].clone() for k in ('s_inputs', 's_trunk', 'z_trunk')}
        online = tuple(original[k] + 0.7 for k in original_tensors)
        noise, atom_noise = torch.randn(2, 3, 3), torch.randn(2, 6, 3)
        args = dict(reference_present=False, noise=noise, atom_noise=atom_noise, sampler='sde_eta10')
        torch.manual_seed(72)
        fixed = model.generate(batch, online, **args)
        model.__class__ = joint_model_class(ONLINE_CONTEXT)
        for contexts in (online, tuple(v - 0.2 for v in online)):
            routed = dict(batch, codec_context=dict(original, **dict(zip(original_tensors, contexts))))
            torch.manual_seed(72)
            expected = JointEnsembleModel.generate(model, routed, contexts, **args)
            torch.manual_seed(72)
            actual = model.generate(batch, contexts, **args)
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                self.assertTrue(torch.isfinite(actual[key]).all())
            self.assertIs(batch['codec_context'], original)
        self.assertFalse(torch.allclose(fixed['proposal'], actual['proposal']))
        for k, value in original_tensors.items():
            torch.testing.assert_close(original[k], value, atol=0, rtol=0)

    def test_contract_is_explicit_and_unknown_rejected_before_hashing(self):
        self.assertIs(joint_model_class(FIXED_CONTEXT), JointEnsembleModel)
        with patch('oridynamic.inference_runtime.load_weights', return_value={
                'metadata': {'condition_contract': 'unknown'}}), \
             patch('oridynamic.inference_runtime.file_info') as info:
            with self.assertRaisesRegex(ValueError, 'codec'):
                load_parent({'joint': Path('unused.pt')})
            info.assert_not_called()


if __name__ == '__main__':
    unittest.main()
