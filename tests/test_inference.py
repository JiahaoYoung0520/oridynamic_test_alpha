"""CPU接口/原生小模块回归；全尺寸GPU由用户运行。"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ["PROTENIX_DISABLE_FUSED_LN"] = "1"
from predict import select_runtime
import predict
ROOT = Path(predict.__file__).resolve().parent
select_runtime(ROOT)
import torch
import numpy as np
from inference_runtime import choose_anchor, make_batch, map_structure, write_coordinates


class InferenceTests(unittest.TestCase):
    def test_selection_rejects_invalid_and_stable_tie(self):
        xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.assertEqual(choose_anchor([xyz, xyz+1], [.5, .5], 4), 0)
        self.assertEqual(choose_anchor([xyz*float("nan"), xyz, xyz[:3]], [1., .1, 3.], 4), 1)
        self.assertEqual(choose_anchor([xyz*0, xyz], [3., .1], 4), 1)
        with self.assertRaises(ValueError):
            choose_anchor([xyz], [float("nan")], 4)

    def test_native_small_joint_runs_without_targets(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from fixture_inference import fixture
        from runtime_utils import generate_samples_chunked
        torch.set_num_threads(1)
        model, old = fixture(checkpoint=False)
        model.eval().requires_grad_(False)
        teacher = old["codec_context"]
        anchor = old["canonical_frame"].from_canonical(teacher["reference_coordinate"])
        batch = make_batch(old["features"], anchor,
                           tuple(teacher[k] for k in ("s_inputs", "s_trunk", "z_trunk")))
        self.assertNotIn("target", batch)
        self.assertNotIn("frames", batch)
        with torch.no_grad():
            online = model.online_context(batch["features"])
            torch.manual_seed(19)
            got = generate_samples_chunked(model, batch, online, reference_present=False,
                                          samples=3, chunk_size=1, sampler="sde_eta10")
            torch.manual_seed(19)
            prior = generate_samples_chunked(model, old, online, reference_present=False,
                                            samples=3, chunk_size=1, sampler="sde_eta10")
        self.assertEqual(got["final"].shape, (3, 6, 3))
        torch.testing.assert_close(got["final"], prior["final"], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(batch["codec_context"]["input_feature_dict"]["ref_pos"], old["features"]["ref_pos"])

    def test_structure_mapping_and_cif_roundtrip(self):
        import biotite.structure as struc
        from runtime_utils import build_reference_identities
        atoms = struc.AtomArray(4)
        atoms.chain_id = ["A"]*4
        atoms.res_id = [42]*4
        atoms.res_name = ["ALA"]*4
        atoms.atom_name = ["N", "CA", "C", "O"]
        atoms.element = ["N", "C", "C", "O"]
        atoms.set_annotation("entity_id_int", np.zeros(4, dtype=int))
        atoms.set_annotation("sym_id_int", np.zeros(4, dtype=int))
        xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        identities = build_reference_identities(atoms)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"anchor.cif"
            error = write_coordinates(path, atoms, identities, xyz[None])
            self.assertLessEqual(error, 5.1e-4)
            expected = atoms.copy()
            expected.res_id[:] = 1
            torch.testing.assert_close(map_structure(path, expected), xyz)
            expected.atom_name[-1] = "CB"
            with self.assertRaises(ValueError):
                map_structure(path, expected)
            with self.assertRaises(ValueError):
                map_structure(path, atoms, chain="B")
            with self.assertRaises(ValueError):
                write_coordinates(path, atoms, identities, xyz[None])

    def test_no_training_interface_and_local_imports(self):
        from fixture_inference import fixture
        from protenix.model.protenix import Protenix
        from protenix.model import generator
        model, batch = fixture(checkpoint=False)
        with self.assertRaises(RuntimeError):
            model.train()
        with self.assertRaises(RuntimeError):
            model(batch)
        self.assertFalse(hasattr(Protenix, 'main_train_loop'))
        self.assertFalse(hasattr(generator, 'sample_diffusion_training'))
        self.assertFalse(hasattr(generator, 'TrainingNoiseSampler'))
        self.assertFalse(hasattr(model, 'native_loss'))
        self.assertFalse(hasattr(model, 'legacy_replay'))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        select_runtime(ROOT)
        with self.assertRaises(ValueError):
            select_runtime(ROOT.parent)


if __name__ == "__main__":
    unittest.main()
