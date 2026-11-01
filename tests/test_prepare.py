"""准备接口的小检查；不证明真实PTX/Joint推理通过。"""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from oridynamic.prepare import build_request, seed_list, write_request


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fasta = self.root / "query.fasta"
        self.fasta.write_text(">query\nACDEFG\n")
        self.args = argparse.Namespace(fasta=self.fasta, name=None, seeds="0,1,2,3,4",
            samples_per_seed=50, sample_chunk_size=1, msa_a3m=None,
            msa_server_mode="protenix", initial_structure=None)

    def test_native_input_and_budget(self):
        native, request = build_request(self.args)
        self.assertEqual(native[0]["sequences"][0]["proteinChain"],
                         {"sequence": "ACDEFG", "count": 1})
        self.assertEqual(request["ensemble"]["total_samples"], 250)
        self.assertFalse(request["ready_for_inference"])
        self.args.sample_chunk_size = 16
        self.assertEqual(build_request(self.args)[1]["ensemble"]["total_samples"], 250)

    def test_invalid_budget_and_multichain(self):
        for bad in ("", "1,1", "-1", "4294967296", "x"):
            with self.assertRaises(ValueError):
                seed_list(bad)
        self.args.samples_per_seed = 0
        with self.assertRaises(ValueError):
            build_request(self.args)
        self.fasta.write_text(">a\nAAA\n>b\nCCC\n")
        with self.assertRaisesRegex(ValueError, "one protein sequence"):
            build_request(self.args)

    def test_structure_and_msa_are_only_prechecked(self):
        self.args.initial_structure = self.root / "anchor.cif"
        self.args.initial_structure.write_text("data_fixture_not_validated\n")
        self.args.msa_a3m = self.root / "query.a3m"
        self.args.msa_a3m.write_text(">query\nACDEFG\n>match\nACDEFG\n")
        native, request = build_request(self.args)
        self.assertEqual(request["anchor"]["source"], "user_structure")
        self.assertIn("not_checked", request["anchor"]["validation"])
        self.assertIn("unpairedMsaPath", native[0]["sequences"][0]["proteinChain"])
        self.args.msa_a3m.write_text(">wrong\nACDFFG\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            build_request(self.args)

    def test_new_output_only(self):
        native, request = build_request(self.args)
        out = self.root / "prepared"
        write_request(out, native, request)
        before = (out / "request.json").read_bytes()
        with self.assertRaises(ValueError):
            write_request(out, native, request)
        self.assertEqual((out / "request.json").read_bytes(), before)
        self.assertEqual(json.loads(before)["status"], "prepared_not_executed")
        link = self.root / "linked"
        link.symlink_to(out, target_is_directory=True)
        with self.assertRaises(ValueError):
            write_request(link, native, request)


if __name__ == "__main__":
    unittest.main()
