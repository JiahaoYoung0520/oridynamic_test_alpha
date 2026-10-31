"""标准库假搜索器测试；不搜索真实数据库，不导入模型。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from oridynamic.local_msa import (add_arguments, attach_local_msa, command, preflight,
                       run_search, search_environment, validate_a3m, validate_modes)


class LocalMsaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="msa test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fasta = self.root / "input.fasta"
        self.fasta.write_text(">some_protein\nACDEFG\n")
        self.database = self.root / "db"
        self.database.mkdir()
        for name in ("uniref30_2302_db", "colabfold_envdb_202108_db"):
            (self.database / (name + ".dbtype")).write_bytes(b"\x00\x00\x00\x00")
        self.binary = self.root / "fake_search"
        self.binary.write_text(f"#!{sys.executable}\n" + '''import sys
from pathlib import Path
if "--help" in sys.argv:
    print("--gpu --gpu-server --db3 --use-templates --unpack")
elif "version" in sys.argv:
    print("test-only-mmseqs")
else:
    out = Path(sys.argv[3]); out.mkdir()
    (out / "query.a3m").write_text(">query\\nACDEFG\\n>hit\\nACdDE-G\\n")
    print("fake search completed (NOT a real database search)")
''')
        self.binary.chmod(0o755)
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        self.args = parser.parse_args(["--msa-local-db", str(self.database),
            "--msa-search-executable", str(self.binary), "--msa-mmseqs", str(self.binary)])
        self.args.fasta = self.fasta
        self.args.out = self.root / "run"

    def test_command_database_role_and_resources(self):
        report = preflight(self.args)
        cmd = command(self.args, self.args.out, report)
        self.assertEqual(cmd[cmd.index("--db3")+1], "colabfold_envdb_202108_db")
        self.assertNotIn("--db2", cmd)
        self.assertEqual(cmd[cmd.index("--use-templates")+1], "0")
        self.assertEqual(cmd[cmd.index("--gpu-server")+1], "0")
        self.assertLessEqual(report["threads"], 16)
        self.assertFalse(report["server_used"])
        self.assertFalse(self.args.out.exists())

    def test_search_attach_and_read_only_database(self):
        before = {p.name: p.read_bytes() for p in self.database.iterdir()}
        native = [{"sequences": [{"proteinChain": {"sequence": "ACDEFG", "count": 1}}]}]
        request = {"msa": {"mode": "auto"}}
        attach_local_msa(self.args, native, request)
        self.assertEqual(self.args.msa_a3m, self.args.out / "msa/unpaired.a3m")
        self.assertEqual(native[0]["sequences"][0]["proteinChain"]["unpairedMsaPath"], str(self.args.msa_a3m))
        self.assertEqual(request["msa"]["mode"], "local")
        self.assertFalse(request["msa"]["server_used"])
        report = json.loads((self.args.out / "msa/search.json").read_text())
        self.assertEqual((report["status"], report["n_records"]), ("complete", 2))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.database.iterdir()})
        with self.assertRaisesRegex(ValueError, "拒绝"):
            run_search(self.args, self.args.out / "msa")

    def test_failure_no_fallback(self):
        with patch("oridynamic.local_msa.validate_a3m", side_effect=ValueError("wrong query")):
            with self.assertRaisesRegex(ValueError, "wrong query"):
                run_search(self.args, self.args.out)
        report = json.loads((self.args.out / "search.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse((self.args.out / "unpaired.a3m").exists())
        self.assertFalse(report["server_used"])

    def test_a3m_query_columns_metadata(self):
        path = self.root / "result.a3m"
        path.write_text("#6\t1\n>query\nACDEFG\n>hit\nAC.deDE-G\n")
        normalized, n = validate_a3m(path, "ACDEFG")
        self.assertEqual(n, 2)
        self.assertNotIn("#", normalized)
        self.assertNotIn(".", normalized)
        for bad in (">query\nACDEFA\n", ">query\nACDEFG\n>hit\nACD\n", "#6\t2\n>query\nACDEFG\n"):
            path.write_text(bad)
            with self.assertRaises(ValueError):
                validate_a3m(path, "ACDEFG")

    def test_missing_database_and_executable(self):
        self.args.msa_search_executable = str(self.root / "absent")
        with self.assertRaisesRegex(ValueError, "未找到"):
            preflight(self.args)
        self.args.msa_search_executable = str(self.binary)
        self.args.msa_env_db = "wrong"
        with self.assertRaises(FileNotFoundError):
            preflight(self.args)

    def test_output_under_database_and_broken_symlink_rejected(self):
        with self.assertRaisesRegex(ValueError, "数据库目录内"):
            run_search(self.args, self.database / "bad")
        link = self.root / "broken"
        link.symlink_to(self.root / "not_created")
        with self.assertRaisesRegex(ValueError, "拒绝"):
            run_search(self.args, link)
        self.args.out = self.database / "inference"
        with self.assertRaisesRegex(ValueError, "数据库目录内"):
            preflight(self.args)
        self.assertFalse(self.args.out.exists())

    def test_conflicts_and_check_is_not_search(self):
        args = argparse.Namespace(msa_a3m=None, single_sequence=False, msa_local_db=self.database, check=False)
        validate_modes(args)
        args.msa_a3m = self.fasta
        with self.assertRaises(ValueError):
            validate_modes(args)
        args.msa_a3m = None
        args.check = True
        with self.assertRaisesRegex(ValueError, "不搜索"):
            validate_modes(args)

    def test_environment_isolation(self):
        with patch.dict(os.environ, PYTHONPATH="/wrong/worktree", PYTHONHOME="/wrong/env", CUDA_VISIBLE_DEVICES="3"):
            env = search_environment(self.args.out, 8, "gpu")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("PYTHONHOME", env)
            self.assertEqual(env["OMP_NUM_THREADS"], "8")
            self.assertEqual(search_environment(self.args.out, 8, "cpu")["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(os.environ["PYTHONPATH"], "/wrong/worktree")

    def test_predict_search_failure_before_runtime_import(self):
        # 只使用release源码；伪造权重，搜索失败后绝不进入模型加载。
        assets = self.root / "assets/common"; assets.mkdir(parents=True)
        for name in ("components.cif", "components.cif.rdkit_mol.pkl"):
            (assets / name).touch()
        weight = self.root / "fake.pt"; weight.write_text("not a checkpoint")
        config = self.root / "paths.json"
        config.write_text(json.dumps(dict(checkpoints={k:str(weight) for k in ("native", "codec", "joint")})))
        self.binary.write_text(f"#!{sys.executable}\nimport sys\nprint('--gpu --gpu-server --db3 --use-templates --unpack')\nsys.exit(0 if '--help' in sys.argv or 'version' in sys.argv else 7)\n")
        cmd = [sys.executable, "-B", "-m", "oridynamic.predict", "--fasta", str(self.fasta), "--paths", str(config),
               "--assets-root", str(assets.parent), "--msa-local-db", str(self.database),
               "--msa-search-executable", str(self.binary), "--msa-mmseqs", str(self.binary), "--out", str(self.args.out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        failure = json.loads((self.args.out / "failure.json").read_text())
        self.assertIn("code=7", failure["error"])
        self.assertFalse((self.args.out / "completion.json").exists())

    def test_docker_wrapper_without_docker_or_gpu(self):
        fake_docker = self.root / "docker"
        capture = self.root / "docker_args.json"
        fake_docker.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
if sys.argv[1:3] == ["image", "inspect"]:
    print("sha256:fake-test-image")
else:
    Path(os.environ["MSA_TEST_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))
    mount = next(a for a in sys.argv if a.endswith("dst=/work"))
    out = Path(mount.split("src=")[1].split(",dst=")[0]) / "msa"
    out.mkdir(); (out / "unpaired.a3m").write_text(">query\\nACDEFG\\n")
''')
        fake_docker.chmod(0o755)
        env = dict(os.environ, PATH=str(self.root)+os.pathsep+os.environ["PATH"],
                   MSA_TEST_CAPTURE=str(capture), MSA_DEVICE="gpu", MSA_GPUS="2,3")
        wrapper = Path(__file__).resolve().parents[1] / "tools/search_msa_docker.sh"
        cmd = ["bash", str(wrapper), str(self.fasta), str(self.database), str(self.args.out)]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads(capture.read_text())
        self.assertEqual(argv[argv.index("--network")+1], "none")
        self.assertEqual(argv[argv.index("--gpus")+1], '"device=2,3"')
        self.assertIn(f"type=bind,src={self.database},dst=/db,readonly", argv)
        self.assertEqual(subprocess.run(cmd, env=env, capture_output=True).returncode, 2)


if __name__ == "__main__":
    unittest.main()
