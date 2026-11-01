"""用户入口；只调度已有推理/MSA，不引入训练或新的采样实现。"""
import argparse
import os
import sys

from oridynamic import __version__


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="oridynamic", description="OriDynamic: all-atom ensembles for monomers, protein-ligand complexes, and protein-protein assemblies")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("command", nargs="?", choices=("sample", "msa", "doctor"),
                   help="sample: generate ensembles; msa: local MSA search; doctor: read-only environment/resource checks")
    if not argv or argv[0] in ("-h", "--help", "--version"):
        p.parse_args(argv)
        if not argv: p.print_help()
        return 0
    # 保留python -m oridynamic --fasta ...旧调用，不改变其默认采样配置。
    command = "sample" if argv[0].startswith("-") else argv.pop(0)
    if command == "sample":
        from oridynamic.batch_predict import main as sample
        for flag, variable in (("--paths", "ORIDYNAMIC_PATHS"), ("--assets-root", "ORIDYNAMIC_ASSETS")):
            if not any(v == flag or v.startswith(flag + "=") for v in argv) and os.environ.get(variable):
                argv.extend((flag, os.environ[variable]))
        sample(argv)
        return 0
    if command == "msa":
        from oridynamic.local_msa import main as search
        previous = sys.argv
        try:
            sys.argv = ["oridynamic msa", *argv]
            search()
        finally:
            sys.argv = previous
        return 0
    if command == "doctor":
        from oridynamic.doctor import main as doctor
        return doctor(argv)
    p.error(f"Unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
