"""离线构建wheel/sdist并在临时环境检查；不下载依赖/权重、不启动GPU。"""
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def copy_source(destination):
    """仅向临时构建目录复制安装源码与公开示例，不带入权重或运行结果。"""
    destination.mkdir()
    for name in ('README.md', 'INSTALL.md', 'LICENSE-PTX', 'requirements-inference.txt',
                 'pyproject.toml', 'MANIFEST.in'):
        shutil.copyfile(ROOT / name, destination / name)
    trees = {'oridynamic': {'.py'}, 'protenix': {'.py'}, 'runner': {'.py'},
             'configs': {'.py', '.json'}, 'docs': {'.md'}}
    files = []
    for directory, suffixes in trees.items():
        files.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts
                     and p.name != 'local_paths.example.json')
    files.extend(ROOT / 'examples' / name for name in
                 ('README.md', 'monomer.fasta', 'batch.fasta', 'five_cases.json', 'five_cases_with_msa.json'))
    files.extend((ROOT / 'examples/msas').glob('*.a3m'))
    for path in files:
        assert not path.is_symlink(), path
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def run(command, cwd):
    subprocess.run(command, cwd=cwd, check=True)


def main():
    with tempfile.TemporaryDirectory(prefix='oridynamic-package-') as tmp:
        tmp = Path(tmp); source = tmp / 'source'; wheels = tmp / 'wheels'
        copy_source(source)
        run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--no-index',
             '--wheel-dir', str(wheels), str(source)], tmp)
        wheel, = wheels.glob('*.whl')
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
        for required in ('oridynamic/cli.py', 'oridynamic/batch_runtime.py', 'oridynamic/complex_inputs.py', 'configs/inference_defaults.json',
                         'configs/base_diffusion_signature.json', 'protenix/model/modules/joint_ensemble.py'):
            assert required in names, required
        assert not any(name.startswith(('runs/', 'inputs/', 'weights/', 'assets/', 'tests/', 'tools/'))
                       or name.endswith(('.pt', '.npz', '.cif', '.pdb', '.xtc', '.hdf5')) for name in names)
        run([sys.executable, '-c', 'from setuptools.build_meta import build_sdist; build_sdist("dist")'], source)
        sdist, = (source / 'dist').glob('*.tar.gz')
        with tarfile.open(sdist) as archive:
            parts = [Path(name).parts[1:] for name in archive.getnames()]
        assert not any(set(p) & {'runs', 'inputs', 'weights', 'assets', 'datasets'} for p in parts)
        assert sum(tuple(p[:2]) == ('examples', 'msas') and p[-1].endswith('.a3m') for p in parts) == 6
        assert ('examples', 'five_cases_with_msa.json') in [tuple(p) for p in parts]
        env = tmp / 'env'
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env)
        python = str(env / 'bin/python')
        run([python, '-m', 'pip', 'install', '--no-deps', '--no-index', str(wheel)], tmp)
        code = '''
from pathlib import Path
import sys
import oridynamic, configs, protenix, runner
from oridynamic import batch_predict, inference_runtime, batch_runtime, complex_inputs
for module in (oridynamic, batch_predict, inference_runtime, batch_runtime, complex_inputs, configs, protenix, runner):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix)), module.__file__
assert (Path(inference_runtime.__file__).parents[1]/'configs/inference_defaults.json').is_file()
print('[PASS] 已安装wheel实际导入，无源码目录或研究仓库回退')
'''
        run([python, '-I', '-B', '-c', code], tmp)
        run([str(env / 'bin/oridynamic'), '--help'], tmp)
        run([str(env / 'bin/oridynamic'), 'sample', '--help'], tmp)
        run([str(env / 'bin/oridynamic'), 'doctor'], tmp)
        # discover只将tests目录加入路径，生产模块必须来自已安装wheel。
        run([python, '-I', '-B', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-q'], tmp)
        print(json.dumps(dict(status='pass', wheel_files=len(names), source_and_wheel_private_data=False,
                              gpu_tested=False, installed_runtime_verified=True)))


if __name__ == '__main__': main()
