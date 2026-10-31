"""离线构建wheel/sdist并在临时环境检查；不下载依赖/权重、不启动GPU。"""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.release_bundle import export


def run(command, cwd):
    subprocess.run(command, cwd=cwd, check=True)


def main():
    with tempfile.TemporaryDirectory(prefix='oridynamic-package-') as tmp:
        tmp = Path(tmp); source = tmp / 'source'; wheels = tmp / 'wheels'
        export(source)
        run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--no-index',
             '--wheel-dir', str(wheels), str(source)], tmp)
        wheel, = wheels.glob('*.whl')
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
        for required in ('oridynamic/cli.py', 'batch_runtime.py', 'complex_inputs.py', 'configs/inference_defaults.json',
                         'configs/base_diffusion_signature.json', 'protenix/model/modules/joint_ensemble.py'):
            assert required in names, required
        assert not any(name.startswith(('runs/', 'inputs/', 'weights/', 'assets/', 'tests/', 'tools/'))
                       or name.endswith(('.pt', '.npz', '.cif', '.pdb', '.xtc', '.hdf5')) for name in names)
        run([sys.executable, '-c', 'from setuptools.build_meta import build_sdist; build_sdist("dist")'], source)
        sdist, = (source / 'dist').glob('*.tar.gz')
        with tarfile.open(sdist) as archive:
            parts = [Path(name).parts[1:] for name in archive.getnames()]
        assert not any(set(p) & {'runs', 'inputs', 'weights', 'assets', 'datasets'} for p in parts)
        env = tmp / 'env'
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env)
        python = str(env / 'bin/python')
        run([python, '-m', 'pip', 'install', '--no-deps', '--no-index', str(wheel)], tmp)
        code = '''
from pathlib import Path
import sys
import oridynamic, batch_predict, inference_runtime, batch_runtime, complex_inputs, configs, protenix, runner
for module in (oridynamic, batch_predict, inference_runtime, batch_runtime, complex_inputs, configs, protenix, runner):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix)), module.__file__
assert (Path(inference_runtime.__file__).parent/'configs/inference_defaults.json').is_file()
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
