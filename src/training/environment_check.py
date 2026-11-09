"""Fail before training when the runtime differs from an explicit environment target."""
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def collect_environment():
    versions = {'python': platform.python_version()}
    modules = {}
    for name in ('torch', 'numpy', 'pandas', 'pyarrow'):
        modules[name] = importlib.import_module(name)
        versions[name] = str(modules[name].__version__)
    torch = modules['torch']
    cuda = torch.cuda.is_available()
    return {'versions': versions, 'cuda_available': cuda,
            'cuda_build': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_names': [torch.cuda.get_device_name(i)
                          for i in range(torch.cuda.device_count())] if cuda else [],
            'platform': platform.platform(), 'python_executable': sys.executable}


def environment_mismatches(actual, expected):
    required = {'python', 'torch', 'numpy', 'pandas', 'pyarrow'}
    if set(expected.get('versions', {})) != required:
        raise ValueError('Environment target must specify Python and all four library versions')
    if not all(isinstance(v, str) and v for v in expected['versions'].values()):
        raise ValueError('Environment target versions must be nonempty strings')
    if type(expected.get('require_cuda')) is not bool:
        raise ValueError('Environment target must declare require_cuda as a boolean')
    found = [f'{name}: actual {actual.get("versions", {}).get(name)!r}, expected {version!r}'
             for name, version in expected['versions'].items()
             if actual.get('versions', {}).get(name) != version]
    if expected['require_cuda'] and actual.get('cuda_available') is not True:
        found.append('CUDA is required but unavailable')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', default='configs/colab_environment.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    expected = json.loads(Path(args.spec).read_text())
    actual = collect_environment()
    mismatches = environment_mismatches(actual, expected)
    report = {'status': 'FAIL' if mismatches else 'PASS',
              'checked_utc': datetime.now(timezone.utc).isoformat(),
              'expected': expected, 'actual': actual, 'mismatches': mismatches,
              'pip_freeze': subprocess.check_output(
                  [sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if mismatches:
        parser.exit(2, 'Environment mismatch: stop before training. Restore the declared runtime '
                    'or agree on a new environment target for a separately recorded experiment. '
                    'Do not mix environments across grid/repeat runs.\n')


if __name__ == '__main__':
    main()
