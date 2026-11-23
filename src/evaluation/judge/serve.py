"""Launch the pinned judge as a loopback-only vLLM API in a dedicated GPU environment."""
import argparse
import importlib.metadata
import json
import shlex
import subprocess
import sys
from pathlib import Path

from src.evaluation.judge.run import CONFIG, ROOT, RUBRIC, committed_protocol, load_config


def command(config, port=8001):
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', config['model_id'], '--revision', config['revision'],
        '--tokenizer-revision', config['revision'], '--served-model-name', config['served_model_name'],
        '--tokenizer-mode', 'mistral', '--config-format', 'mistral', '--load-format', 'mistral',
        '--dtype', 'bfloat16', '--max-model-len', str(config['max_model_len']),
        '--max-num-seqs', '20', '--gpu-memory-utilization', '0.85',
        '--host', '127.0.0.1', '--port', str(port), '--disable-log-requests']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8001)
    ap.add_argument('--record-dir', default='results/judge_serving')
    ap.add_argument('--print-command', action='store_true')
    args = ap.parse_args()
    config = load_config()
    cmd = command(config, args.port)
    if args.print_command:
        print(shlex.join(cmd))
        return
    protocol = committed_protocol(CONFIG, RUBRIC)
    if importlib.metadata.version('vllm') != config['vllm_version']:
        raise RuntimeError('Install the pinned serving requirements in a dedicated environment')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU supporting BF16 is required; use an A100 for the pilot')
    memory = torch.cuda.get_device_properties(0).total_memory / 2**30
    if memory < 22:
        raise RuntimeError('Insufficient memory for this BF16 configuration; do not silently quantize the judge')
    record = Path(args.record_dir)
    record.mkdir(parents=True, exist_ok=False)
    metadata = {'model': config, 'protocol': protocol, 'command': cmd,
                'gpu': torch.cuda.get_device_name(0), 'gpu_memory_gib': memory,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'status': 'LAUNCH_REQUESTED_NOT_YET_HEALTH_CHECKED'}
    (record/'serving_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (record/'environment.txt').write_text(freeze.stdout)
    # Health/readiness and model alias are checked by the client before scoring.
    raise SystemExit(subprocess.run(cmd, cwd=ROOT).returncode)


if __name__ == '__main__':
    main()
