"""Score text with 20 independent sampled JSON responses per case (G-Eval adaptation)."""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'configs/judge/ministral.json'
RUBRIC = ROOT / 'configs/judge/rubric.txt'
CRITERIA = ('faithfulness', 'relevance', 'coherence', 'fluency')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text())
    if config['judge_family'] != 'mistral' or not config['model_id'].startswith('mistralai/'):
        raise ValueError('Expected the agreed non-Qwen Mistral judge')
    if not re.fullmatch(r'[a-f0-9]{40}', config['revision']):
        raise ValueError('An immutable model revision is required')
    if config['criteria'] != list(CRITERIA) or config['samples'] != 20:
        raise ValueError('The committed protocol requires four criteria and 20 samples')
    if config['scoring_method'] != 'mean_20_samples' or not 0 < config['temperature'] <= 2:
        raise ValueError('Expected stochastic mean-of-20 scoring')
    if config['seed'] != 2026:
        raise ValueError('Expected committed seed 2026')
    return config


def committed_protocol(config_path, rubric_path):
    """Real scoring requires the exact protocol and runner to be in Git already."""
    paths = [Path(config_path).resolve(), Path(rubric_path).resolve(), Path(__file__).resolve(),
             Path(__file__).with_name('serve.py'), Path(__file__).with_name('requirements-serving.txt'),
             ROOT / 'docs/metrics_commitment.md']
    hashes = {}
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            frozen = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'HEAD:{relative}'], stderr=subprocess.DEVNULL)
            if frozen != path.read_bytes():
                raise ValueError('Commit the protocol and runner before real scoring')
            hashes[relative] = digest(path)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError('Real scoring requires a Git checkout with committed protocol files') from exc
    return {'commit': commit, 'files_sha256': hashes}


def validate_cases(cases, config, mode):
    if not cases:
        raise ValueError('No cases supplied')
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('Each case must be an object')
        cid = case.get('case_id')
        if not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', cid) or cid in seen:
            raise ValueError('Use unique opaque case IDs')
        seen.add(cid)
        if case.get('task') not in ('explanation', 'profile'):
            raise ValueError('Unknown evaluation task')
        family = case.get('generator_family')
        if not isinstance(family, str) or not family.strip() or family.strip().lower() in ('mistral', 'ministral'):
            raise ValueError('Judge and generator must be different model families')
        for field in ('request', 'answer'):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError('Request and answer must be nonempty strings')
        evidence = case.get('evidence')
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('Grounding requires evidence')
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {'product_id', 'text'}:
                raise ValueError('Evidence requires product_id and text')
            if not all(isinstance(item[k], str) and item[k].strip() for k in item):
                raise ValueError('Evidence fields must be nonempty strings')
        ids = case.get('recommended_product_ids')
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
            raise ValueError('Supply a unique product-ID list (empty for profiles or no-match)')
        if case.get('synthetic') is not (mode == 'smoke'):
            raise ValueError('Smoke requires synthetic cases; evaluation refuses synthetic cases')
    if mode == 'evaluation':
        if len(cases) != 100 or any(sum(c['task'] == t for c in cases) != 50 for t in ('explanation', 'profile')):
            raise ValueError('Final protocol requires 50 explanations and 50 profiles')


def payload(case, config, rubric, seed):
    # Generator names and human scores are deliberately withheld from the judge.
    material = {k: case[k] for k in ('task', 'request', 'evidence', 'answer', 'recommended_product_ids')}
    return {'model': config['served_model_name'], 'messages': [
        {'role': 'system', 'content': rubric},
        {'role': 'user', 'content': json.dumps(material, ensure_ascii=False)}],
        'temperature': config['temperature'], 'top_p': config['top_p'],
        'seed': seed, 'n': config['samples'], 'max_tokens': config['max_tokens'],
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'quality_scores', 'strict': True, 'schema': {
                'type': 'object', 'properties': {k: {'type': 'integer', 'minimum': 1, 'maximum': 5} for k in CRITERIA},
                'required': list(CRITERIA), 'additionalProperties': False}}}}


def parse_scores(response, config):
    if response.get('model') != config['served_model_name']:
        raise ValueError('Endpoint returned a different model alias')
    choices = response.get('choices', [])
    if len(choices) != 20 or sorted(x.get('index', -1) for x in choices) != list(range(20)):
        raise ValueError('Exactly 20 distinct indexed samples are required')
    samples = []
    for choice in sorted(choices, key=lambda c: c['index']):
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Truncated or incomplete judge response')
        value = json.loads(choice['message']['content'])
        if not isinstance(value, dict) or set(value) != set(CRITERIA):
            raise ValueError('Judge returned invalid criterion keys')
        if any(type(v) is not int or not 1 <= v <= 5 for v in value.values()):
            raise ValueError('Scores must be integers from 1 to 5')
        samples.append(value)
    return {'samples': samples, 'scores': {k: sum(s[k] for s in samples) / 20 for k in CRITERIA}}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Endpoint redirects are not permitted')


class Endpoint:
    def __init__(self, base_url, api_key=None, timeout=300):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ('http', 'https') or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Use an HTTP(S) API base URL without credentials or query parameters')
        if parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('Remote endpoints must use HTTPS')
        self.base_url, self.api_key, self.timeout = base_url.rstrip('/'), api_key, timeout
        self.opener = urllib.request.build_opener(NoRedirect)

    def request(self, path, data=None):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        req = urllib.request.Request(self.base_url + path,
            data=None if data is None else json.dumps(data).encode(), headers=headers)
        try:
            with self.opener.open(req, timeout=self.timeout) as stream:
                return json.load(stream)
        except (urllib.error.URLError, TimeoutError) as exc:
            # Do not put endpoint response bodies or authentication headers into reports.
            raise RuntimeError('Judge endpoint request failed') from None

    def verify(self, config):
        models = self.request('/models')
        if config['served_model_name'] not in [m['id'] for m in models.get('data', [])]:
            raise ValueError('Pinned judge alias is not served by this endpoint')

    def __call__(self, request):
        return self.request('/chat/completions', request)


def run(cases_path, output, endpoint, mode='smoke', config_path=CONFIG, rubric_path=RUBRIC):
    config = load_config(config_path)
    cases = [json.loads(line) for line in Path(cases_path).read_text().splitlines() if line.strip()]
    validate_cases(cases, config, mode)
    protocol = committed_protocol(config_path, rubric_path)
    rubric = Path(rubric_path).read_text()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'mode': mode, 'synthetic_smoke': mode == 'smoke',
              'judge': config, 'protocol': protocol, 'input_sha256': digest(cases_path),
              'python': platform.python_version(), 'cases_expected': len(cases), 'cases_completed': 0,
              'endpoint_identity': 'served alias checked; remote weights not independently attested',
              'calibration_status': 'PENDING independent human ratings; not a reliability claim'}
    started = time.monotonic()
    failed = False
    try:
        endpoint.verify(config)
        with (output/'scores.jsonl').open('x') as scores_file:
            for index, case in enumerate(sorted(cases, key=lambda c: c['case_id'])):
                seed = config['seed'] + index
                scores = parse_scores(endpoint(payload(case, config, rubric, seed)), config)
                record = {'case_id': case['case_id'], 'task': case['task'], 'seed': seed,
                          'case_sha256': hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(), **scores}
                scores_file.write(json.dumps(record) + '\n')
                scores_file.flush()
                report['cases_completed'] += 1
    except Exception as exc:
        failed = True
        report.update(error_type=type(exc).__name__, error='Scoring stopped; incomplete run must not be reported as a final result.')
    report.update(status='FAIL' if failed else 'PASS', seconds=round(time.monotonic()-started, 3))
    (output/'run_metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:8001/v1')
    ap.add_argument('--mode', choices=['smoke', 'evaluation'], default='smoke')
    args = ap.parse_args()
    result = run(args.cases, args.output, Endpoint(args.base_url, os.environ.get('JUDGE_API_KEY')), args.mode)
    print(result['status'], result['cases_completed'], '/', result['cases_expected'], 'cases')
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
