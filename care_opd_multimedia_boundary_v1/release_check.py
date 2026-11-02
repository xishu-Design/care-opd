"""Run CPU tests in source and extracted package before publishing immutable dist files."""
from __future__ import annotations

import argparse
import ast
import contextlib
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

import boundary_common as c
import build_bundle
import check_package


def test_suite(folder):
    process = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                              '-s', '.', '-p', 'test_boundary*.py', '-v'], cwd=folder,
                             capture_output=True, encoding='utf-8', errors='replace',
                             env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    log = process.stdout + process.stderr
    match = re.search(r'Ran (\d+) tests', log)
    if process.returncode or match is None:
        print(log)
        raise RuntimeError('CPU test suite failed: ' + str(folder))
    return {'status': 'PASS', 'test_count': int(match.group(1)), 'log': log}


def release(output):
    root = Path(__file__).resolve().parent
    syntax = []
    for path in sorted(root.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=path.name)
        syntax.append(path.name)
    source_tests = test_suite(root)
    bash = shutil.which('bash')
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    if git_bash.is_file():
        bash = str(git_bash)
    shell_check = {'status': 'NOT_RUN', 'reason': 'bash unavailable'}
    if bash:
        process = subprocess.run([bash, '-n', str(root / 'run_boundary_gpu1.sh')],
                                 capture_output=True, encoding='utf-8', errors='replace')
        if process.returncode:
            raise RuntimeError('bash syntax check failed: ' + process.stderr)
        shell_check = {'status': 'PASS', 'command': 'bash -n run_boundary_gpu1.sh'}
    with tempfile.TemporaryDirectory(prefix='boundary-release-check-') as temporary:
        stage = Path(temporary)
        with contextlib.redirect_stdout(io.StringIO()):
            package = build_bundle.build(stage / 'dist')
        extracted = stage / 'extracted'
        extracted.mkdir()
        with tarfile.open(package['archive'], 'r:gz') as archive:
            for member in archive.getmembers():
                target = (extracted / member.name).resolve()
                if not member.isfile() or not target.is_relative_to(extracted):
                    raise ValueError('Unsafe archive member')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
        with contextlib.redirect_stdout(io.StringIO()):
            inventory = check_package.verify(extracted)
        archive_tests = test_suite(extracted)
        for entry in ('boundary_run.py', 'boundary_prepare.py', 'boundary_fetch.py'):
            process = subprocess.run([sys.executable, '-B', str(extracted / entry), '--help'],
                                     capture_output=True, encoding='utf-8', errors='replace')
            if process.returncode:
                raise RuntimeError('Extracted entrypoint failed: ' + entry + process.stderr)
    final = build_bundle.build(output)
    if final['sha256'] != package['sha256']:
        raise RuntimeError('Source changed during release QA')
    report = {'schema_version': 'care_opd.multimedia_boundary.release_qa.v1',
              'status': 'LOCAL_CPU_PACKAGE_PASS', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'syntax_checked': syntax, 'shell_syntax': shell_check,
              'source_tests': source_tests, 'extracted_archive_tests': archive_tests,
              'archive': final, 'manifest_file_count': len(inventory['files']),
              'actual_dataset_download_tested': False, 'actual_chartqa_24_block_selection_tested': False,
              'real_gpu_model_inference_tested': False, 'scientific_results_generated': False,
              'outstanding_server_checks': ['real pinned validation selection', 'GPU preflight', 'single block inference smoke'],
              'integrity_controls': ['private gold not parsed by generation', 'first completed output immutable',
                                     'whole block paired bootstrap', 'identical observation cache for all arms',
                                     'no deletion or overwrite of previous experiments', 'negative outputs retained']}
    c.json_once(Path(output) / 'RELEASE_QA_REPORT.json', report)
    c.write_once(Path(output) / 'SERVER_COMMANDS_GPU1.md',
                 (root / 'SERVER_COMMANDS_GPU1.md').read_bytes().replace(b'\r\n', b'\n'))
    print('RELEASE_LOCAL_CPU_PASS', source_tests['test_count'], archive_tests['test_count'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'dist')
    release(parser.parse_args().output)
