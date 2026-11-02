"""Build a deterministic, source-only deployment archive and validate its inventory."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import boundary_common as c

ROOT = Path(__file__).resolve().parent
ARCHIVE_NAME = 'CARE_OPD_MULTIMEDIA_BOUNDARY_GPU1_V1.tar.gz'
SOURCE_NAMES = (
    'README.md', 'SERVER_COMMANDS_GPU1.md', 'requirements_data.txt',
    'boundary_common.py', 'boundary_prepare.py', 'boundary_fetch.py',
    'boundary_run.py', 'boundary_score.py', 'check_package.py', 'run_boundary_gpu1.sh',
    'test_boundary_prepare.py', 'test_boundary_score.py', 'test_boundary_run.py',
    'test_boundary_package.py', 'build_bundle.py', 'release_check.py',
)


def build(output, runtime_dir=None):
    runtime = Path(runtime_dir) if runtime_dir else ROOT / 'runtime'
    if not runtime.is_dir():
        runtime = ROOT.parent / 'care_opd/mechanism_experiments/visual_evidence_transfer'
    files = {name: (ROOT / name).read_bytes().replace(b'\r\n', b'\n') for name in SOURCE_NAMES}
    for name in ('evaluate.py', 'common.py'):
        files['runtime/' + name] = (runtime / name).read_bytes().replace(b'\r\n', b'\n')
    # Source-only artifacts: no checkpoints, clinical records, labels or model outputs.
    manifest = {'schema_version': 'care_opd.multimedia_boundary.package.v1',
                'package_name': ARCHIVE_NAME, 'default_physical_gpu': 1,
                'training': False, 'rag_index_required': False, 'evaluation_use': 'development_pilot',
                'runtime_origin': 'local_visual_evidence_transfer_HF_helpers',
                'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                          for name, data in sorted(files.items())}}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')
    files['PACKAGE_MANIFEST.json'] = manifest_bytes
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', filename='', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as archive:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.uid, info.gid = len(data), 0, 0, 0
                info.uname = info.gname = ''
                info.mode = 0o700 if name.endswith(('.sh', '.py')) else 0o600
                archive.addfile(info, io.BytesIO(data))
    data = buffer.getvalue()
    # Validate exact membership and bytes in memory, no untrusted extraction.
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        if sorted(archive.getnames()) != sorted(files):
            raise RuntimeError('Archive member mismatch')
        for member in archive.getmembers():
            if not member.isfile() or archive.extractfile(member).read() != files[member.name]:
                raise RuntimeError('Archive round-trip mismatch: ' + member.name)
    output = Path(output)
    target = output / ARCHIVE_NAME
    c.write_once(target, data)
    sha = hashlib.sha256(data).hexdigest()
    c.write_once(output / (ARCHIVE_NAME + '.sha256'), (sha + '  ' + ARCHIVE_NAME + '\n').encode('ascii'))
    c.write_once(output / 'PACKAGE_MANIFEST.json', manifest_bytes)
    result = {'archive': str(target.resolve()), 'sha256': sha, 'bytes': len(data), 'members': len(files)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    parser.add_argument('--runtime-dir', type=Path)
    args = parser.parse_args()
    build(args.output, args.runtime_dir)
