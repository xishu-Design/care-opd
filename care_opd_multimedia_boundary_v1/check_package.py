"""Verify the package inventory without importing model libraries."""
from pathlib import Path
import argparse
import hashlib
import json


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 'care_opd.multimedia_boundary.package.v1':
        raise ValueError('Unsupported package manifest')
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing/unsafe member: ' + relative)
        data = path.read_bytes()
        if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise ValueError('Changed package member: ' + relative)
    print('PACKAGE_OK', len(manifest['files']), 'files')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    verify(parser.parse_args().root)
