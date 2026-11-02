"""Source-package consistency tests without Torch, network, or model generation."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import tarfile
import unittest

import build_bundle
import check_package


class PackageTests(unittest.TestCase):
    def test_archive_roundtrip_deterministic_and_source_only(self):
        with tempfile.TemporaryDirectory(prefix='boundary-package-test-') as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                first = build_bundle.build(root / 'first')
                second = build_bundle.build(root / 'second')
            self.assertEqual(first['sha256'], second['sha256'])
            archive_path = Path(first['archive'])
            self.assertEqual(hashlib.sha256(archive_path.read_bytes()).hexdigest(), first['sha256'])
            extracted = root / 'extracted'
            extracted.mkdir()
            with tarfile.open(archive_path, 'r:gz') as archive:
                names = archive.getnames()
                self.assertTrue({'runtime/evaluate.py', 'runtime/common.py', 'run_boundary_gpu1.sh'}.issubset(names))
                self.assertFalse(any(name.endswith(('.jsonl', '.safetensors', '.bin')) for name in names))
                for member in archive.getmembers():
                    target = (extracted / member.name).resolve()
                    self.assertTrue(target.is_relative_to(extracted))
                    self.assertTrue(member.isfile())
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
                    self.assertNotIn(b'\r\n', target.read_bytes())
            with contextlib.redirect_stdout(io.StringIO()):
                check_package.verify(extracted)
            with (extracted / 'boundary_run.py').open('ab') as stream:
                stream.write(b'\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'Changed package'):
                check_package.verify(extracted)


if __name__ == '__main__':
    unittest.main()
