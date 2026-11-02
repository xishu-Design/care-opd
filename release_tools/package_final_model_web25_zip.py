#!/usr/bin/env python3
"""Create ZIP-compressed model volumes that each fit GitHub's web uploader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


WEIGHT_BYTES = 8_411_558_400
WEIGHT_SHA256 = "87670f85af0db2abf466623ef5fcc2e4f1ac727aa07a2e58c2b15ff246a09fc4"
MODEL_FINGERPRINT = "4cf77e31795163f32fa4e6715d42d06fec88ff98801b154934795b98281ecdcd"
MAX_WEB_BYTES = 25_000_000


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


RESTORE_SCRIPT = r'''#!/usr/bin/env python3
"""Verify ZIP volumes and reconstruct model/model.safetensors."""
from __future__ import annotations
import argparse, hashlib, json, os, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""): h.update(b)
    return h.hexdigest()

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--verify-only",action="store_true")
    args=ap.parse_args()
    m=json.loads((ROOT/"web25_manifest.json").read_text(encoding="utf-8"))
    out=ROOT/m["reassembly"]["output"]
    tmp=out.with_name(out.name+".tmp")
    if not args.verify_only and (out.exists() or tmp.exists()):
        raise SystemExit(f"refusing to overwrite: {out} / {tmp}")
    full=hashlib.sha256(); total=0
    dst=None
    try:
        if not args.verify_only:
            out.parent.mkdir(parents=True,exist_ok=True)
            dst=tmp.open("xb")
        for item in m["volumes"]:
            zp=ROOT/item["archive_path"]
            if not zp.is_file() or zp.stat().st_size!=item["archive_bytes"]:
                raise RuntimeError(f"missing/bad archive length: {zp}")
            if digest(zp)!=item["archive_sha256"]:
                raise RuntimeError(f"bad archive sha256: {zp}")
            with zipfile.ZipFile(zp,"r") as zf:
                if zf.namelist()!=[item["member_name"]]:
                    raise RuntimeError(f"bad member list: {zp}")
                member=hashlib.sha256(); member_bytes=0
                with zf.open(item["member_name"],"r") as src:
                    while True:
                        block=src.read(8*1024*1024)
                        if not block: break
                        member.update(block); full.update(block)
                        member_bytes+=len(block); total+=len(block)
                        if dst is not None: dst.write(block)
                if member_bytes!=item["member_bytes"] or member.hexdigest()!=item["member_sha256"]:
                    raise RuntimeError(f"bad member content: {zp}")
            print(f"verified {item['index']:04d}/{len(m['volumes']):04d}: {zp.name}")
        artifact=m["artifact"]
        if total!=artifact["bytes"] or full.hexdigest()!=artifact["sha256"]:
            raise RuntimeError("complete weight hash/length mismatch")
        if dst is not None:
            dst.flush(); os.fsync(dst.fileno()); dst.close(); dst=None
            os.replace(tmp,out)
    except BaseException:
        if dst is not None: dst.close()
        if tmp.exists(): tmp.unlink()
        raise
    print(("PACKAGE_VERIFY" if args.verify_only else "ASSEMBLY")+"=PASS")

if __name__=="__main__": main()
'''


README = f"""# CARE-OPD QZBC 4B — GitHub web-upload ZIP volumes

This package is the web-uploader variant of the CARE-OPD merged checkpoint.
Every ZIP file is smaller than 25,000,000 bytes.  The weight was divided into
20 MiB raw segments and each segment was independently compressed as ZIP.

## Upload

1. Create an empty GitHub repository.
2. Upload the small metadata files and directories except `archives/`.
3. Upload `archives/model/` ZIP files in manageable batches (for example 50
   files at a time).  Preserve every filename exactly.
4. Confirm all archives listed in `web25_manifest.json` are present.

GitHub's web interface is appropriate only because each archive is below its
25 MB limit.  There are hundreds of volumes; command-line Git or a model hub is
still more reliable for a package of this size.

## Restore

Windows:

```powershell
./scripts/restore.ps1
```

Linux/macOS:

```bash
bash scripts/restore.sh
```

Or run:

```bash
python scripts/restore_from_zip_parts.py
```

The script checks each ZIP SHA-256, each uncompressed member SHA-256, and the
complete reconstructed weight before publishing `model/model.safetensors`.

Complete weight SHA-256:

```text
{WEIGHT_SHA256}
```

Formal evaluation model fingerprint:

```text
{MODEL_FINGERPRINT}
```

See `MODEL_CARD.md`, `LICENSE`, `NOTICE`, and `PUBLISHING_CHECKLIST.md` before
making the repository public.
"""


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-model",type=Path,default=Path("final_model/model.safetensors"))
    ap.add_argument("--metadata-source",type=Path,default=Path("github_release/care-opd-qzbc-4b-final"))
    ap.add_argument("--output",type=Path,default=Path("github_release/care-opd-qzbc-4b-final-web25"))
    ap.add_argument("--raw-part-mib",type=int,default=20)
    args=ap.parse_args()
    source=args.source_model.resolve(); metadata=args.metadata_source.resolve(); output=args.output.resolve()
    staging=output.with_name(output.name+".building")
    raw_bytes=args.raw_part_mib*1024*1024
    if not 1<=args.raw_part_mib<=20: raise SystemExit("raw part size must be 1..20 MiB")
    if output.exists() or staging.exists(): raise SystemExit(f"refusing existing output/staging: {output} / {staging}")
    if source.stat().st_size!=WEIGHT_BYTES or sha256(source)!=WEIGHT_SHA256:
        raise SystemExit("source model does not match the formal checkpoint")
    try:
        staging.mkdir(parents=True)
        # Copy the already sanitized public metadata, never the 48 MiB raw parts.
        for name in ("LICENSE","NOTICE","MODEL_CARD.md","DEPLOYMENT.md","PUBLISHING_CHECKLIST.md"):
            shutil.copyfile(metadata/name,staging/name)
        shutil.copytree(metadata/"model",staging/"model")
        (staging/"scripts").mkdir()
        write_text(staging/"README.md",README)
        write_text(staging/"scripts"/"restore_from_zip_parts.py",RESTORE_SCRIPT)
        write_text(staging/"scripts"/"restore.ps1","$ErrorActionPreference='Stop'\n$py=if($env:PYTHON_BIN){$env:PYTHON_BIN}else{'python'}\n& $py \"$PSScriptRoot/restore_from_zip_parts.py\" @args\nexit $LASTEXITCODE")
        write_text(staging/"scripts"/"restore.sh",'#!/usr/bin/env bash\nset -euo pipefail\nPYTHON_BIN="${PYTHON_BIN:-python3}"\nexec "$PYTHON_BIN" "$(dirname "$0")/restore_from_zip_parts.py" "$@"')
        write_text(staging/".gitignore","/model/model.safetensors\n/model/model.safetensors.tmp*\n__pycache__/\n*.pyc")
        write_text(staging/".gitattributes","*.zip binary -diff -merge -delta")
        archive_dir=staging/"archives"/"model"; archive_dir.mkdir(parents=True)
        count=(WEIGHT_BYTES+raw_bytes-1)//raw_bytes
        volumes=[]; offset=0
        print(f"Creating {count} ZIP volumes from {source}...")
        with source.open("rb") as src:
            for index in range(1,count+1):
                data=src.read(raw_bytes)
                if not data: raise RuntimeError("unexpected EOF")
                member=f"model.safetensors.part-{index:04d}-of-{count:04d}"
                archive=archive_dir/f"care-opd-weight-{index:04d}-of-{count:04d}.zip"
                with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=False) as zf:
                    zf.writestr(member,data)
                if archive.stat().st_size>=MAX_WEB_BYTES:
                    raise RuntimeError(f"archive exceeds web limit: {archive} {archive.stat().st_size}")
                volumes.append({
                    "index":index,"offset_bytes":offset,
                    "archive_path":archive.relative_to(staging).as_posix(),
                    "archive_bytes":archive.stat().st_size,"archive_sha256":sha256(archive),
                    "member_name":member,"member_bytes":len(data),
                    "member_sha256":hashlib.sha256(data).hexdigest(),
                })
                offset+=len(data)
                print(f"  {index:04d}/{count:04d} {archive.name} {archive.stat().st_size}")
        if offset!=WEIGHT_BYTES: raise RuntimeError("split byte count mismatch")
        manifest={
            "schema_version":"care_opd.github_web25_zip_package.v1","status":"COMPLETE",
            "created_at_utc":datetime.now(timezone.utc).isoformat(),
            "artifact":{"output":"model/model.safetensors","bytes":WEIGHT_BYTES,"sha256":WEIGHT_SHA256,"format":"safetensors"},
            "evaluation_model_fingerprint":MODEL_FINGERPRINT,
            "web_constraint":{"maximum_file_bytes_exclusive":MAX_WEB_BYTES,"largest_archive_bytes":max(v["archive_bytes"] for v in volumes)},
            "chunking":{"raw_part_bytes":raw_bytes,"archive_format":"zip-deflate-level-1","volume_count":count},
            "volumes":volumes,
            "reassembly":{"script":"scripts/restore_from_zip_parts.py","output":"model/model.safetensors","overwrite_policy":"refuse"},
        }
        write_text(staging/"web25_manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2))
        # Convenient metadata bundle; it intentionally excludes the weight ZIP volumes.
        meta_zip=staging/"care-opd-metadata.zip"
        with zipfile.ZipFile(meta_zip,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
            for path in sorted(p for p in staging.rglob("*") if p.is_file() and "archives" not in p.parts and p!=meta_zip):
                zf.write(path,path.relative_to(staging).as_posix())
        if meta_zip.stat().st_size>=MAX_WEB_BYTES: raise RuntimeError("metadata ZIP exceeds web limit")
        sums=[]
        for path in sorted(p for p in staging.rglob("*") if p.is_file() and p.name!="SHA256SUMS"):
            sums.append(f"{sha256(path)}  {path.relative_to(staging).as_posix()}")
        write_text(staging/"SHA256SUMS","\n".join(sums))
        too_large=[p for p in staging.rglob("*") if p.is_file() and p.stat().st_size>=MAX_WEB_BYTES]
        if too_large: raise RuntimeError(f"files still exceed web limit: {too_large[:3]}")
        os.replace(staging,output)
        print(f"WEB25_PACKAGE=COMPLETE path={output} volumes={count}")
    except BaseException:
        print(f"WEB25_PACKAGE=FAILED staging retained: {staging}")
        raise


if __name__=="__main__": main()
