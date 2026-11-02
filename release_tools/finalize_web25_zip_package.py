#!/usr/bin/env python3
"""Finalize a completed web25 staging directory after an interrupted builder."""
from __future__ import annotations
import hashlib, json, os, zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STAGING=ROOT/"github_release/care-opd-qzbc-4b-final-web25.building"
OUTPUT=ROOT/"github_release/care-opd-qzbc-4b-final-web25"
WEIGHT_BYTES=8_411_558_400
WEIGHT_SHA256="87670f85af0db2abf466623ef5fcc2e4f1ac727aa07a2e58c2b15ff246a09fc4"
MODEL_FINGERPRINT="4cf77e31795163f32fa4e6715d42d06fec88ff98801b154934795b98281ecdcd"
MAX_WEB_BYTES=25_000_000

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def write(path: Path,text: str) -> None:
    path.write_text(text.rstrip()+"\n",encoding="utf-8",newline="\n")

def main() -> None:
    if OUTPUT.exists() or not STAGING.is_dir():
        raise SystemExit(f"unexpected output/staging state: {OUTPUT} / {STAGING}")
    archives=sorted((STAGING/"archives/model").glob("care-opd-weight-*-of-0402.zip"))
    if len(archives)!=402: raise SystemExit(f"expected 402 archives, found {len(archives)}")
    volumes=[]; offset=0; full=hashlib.sha256()
    for index,archive in enumerate(archives,1):
        if archive.stat().st_size>=MAX_WEB_BYTES: raise RuntimeError(f"oversize: {archive}")
        with zipfile.ZipFile(archive,"r") as zf:
            names=zf.namelist()
            expected=f"model.safetensors.part-{index:04d}-of-0402"
            if names!=[expected]: raise RuntimeError(f"bad member: {archive} {names}")
            member=hashlib.sha256(); n=0
            with zf.open(expected,"r") as src:
                while True:
                    block=src.read(8*1024*1024)
                    if not block: break
                    member.update(block); full.update(block); n+=len(block)
        volumes.append({
            "index":index,"offset_bytes":offset,
            "archive_path":archive.relative_to(STAGING).as_posix(),
            "archive_bytes":archive.stat().st_size,"archive_sha256":digest(archive),
            "member_name":expected,"member_bytes":n,"member_sha256":member.hexdigest(),
        })
        offset+=n
        print(f"verified {index:04d}/0402 {archive.name}")
    if offset!=WEIGHT_BYTES or full.hexdigest()!=WEIGHT_SHA256:
        raise RuntimeError(f"full mismatch bytes={offset} sha={full.hexdigest()}")
    manifest={
        "schema_version":"care_opd.github_web25_zip_package.v1","status":"COMPLETE",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "artifact":{"output":"model/model.safetensors","bytes":WEIGHT_BYTES,"sha256":WEIGHT_SHA256,"format":"safetensors"},
        "evaluation_model_fingerprint":MODEL_FINGERPRINT,
        "web_constraint":{"maximum_file_bytes_exclusive":MAX_WEB_BYTES,"largest_archive_bytes":max(v["archive_bytes"] for v in volumes)},
        "chunking":{"raw_part_bytes":20*1024*1024,"archive_format":"zip-deflate-level-1","volume_count":402},
        "volumes":volumes,
        "reassembly":{"script":"scripts/restore_from_zip_parts.py","output":"model/model.safetensors","overwrite_policy":"refuse"},
    }
    write(STAGING/"web25_manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    meta_zip=STAGING/"care-opd-metadata.zip"
    with zipfile.ZipFile(meta_zip,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for path in sorted(p for p in STAGING.rglob("*") if p.is_file() and "archives" not in p.parts and p!=meta_zip):
            zf.write(path,path.relative_to(STAGING).as_posix())
    if meta_zip.stat().st_size>=MAX_WEB_BYTES: raise RuntimeError("metadata ZIP oversize")
    sums=[]
    for path in sorted(p for p in STAGING.rglob("*") if p.is_file() and p.name!="SHA256SUMS"):
        sums.append(f"{digest(path)}  {path.relative_to(STAGING).as_posix()}")
    write(STAGING/"SHA256SUMS","\n".join(sums))
    too_large=[p for p in STAGING.rglob("*") if p.is_file() and p.stat().st_size>=MAX_WEB_BYTES]
    if too_large: raise RuntimeError(f"oversize package files: {too_large[:3]}")
    os.replace(STAGING,OUTPUT)
    print(f"WEB25_PACKAGE=COMPLETE path={OUTPUT} volumes={len(volumes)}")

if __name__=="__main__": main()
