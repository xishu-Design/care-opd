from __future__ import annotations

"""Read-only, content-level scan for the final public upload directory."""

import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

import fitz
from PIL import Image


TEXT_SUFFIXES = {".md", ".csv", ".json", ".txt", ".yaml", ".yml", ".svg", ".sha256"}

FORBIDDEN = {
    "internal_state": re.compile(
        r"build_internal|RELEASE_CANDIDATE|FINAL_RELEASE_STATUS|V6_RELEASE_AUDIT|"
        r"upload_documents_ready|upload_release_candidate|upload_ready",
        re.I,
    ),
    "internal_source_path": re.compile(
        r"competition_submission_v\d+|supporting_public_v\d+|"
        r"[A-Za-z]:[\\/]lcx[\\/]opdrevi|[A-Za-z]:[\\/]Users[\\/]|"
        r"/(?:root|home/[^/\s]+)/|file://",
        re.I,
    ),
    "draft_or_team_language": re.compile(
        r"\bDRAFT\b|\bTODO\b|待填写|待补充|待确认|尚待补|封版|"
        r"给队员|供队员|团队内部|文档内部指令|不视为用户要求|答辩口径|"
        r"评委问到时|现场照着|提交前请|internal build records|presenter scripts",
        re.I,
    ),
    "anonymous_identity": re.compile(
        r"指导教师|导师姓名|所在学校|学校名称|学院名称|校徽"
    ),
    "private_payload": re.compile(
        r"source_text_private|raw_response_private|clinical_gold\.private|cases\.private",
        re.I,
    ),
    "secret": re.compile(
        r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|"
        r"Authorization\s*:\s*Bearer\s+\S+|"
        r"\bhf_[A-Za-z0-9]{20,}\b|\bsk-[A-Za-z0-9_-]{20,}\b",
        re.I,
    ),
    "submission_version": re.compile(r"\bV6(?:\.0)?\b|版本号码|文档版本", re.I),
}

PII = {
    "mobile": re.compile(r"(?<![A-Za-z0-9.])1[3-9][0-9]{9}(?![A-Za-z0-9.])"),
    "identity_card": re.compile(
        r"(?<![A-Za-z0-9.])[1-9][0-9]{5}(?:19|20)[0-9]{2}"
        r"(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])"
        r"[0-9]{3}[0-9Xx](?![A-Za-z0-9.])"
    ),
    "email": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_findings(label: str, text: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for name, pattern in {**FORBIDDEN, **PII}.items():
        match = pattern.search(text)
        if match:
            findings.append(
                {"file": label, "rule": name, "fragment": match.group(0)[:120]}
            )
    return findings


def image_metadata_findings(label: str, data: bytes) -> list[dict[str, str]]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            metadata = json.dumps(
                {"info": image.info, "exif": dict(image.getexif())},
                ensure_ascii=False,
                default=str,
            )
    except Exception:
        return []
    return text_findings(label + "#image-metadata", metadata)


def scan_pdf(label: str, data: bytes) -> tuple[dict[str, object], list[dict[str, str]]]:
    findings: list[dict[str, str]] = []
    with fitz.open(stream=data, filetype="pdf") as document:
        visible = "\n".join(page.get_text("text") for page in document)
        metadata = json.dumps(document.metadata, ensure_ascii=False)
        findings.extend(text_findings(label, visible + "\n" + metadata))

        nearly_blank = [
            index + 1
            for index, page in enumerate(document)
            if len(page.get_text("text").strip()) < 55
        ]
        non_a4 = [
            index + 1
            for index, page in enumerate(document)
            if abs(page.rect.width - 595.3) > 2 or abs(page.rect.height - 841.9) > 2
        ]
        non_http_links: list[dict[str, object]] = []
        image_xrefs: set[int] = set()
        for page_number, page in enumerate(document, start=1):
            for link in page.get_links():
                uri = link.get("uri") or ""
                if uri and not re.match(r"^(?:https?|mailto):", uri, re.I):
                    non_http_links.append({"page": page_number, "uri": uri})
            image_xrefs.update(item[0] for item in page.get_images(full=True))
        for xref in image_xrefs:
            try:
                image = document.extract_image(xref)["image"]
            except Exception:
                continue
            findings.extend(image_metadata_findings(f"{label}#xref={xref}", image))

        active_markers = [
            marker.decode("ascii")
            for marker in (b"/JavaScript", b"/JS", b"/Launch", b"/EmbeddedFile")
            if marker in data
        ]
        report = {
            "file": label,
            "pages": len(document),
            "nearly_blank_pages": nearly_blank,
            "non_a4_pages": non_a4,
            "embedded_files": document.embfile_names(),
            "non_http_links": non_http_links,
            "active_content_markers": active_markers,
            "image_count": len(image_xrefs),
            "metadata": document.metadata,
        }
    return report, findings


def scan_zip(path: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    findings: list[dict[str, str]] = []
    pdf_reports: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            findings.append({"file": path.name, "rule": "duplicate_zip_member", "fragment": ""})
        roots: set[str] = set()
        encrypted: list[str] = []
        unsafe: list[str] = []
        payloads: dict[str, bytes] = {}
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            if pure.parts:
                roots.add(pure.parts[0])
            if pure.is_absolute() or ".." in pure.parts or "\\" in info.filename:
                unsafe.append(info.filename)
            if info.flag_bits & 0x1:
                encrypted.append(info.filename)
            if info.is_dir():
                continue
            data = archive.read(info)
            payloads[info.filename] = data
            findings.extend(text_findings(info.filename + "#path", info.filename))
            suffix = pure.suffix.lower()
            if suffix in TEXT_SUFFIXES:
                findings.extend(
                    text_findings(info.filename, data.decode("utf-8", errors="replace"))
                )
            elif suffix == ".pdf":
                report, pdf_findings = scan_pdf(info.filename, data)
                pdf_reports.append(report)
                findings.extend(pdf_findings)
            elif suffix in {".png", ".jpg", ".jpeg"}:
                findings.extend(image_metadata_findings(info.filename, data))

        checksum_names = [name for name in names if name.endswith("/SHA256SUMS.txt")]
        manifest_names = [
            name for name in names if name.endswith("/PUBLIC_EVIDENCE_MANIFEST.json")
        ]
        checksum_errors: list[str] = []
        manifest_errors: list[str] = []
        if len(checksum_names) != 1:
            checksum_errors.append(f"checksum_file_count={len(checksum_names)}")
        else:
            checksum_name = checksum_names[0]
            root = str(PurePosixPath(checksum_name).parent)
            lines = payloads[checksum_name].decode("utf-8").splitlines()
            for line in lines:
                match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
                if not match:
                    checksum_errors.append("malformed_checksum_line")
                    continue
                expected, relative = match.groups()
                member = f"{root}/{relative}"
                if member not in payloads:
                    checksum_errors.append(f"missing:{relative}")
                elif sha256_bytes(payloads[member]) != expected:
                    checksum_errors.append(f"hash:{relative}")
        if len(manifest_names) != 1:
            manifest_errors.append(f"manifest_file_count={len(manifest_names)}")
        else:
            manifest_name = manifest_names[0]
            root = str(PurePosixPath(manifest_name).parent)
            manifest = json.loads(payloads[manifest_name].decode("utf-8"))
            entries = manifest.get("files", [])
            if manifest.get("file_count") != len(entries):
                manifest_errors.append("file_count")
            if manifest.get("schema") != "shennong-sanyan.public-evidence-manifest":
                manifest_errors.append("schema")
            for entry in entries:
                relative = entry["path"]
                member = f"{root}/{relative}"
                data = payloads.get(member)
                if data is None:
                    manifest_errors.append(f"missing:{relative}")
                    continue
                if len(data) != int(entry["bytes"]):
                    manifest_errors.append(f"bytes:{relative}")
                if sha256_bytes(data) != entry["sha256"]:
                    manifest_errors.append(f"hash:{relative}")

        if unsafe:
            findings.append({"file": path.name, "rule": "unsafe_zip_path", "fragment": unsafe[0]})
        if encrypted:
            findings.append({"file": path.name, "rule": "encrypted_zip_member", "fragment": encrypted[0]})
        if checksum_errors:
            findings.append({"file": path.name, "rule": "checksum", "fragment": checksum_errors[0]})
        if manifest_errors:
            findings.append({"file": path.name, "rule": "manifest", "fragment": manifest_errors[0]})
        report = {
            "file": path.name,
            "member_count": len(names),
            "single_root": len(roots) == 1,
            "roots": sorted(roots),
            "archive_comment_empty": not archive.comment,
            "encrypted_member_count": len(encrypted),
            "unsafe_member_count": len(unsafe),
            "checksum_status": "PASS" if not checksum_errors else "FAIL",
            "manifest_status": "PASS" if not manifest_errors else "FAIL",
            "nested_pdfs": pdf_reports,
        }
    return report, findings


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    upload = Path(sys.argv[1] if len(sys.argv) > 1 else "formal/competition_submission_v6/正式上传包")
    files = sorted(path for path in upload.iterdir() if path.is_file())
    findings: list[dict[str, str]] = []
    pdf_reports: list[dict[str, object]] = []
    zip_reports: list[dict[str, object]] = []

    expected = {
        "智在必得队_神农三言_参赛作品简介.pdf",
        "智在必得队_神农三言_项目文档.pdf",
        "智在必得队_神农三言_其他.zip",
    }
    actual = {path.name for path in files}
    if actual != expected:
        findings.append(
            {
                "file": str(upload),
                "rule": "unexpected_top_level_set",
                "fragment": json.dumps(sorted(actual), ensure_ascii=False),
            }
        )

    for path in files:
        findings.extend(text_findings(path.name + "#path", path.name))
        if path.suffix.lower() == ".pdf":
            report, pdf_findings = scan_pdf(path.name, path.read_bytes())
            pdf_reports.append(report)
            findings.extend(pdf_findings)
        elif path.suffix.lower() == ".zip":
            report, zip_findings = scan_zip(path)
            zip_reports.append(report)
            findings.extend(zip_findings)

    report = {
        "status": "PASS" if not findings else "FAIL",
        "upload_directory": str(upload.resolve()),
        "top_level_file_count": len(files),
        "top_level_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
        "pdfs": pdf_reports,
        "archives": zip_reports,
        "finding_count": len(findings),
        "findings": findings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
