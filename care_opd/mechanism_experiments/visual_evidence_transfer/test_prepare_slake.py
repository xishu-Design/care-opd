#!/usr/bin/env python3
"""CPU-only tests for frozen SLAKE preparation and archive safety."""

from __future__ import annotations

import json
import hashlib
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

import fetch_assets
import prepare_slake


def _row(
    qid: str,
    image: str,
    answer: str,
    question: str,
    *,
    img_id: str | None = None,
    base_type: str = "vqa",
    answer_type: str = "CLOSED",
    q_lang: str = "zh",
) -> dict[str, object]:
    value: dict[str, object] = {
        "qid": qid,
        "img_name": image,
        "answer": answer,
        "question": question,
        "base_type": base_type,
        "answer_type": answer_type,
        "q_lang": q_lang,
    }
    if img_id is not None:
        value["img_id"] = img_id
    return value


class DatasetFixture:
    def __init__(self, root: Path) -> None:
        self.root = root

    def image(self, relative: str, payload: bytes) -> None:
        normalized = relative.replace("\\", "/")
        if normalized.startswith("imgs/"):
            normalized = normalized[len("imgs/") :]
        path = self.root / "imgs" / Path(*normalized.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def split(self, name: str, rows: list[dict[str, object]]) -> None:
        (self.root / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )


class PrepareSlakeTests(unittest.TestCase):
    def test_validation_binary_filter_pairing_and_private_gold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = DatasetFixture(root)
            fixture.image("xmlab1/source.jpg", b"image-one")
            fixture.image("xmlab2/source.jpg", b"image-two")
            fixture.image("xmlab3/source.jpg", b"image-three")
            fixture.image("xmlab4/source.jpg", b"image-four")
            fixture.image("xmlab5/source.jpg", b"image-five")
            fixture.split(
                "validation",
                [
                    _row("101", "xmlab1/source.jpg", "是", "图中是否存在异常？", img_id="1"),
                    _row("102", "imgs/xmlab2/source.jpg", "没有", "图中是否存在异常？", img_id="2"),
                    _row("103", "xmlab3/source.jpg", "YES", "器官边缘是否清楚？", img_id="3"),
                    _row("104", "xmlab4/source.jpg", "肺", "检查部位是什么？", img_id="4"),
                    _row(
                        "105",
                        "xmlab5/source.jpg",
                        "NO",
                        "图中是否存在异常？",
                        img_id="5",
                        q_lang="en",
                    ),
                ],
            )
            fixture.split("train", [_row("1", "xmlab4/source.jpg", "YES", "训练题")])
            fixture.split("test", [_row("2", "xmlab5/source.jpg", "NO", "测试题")])
            output = root / "prepared"

            result = prepare_slake.prepare(
                dataset_root=root,
                output_dir=output,
                split="validation",
            )
            self.assertEqual(result["status"], "CREATED")
            tasks = [
                json.loads(line)
                for line in (output / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            gold = [
                json.loads(line)
                for line in (output / "gold.private.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            manifest = json.loads(
                (output / "selection_manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["dataset"]["split"], "validation")
            self.assertEqual(manifest["dataset"]["split_role"], "pilot")
            self.assertEqual(manifest["selection"]["selected_families"], 3)
            self.assertEqual(manifest["counts"]["tasks"], 8)
            self.assertEqual(manifest["counts"]["conditions"]["swapped_image"], 2)
            self.assertIn("tasks_sha256", manifest)
            self.assertIn("gold_sha256", manifest)
            self.assertEqual(len({row["task_id"] for row in tasks}), len(tasks))
            self.assertEqual(
                [row["task_id"] for row in tasks], [row["task_id"] for row in gold]
            )
            self.assertFalse(
                any(
                    forbidden in task
                    for task in tasks
                    for forbidden in ("answer", "gold", "original_answer")
                )
            )
            for family_id in {row["family_id"] for row in tasks}:
                family = [row for row in tasks if row["family_id"] == family_id]
                self.assertEqual(len({row["context_value"] for row in family}), 1)
                self.assertRegex(family[0]["context_value"], r"^[A-Z2-9]{8}$")
                no_image = next(row for row in family if row["condition"] == "no_image")
                self.assertIsNone(no_image["image_path"])
                self.assertIsNone(no_image["image_sha256"])
                no_image_gold = next(row for row in gold if row["task_id"] == no_image["task_id"])
                self.assertEqual(no_image_gold["answer"], "ABSTAIN")

            paired_tasks = [row for row in tasks if row["family_id"] in {"101", "102"}]
            self.assertEqual(len({row["cluster_id"] for row in paired_tasks}), 1)
            for task in tasks:
                if task["condition"] != "no_image":
                    self.assertTrue(Path(task["image_path"]).is_absolute())
                    self.assertEqual(len(task["image_sha256"]), 64)
            swapped_gold = [row for row in gold if row["condition"] == "swapped_image"]
            self.assertEqual({row["answer"] for row in swapped_gold}, {"YES", "NO"})
            self.assertTrue(all(row["counterpart_qid"] for row in swapped_gold))

            repeated = prepare_slake.prepare(
                dataset_root=root,
                output_dir=output,
                split="validation",
            )
            self.assertEqual(repeated["status"], "IDEMPOTENT_EXISTING")
            with self.assertRaises(FileExistsError):
                prepare_slake.prepare(
                    dataset_root=root,
                    output_dir=output,
                    split="validation",
                    max_items=1,
                )

    def test_formal_test_excludes_validation_images_and_warns_on_train_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = DatasetFixture(root)
            for name, payload in {
                "common/source.jpg": b"common-image",
                "val_only/source.jpg": b"validation-only",
                "test_yes/source.jpg": b"test-yes",
                "test_no/source.jpg": b"test-no",
                "train_only/source.jpg": b"train-only",
            }.items():
                fixture.image(name, payload)
            fixture.split(
                "validation",
                [
                    _row("v1", "common/source.jpg", "YES", "可见病灶吗？", img_id="11"),
                    _row("v2", "val_only/source.jpg", "NO", "有积液吗？", img_id="12"),
                ],
            )
            fixture.split(
                "test",
                [
                    _row("drop", "imgs/common/source.jpg", "YES", "可见病灶吗？", img_id="11"),
                    _row("keep-y", "test_yes/source.jpg", "YES", "图中有异常吗？", img_id="21"),
                    _row("keep-n", "test_no/source.jpg", "NO", "图中有异常吗？", img_id="22"),
                ],
            )
            fixture.split(
                "train",
                [
                    _row("tr1", "test_yes/source.jpg", "YES", "训练重叠", img_id="21"),
                    _row("tr2", "train_only/source.jpg", "NO", "训练独有", img_id="31"),
                ],
            )

            tasks, gold, manifest = prepare_slake.build_outputs(
                dataset_root=root,
                split="test",
            )
            self.assertEqual(
                manifest["dataset"]["split"], "official_test_minus_validation_images"
            )
            self.assertEqual(manifest["dataset"]["source_split"], "test")
            self.assertEqual({row["family_id"] for row in tasks}, {"keep-y", "keep-n"})
            exclusion = manifest["derived_test_exclusion"]
            self.assertEqual(exclusion["excluded_unique_image_count"], 1)
            self.assertEqual(exclusion["excluded_row_count"], 1)
            self.assertEqual(exclusion["excluded_qids"], ["drop"])
            self.assertEqual(exclusion["remaining_validation_overlap_count"], 0)
            overlap = manifest["split_image_overlap_audit"]
            self.assertEqual(overlap["pairwise_overlap_counts"]["validation__test"], 1)
            self.assertEqual(overlap["pairwise_overlap_counts"]["train__test"], 1)
            self.assertTrue(overlap["train_overlap_is_nonblocking_warning"])
            self.assertIn("not an upstream image-disjoint split", manifest["dataset"]["boundary"])
            self.assertEqual(
                sum(row["condition"] == "swapped_image" for row in tasks), 2
            )
            self.assertEqual(
                {row["answer"] for row in gold if row["condition"] == "swapped_image"},
                {"YES", "NO"},
            )

    def test_train_overlap_does_not_block_validation_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = DatasetFixture(root)
            fixture.image("shared/source.jpg", b"shared")
            row = _row("v", "shared/source.jpg", "YES", "正常吗？", img_id="1")
            fixture.split("validation", [row])
            fixture.split("train", [_row("tr", "shared/source.jpg", "YES", "正常吗？", img_id="1")])
            fixture.split("test", [_row("te", "shared/source.jpg", "YES", "正常吗？", img_id="1")])
            _, _, manifest = prepare_slake.build_outputs(dataset_root=root, split="validation")
            self.assertTrue(
                manifest["split_image_overlap_audit"]["train_overlap_is_nonblocking_warning"]
            )

    def test_same_image_question_conflict_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = DatasetFixture(root)
            fixture.image("x/source.jpg", b"x")
            fixture.split(
                "validation",
                [
                    _row("1", "x/source.jpg", "YES", "异常吗？"),
                    _row("2", "x/source.jpg", "NO", "异常吗？"),
                ],
            )
            with self.assertRaisesRegex(prepare_slake.PreparationError, "answer conflict"):
                prepare_slake.build_outputs(dataset_root=root, split="validation")

    def test_byte_identical_aliases_are_clustered_and_conflicts_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = DatasetFixture(root)
            fixture.image("alias_a/source.jpg", b"same-raw-pixels")
            fixture.image("alias_b/source.jpg", b"same-raw-pixels")
            fixture.split(
                "validation",
                [
                    _row("1", "alias_a/source.jpg", "YES", "问题甲？", img_id="1"),
                    _row("2", "alias_b/source.jpg", "NO", "问题乙？", img_id="2"),
                ],
            )
            tasks, _, manifest = prepare_slake.build_outputs(
                dataset_root=root, split="validation"
            )
            self.assertEqual(
                len({row["cluster_id"] for row in tasks}),
                1,
                "byte-identical aliases must stay in one bootstrap component",
            )
            self.assertFalse(any(row["condition"] == "swapped_image" for row in tasks))
            self.assertTrue(
                manifest["bootstrap_clustering"][
                    "byte_identical_image_references_are_connected"
                ]
            )

            fixture.split(
                "validation",
                [
                    _row("1", "alias_a/source.jpg", "YES", "同一问题？", img_id="1"),
                    _row("2", "alias_b/source.jpg", "NO", "同一问题？", img_id="2"),
                ],
            )
            with self.assertRaisesRegex(
                prepare_slake.PreparationError, "byte-identical-image"
            ):
                prepare_slake.build_outputs(dataset_root=root, split="validation")

    def test_identical_content_is_explicitly_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = DatasetFixture(root)
            fixture.image("x/source.jpg", b"x")
            fixture.split(
                "validation",
                [
                    _row("1", "x/source.jpg", "YES", "异常吗？"),
                    _row("2", "x/source.jpg", "YES", "异常吗？"),
                ],
            )
            tasks, _, manifest = prepare_slake.build_outputs(
                dataset_root=root, split="validation"
            )
            self.assertEqual({row["family_id"] for row in tasks}, {"1"})
            self.assertEqual(
                manifest["deduplication_audit"]["duplicate_content_rows_removed"], 1
            )

    def test_unsafe_image_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = DatasetFixture(root)
            fixture.split(
                "validation",
                [_row("1", "../outside.jpg", "YES", "异常吗？")],
            )
            with self.assertRaisesRegex(prepare_slake.PreparationError, "unsafe image"):
                prepare_slake.build_outputs(dataset_root=root, split="validation")

    def test_test_requires_validation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = DatasetFixture(root)
            fixture.image("x/source.jpg", b"x")
            fixture.split("test", [_row("1", "x/source.jpg", "YES", "异常吗？")])
            with self.assertRaisesRegex(
                prepare_slake.PreparationError, "requires validation.json"
            ):
                prepare_slake.build_outputs(dataset_root=root, split="test")

    def test_source_commit_is_claimed_only_when_asset_manifest_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = DatasetFixture(root)
            fixture.image("x/source.jpg", b"x")
            fixture.split("validation", [_row("1", "x/source.jpg", "YES", "异常吗？")])

            _, _, unverified = prepare_slake.build_outputs(
                dataset_root=root, split="validation"
            )
            self.assertEqual(
                unverified["dataset"]["verification_status"], "UNVERIFIED_LOCAL_COPY"
            )
            self.assertIsNone(unverified["dataset"]["verified_source_commit"])

            source = root / "validation.json"
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (root / "slake_assets_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "repo_id": prepare_slake.DATASET_REPO,
                        "resolved_commit": prepare_slake.DATASET_REVISION,
                        "downloaded_files": [
                            {
                                "path": "validation.json",
                                "size_bytes": source.stat().st_size,
                                "sha256": digest,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            _, _, verified = prepare_slake.build_outputs(
                dataset_root=root, split="validation"
            )
            self.assertEqual(
                verified["dataset"]["verification_status"],
                "VERIFIED_PINNED_HF_ASSET",
            )
            self.assertEqual(
                verified["dataset"]["verified_source_commit"],
                prepare_slake.DATASET_REVISION,
            )

            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                prepare_slake.PreparationError, "does not match"
            ):
                prepare_slake.build_outputs(dataset_root=root, split="validation")


class ZipSafetyTests(unittest.TestCase):
    def test_safe_extract_and_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
                handle.writestr("imgs/a/source.jpg", b"pixels")
            result = fetch_assets.safe_extract_zip(archive, root / "out")
            self.assertEqual(result["file_count"], 1)
            self.assertEqual((root / "out/imgs/a/source.jpg").read_bytes(), b"pixels")

            limited = root / "limited.zip"
            with zipfile.ZipFile(limited, "w", compression=zipfile.ZIP_STORED) as handle:
                handle.writestr("large.bin", b"123456")
            with self.assertRaises(fetch_assets.AssetError):
                fetch_assets.safe_extract_zip(
                    limited, root / "limited-out", max_total_bytes=4
                )

    def test_traversal_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traversal = root / "traversal.zip"
            with zipfile.ZipFile(traversal, "w") as handle:
                handle.writestr("../escape.txt", b"bad")
            with self.assertRaises(fetch_assets.AssetError):
                fetch_assets.safe_extract_zip(traversal, root / "out1")

            symlink = root / "symlink.zip"
            with zipfile.ZipFile(symlink, "w") as handle:
                info = zipfile.ZipInfo("imgs/link")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                handle.writestr(info, "../../outside")
            with self.assertRaises(fetch_assets.AssetError):
                fetch_assets.safe_extract_zip(symlink, root / "out2")


if __name__ == "__main__":
    unittest.main()
