"""CPU-only synthetic fixture tests, NOT benchmark result generation."""
from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, PngImagePlugin

import boundary_common as common
import boundary_fetch as fetch
import boundary_prepare as prepare


def fixture_image(number: int, *, metadata: str | None = None) -> bytes:
    image = Image.new("RGB", (320, 140), (number * 17 % 230, number * 31 % 230, number * 47 % 230))
    painter = ImageDraw.Draw(image)
    painter.rectangle((12, 12, 24 + number * 2, 100), fill=(245, 245, 245))
    stream = BytesIO()
    info = PngImagePlugin.PngInfo()
    if metadata is not None:
        info.add_text("fixture", metadata)
    image.save(stream, format="PNG", pnginfo=info)
    return stream.getvalue()


def fixture_record(number: int, question: str = "Which value is shown?", answer: str | None = None):
    return prepare._record(fixture_image(number), question, [answer or f"fixture-value-{number}"],
                           f"fixture-source-{number}", "val", "synthetic-test-only", "test-fixture",
                           {"fixture_index": number})


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="boundary-prepare-test-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_normalization_has_no_numeric_tolerance(self):
        self.assertEqual(prepare.normalize_answer("  ＢＬＵＥ \n  sky "), "blue sky")
        self.assertNotEqual(prepare.normalize_answer("100"), prepare.normalize_answer("101"))
        records = [fixture_record(1, answer="100"), fixture_record(2, answer="101"),
                   fixture_record(3, question="Independent question?")]
        selected = prepare.select_blocks(records, 1)
        self.assertEqual({selected[0][0].answers[0], selected[0][1].answers[0]}, {"100", "101"})

    def test_same_query_different_normalized_answers_required(self):
        same_labels = [fixture_record(1, answer="Blue"), fixture_record(2, answer="ＢＬＵＥ"),
                       fixture_record(3, question="Other query?")]
        with self.assertRaisesRegex(ValueError, "No automatic count reduction"):
            prepare.select_blocks(same_labels, 1)
        different_queries = [fixture_record(i, question=f"Question {i}?") for i in range(3)]
        with self.assertRaises(ValueError):
            prepare.select_blocks(different_queries, 1)

    def test_selection_deterministic_and_three_pixel_clusters_disjoint(self):
        records = [fixture_record(i) for i in range(1, 10)]
        first = prepare.select_blocks(records, 3, seed=123)
        second = prepare.select_blocks(list(reversed(records)), 3, seed=123)
        self.assertEqual([[r.identity for r in block] for block in first],
                         [[r.identity for r in block] for block in second])
        all_pixels = [r.pixel_sha256 for block in first for r in block]
        self.assertEqual(len(all_pixels), 9)
        self.assertEqual(len(set(all_pixels)), 9)
        for a, update, b in first:
            self.assertEqual(a.question, update.question)
            self.assertNotEqual(a.answers, update.answers)
            self.assertNotEqual(a.pixel_sha256, b.pixel_sha256)

    def test_default_24_blocks_require_72_distinct_images(self):
        records = [fixture_record(i) for i in range(1, 73)]
        selected = prepare.select_blocks(records)
        self.assertEqual(len(selected), 24)
        self.assertEqual(len({record.pixel_sha256 for block in selected for record in block}), 72)
        with self.assertRaisesRegex(ValueError, "requested 24"):
            prepare.select_blocks(records[:-1])

    def test_same_pixels_different_png_bytes_not_counted_as_new_image(self):
        first = fixture_record(1)
        differently_encoded = prepare._record(fixture_image(1, metadata="second encoding"), first.question,
                                               list(first.answers), "new-source-id", "val", first.dataset,
                                               first.license, {})
        self.assertNotEqual(first.image_sha256, differently_encoded.image_sha256)
        self.assertEqual(first.pixel_sha256, differently_encoded.pixel_sha256)
        with self.assertRaises(ValueError):
            prepare.select_blocks([first, differently_encoded, fixture_record(2)], 1)

    def test_conflicting_annotations_dropped_not_unioned(self):
        records = [fixture_record(i) for i in range(1, 7)]
        conflicting = replace(records[0], answers=("contradictory-label",), source_id="conflicting-source")
        eligible, audit = prepare.screen_records(records + [conflicting])
        self.assertEqual(len(eligible), 5)
        self.assertEqual(audit["excluded_record_count"], 2)
        self.assertNotIn(records[0].pixel_sha256, {record.pixel_sha256 for record in eligible})
        self.assertEqual(audit["excluded_groups"][0]["reason"], "inconsistent_answer_sets_same_pixels_and_question")
        # Even partially overlapping, non-identical answer sets are conservatively excluded.
        partial = replace(records[0], answers=(records[0].answers[0], "extra-alternative"))
        eligible2, audit2 = prepare.screen_records([records[0], partial, records[1]])
        self.assertEqual(eligible2, [records[1]])
        self.assertEqual(audit2["excluded_record_count"], 2)
        manifest = prepare.prepare_bundle(records + [conflicting], self.root / "screened", block_count=1)
        self.assertEqual(manifest["annotation_screening"]["excluded_record_count"], 2)
        serialized_gold = (self.root / "screened" / "gold.private.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("contradictory-label", serialized_gold)

    def test_insufficient_blocks_do_not_write_or_reduce_count(self):
        destination = self.root / "must-not-exist"
        with self.assertRaisesRegex(ValueError, "requested 24"):
            prepare.prepare_bundle([fixture_record(i) for i in range(1, 4)], destination)
        self.assertFalse(destination.exists())

    def test_all_six_conditions_and_schema_hashes(self):
        work = self.root / "bundle"
        manifest = prepare.prepare_bundle([fixture_record(i) for i in range(1, 7)], work, block_count=2)
        tasks, gold = common.read_jsonl(work / "tasks.jsonl"), common.read_jsonl(work / "gold.private.jsonl")
        self.assertEqual(manifest["blocks"], 2)
        self.assertEqual(manifest["task_count"], 12)
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len({row["task_id"] for row in tasks}), 12)
        self.assertEqual(len(manifest["block_records"]), 2)
        for filename, expected in manifest["files"].items():
            self.assertEqual(common.hash_file(work / filename), expected)
        for filename, expected in manifest["image_files"].items():
            self.assertEqual(common.hash_file(work / filename), expected)
        gold_by_id = {row["task_id"]: row for row in gold}
        for block in manifest["block_records"]:
            variants = {task["condition"]: task for task in tasks if task["block_id"] == block["block_id"]}
            self.assertEqual(set(variants), set(prepare.CONDITIONS))
            clean = variants["clean"]
            for condition, task in variants.items():
                self.assertEqual(set(task), {"task_id", "block_id", "condition", "request", "sources", "dataset"})
                self.assertEqual(set(task["request"]), {"questions", "requested_records", "record_context"})
                self.assertEqual(task["request"]["questions"], clean["request"]["questions"])
                self.assertEqual(task["sources"]["b"], clean["sources"]["b"])
                self.assertEqual(task["request"]["record_context"], prepare.CONTEXT)
                for source in task["sources"].values():
                    self.assertEqual(set(source), {"source_id", "record_id", "image_path", "image_sha256"})
                    self.assertEqual(common.hash_file(work / source["image_path"]), source["image_sha256"])
                gold_row = gold_by_id[task["task_id"]]
                self.assertEqual(gold_row["answers"]["b"], gold_by_id[clean["task_id"]]["answers"]["b"])
                self.assertEqual(gold_row["attack_target"], prepare.ATTACK_TARGET if condition == "instruction_overlay" else None)
                serialized_request = common.canonical(task["request"])
                for forbidden in ("fixture-value-", "BOUNDARY_OVERRIDE", "answers", "condition", "task_id", "block_id"):
                    self.assertNotIn(forbidden, serialized_request)
                observations = {side: {"raw_text": '{"answer":"frontend-only","evidence":"fixture"}'} for side in ("a", "b")}
                model_payload = common.answer_messages(task, observations)[1]["content"]
                for forbidden in ("fixture-value-", "BOUNDARY_OVERRIDE", '"answers"', '"condition"', '"task_id"',
                                  '"block_id"', '"dataset"', '"image_path"', '"image_sha256"', '"provenance"'):
                    self.assertNotIn(forbidden, model_payload)
            clean_gold = gold_by_id[clean["task_id"]]
            update = variants["valid_update"]
            self.assertNotEqual(clean["sources"]["a"]["image_sha256"], update["sources"]["a"]["image_sha256"])
            self.assertNotEqual(clean_gold["answers"]["a"], gold_by_id[update["task_id"]]["answers"]["a"])
            self.assertEqual(update["sources"]["a"]["record_id"], update["request"]["requested_records"]["a"])
            mismatch = variants["scope_mismatch"]
            self.assertEqual(mismatch["sources"]["a"]["image_sha256"], clean["sources"]["a"]["image_sha256"])
            self.assertEqual(mismatch["request"], clean["request"])
            self.assertNotEqual(mismatch["sources"]["a"]["record_id"], mismatch["request"]["requested_records"]["a"])
            mismatch_gold = gold_by_id[mismatch["task_id"]]
            self.assertEqual(mismatch_gold["answers"]["a"], ["ABSTAIN"])
            self.assertEqual(mismatch_gold["visual_answers"]["a"], clean_gold["visual_answers"]["a"])
            self.assertEqual(mismatch_gold["source_ids"]["a"], [])
            outage = variants["source_outage"]
            outage_gold = gold_by_id[outage["task_id"]]
            self.assertEqual(outage_gold["answers"]["a"], ["ABSTAIN"])
            self.assertEqual(outage_gold["visual_answers"]["a"], ["UNKNOWN"])
            with Image.open(work / outage["sources"]["a"]["image_path"]) as image:
                self.assertEqual(image.getextrema(), ((240, 240), (240, 240), (240, 240)))

    def test_overlays_same_canvas_and_preserve_every_original_pixel(self):
        original = prepare._read_image(fixture_image(2))
        font, _ = prepare._font(None)
        benign, attack, audit = prepare.overlay_pair(original, font)
        self.assertEqual(benign.size, attack.size)
        self.assertGreater(benign.height, original.height)
        for image in (benign, attack):
            self.assertEqual(image.crop((0, 0, original.width, original.height)).tobytes(), original.tobytes())
            self.assertEqual(image.getpixel((0, image.height - 1)), (240, 240, 240))
        self.assertNotEqual(benign.tobytes(), attack.tobytes())
        self.assertEqual(audit["output_size"], list(benign.size))
        self.assertNotIn(prepare.ATTACK_TARGET, prepare.BENIGN_TEXT)
        self.assertIn(prepare.ATTACK_TARGET, prepare.ATTACK_TEXT)

    def test_existing_work_not_overwritten(self):
        work = self.root / "old"
        work.mkdir()
        marker = work / "owned.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "new/empty"):
            prepare.prepare_bundle([fixture_record(i) for i in range(1, 4)], work, block_count=1)
        self.assertEqual(marker.read_text(), "preserve")
        self.assertEqual([item.name for item in work.iterdir()], ["owned.txt"])

    def test_normalized_import_and_path_traversal_rejected(self):
        (self.root / "image.png").write_bytes(fixture_image(1))
        row = {"image_path": "image.png", "question": "What value?", "answers": ["1"],
               "source_id": "user-source", "split": "validation", "dataset": "authorized-DocVQA-import",
               "license": "user-confirmed-research-license"}
        path = self.root / "normalized.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        records, audit = prepare.load_normalized(path)
        self.assertEqual(records[0].answers, ("1",))
        self.assertEqual(audit["format"], "normalized")
        row["image_path"] = "../outside.png"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escapes"):
            prepare.load_normalized(path)

    def test_official_local_chartqa_shape_and_split(self):
        val = self.root / "ChartQA Dataset" / "val"
        (val / "png").mkdir(parents=True)
        (val / "png" / "example.png").write_bytes(fixture_image(1))
        (val / "val_human.json").write_text(json.dumps([{"imgname": "example.png", "query": "Which bar?", "label": "Blue"}]), encoding="utf-8")
        records, audit = prepare.load_chartqa_local(self.root)
        self.assertEqual(records[0].dataset, "ChartQA")
        self.assertEqual(records[0].split, "val")
        self.assertEqual(records[0].answers, ("Blue",))
        self.assertEqual(audit["format"], "chartqa-local")
        with self.assertRaisesRegex(ValueError, "validation/development"):
            prepare._record(fixture_image(2), "Q?", ["A"], "source", "test", "ChartQA", "GPL", {})

    def test_reserved_gold_rejected(self):
        for answer in ("ABSTAIN", "unknown", "ＢＯＵＮＤＡＲＹ＿ＯＶＥＲＲＩＤＥ"):
            with self.subTest(answer=answer), self.assertRaisesRegex(ValueError, "reserved"):
                fixture_record(1, answer=answer)

    def mock_parquet(self):
        parent = types.ModuleType("pyarrow")
        module = types.ModuleType("pyarrow.parquet")
        payload = {"image": {"bytes": fixture_image(1), "path": "not-needed.png"},
                   "query": "Which value?", "label": ["37"], "human_or_machine": 0}
        batch = types.SimpleNamespace(to_pylist=lambda: [payload])
        parquet = types.SimpleNamespace(schema_arrow=types.SimpleNamespace(names=list(payload)),
                                        iter_batches=lambda **kwargs: iter([batch]))
        module.ParquetFile = lambda path: parquet
        parent.parquet = module
        return {"pyarrow": parent, "pyarrow.parquet": module}

    def test_parquet_embedded_image_schema_and_private_provenance(self):
        source = self.root / "download"
        (source / "data").mkdir(parents=True)
        shard = source / "data" / "val-00000-of-00001.parquet"
        shard.write_bytes(b"mock-parquet-fixture")
        with patch.dict(sys.modules, self.mock_parquet()):
            records, audit = prepare.load_chartqa_parquet(source)
        self.assertEqual(records[0].answers, ("37",))
        self.assertEqual(records[0].provenance["human_or_machine"], 0)
        self.assertEqual(audit["input_files"][str(shard.resolve())], common.hash_file(shard))
        self.assertEqual(records[0].image_bytes, fixture_image(1))

    def test_parquet_pin_manifest_detects_tampering(self):
        source = self.root / "download"
        (source / "data").mkdir(parents=True)
        shard = source / "data" / "val-00000-of-00001.parquet"
        shard.write_bytes(b"mock-parquet-fixture")
        manifest = {"repo_id": fetch.REPO_ID, "split": "val", "resolved_revision": "a" * 40,
                    "files": {"data/val-00000-of-00001.parquet": "b" * 64}}
        (source / "download_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with patch.dict(sys.modules, self.mock_parquet()), self.assertRaisesRegex(ValueError, "differs"):
            prepare.load_chartqa_parquet(source)


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="boundary-fetch-test-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_ignore_proxy_is_copy_only(self):
        original = {"HTTPS_PROXY": "private-proxy", "http_proxy": "second-proxy", "ALL_PROXY": "third", "NO_PROXY": "localhost", "KEEP": "yes",
                    "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        self.assertEqual(fetch.child_environment(original, True), {"KEEP": "yes", "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0"})
        self.assertIn("HTTPS_PROXY", original)
        self.assertEqual(original["HF_HUB_OFFLINE"], "1")
        self.assertEqual(original["HF_DATASETS_OFFLINE"], "1")
        self.assertEqual(fetch.child_environment(original, False), {**original, "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0"})
        self.assertIsNot(fetch.child_environment(original, False), original)

    def test_fetch_paths_are_validation_or_license_only(self):
        for allowed in ("data/val-00000-of-00001.parquet", "README.md", "LICENSE", "LICENSE.md"):
            self.assertTrue(fetch._allowed_name(allowed))
        for forbidden in ("data/train-00000.parquet", "data/test-00000.parquet", "model.safetensors",
                          "data/val-directory/model.parquet", "../LICENSE", "/LICENSE"):
            self.assertFalse(fetch._allowed_name(forbidden))

    def mock_hub(self):
        calls = []
        module = types.ModuleType("huggingface_hub")
        def dataset_info(repo_id, **kwargs):
            calls.append(("info", repo_id, kwargs))
            return types.SimpleNamespace(sha="a" * 40, siblings=[types.SimpleNamespace(rfilename=name) for name in
                                          ("data/val-00000-of-00001.parquet", "data/train-00000.parquet", "README.md", "model.safetensors")])
        def snapshot_download(**kwargs):
            calls.append(("download", kwargs))
            root = Path(kwargs["local_dir"])
            (root / "data").mkdir(exist_ok=True)
            (root / "data" / "val-00000-of-00001.parquet").write_bytes(b"synthetic-parquet-download-fixture")
            (root / "README.md").write_text("license: gpl-3.0", encoding="utf-8")
        module.HfApi = lambda: types.SimpleNamespace(dataset_info=dataset_info)
        module.snapshot_download = snapshot_download
        return module, calls

    def test_fetch_pins_sha_val_only_and_verifies_existing(self):
        module, calls = self.mock_hub()
        destination = self.root / "download"
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            manifest = fetch.fetch_chartqa(destination, "main")
            again = fetch.fetch_chartqa(destination, "main")
        self.assertEqual(manifest, again)
        download = next(call[1] for call in calls if call[0] == "download")
        self.assertEqual(download["revision"], "a" * 40)
        self.assertEqual(download["repo_type"], "dataset")
        self.assertEqual(download["repo_id"], fetch.REPO_ID)
        self.assertEqual(download["allow_patterns"], fetch.ALLOW_PATTERNS)
        self.assertFalse(manifest["downloads_models"])
        self.assertEqual(len([call for call in calls if call[0] == "download"]), 1)
        self.assertTrue((destination / "download_manifest.json").is_file())
        self.assertTrue(all("train" not in name and "model" not in name for name in manifest["files"]))
        (destination / "data" / "val-00000-of-00001.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "incomplete/changed"):
            fetch.fetch_chartqa(destination, "main")

    def test_fetch_nonempty_destination_is_not_repurposed(self):
        destination = self.root / "existing"
        destination.mkdir()
        marker = destination / "user-owned.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "new/empty"):
            fetch.fetch_chartqa(destination)
        self.assertEqual(marker.read_text(), "keep")

    def test_failed_download_resumes_locked_sha_without_resolving_main_again(self):
        module, calls = self.mock_hub()
        original_download = module.snapshot_download
        attempts = []
        def interrupted_download(**kwargs):
            attempts.append(kwargs["revision"])
            original_download(**kwargs)
            if len(attempts) == 1:
                # Partial cached state is intentionally retained, like HF local_dir.
                cache = Path(kwargs["local_dir"]) / ".cache" / "huggingface"
                cache.mkdir(parents=True)
                (cache / "partial.download").write_bytes(b"partial")
                raise ConnectionError("synthetic network interruption")
        module.snapshot_download = interrupted_download
        destination = self.root / "resume"
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            with self.assertRaisesRegex(ConnectionError, "network interruption"):
                fetch.fetch_chartqa(destination, "main")
            locked = common.read_json(destination / fetch.REQUEST_NAME)
            self.assertEqual(locked["resolved_revision"], "a" * 40)
            self.assertFalse((destination / "download_manifest.json").exists())
            # A moving main (or unavailable metadata API) must not affect resume.
            module.HfApi = lambda: self.fail("Resume must not re-resolve main")
            manifest = fetch.fetch_chartqa(destination, "main")
        self.assertEqual(attempts, ["a" * 40, "a" * 40])
        self.assertEqual(manifest["resolved_revision"], "a" * 40)
        self.assertNotIn(fetch.REQUEST_NAME, manifest["files"])
        self.assertEqual(manifest["download_request_sha256"], common.hash_file(destination / fetch.REQUEST_NAME))
        self.assertEqual(len([call for call in calls if call[0] == "info"]), 1)

    def test_incomplete_download_rejects_changed_revision_and_foreign_file(self):
        module, _ = self.mock_hub()
        original_download = module.snapshot_download
        def interrupted_download(**kwargs):
            original_download(**kwargs)
            raise ConnectionError("interrupted")
        module.snapshot_download = interrupted_download
        destination = self.root / "partial"
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            with self.assertRaises(ConnectionError):
                fetch.fetch_chartqa(destination)
        with self.assertRaisesRegex(ValueError, "another revision"):
            fetch.fetch_chartqa(destination, "b" * 40)
        foreign = destination / "model.safetensors"
        foreign.write_bytes(b"do-not-touch")
        with self.assertRaisesRegex(ValueError, "non-allowed"):
            fetch.fetch_chartqa(destination)
        self.assertEqual(foreign.read_bytes(), b"do-not-touch")

    def test_fetch_proxy_override_occurs_in_subprocess(self):
        with patch.dict("os.environ", {"HTTPS_PROXY": "parent-secret-value"}), patch.object(fetch.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(fetch.main(["--destination", str(self.root / "child"), "--ignore-proxy"]), 0)
            kwargs = run.call_args.kwargs
            self.assertNotIn("HTTPS_PROXY", kwargs["env"])
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "0")
            self.assertEqual(kwargs["env"]["HF_DATASETS_OFFLINE"], "0")
            import os
            self.assertEqual(os.environ["HTTPS_PROXY"], "parent-secret-value")
            self.assertIn("--child", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
