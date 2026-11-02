"""Small synthetic tests; these do not depend on private caches or models."""
import copy
import unittest

import probe_core as p


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "request": {"questions": {"a": "A question?", "b": "B question?"},
                        "requested_records": {"a": "record-a", "b": "record-b"},
                        "record_context": "Context control"},
            "sources": {s: {"source_id": "image:" + s, "record_id": "record-" + s}
                        for s in ("a", "b")}}
        self.obs = {"a": {"raw_text": '{"answer":"UNKNOWN"'},
                    "b": {"raw_text": '```json\n{"answer":"7"}\n```'}}

    def test_lossless_malformed_observation(self):
        new = p.adapter_messages(self.task, self.obs)
        old = p.c.answer_messages(self.task, self.obs)
        self.assertEqual(new[0], old[0])
        self.assertEqual(p.roundtrip_payload(new), p.c.strict_json(old[1]["content"]))
        payload = p.c.strict_json(new[1]["content"])
        self.assertEqual(payload["evidence_packet"]["retrieved_evidence"][0]["summary"], self.obs["a"]["raw_text"])

    def test_no_answers_status_or_role_injection(self):
        modified = copy.deepcopy(self.task)
        modified.update(gold="SECRET", condition="source_outage", reference_answer="GOLD")
        self.assertEqual(p.adapter_messages(modified, self.obs), p.adapter_messages(self.task, self.obs))
        payload = p.c.strict_json(p.adapter_messages(modified, self.obs)[1]["content"])
        self.assertEqual(set(payload), {"questions", "requested_records", "record_context", "evidence_packet"})
        self.assertEqual(set(payload["evidence_packet"]["retrieved_evidence"][0]),
                         {"source_id", "record_id", "summary", "origin"})

    def test_scope_mismatch_is_not_auto_repaired(self):
        self.task["sources"]["a"]["record_id"] = "another-record"
        payload = p.roundtrip_payload(p.adapter_messages(self.task, self.obs))
        self.assertEqual(payload["evidence"]["a"]["record_id"], "another-record")
        self.assertEqual(payload["requested_records"]["a"], "record-a")

    def test_hash_selection_order_invariant(self):
        tasks = [{"block_id": str(n)} for n in range(10)]
        self.assertEqual(p.selected_block_ids(tasks), p.selected_block_ids(tasks[::-1]))
        self.assertEqual(len(p.selected_block_ids(tasks)), 3)

    def test_real_no_evidence_string_preserved(self):
        self.obs["a"]["raw_text"] = ""
        value = p.roundtrip_payload(p.adapter_messages(self.task, self.obs))
        self.assertEqual(value["evidence"]["a"]["observation"], "")


if __name__ == "__main__":
    unittest.main()
