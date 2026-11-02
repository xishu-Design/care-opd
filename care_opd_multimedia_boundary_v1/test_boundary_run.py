"""CPU integration tests. Fake model outputs are fixtures, never experiment results."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import boundary_common as c
import boundary_prepare as prepare
import boundary_run as runner
from test_boundary_prepare import fixture_record


class FakeRuntime:
    def __init__(self, *, truncated=False, image_seen=True):
        self.calls = []
        self.truncated = truncated
        self.image_seen = image_seen
        self.changed = False
        self.closed = 0
        self.open_count = 0

    def check_stack(self, config):
        assert set(config['models']) == {'vlm', *c.ARMS}

    def environment(self):
        return {'python': 'fake-fixture', 'torch': 'none', 'gpu_name': 'fake-test-device',
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '1'),
                'gpu_uuid': 'fixture-' + os.environ.get('CUDA_VISIBLE_DEVICES', '1')}

    def fingerprint_model(self, path, previous=None, full=False):
        return {'sha256': c.digest([path.name, self.changed]), 'files': {}, 'stat': {}}

    def HFGenerator(self, model, generation, visual):
        parent = self
        self.open_count += 1
        assert self.open_count == 1, 'Generators must not co-reside on one GPU'

        class Generator:
            def generate(self, messages, image_path, max_tokens):
                parent.calls.append((visual, model, copy.deepcopy(messages), image_path, max_tokens))
                if visual:
                    return {'raw_text': '{"answer":"fixture-answer","evidence":"fixture","transcribed_text":[]}',
                            'finish_reason': 'stop', 'image_seen': parent.image_seen}
                return {'raw_text': '' if parent.truncated else 'not JSON: retained first output',
                        'finish_reason': 'length' if parent.truncated else 'stop', 'image_seen': False}

            def close(self):
                parent.closed += 1
                parent.open_count -= 1

        return Generator()


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='boundary-run-test-')
        self.root = Path(self.temp.name)
        self.work = self.root / 'work'
        prepare.prepare_bundle([fixture_record(i) for i in range(1, 9)], self.work, block_count=2)
        self.runtime_dir = self.root / 'runtime'
        self.runtime_dir.mkdir()
        for name in ('evaluate.py', 'common.py'):
            (self.runtime_dir / name).write_text('# fixture only\n', encoding='utf-8')
        self.models = {}
        for arm in ('vlm', *c.ARMS):
            path = self.root / 'models' / arm
            path.mkdir(parents=True)
            (path / 'config.json').write_text('{}', encoding='utf-8')
            self.models[arm] = str(path)
        self.source = self.root / 'source.json'
        c.json_once(self.source, {'models': self.models})
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        runner.init(self.work, self.source, self.runtime_dir)

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.temp.cleanup()

    def test_real_preparation_accepts_changed_record_for_valid_update(self):
        tasks, selection = runner.load_prepared(self.work)
        self.assertEqual(selection['blocks'], 2)
        self.assertEqual(len(tasks), 12)
        clean = next(t for t in tasks if t['condition'] == 'clean')
        update = next(t for t in tasks if t['block_id'] == clean['block_id'] and t['condition'] == 'valid_update')
        self.assertNotEqual(clean['request']['requested_records']['a'], update['request']['requested_records']['a'])

    def test_smoke_resume_freezes_invalid_first_outputs_and_scores_all_rows(self):
        runtime = FakeRuntime(truncated=True)
        runner.run(self.work, 1, runtime=runtime)
        files = [self.work / 'responses' / arm / 'responses.jsonl' for arm in (*c.ARMS, *c.BASELINES)]
        prefixes = {p: p.read_bytes() for p in files}
        self.assertTrue(all(len(c.read_jsonl(p)) == 6 for p in files))
        with self.assertRaisesRegex(ValueError, 'incomplete|responses'):
            runner.score(self.work)
        runner.run(self.work, runtime=runtime)
        for path in files:
            self.assertTrue(path.read_bytes().startswith(prefixes[path]))
            self.assertEqual(len(c.read_jsonl(path)), 12)
        calls = len(runtime.calls)
        snapshots = {p: p.read_bytes() for p in files}
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            runner.run(self.work, runtime=runtime)
        self.assertEqual(len(runtime.calls), calls)
        self.assertEqual(snapshots, {p: p.read_bytes() for p in files})
        report = runner.score(self.work)
        for arm in c.ARMS:
            overall = report['arms'][arm]['semantic']['overall']
            self.assertEqual(overall['joint_visual_safe_useful']['denominator'], 12)
            self.assertEqual(overall['truncated']['count'], 12)
            self.assertEqual(overall['joint_visual_safe_useful']['count'], 0)
        self.assertEqual(report['arms']['always_abstain']['semantic']['overall']['joint_visual_safe_useful']['count'], 0)
        self.assertEqual(runtime.open_count, 0)
        self.assertEqual(len(list((self.work / 'summary').glob('*/metrics.csv'))), 1)
        runner.score(self.work)  # Identical scoring is idempotent and immutable.

    def test_inference_does_not_parse_gold(self):
        original = c.read_jsonl
        def guard(path):
            if Path(path).name == 'gold.private.jsonl':
                self.fail('Inference attempted to parse private gold')
            return original(path)
        runtime = FakeRuntime()
        with patch.object(c, 'read_jsonl', side_effect=guard):
            runner.run(self.work, 1, runtime=runtime)
        for visual, model, messages, image_path, budget in runtime.calls:
            combined = c.canonical(messages)
            self.assertNotIn('fixture-value-', combined)
            self.assertNotIn('gold.private', combined)
            if not visual:
                self.assertIsNone(image_path)
                payload = c.strict_json(messages[1]['content'])
                self.assertEqual(set(payload), {'questions', 'requested_records', 'record_context', 'evidence'})
                self.assertEqual(budget, 1024)

    def test_vlm_actually_requires_image(self):
        runtime = FakeRuntime(image_seen=False)
        with self.assertRaisesRegex(RuntimeError, 'real pixels'):
            runner.run(self.work, 1, runtime=runtime)
        self.assertFalse((self.work / 'observations.jsonl').exists())
        self.assertTrue((self.work / 'infrastructure_errors.jsonl').exists())
        self.assertEqual(runtime.open_count, 0)

    def test_observation_cache_reuses_scope_and_preserved_b(self):
        runtime = FakeRuntime()
        runner.run(self.work, runtime=runtime)
        tasks, _ = runner.load_prepared(self.work)
        unique = {c.observation_spec(t, side)[0] for t in tasks for side in ('a', 'b')}
        self.assertEqual(sum(call[0] for call in runtime.calls), len(unique))
        self.assertLess(len(unique), len(tasks) * 2)
        for block in {t['block_id'] for t in tasks}:
            rows = [t for t in tasks if t['block_id'] == block]
            self.assertEqual(len({c.observation_spec(t, 'b')[0] for t in rows}), 1)

    def test_changed_model_fingerprint_rejected(self):
        runtime = FakeRuntime()
        runner.run(self.work, 1, runtime=runtime)
        runtime.changed = True
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            runner.run(self.work, runtime=runtime)

    def test_changed_code_rejected(self):
        with patch.object(runner, 'code_hashes', return_value={'changed': 'fixture'}):
            with self.assertRaisesRegex(ValueError, 'code changed'):
                runner.load_config(self.work)

    def test_changed_task_bytes_rejected(self):
        with (self.work / 'tasks.jsonl').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            runner.load_prepared(self.work)

    def test_changed_first_output_rejected(self):
        runtime = FakeRuntime()
        runner.run(self.work, 1, runtime=runtime)
        path = self.work / 'responses' / c.ARMS[0] / 'responses.jsonl'
        rows = c.read_jsonl(path)
        rows[0]['raw_text'] = '{"edited":"after inspection"}'
        path.write_text(''.join(c.canonical(row) + '\n' for row in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'first output changed'):
            runner.run(self.work, runtime=runtime)

    def test_truncated_ledger_is_not_silently_discarded(self):
        path = self.root / 'broken.jsonl'
        path.write_bytes(b'{"task_id":"a"')
        with self.assertRaisesRegex(ValueError, 'unterminated'):
            c.ledger(path)

    def test_lock_does_not_automatically_remove_other_owner(self):
        with c.run_lock(self.work):
            with self.assertRaisesRegex(RuntimeError, 'lock exists'):
                with c.run_lock(self.work):
                    self.fail('second writer entered')
        self.assertFalse((self.work / '.boundary.lock').exists())

    def test_inner_whitelist_excludes_extra_request_key(self):
        task = c.read_jsonl(self.work / 'tasks.jsonl')[0]
        task['request']['questions']['gold'] = 'forbidden'
        obs = {'a': {'raw_text': '{}'}, 'b': {'raw_text': '{}'}}
        payload = c.strict_json(c.answer_messages(task, obs)[1]['content'])
        self.assertEqual(set(payload['questions']), {'a', 'b'})
        self.assertNotIn('forbidden', c.canonical(payload))

    def test_rule_baseline_never_uses_condition_or_gold(self):
        task = c.read_jsonl(self.work / 'tasks.jsonl')[0]
        obs = {side: {'raw_text': '{"answer":"100"}', 'finish_reason': 'stop'} for side in ('a', 'b')}
        response = c.rule_response(task, obs)
        altered = copy.deepcopy(task)
        altered['condition'] = 'source_outage'
        altered['gold'] = {'a': 'ABSTAIN'}
        self.assertEqual(response, c.rule_response(altered, obs))
        obs['a']['raw_text'] = '{"answer":"UNKNOWN"}'
        parsed = c.strict_json(c.rule_response(task, obs))
        self.assertEqual(parsed['sections'][0]['content'], 'ABSTAIN')
        self.assertEqual(parsed['sections'][1]['content'], '100')


if __name__ == '__main__':
    unittest.main()
