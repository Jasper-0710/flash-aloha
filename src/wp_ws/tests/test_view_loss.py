"""Synthetic logs only; optional PNG test is CPU/headless. Never train or use W&B."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.wp_ws import view_loss as viewer


class ViewLossTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for module in (viewer, viewer.metrics):
            for name, folder in (('LOG_DIR', 'logs'), ('DATA_DIR', 'data'), ('CKPT_DIR', 'ckpts')):
                mocked = patch.object(module, name, self.root / folder)
                mocked.start()
                self.addCleanup(mocked.stop)
        self.source = viewer.LOG_DIR / 'wp_ws' / 'flip_the_pillow' / 'synthetic'
        self.source.mkdir(parents=True)
        self.manifest = {'format': 'wp_ws.act_baseline.v1', 'model_variant': 'decoder_only_ACT',
                         'data_config': {'task_name': 'flip_the_pillow'},
                         'validation_selection': {'mode': 'full_fixed_validation'},
                         'batch_size': 4, 'steps_requested': 30000}
        (self.source / 'run_config.json').write_text(json.dumps(self.manifest))
        self.train = {'event': 'train', 'step': 10, 'normalized_l1': 0.6}
        self.val = {'event': 'validation', 'step': 10, 'normalized_l1': 0.8,
                    'samples': 5823, 'mode': 'full_fixed_validation'}
        self.new = {'event': 'train', 'step': 20, 'normalized_l1': 0.4}

    def write(self, *records, suffix=b''):
        content = b''.join(json.dumps(row).encode() + b'\n' for row in records)
        (self.source / 'metrics.jsonl').write_bytes(content + suffix)

    def hashes(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.source.iterdir() if p.is_file()}

    def run_viewer(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            code = viewer.main(list(arguments))
        self.assertEqual(code, 0)
        return output.getvalue()

    def test_snapshot_reports_partial_line_without_changing_source(self):
        self.write(self.train, self.val, suffix=b'{"event":"train"')
        before = self.hashes()
        records, pending = viewer.snapshot(self.source)
        self.assertEqual(records, [self.train, self.val])
        self.assertTrue(pending)
        self.assertEqual(before, self.hashes())

    def test_snapshot_handles_missing_or_only_partial_metrics(self):
        self.assertEqual(viewer.snapshot(self.source), ([], False))
        self.write(suffix=b'{"event":')
        self.assertEqual(viewer.snapshot(self.source), ([], True))

    def test_rejects_malformed_complete_line_and_nonfinite_loss(self):
        self.write(suffix=b'not-json\n')
        with self.assertRaises(ValueError):
            viewer.snapshot(self.source)
        self.write(dict(self.train, normalized_l1=float('nan')))
        with self.assertRaises(ValueError):
            viewer.snapshot(self.source)

    def test_summary_latest_best_and_recent_records(self):
        self.write(self.train, self.val, self.new,
                   dict(self.val, step=20, normalized_l1=0.9))
        before = self.hashes()
        text = self.run_viewer('--run-name', 'synthetic', '--last', '2')
        self.assertIn('Requested steps: 30000', text)
        self.assertIn('best val: step=10 normalized_l1=0.800000', text)
        self.assertIn('Latest train: step=20 normalized_l1=0.400000', text)
        table = text.split('Recent losses', 1)[1]
        self.assertNotIn('step=      10', table)
        self.assertEqual(before, self.hashes())
        self.assertIn('not proof', text)

    def test_list_run_with_requested_steps(self):
        self.write(self.train)
        text = self.run_viewer('--list')
        self.assertIn('synthetic  requested=30000  latest_train_step=10', text)

    def test_list_excludes_preflight(self):
        self.manifest.pop('steps_requested')
        (self.source / 'run_config.json').write_text(json.dumps(self.manifest))
        text = self.run_viewer('--list')
        self.assertIn('No ACT training runs found', text)

    def test_follow_rejects_replacement_between_snapshot_and_follow(self):
        self.write(self.train)
        original = viewer.metrics.metric_records

        def replace_file(source, **kwargs):
            replacement = self.source / 'replacement.jsonl'
            replacement.write_text(json.dumps(self.new) + '\n'
                                   + '{"event":"completed","steps":20}\n')
            replacement.replace(self.source / 'metrics.jsonl')
            yield from original(source, **kwargs)

        args = viewer.build_parser().parse_args(['--run-name', 'synthetic', '--follow'])
        with patch.object(viewer.metrics, 'metric_records', replace_file):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'replaced/truncated'):
                viewer.view(args)

    def test_completed_and_failed_states(self):
        self.write(self.train)
        (self.source / 'COMPLETED.json').write_text('{}')
        self.assertEqual(viewer.state(self.source, []), 'completed')
        (self.source / 'FAILED.json').write_text('{}')
        self.assertEqual(viewer.state(self.source, []), 'failed')

    def test_follow_skips_snapshot_and_waits_for_partial_line(self):
        self.write(self.train, suffix=b'{"event":"train",')
        original = viewer.metrics.metric_records

        def finish_partial_line(_seconds):
            with (self.source / 'metrics.jsonl').open('ab') as stream:
                stream.write(b'"step":20,"normalized_l1":0.4}\n')
                stream.write(b'{"event":"completed","steps":20}\n')
            (self.source / 'COMPLETED.json').write_text('{}')

        def read_records(source, **kwargs):
            yield from original(source, **kwargs, sleep=finish_partial_line)

        with patch.object(viewer.metrics, 'metric_records', read_records):
            text = self.run_viewer('--run-name', 'synthetic', '--follow')
        self.assertIn('incomplete final JSONL record', text)
        self.assertEqual(text.count('train  step=      10'), 1)
        self.assertEqual(text.count('train  step=      20'), 1)
        self.assertIn('State: completed', text)

    def test_ctrl_c_only_stops_viewer(self):
        self.write(self.train)
        before = self.hashes()

        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt
            yield  # Make this a generator, like the production reader.

        with patch.object(viewer.metrics, 'metric_records', interrupted):
            text = self.run_viewer('--run-name', 'synthetic', '--follow')
        self.assertIn('original training process was not touched', text)
        self.assertEqual(before, self.hashes())

    def test_rejects_escaping_source_file(self):
        other = self.root / 'outside.jsonl'
        other.write_text(json.dumps(self.train) + '\n')
        (self.source / 'metrics.jsonl').symlink_to(other)
        with self.assertRaises(ValueError):
            viewer.snapshot(self.source)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional matplotlib unavailable')
    def test_png_plot_is_separate_and_source_unchanged(self):
        self.write(self.train, self.val, self.new)
        before = self.hashes()
        previous = os.environ.get('MPLCONFIGDIR')
        text = self.run_viewer('--run-name', 'synthetic', '--plot')
        target = viewer.LOG_DIR / 'wp_ws' / '_loss_views' / 'flip_the_pillow' / 'synthetic' / 'loss.png'
        self.assertTrue(target.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertIn(str(target), text)
        self.assertEqual(before, self.hashes())
        self.assertEqual(os.environ.get('MPLCONFIGDIR'), previous)
        self.assertFalse(list(target.parent.glob('mpl-cache-*')))

    def test_plot_cannot_escape_namespace_or_overlap_source(self):
        self.write(self.train)
        namespace = viewer.LOG_DIR / 'wp_ws' / '_loss_views' / 'flip_the_pillow'
        namespace.mkdir(parents=True)
        (namespace / 'synthetic').symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            viewer.save_plot(self.source, 'flip_the_pillow', 'synthetic',
                             {'validation_mode': 'full_fixed_validation'}, [self.train])


if __name__ == '__main__':
    unittest.main()
