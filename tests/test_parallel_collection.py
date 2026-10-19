"""Spawn/failure tests. Synthetic worker faults are NOT training examples."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from collect_exploration_trajs import collect_exploration, _exploration_worker
from collection_workers import CHILD_THREAD_LIMITS, run_workers
from exploration import ExplorationScene
from inspect_grasp_dataset import validate_dataset


def echo_worker(pipe, stop):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        while True:
            job = pipe.recv()
            if job is None or stop.is_set():
                return
            # Ensure the coordinator handles a later episode finishing first.
            if job['id'] == 0:
                time.sleep(.5)
            pipe.send({'status': 'complete', 'row': {
                'id': job['id'], 'pid': os.getpid(),
                'limits': {k: os.environ.get(k) for k in CHILD_THREAD_LIMITS}}})
    finally:
        pipe.close()


def crash_worker(pipe, stop):
    pipe.recv()
    os._exit(17)


def wrong_id_worker(pipe, stop):
    pipe.recv()
    pipe.send({'status': 'complete', 'row': {'id': 9999}})
    pipe.close()


def fault_after_physics(env, action):
    # Execute real physics, then inject an exception: no successor may be faked.
    ExplorationScene.__bases__[0].step(env, action)
    raise RuntimeError('injected spawned worker physics error')


def failing_physics_worker(pipe, stop):
    with patch.object(ExplorationScene, 'step', fault_after_physics):
        _exploration_worker(pipe, stop)


class ParallelCollectionTests(unittest.TestCase):
    def setUp(self):
        self.before = {p.pid for p in mp.active_children()}

    def tearDown(self):
        self.assertEqual({p.pid for p in mp.active_children()} - self.before, set(),
                         'collection left child writers running')

    def test_dispatch_handles_out_of_order_completion_and_restores_environment(self):
        original = {k: os.environ.get(k) for k in CHILD_THREAD_LIMITS}
        rows = []
        run_workers([{'id': i} for i in range(6)], echo_worker, 2, rows.append)
        self.assertEqual(sorted(r['id'] for r in rows), list(range(6)))
        self.assertEqual(len({r['pid'] for r in rows}), 2)
        self.assertNotEqual(rows[0]['id'], 0)
        self.assertTrue(all(r['limits'] == CHILD_THREAD_LIMITS for r in rows))
        self.assertEqual(original, {k: os.environ.get(k) for k in CHILD_THREAD_LIMITS})

    def test_abrupt_worker_death_returns_error(self):
        with self.assertRaisesRegex(RuntimeError, 'Worker exited'):
            run_workers([{'id': i} for i in range(2)], crash_worker, 2, lambda row: None)

    def test_wrong_episode_id_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'wrong episode id'):
            run_workers([{'id': i} for i in range(2)], wrong_id_worker, 2, lambda row: None)

    def test_parent_interrupt_stops_all_children(self):
        def interrupted(row):
            raise KeyboardInterrupt('test interrupt')
        with self.assertRaises(KeyboardInterrupt):
            run_workers([{'id': i} for i in range(8)], echo_worker, 2, interrupted)

    def test_serial_two_and_four_workers_match_by_episode_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = []
            for workers in (1, 2, 4):
                folder = root / f'w{workers}'
                manifest = collect_exploration(folder, n_trajectories=4, seed=31,
                                               max_seconds=1.5, workers=workers)
                manifests.append(manifest)
                self.assertTrue(validate_dataset(folder)['valid'])
                self.assertEqual([r['id'] for r in manifest['episodes']], [0, 1, 2, 3])
                self.assertEqual(manifest['execution']['workers_used'], workers)
                self.assertEqual(len({r['worker_pid'] for r in manifest['episodes']}), workers)
                self.assertGreater(manifest['summary']['actions_per_wall_second'], 0)
            for manifest in manifests[1:]:
                self.assertEqual(manifest['mode_schedule'], manifests[0]['mode_schedule'])
                self.assertEqual(manifest['summary']['class_counts'], manifests[0]['summary']['class_counts'])
            for index in range(4):
                baseline = root / 'w1' / f'episode_{index:04d}'
                baseline_meta = json.loads((baseline / 'metadata.json').read_text())
                for workers in (2, 4):
                    other = root / f'w{workers}' / baseline.name
                    metadata = json.loads((other / 'metadata.json').read_text())
                    for key in ('start_seed', 'action_seed', 'hover_goal_xyz', 'start_sample', 'outcome'):
                        self.assertEqual(baseline_meta[key], metadata[key], key)
                    with np.load(baseline / 'trajectory.npz') as a, np.load(other / 'trajectory.npz') as b:
                        self.assertEqual(set(a.files), set(b.files))
                        for key in a.files:
                            np.testing.assert_array_equal(a[key], b[key], err_msg=f'{index}/{workers}/{key}')
                    for frame in (0, 15, 30):
                        name = f'images/{frame:06d}.png'
                        self.assertEqual((baseline / name).read_bytes(), (other / name).read_bytes())

    def test_invalid_worker_counts_create_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'absent'
            for workers in (0, -1, 5, 2.0, True):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    collect_exploration(output, workers=workers)
                self.assertFalse(output.exists())

    def test_more_workers_than_episodes_is_capped(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'data'
            result = collect_exploration(folder, n_trajectories=1, max_seconds=.05, workers=4)
            self.assertEqual(result['execution']['workers_requested'], 4)
            self.assertEqual(result['execution']['workers_used'], 1)
            self.assertTrue(validate_dataset(folder)['valid'])

    def test_actual_cli_entrypoint_spawns_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'cli_data'
            result = subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[1] / 'collect_exploration_trajs.py'),
                '--workers', '2', '--n-trajectories', '4', '--max-seconds', '.1',
                '--output-dir', str(folder)], capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('2 worker(s)', result.stdout)
            self.assertTrue(validate_dataset(folder)['valid'])
            manifest = json.loads((folder / 'manifest.json').read_text())
            self.assertEqual(len({row['worker_pid'] for row in manifest['episodes']}), 2)

    def test_worker_physics_error_quarantines_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'data'
            with patch('collect_exploration_trajs._exploration_worker', failing_physics_worker):
                with self.assertRaisesRegex(RuntimeError, 'injected spawned worker'):
                    collect_exploration(root, n_trajectories=2, max_seconds=1, workers=2)
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'interrupted')
            self.assertFalse(validate_dataset(root)['valid'])
            self.assertTrue(manifest['incomplete_episodes'])
            for entry in manifest['incomplete_episodes']:
                episode = root / entry['path']
                meta = json.loads((episode / 'metadata.json').read_text())
                self.assertEqual(meta['status'], 'error')
                self.assertTrue((episode / 'partial_trajectory.npz').exists())

    def test_crash_marks_manifest_interrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'data'
            with patch('collect_exploration_trajs._exploration_worker', crash_worker):
                with self.assertRaisesRegex(RuntimeError, 'Worker exited'):
                    collect_exploration(root, n_trajectories=2, max_seconds=1, workers=2)
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'interrupted')
            self.assertFalse(validate_dataset(root)['valid'])

    def test_parent_manifest_error_preserves_completed_episode(self):
        from collect_grasp_trajs import write_json as real_write
        def interrupt_after_complete(path, content):
            if Path(path).name == 'manifest.json' and content['status'] == 'collecting' and content['episodes']:
                raise KeyboardInterrupt('test parent interrupt after a completed episode')
            real_write(path, content)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'data'
            with patch('collect_exploration_trajs.write_json', interrupt_after_complete):
                with self.assertRaises(KeyboardInterrupt):
                    collect_exploration(root, n_trajectories=4, max_seconds=.1, workers=2)
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'interrupted')
            self.assertGreaterEqual(len(manifest['episodes']), 1)
            for row in manifest['episodes']:
                meta = json.loads((root / row['path'] / 'metadata.json').read_text())
                self.assertEqual(meta['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
