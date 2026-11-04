import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.slurm_remote import run
from scripts.slurm_train import prepare
from scripts.slurm_worker import lease_valid
from scripts.setup_training import disk_bytes, bounded_install


class SlurmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rid = 'a' * 32
        self.profile = {'root': str(self.root), 'resources': {'partition': 'gpu', 'gres': 'gpu:1'},
                        'smoke_resources': {'partition': 'all', 'time': '00:03:00'},
                        'models': {'smolvla_sft': {'python': os.sys.executable}}}
        self.exp = {'recipe': 'smolvla_sft', 'steps': 10, 'seed': 42, 'learning_rate': .001, 'dataset': 'lab/$(touch bad); data'}

    def tearDown(self):
        self.temp.cleanup()

    def test_submit_is_idempotent_and_input_stays_in_json(self):
        payload = prepare(self.profile, self.exp, self.rid)
        self.assertNotIn(self.exp['dataset'], str(payload['launch']['command']))
        with patch('scripts.slurm_remote.call', return_value='1234') as caller:
            first = run(payload)
            second = run(payload)
        self.assertEqual(first, second)
        self.assertEqual(caller.call_count, 1)
        self.assertEqual(caller.call_args[0][0][0], 'sbatch')
        self.assertNotIn(self.exp['dataset'], caller.call_args[1]['input'])
        directory = Path(first['directory'])
        self.assertEqual(json.loads((directory / 'experiment.json').read_text()), self.exp)
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)

    def test_uncertain_submission_is_never_repeated(self):
        payload = prepare(self.profile, self.exp, self.rid)
        with patch('scripts.slurm_remote.call', side_effect=OSError('connection lost')):
            with self.assertRaises(OSError):
                run(payload)
        with patch('scripts.slurm_remote.call') as caller:
            with self.assertRaisesRegex(ValueError, 'uncertain'):
                run(payload)
            caller.assert_not_called()

    def test_status_streams_log_with_offset_and_exit_code(self):
        payload = prepare(self.profile, self.exp, self.rid)
        with patch('scripts.slurm_remote.call', return_value='1234'):
            job = run(payload)
        (Path(job['directory']) / 'stdout.log').write_text('first\nsecond\n')
        with patch('scripts.slurm_remote.call', side_effect=['', '1234|COMPLETED|0:0|']):
            result = run({'action': 'status', 'root': str(self.root), 'run_id': self.rid, 'offset': 6})
        self.assertEqual(result['log'], 'second\n')
        self.assertEqual(result['offset'], 13)
        self.assertEqual(result['exit_code'], '0:0')
        self.assertEqual(result['state'], 'COMPLETED')

    def test_cancel_prevents_late_submission_and_expires_worker_lease(self):
        with patch('scripts.slurm_remote.call', return_value='') as caller:
            run({'action': 'cancel', 'root': str(self.root), 'run_id': self.rid})
        self.assertEqual(caller.call_args[0][0], ['scancel', '--name=armature-' + self.rid])
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            run(prepare(self.profile, self.exp, self.rid))
        self.assertFalse(lease_valid(self.root / 'runs' / self.rid))

    def test_lease_expiry_and_renewal(self):
        self.assertFalse(lease_valid(self.root))
        (self.root / 'lease').touch()
        now = time.time()
        self.assertTrue(lease_valid(self.root, now))
        self.assertFalse(lease_valid(self.root, now + 121))

    def test_storage_guard_handles_vanished_temporary_files(self):
        output = subprocess.CompletedProcess([], 1, '2048\t/project\n', 'du: /project/.tmp: No such file or directory\n')
        with patch('scripts.setup_training.subprocess.run', return_value=output):
            self.assertEqual(disk_bytes('/project'), 2 * 1024**2)
        output.stderr = 'du: /project/cache: Permission denied\n'
        with patch('scripts.setup_training.subprocess.run', return_value=output):
            with self.assertRaises(RuntimeError):
                disk_bytes('/project')

    def test_storage_guard_stops_installation_at_limit(self):
        process = subprocess.Popen([os.sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        with patch('scripts.setup_training.disk_bytes', side_effect=[0, 2 * 1024**3]), patch('scripts.setup_training.subprocess.Popen', return_value=process):
            with self.assertRaisesRegex(SystemExit, 'storage limit reached'):
                bounded_install(['unused'], self.root, 1)
        self.assertIsNotNone(process.poll())

    def test_rejects_traversal_and_unregistered_source(self):
        payload = prepare(self.profile, self.exp, self.rid)
        for rid in ['../escape', '-1', '0;echo bad']:
            with self.assertRaises(ValueError):
                run({**payload, 'run_id': rid})
        payload['files'] = {'../../escape.py': 'bad'}
        with self.assertRaises(ValueError):
            run(payload)
        self.assertFalse((self.root / 'escape.py').exists())


if __name__ == '__main__':
    unittest.main()
