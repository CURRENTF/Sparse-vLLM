"""Guard against stopping shared SSH sessions or unbounded outage recovery."""
import unittest
from unittest.mock import patch

from scripts.official_experiments.chain_cache_miniswe import reverse_ssh_watchdog as w


class ReverseSSHRecovery(unittest.TestCase):
    def test_shared_listener_owner_is_never_eligible(self):
        dedicated = 'LISTEN 0 128 127.0.0.1:22025 0.0.0.0:* users:(("sshd",pid=123,fd=5))'
        self.assertEqual(w.dedicated_pid(dedicated, 22025), 123)
        shared = dedicated + '\n' + dedicated.replace(':22025', ':22026')
        with self.assertRaisesRegex(RuntimeError, 'other listeners'):
            w.dedicated_pid(shared, 22025)
        with self.assertRaises(RuntimeError):
            w.dedicated_pid(dedicated.replace('127.0.0.1', '0.0.0.0'), 22025)
        with self.assertRaises(RuntimeError):
            w.dedicated_pid(dedicated.replace('sshd', 'python'), 22025)

    def check(self, state, **kwargs):
        return w.check_port(22025, state, timeout=1, failures=3,
                            max_rebuilds=2, expected_uid=0, **kwargs)

    def test_transient_probe_failure_does_not_reset_connection(self):
        with patch.object(w, 'probe', side_effect=TimeoutError('stalled')), \
                patch.object(w, 'reset_listener') as reset:
            state = self.check({})
            self.check(state)
            reset.assert_not_called()

    def test_rebuilds_are_bounded_and_success_opens_a_new_outage_budget(self):
        state = {}
        with patch.object(w, 'probe', side_effect=TimeoutError('stalled')), \
                patch.object(w, 'reset_listener', return_value=123) as reset:
            for _ in range(30):
                self.check(state)
            self.assertEqual(reset.call_count, 2)
            self.assertTrue(state['exhausted'])
        with patch.object(w, 'probe'), patch.object(w, 'reset_listener') as reset:
            self.check(state)
            self.assertTrue(state['healthy'])
            self.assertEqual(state['rebuilds'], 0)
            reset.assert_not_called()

    def test_unsafe_reset_is_explicit_and_not_counted_as_rebuilt(self):
        with patch.object(w, 'probe', side_effect=TimeoutError('stalled')), \
                patch.object(w, 'reset_listener', side_effect=RuntimeError('shared owner')):
            state = self.check({'failures': 2})
            self.assertFalse(state['healthy'])
            self.assertEqual(state['reset_error'], 'shared owner')
            self.assertNotIn('last_reset_pid', state)


if __name__ == '__main__':
    unittest.main()
