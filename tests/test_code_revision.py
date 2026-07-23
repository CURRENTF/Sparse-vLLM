import unittest
from unittest.mock import patch

from sparsevllm.utils.code_revision import code_revision_info


class CodeRevisionTest(unittest.TestCase):
    def tearDown(self):
        code_revision_info.cache_clear()

    @patch("sparsevllm.utils.code_revision._git_command", return_value=None)
    @patch("sparsevllm.utils.code_revision.version", return_value="0.1.0")
    def test_uses_published_distribution_name(self, mock_version, _mock_git):
        code_revision_info.cache_clear()

        revision = code_revision_info()

        mock_version.assert_called_once_with("deltakv")
        self.assertEqual(revision["package_version"], "0.1.0")
        self.assertIsNone(revision["git_commit"])


if __name__ == "__main__":
    unittest.main()
