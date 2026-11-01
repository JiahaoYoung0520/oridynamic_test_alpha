"""下载与遗留在线 MSA 入口回归；无真实网络请求。"""
from contextlib import redirect_stderr
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import requests
from runner.download import download_from_url
from protenix.web_service.colab_request_utils import service_auth, run_mmseqs2_service


class ResourceTests(unittest.TestCase):
    def test_download_hash_failure_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'resource'; path.write_bytes(b'original')
            response = io.BytesIO(b'new content')
            response.headers = {'Content-Length': '11'}
            with patch('urllib.request.urlopen', return_value=response), self.assertRaisesRegex(ValueError, 'SHA-256'):
                download_from_url('https://example.org/resource', str(path), expected_sha256='0'*64)
            self.assertEqual(path.read_bytes(), b'original')
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            response = io.BytesIO(b'new content'); response.headers = {'Content-Length': '11'}
            with patch('urllib.request.urlopen', return_value=response):
                download_from_url('https://example.org/resource', str(path),
                                  expected_sha256=hashlib.sha256(b'new content').hexdigest())
            self.assertEqual(path.read_bytes(), b'new content')

    def test_truncated_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'resource'
            response = io.BytesIO(b'bad'); response.headers = {'Content-Length': '100'}
            with patch('urllib.request.urlopen', return_value=response), self.assertRaisesRegex(ValueError, 'incomplete'):
                download_from_url('https://example.org/resource', str(path))
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_msa_auth_is_optional_and_requires_a_pair(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(service_auth())
            with patch.dict(os.environ, {'MSA_SERVICE_USERNAME': 'test'}):
                with self.assertRaises(ValueError): service_auth()
            with patch.dict(os.environ, {'MSA_SERVICE_USERNAME': 'test', 'MSA_SERVICE_PASSWORD': 'dummy'}):
                self.assertEqual(service_auth().username, 'test')

    def test_custom_msa_endpoint_and_bounded_timeouts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
             patch('requests.post', side_effect=requests.exceptions.Timeout) as post, \
             patch('time.sleep'), redirect_stderr(io.StringIO()):
            with self.assertRaises(requests.exceptions.Timeout):
                run_mmseqs2_service('ACDE', str(Path(tmp)/'msa'),
                    host_url='https://mirror.example/msa/', server_mode='protenix', user_agent='test')
            self.assertEqual(post.call_count, 6)
            self.assertEqual(post.call_args.args[0], 'https://mirror.example/msa/ticket/msa')
            self.assertIsNone(post.call_args.kwargs['auth'])


if __name__ == '__main__': unittest.main()
