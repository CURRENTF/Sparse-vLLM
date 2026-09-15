"""HTTP instrumentation must not change responses or hide execution failures."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.official_experiments.chain_cache_miniswe.request_logging import RequestLoggingMiddleware


class RequestLoggingContracts(unittest.TestCase):
    def test_forwards_chunked_body_unchanged_and_records_usage(self):
        async def run(root):
            sent = []
            incoming = {"type": "http.request", "body": b'{"stream":false}'}
            messages = [
                {"type": "http.response.start", "status": 200, "headers": []},
                {"type": "http.response.body", "body": b'{"id":"test",', "more_body": True},
                {"type": "http.response.body", "body": b'"usage":{"completion_tokens":2}}'},
            ]
            async def receive():
                return incoming
            async def send(message):
                sent.append(message)
            async def app(scope, receive, send):
                self.assertIs(await receive(), incoming)
                for message in messages:
                    await send(message)
            await RequestLoggingMiddleware(app)({"type": "http", "path": "/v1/chat/completions"}, receive, send)
            self.assertEqual(sent, messages)
            record = json.loads(next(Path(root).glob("*.json")).read_text())
            self.assertEqual(record["response"]["usage"]["completion_tokens"], 2)
            self.assertEqual(record["request"], {"stream": False})
            self.assertEqual(record["status"], "success")
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, MINISWE_REQUEST_LOG_DIR=root):
            asyncio.run(run(root))

    def test_failure_is_logged_and_propagated(self):
        async def app(*args):
            raise RuntimeError("original model failure")
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, MINISWE_REQUEST_LOG_DIR=root):
            with self.assertRaisesRegex(RuntimeError, "original model failure"):
                asyncio.run(RequestLoggingMiddleware(app)(
                    {"type": "http", "path": "/v1/chat/completions"}, None, None))
            record = json.loads(next(Path(root).glob("*.json")).read_text())
            self.assertEqual(record["status"], "failed")
            self.assertIn("original model failure", record["error"])
