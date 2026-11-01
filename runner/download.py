# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download a resource without deserializing its contents."""
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


def download_from_url(tos_url: str, destination: str, *, expected_sha256=None) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(prefix=target.name + '.', suffix='.part', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(tos_url, timeout=60) as response:
            digest = hashlib.sha256()
            size = 0
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                size += len(block)
            expected_size = response.headers.get('Content-Length')
            if size == 0 or (expected_size is not None and size != int(expected_size)):
                raise ValueError('Empty or incomplete download')
            if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                raise ValueError('Downloaded resource SHA-256 does not match')
        os.replace(partial, target)
    finally:
        Path(partial).unlink(missing_ok=True)
