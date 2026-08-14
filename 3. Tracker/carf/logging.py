"""Versioned JSONL logging for CARF write audits."""

import json
import os
import threading


SCHEMA_VERSION = 'carf.audit.v2'


class AuditJSONLWriter:
    def __init__(self, path, schema_version=SCHEMA_VERSION):
        self.path = path
        self.schema_version = schema_version
        self._lock = threading.Lock()
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._file = open(path, 'w', encoding='utf-8')

    def write(self, record):
        payload = {'schema_version': self.schema_version, **record}
        with self._lock:
            self._file.write(json.dumps(payload, sort_keys=True,
                                        separators=(',', ':')) + '\n')
            self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
