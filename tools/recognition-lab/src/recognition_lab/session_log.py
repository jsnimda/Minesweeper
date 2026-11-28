"""Bounded plain-text diagnostics for one application session."""
from collections import deque
from datetime import datetime


class SessionLog:
    def __init__(self, capacity=1000):
        self.entries = deque(maxlen=capacity)

    def add(self, severity, operation, message, job=None):
        timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
        identity = f' #{job}' if job is not None else ''
        message = str(message)
        if len(message) > 16000:
            message = message[:16000] + ' [truncated]'
        self.entries.append(f'{timestamp} | {severity.upper()} | {operation}{identity} | {message}')

    def text(self):
        return '\n'.join(self.entries)

    def clear(self):
        self.entries.clear()
