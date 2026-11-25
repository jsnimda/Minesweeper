import codecs
import errno
import os
import pty
import subprocess
import sys

from collections.abc import Sequence


def run_pty(
    args: Sequence[str],
    *,
    check: bool = False,
) -> subprocess.CompletedProcess:
    master, slave = pty.openpty()

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""

    try:
        with subprocess.Popen(args, stderr=slave) as process:
            os.close(slave)
            slave = None

            while True:
                try:
                    data = os.read(master, 4096)
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise

                if not data:
                    break

                buffer += decoder.decode(data)

                while True:
                    end = next(
                        (
                            i for i, char in enumerate(buffer)
                            if char in "\r\n"
                        ),
                        -1,
                    )

                    if end == -1:
                        break

                    sys.stderr.write(buffer[:end + 1])
                    sys.stderr.flush()

                    buffer = buffer[end + 1:]

            buffer += decoder.decode(b"", final=True)

            if buffer:
                sys.stderr.write(buffer)
                sys.stderr.flush()

            returncode = process.wait()

    finally:
        os.close(master)

        if slave is not None:
            os.close(slave)

    result = subprocess.CompletedProcess(args, returncode)

    if check:
        result.check_returncode()

    return result