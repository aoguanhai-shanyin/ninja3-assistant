"""Lint Python blobs in the Git index without reading unstaged file contents."""

import os
import subprocess
import sys


def main() -> int:
    changed = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z", "--", "src", "tests", "tools"]
    )
    returncode = 0
    for filename in changed.split(b"\0"):
        if not filename.endswith(b".py"):
            continue
        path = os.fsdecode(filename)
        contents = subprocess.check_output(["git", "show", f":{path}"])
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--quiet", "--force-exclude", "--stdin-filename", path, "-"],
            input=contents,
        )
        if result.returncode not in (0, 1):
            return result.returncode
        returncode = result.returncode or returncode
    return returncode


if __name__ == "__main__":
    sys.exit(main())
