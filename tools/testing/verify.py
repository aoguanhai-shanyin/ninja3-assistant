"""Build and run the asset-free Windows verification checks."""
import subprocess
import sys

from ninja3_assistant.paths import project_root


def main():
    if sys.platform != 'win32':
        sys.exit('verify requires Windows. Run python -m tools.testing.run_portable_tests for the Python-only suite.')

    root = project_root()
    steps = (
        ('Compile NinjaTouch', ['cmd.exe', '/d', '/c', r'native_touch\build.cmd']),
        ('Portable regression', [sys.executable, '-m', 'tools.testing.run_portable_tests']),
        ('Native mock regression', [
            sys.executable, '-m', 'unittest',
            'tests.device.test_native_touch_protocol', 'tests.device.test_native_plans',
        ]),
    )
    for name, command in steps:
        print(f'== {name} ==', flush=True)
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
