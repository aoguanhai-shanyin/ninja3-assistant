import subprocess
import sys
import tempfile
import unittest

from ninja3_assistant.device.input_owner import input_owner


TRY_ACQUIRE = '''
import sys
from ninja3_assistant.device.input_owner import InputBusy, input_owner

try:
    with input_owner(sys.argv[1], int(sys.argv[2])):
        print('acquired')
except InputBusy:
    print('busy')
'''


@unittest.skipUnless(sys.platform == 'win32', 'requires the Windows named mutex')
class InputOwnerTest(unittest.TestCase):
    def test_separate_processes_exclude_the_same_instance_and_allow_another_index(self):
        def attempt(folder, index):
            result = subprocess.run([sys.executable, '-c', TRY_ACQUIRE, folder, str(index)],
                                    capture_output=True, encoding='utf-8', timeout=5,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout.strip()

        with tempfile.TemporaryDirectory() as folder:
            with input_owner(folder, 1):
                self.assertEqual(attempt(folder, 1), 'busy')
                self.assertEqual(attempt(folder, 2), 'acquired')
            self.assertEqual(attempt(folder, 1), 'acquired')
