import sys
import unittest
from unittest.mock import patch

from ninja3_assistant.device.native_touch import executable_path
from ninja3_assistant.paths import project_root, resource_root


class PackagePathsTest(unittest.TestCase):
    def test_source_root_contains_project_metadata_and_assets_manifest(self):
        root = project_root()
        self.assertTrue((root/'pyproject.toml').is_file())
        self.assertTrue((root/'assets/asset-manifest.json').is_file())
        self.assertTrue((root/'src/ninja3_assistant').is_dir())

    def test_frozen_resource_root_uses_bundle_without_changing_source_root(self):
        root = project_root()
        with patch.object(sys,'_MEIPASS',str(root/'build/example-bundle'),create=True):
            self.assertEqual(resource_root(),root/'build/example-bundle')
            self.assertEqual(project_root(),root)

    def test_native_source_executable_stays_in_project_build_directory(self):
        with patch.object(sys,'frozen',False,create=True):
            self.assertEqual(executable_path(),project_root()/'build/ninja-touch/NinjaTouch.exe')
