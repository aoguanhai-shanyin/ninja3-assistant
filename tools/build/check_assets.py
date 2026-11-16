"""Check local-only runtime assets without connecting to the emulator."""
from ninja3_assistant.paths import project_root
import json
import sys


def main():
    root = project_root()/'assets'
    manifest = json.loads((root/'asset-manifest.json').read_text(encoding='utf-8'))
    missing = [entry['path'] for entry in manifest['files'] if not (root/entry['path']).is_file()]
    if missing:
        print('Missing local recognition assets: '+str(len(missing)))
        for name in missing:
            print('  '+name)
        print('See assets/README.md. Public core tests do not require these assets.')
        return 1
    print('Local runtime assets present: '+str(len(manifest['files'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
