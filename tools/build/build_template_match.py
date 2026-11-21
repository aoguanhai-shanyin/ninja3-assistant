"""Generate the project-wide WinML template-matching graph.

The graph holds every branch structure but no template pixels: those are
external initializers injected by the Worker at session creation. Re-run this
whenever the catalog gains or loses a template.
"""
from pathlib import Path
from tempfile import NamedTemporaryFile

from ninja3_assistant.paths import project_root
from ninja3_assistant.vision.template_match import CATALOG, MODEL_NAME, build_model


def main():
    assets = project_root() / 'assets'
    path = assets / MODEL_NAME
    model = build_model()
    file = NamedTemporaryFile(dir=assets, prefix=path.name + '.', delete=False)
    temporary = Path(file.name)
    try:
        with file:
            file.write(model.SerializeToString())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Generated {path.name} for {len(CATALOG)} templates')


if __name__ == '__main__':
    main()
