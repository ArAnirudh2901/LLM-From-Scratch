from __future__ import annotations
import argparse, json, os, sys

from export_web import sha256

# Checks every model file against model_manifest.json (written by export_web.py): run after copying or moving
# the project. Exit code 1 if anything is missing, truncated or changed.


def check(path: str, manifest: str = 'model_manifest.json') -> None:
    """Raise if `path` is listed in the manifest and its size or SHA-256 differs (no-op if unlisted)."""
    if not os.path.exists(manifest):
        return
    root = os.path.dirname(os.path.abspath(manifest))
    want = json.load(open(manifest)).get(os.path.relpath(os.path.abspath(path), root))
    if want and (os.path.getsize(path) != want['bytes'] or sha256(path) != want['sha256']):
        raise SystemExit(f'{path} is corrupted or was changed after export (see verify_model.py). Re-copy it.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='model_manifest.json')
    args = p.parse_args()
    root = os.path.dirname(os.path.abspath(args.manifest))
    bad = 0
    for rel, want in json.load(open(args.manifest)).items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f'MISSING   {rel}'); bad += 1
        elif os.path.getsize(path) != want['bytes']:
            print(f'TRUNCATED {rel} ({os.path.getsize(path)} of {want["bytes"]} bytes)'); bad += 1
        elif sha256(path) != want['sha256']:
            print(f'CORRUPTED {rel}'); bad += 1
        else:
            print(f'ok        {rel}')
    print('all model files intact' if not bad else f'{bad} file(s) damaged: re-copy them from the original')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
