#!/usr/bin/env python3
"""Portable ZIP64 backup for continuing CIR, not a ZIP of the entire workspace.

python3 scripts/zip_project.py plan
python3 scripts/zip_project.py pack
python3 scripts/zip_project.py restore --input /path/cir-essential.zip --root /new/project
python3 scripts/zip_project.py verify --root /new/project
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from migrate_project import ROOT, environment, file_digest, inventory, safe_target, verify


def select_files(root, profile, include_dataset=False):
    files = inventory(root, include_dataset=include_dataset)
    if profile == 'current-full':
        return files
    selected = {}
    for name, path in files.items():
        # Historical checkpoint experiments are not needed for the next two seeds.
        if name.startswith(('results/reliability_v1/', 'results/supervisor_protocol_v2/')):
            continue
        # A finished run needs best for evaluation and final as completion evidence.
        # Keep last (optimizer) for ALL unfinished runs. Never fabricate final markers.
        if (name.startswith('checkpoints/reliability_v2/')
                and path.name == 'checkpoint_last.pth'
                and (path.parent / 'checkpoint_final.pth').is_file()):
            continue
        selected[name] = path
    return selected


def pack(root, output, profile, python, include_dataset=False):
    files = select_files(root, profile, include_dataset=include_dataset)
    output = output.resolve()
    if output.suffix != '.zip':
        raise ValueError('Output must end in .zip')
    partial = Path(str(output) + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Use a new ZIP filename; no overwrites of backups/partial files')
    env = environment(python)
    previous = root / '.migration/manifest.json'
    source_root = os.environ.get('HUG_SOURCE_ROOT') or (
        json.loads(previous.read_text())['source_root'] if previous.is_file() else str(root))
    manifest = {
        'schema_version': 1, 'archive_format': 'zip64', 'profile': profile,
        'dataset_included': include_dataset,
        'external_dependencies': [] if include_dataset else [{
            'path': 'data/fashion-iq',
            'required_subdirectories': ['images', 'captions', 'image_splits'],
            'note': 'Restore the separately backed-up dataset before setup/preflight/training.',
        }],
        'source_root': source_root, 'python': env['python'],
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip(),
        'git_dirty': bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'])),
        'files': [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for index, (name, path) in enumerate(files.items()):
            before = path.stat()
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | stat.S_IMODE(before.st_mode)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 1
            sha = hashlib.sha256()
            with path.open('rb') as source, archive.open(info, 'w', force_zip64=True) as target:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    target.write(chunk); sha.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'File changed during backup: {path}; stop writers and retry with a new ZIP')
            manifest['files'].append({'path': name, 'size': after.st_size, 'sha256': sha.hexdigest()})
            if index % 1000 == 0 or after.st_size > 100000000:
                print(f'Zipped {index + 1}/{len(files)}: {name}', flush=True)
        pins = ('\n'.join(env['pins']) + '\n').encode()
        name = '.migration/environment.txt'
        archive.writestr(name, pins)
        manifest['files'].append({'path': name, 'size': len(pins), 'sha256': hashlib.sha256(pins).hexdigest()})
        archive.writestr('.migration/manifest.json', json.dumps(manifest, indent=2))
    # No-clobber publish, also when another process creates the destination meanwhile.
    os.link(str(partial), str(output)); partial.unlink()
    print(f'Complete: {output} ({output.stat().st_size / 2**30:.2f} GiB)', flush=True)


def restore(bundle, root):
    root = root.resolve()
    with zipfile.ZipFile(bundle) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP entries')
        for member in members:
            safe_target(root, member.filename)
            mode = member.external_attr >> 16
            if member.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError('Only regular files are allowed')
        meta = archive.getinfo('.migration/manifest.json')
        if meta.file_size > 64 * 1024 * 1024:
            raise ValueError('Oversized manifest')
        manifest = json.loads(archive.read(meta))
        if manifest.get('schema_version') != 1 or manifest.get('archive_format') != 'zip64':
            raise ValueError('Not a CIR ZIP bundle')
        expected = {r['path']: r for r in manifest['files']}
        if len(expected) != len(manifest['files']) or set(names) != set(expected) | {meta.filename}:
            raise ValueError('Manifest and ZIP file list disagree')
        # Validate all destination conflicts before writing any file.
        for member in members:
            target = safe_target(root, member.filename)
            if member.filename in expected:
                row = expected[member.filename]
                if row['size'] != member.file_size:
                    raise ValueError(f'Unexpected file size: {member.filename}')
                expected_sha = row['sha256']
            else:
                expected_sha = hashlib.sha256(archive.read(member)).hexdigest()
            if target.exists() and (not target.is_file() or file_digest(target) != expected_sha):
                raise FileExistsError(f'Conflict: {target}; restore to a fresh directory, never overwrite')
        for index, member in enumerate(members):
            target = safe_target(root, member.filename)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.restore-', delete=False) as out:
                temporary = Path(out.name)
                sha = hashlib.sha256()
                with archive.open(member) as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        out.write(chunk); sha.update(chunk)
            if member.filename in expected and sha.hexdigest() != expected[member.filename]['sha256']:
                raise ValueError(f'Checksum mismatch: {member.filename}; partial retained at {temporary}')
            temporary.chmod(0o755 if (member.external_attr >> 16) & 0o111 else 0o644)
            os.link(str(temporary), str(target)); temporary.unlink()
            if index % 1000 == 0:
                print(f'Restored {index + 1}/{len(members)}', flush=True)
    print('Restore complete. Run verify, then setup_new_machine.sh.')
    if manifest.get('dataset_included') is False:
        print('Dataset NOT bundled: restore images/, captions/, image_splits/ into data/fashion-iq first.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'pack', 'restore', 'verify'])
    parser.add_argument('--profile', choices=['essential', 'current-full'], default='essential')
    parser.add_argument('--include-dataset', action='store_true',
                        help='Opt in to Fashion-IQ assets; omitted by default (dataset backed up separately).')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--python', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    output = args.output or Path('migration_exports') / (
        f"cir-{args.profile}{'' if args.include_dataset else '-no-data'}.zip")
    if not output.is_absolute():
        output = root / output
    if args.action == 'plan':
        files = select_files(root, args.profile, include_dataset=args.include_dataset)
        size = sum(path.stat().st_size for path in files.values()) / 2**30
        print(f'Profile: {args.profile}; {len(files)} files; {size:.2f} GiB BEFORE compression')
        print('Includes code (including data/*.py), Point Dress 42/7/123, reliability_v2, LAVIS, BERT/EVA caches.')
        print(f"Fashion-IQ dataset: {'INCLUDED' if args.include_dataset else 'EXCLUDED; restore separately into data/fashion-iq/'}")
        print('Excludes .git, .venv, credentials, unrelated caches and historical model checkpoints.')
        print(f'Output ZIP: {output.resolve()}')
    elif args.action == 'verify':
        verify(root)
        meta = json.loads((root / '.migration/manifest.json').read_text())
        if meta.get('dataset_included') is False:
            print('ZIP checksums passed; separately stored Fashion-IQ is NOT checked by verify.')
    elif args.action == 'restore':
        if args.input is None:
            parser.error('restore needs --input /path/backup.zip')
        restore(args.input, root)
    else:
        python = args.python or root / '.venv/bin/python'
        if not python.exists() and args.python is None:
            python = root / 'ref/LAVIS/.venv/bin/python'
        if not python.is_file():
            parser.error('Use --python pointing to the CURRENT TRAINING Python, not a random environment')
        pack(root, output, args.profile, python, include_dataset=args.include_dataset)


if __name__ == '__main__':
    main()
