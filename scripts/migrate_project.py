#!/usr/bin/env python3
"""Create/restore a private, checksummed Dress reliability-v2 transfer bundle.

Python standard library only. plan is read-only; pack never overwrites a bundle.
Stop all writers before pack. Only restore bundles you created and trust.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache'}
SKIP_SUFFIXES = {'.pyc', '.orig', '.rej', '.tmp', '.pem', '.key'}


def digest(handle):
    sha = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        sha.update(chunk)
    return sha.hexdigest()


def file_digest(path):
    with Path(path).open('rb') as handle:
        return digest(handle)


def allowed(path):
    return (not set(path.parts) & SKIP_PARTS and path.suffix not in SKIP_SUFFIXES
            and not path.name.startswith('.env')
            and not any(word in path.name.lower() for word in
                        ('credentials', 'client_secret', 'token.json', 'rclone.conf')))


def inventory(root, include_dataset=True):
    files = {}

    def add(source, destination):
        source, destination = Path(source), Path(destination)
        if not source.exists():
            raise FileNotFoundError(f'Required for migration: {source}')
        if source.is_dir():
            for folder, dirs, names in os.walk(source):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_PARTS)
                for name in sorted(names):
                    child = Path(folder) / name
                    rel = child.relative_to(source)
                    if allowed(rel):
                        files[(destination / rel).as_posix()] = child
        elif allowed(destination):
            files[destination.as_posix()] = source

    # Code snapshot complements GitHub; never include .git or OAuth credentials.
    names = subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']
    ).decode().split('\0')
    for name in names:
        path = Path(name)
        if not include_dataset and name.startswith('data/fashion-iq/'):
            continue
        if name and allowed(path) and (root / path).is_file():
            if (root / path).stat().st_size > 10 * 1024 * 1024:
                raise ValueError(f'Unexpected large code file: {path}; fix .gitignore first')
            add(root / path, path)
    for seed in (42, 7, 123):
        path = Path(f'checkpoints/supervisor_protocol_v2/point/dress/seed{seed}/checkpoint_best.pth')
        add(root / path, path)
    for name in ('checkpoints/reliability_v2', 'results/reliability_v2', 'ref/LAVIS'):
        add(root / name, name)
    if include_dataset:
        add(root / 'data/fashion-iq', 'data/fashion-iq')
    pretrained = 'ref/LAVIS/pretrained_weights/blip2_pretrained.pth'
    add(root / pretrained, pretrained)
    # Supervisor source is private on Drive, not published on GitHub.
    add(root / 'word&md&pdf/Thucnghiem_tiep_CIR.docx',
        'word&md&pdf/Thucnghiem_tiep_CIR.docx')
    # Include just the model caches this workflow uses, never a whole user cache.
    local_cache = root / '.migration/cache'
    torch_home = Path(os.environ.get('TORCH_HOME', Path.home() / '.cache/torch'))
    eva = torch_home / 'hub/checkpoints/eva_vit_g.pth'
    local_eva = local_cache / 'torch/hub/checkpoints/eva_vit_g.pth'
    add(local_eva if local_eva.exists() else eva,
        '.migration/cache/torch/hub/checkpoints/eva_vit_g.pth')
    hf_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
    hub = Path(os.environ.get('HUGGINGFACE_HUB_CACHE', hf_home / 'hub'))
    bert_name = 'models--bert-base-uncased'
    bert = local_cache / 'huggingface/hub' / bert_name
    add(bert if bert.exists() else hub / bert_name,
        Path('.migration/cache/huggingface/hub') / bert_name)
    # Preserve small historical summaries for comparison, not every old checkpoint.
    for parent in ('results/reliability_v1', 'results/supervisor_protocol_v2'):
        for path in (root / parent).rglob('*.json'):
            if path.stat().st_size <= 2 * 1024 * 1024:
                add(path, path.relative_to(root))
    return dict(sorted(files.items()))


def environment(python):
    code = """import importlib.metadata as m, json, sys
skip={'salesforce-lavis','pip'}
pins=sorted({d.metadata['Name']+'=='+d.version for d in m.distributions()
             if d.metadata.get('Name') and d.metadata['Name'].lower() not in skip})
print(json.dumps({'python':sys.version,'pins':pins}))
"""
    return json.loads(subprocess.check_output([str(python), '-c', code]))


def pack(root, output, python):
    files = inventory(root)
    env = environment(python)
    previous = root / '.migration/manifest.json'
    source_root = os.environ.get('HUG_SOURCE_ROOT') or (
        json.loads(previous.read_text())['source_root'] if previous.is_file() else str(root))
    manifest = {'schema_version': 1, 'source_root': source_root,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'git_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip(),
                'git_dirty': bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'])),
                'python': env['python'], 'files': []}
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or Path(str(output) + '.partial').exists():
        raise FileExistsError('Use a NEW bundle name; existing/partial bundles are not overwritten')
    partial = Path(str(output) + '.partial')
    # Uncompressed archive: checkpoints are expensive to gzip; Drive keeps original bytes.
    with partial.open('xb') as raw, tarfile.open(fileobj=raw, mode='w', dereference=True) as archive:
        for index, (name, path) in enumerate(files.items()):
            before = path.stat()
            sha = file_digest(path)
            archive.add(str(path), arcname=name, recursive=False)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'File changed during backup: {path}; stop writers and repack')
            manifest['files'].append({'path': name, 'size': after.st_size, 'sha256': sha})
            if index % 1000 == 0 or after.st_size > 100000000:
                print(f'Packed {index + 1}/{len(files)}: {name}', flush=True)
        pins = ('\n'.join(env['pins']) + '\n').encode()
        name = '.migration/environment.txt'
        entry = tarfile.TarInfo(name); entry.size = len(pins)
        archive.addfile(entry, io.BytesIO(pins))
        manifest['files'].append({'path': name, 'size': len(pins),
                                  'sha256': hashlib.sha256(pins).hexdigest()})
        data = json.dumps(manifest, indent=2).encode()
        entry = tarfile.TarInfo('.migration/manifest.json'); entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    partial.rename(output)
    print(f'Bundle complete: {output} ({output.stat().st_size / 2**30:.2f} GiB)', flush=True)


def safe_target(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError(f'Unsafe archive path: {name}')
    target = root.joinpath(*relative.parts)
    # Reject symlinks, including ones pointing within root: no accidental external writes.
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError(f'Symlink in restore target: {target}')
    target.resolve().relative_to(root.resolve())
    return target


def restore(bundle, root):
    with tarfile.open(bundle, 'r:') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError('Bundle must contain unique regular files only')
        for member in members:
            safe_target(root, member.name)
        meta = archive.getmember('.migration/manifest.json')
        if meta.size > 64 * 1024 * 1024:
            raise ValueError('Oversized migration metadata')
        manifest = json.load(archive.extractfile(meta))
        if manifest.get('schema_version') != 1:
            raise ValueError('Unsupported migration schema')
        expected = {row['path']: row for row in manifest['files']}
        if len(expected) != len(manifest['files']):
            raise ValueError('Duplicate manifest entries')
        if set(names) != set(expected) | {meta.name}:
            raise ValueError('Archive and manifest disagree')
        # Check all existing files BEFORE writing anything. Never overwrite user work.
        for member in members:
            target = safe_target(root, member.name)
            if member.name in expected and member.size != expected[member.name]['size']:
                raise ValueError(f'Size mismatch: {member.name}')
            if target.exists():
                sha = digest(archive.extractfile(member))
                if not target.is_file() or file_digest(target) != sha:
                    raise FileExistsError(f'Conflicting file: {target}. Use a fresh matching clone.')
        for index, member in enumerate(members):
            target = safe_target(root, member.name)
            if target.exists():
                if member.name in expected and file_digest(target) != expected[member.name]['sha256']:
                    raise ValueError(f'Checksum mismatch: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write exclusively to temporary file; interrupted restores can be retried.
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.restore-', delete=False) as out:
                temporary = Path(out.name)
                sha = hashlib.sha256()
                handle = archive.extractfile(member)
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    out.write(chunk); sha.update(chunk)
            if member.name in expected and sha.hexdigest() != expected[member.name]['sha256']:
                raise ValueError(f'Checksum mismatch: {member.name}; partial retained at {temporary}')
            temporary.chmod(0o755 if member.mode & 0o111 else 0o644)
            os.link(str(temporary), str(target))  # Atomic no-clobber publication.
            temporary.unlink()
            if index % 1000 == 0:
                print(f'Restored {index + 1}/{len(members)}', flush=True)
    print('Restore complete. Original checkpoint/manifest bytes preserved.')


def verify(root):
    manifest = json.loads((root / '.migration/manifest.json').read_text())
    for row in manifest['files']:
        path = safe_target(root, row['path'])
        if path.stat().st_size != row['size'] or file_digest(path) != row['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
    print(f"Verified {len(manifest['files'])} files (SHA-256).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'pack', 'restore', 'verify'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--python', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.action == 'plan':
        files = inventory(root)
        sizes = {}
        for name, path in files.items():
            group = name.split('/')[0]
            sizes[group] = sizes.get(group, 0) + path.stat().st_size
        print(json.dumps({'files': len(files), 'GiB': {k: round(v / 2**30, 3) for k, v in sizes.items()},
                          'total_GiB': round(sum(sizes.values()) / 2**30, 3)}, indent=2))
    elif args.action == 'verify':
        verify(root)
    elif args.bundle is None:
        parser.error('--bundle is required')
    elif args.action == 'restore':
        restore(args.bundle, root)
    else:
        if args.python is not None and not args.python.is_file():
            parser.error(f'--python does not exist: {args.python}')
        python = args.python or root / '.venv/bin/python'
        if not python.exists():
            python = root / 'ref/LAVIS/.venv/bin/python'
        pack(root, args.bundle, python)


if __name__ == '__main__':
    main()
