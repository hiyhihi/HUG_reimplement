import hashlib
import io
import json
from pathlib import Path
import tarfile
import subprocess
import sys

import pytest

from scripts.migrate_project import allowed, restore, safe_target, verify
from utils.artifact_paths import resolve_artifact_path
from scripts import migrate_project


def make_bundle(path, contents):
    rows = [{'path': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            for name, data in contents.items()]
    meta = json.dumps({'schema_version': 1, 'source_root': '/old/project', 'files': rows}).encode()
    with tarfile.open(path, 'w') as archive:
        for name, data in list(contents.items()) + [('.migration/manifest.json', meta)]:
            entry = tarfile.TarInfo(name); entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))


def test_restore_roundtrip_and_idempotence(tmp_path):
    bundle = tmp_path / 'bundle.tar'; root = tmp_path / 'new'
    make_bundle(bundle, {'checkpoints/point.pth': b'unchanged weights', 'README.md': b'code'})
    restore(bundle, root); verify(root)
    restore(bundle, root); verify(root)
    assert resolve_artifact_path('/old/project/checkpoints/point.pth', root) == root / 'checkpoints/point.pth'


def test_restore_conflict_does_not_write_other_files(tmp_path):
    bundle = tmp_path / 'bundle.tar'; root = tmp_path / 'new'; root.mkdir()
    (root / 'README.md').write_bytes(b'user edits')
    make_bundle(bundle, {'new.txt': b'new', 'README.md': b'old'})
    with pytest.raises(FileExistsError):
        restore(bundle, root)
    assert not (root / 'new.txt').exists()
    assert (root / 'README.md').read_bytes() == b'user edits'


@pytest.mark.parametrize('name', ['../escape', '/absolute/path', 'a/../../escape'])
def test_reject_path_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        safe_target(tmp_path, name)


def test_reject_symlink_target(tmp_path):
    root = tmp_path / 'root'; root.mkdir()
    (root / 'link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_target(root, 'link/file')


def test_artifact_paths_do_not_guess_backbone(tmp_path, monkeypatch):
    monkeypatch.setenv('HUG_SOURCE_ROOT', '/old/project')
    target = tmp_path / 'checkpoints/seed7/best.pth'
    target.parent.mkdir(parents=True); target.touch()
    assert resolve_artifact_path('/old/project/checkpoints/seed7/best.pth', tmp_path) == target
    with pytest.raises(FileNotFoundError):
        resolve_artifact_path('/old/project/checkpoints/seed42/best.pth', tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_artifact_path('/different/project/checkpoints/seed7/best.pth', tmp_path)


@pytest.mark.parametrize('name', ['.env', '.env.local', 'rclone.conf', 'client_secret.json',
                                'ref/LAVIS/.venv/lib/file', 'x.orig', '.git/config'])
def test_no_secrets_or_virtualenv_in_bundle(name):
    assert not allowed(Path(name))


def test_verify_detects_corruption(tmp_path):
    bundle = tmp_path / 'bundle.tar'; root = tmp_path / 'new'
    make_bundle(bundle, {'weights.pth': b'123'})
    restore(bundle, root)
    (root / 'weights.pth').write_bytes(b'456')
    with pytest.raises(ValueError):
        verify(root)


def test_pack_restore_with_exact_bytes(tmp_path, monkeypatch):
    source = tmp_path / 'source'; source.mkdir()
    subprocess.run(['git', 'init', str(source)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test', '-c',
                    'user.email=test@example.com', 'commit', '--allow-empty', '-m', 'fixture'],
                   check=True, capture_output=True)
    weights = source / 'weights.pth'; weights.write_bytes(b'weights and optimizer')
    monkeypatch.setattr(migrate_project, 'inventory', lambda root: {'weights.pth': weights})
    monkeypatch.setattr(migrate_project, 'environment', lambda python: {'python': '3.8', 'pins': ['numpy==1.24.4']})
    bundle = tmp_path / 'bundle.tar'
    migrate_project.pack(source, bundle, Path(sys.executable))
    root = tmp_path / 'new'; restore(bundle, root); verify(root)
    assert weights.read_bytes() == (root / 'weights.pth').read_bytes()
    with pytest.raises(FileExistsError):
        migrate_project.pack(source, bundle, Path(sys.executable))


@pytest.mark.parametrize('email,success', [('wrong@example.com', False), ('huyphan1610@gmail.com', True)])
def test_upload_checks_account_before_copy(tmp_path, monkeypatch, email, success):
    import os
    fake = tmp_path / 'rclone'
    marker = tmp_path / 'calls'
    fake.write_text('#!/usr/bin/env python3\nimport sys,json\n'
                    'if sys.argv[1:3] == ["config","userinfo"]:\n'
                    f' print(json.dumps({{"email": {email!r}}}))\n'
                    'else:\n'
                    f' with open({str(marker)!r},"a") as f: f.write(sys.argv[1]+"\\n")\n')
    fake.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    bundle = tmp_path / 'bundle.tar'; bundle.touch()
    script = Path(__file__).resolve().parents[1] / 'scripts/upload_project_drive.sh'
    result = subprocess.run(['bash', str(script), str(bundle)], capture_output=True)
    assert (result.returncode == 0) == success
    assert marker.exists() == success
    if success:
        assert marker.read_text().splitlines() == ['copyto', 'check']
