import subprocess
import json
import sys
from pathlib import Path
import zipfile

import pytest

from scripts import zip_project


def fixture_bundle(tmp_path, monkeypatch):
    root = tmp_path / 'old'; root.mkdir()
    subprocess.run(['git', 'init', str(root)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c',
                    'user.email=test@example.com', 'commit', '--allow-empty', '-m', 'fixture'],
                   check=True, capture_output=True)
    weight = root / 'point.pth'; weight.write_bytes(b'fixed weights')
    monkeypatch.setattr(zip_project, 'select_files', lambda root, profile, **kwargs: {'point.pth': weight})
    monkeypatch.setattr(zip_project, 'environment', lambda python: {'python': '3.8', 'pins': ['numpy==1.24.4']})
    bundle = tmp_path / 'backup.zip'
    zip_project.pack(root, bundle, 'essential', Path(sys.executable))
    return root, bundle


def test_zip_roundtrip_and_zip64(tmp_path, monkeypatch):
    # Exercise the >4 GiB code path using a tiny threshold, without huge fixtures.
    monkeypatch.setattr(zipfile, 'ZIP64_LIMIT', 8)
    source, bundle = fixture_bundle(tmp_path, monkeypatch)
    target = tmp_path / 'new'
    zip_project.restore(bundle, target); zip_project.verify(target)
    zip_project.restore(bundle, target)
    assert (target / 'point.pth').read_bytes() == (source / 'point.pth').read_bytes()
    with zipfile.ZipFile(bundle) as archive:
        assert archive.getinfo('point.pth').extract_version >= 45
        meta = json.loads(archive.read('.migration/manifest.json'))
        assert meta['dataset_included'] is False
        assert meta['external_dependencies'][0]['path'] == 'data/fashion-iq'
    with pytest.raises(FileExistsError):
        zip_project.pack(source, bundle, 'essential', Path(sys.executable))


def test_zip_refuses_conflicting_files(tmp_path, monkeypatch):
    _, bundle = fixture_bundle(tmp_path, monkeypatch)
    target = tmp_path / 'new'; target.mkdir()
    (target / 'point.pth').write_bytes(b'user weights')
    with pytest.raises(FileExistsError):
        zip_project.restore(bundle, target)
    assert not (target / '.migration').exists()


def test_zip_rejects_traversal(tmp_path):
    bundle = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bundle, 'w') as archive:
        archive.writestr('../escape', b'bad')
    with pytest.raises(ValueError):
        zip_project.restore(bundle, tmp_path / 'new')


def test_profiles_keep_resume_for_unfinished_runs(tmp_path, monkeypatch):
    files = {}
    for seed in (42, 7):
        parent = tmp_path / f'checkpoints/reliability_v2/dress/seed{seed}'
        parent.mkdir(parents=True)
        for name in ['checkpoint_best.pth', 'checkpoint_last.pth']:
            path = parent / name; path.touch()
            files[path.relative_to(tmp_path).as_posix()] = path
        if seed == 42:
            final = parent / 'checkpoint_final.pth'; final.touch()
            files[final.relative_to(tmp_path).as_posix()] = final
    monkeypatch.setattr(zip_project, 'inventory', lambda root, **kwargs: files)
    essential = zip_project.select_files(tmp_path, 'essential')
    assert 'checkpoints/reliability_v2/dress/seed42/checkpoint_last.pth' not in essential
    assert 'checkpoints/reliability_v2/dress/seed7/checkpoint_last.pth' in essential
    assert 'checkpoints/reliability_v2/dress/seed42/checkpoint_final.pth' in essential
    assert zip_project.select_files(tmp_path, 'current-full') == files


@pytest.mark.parametrize('profile', ['essential', 'current-full'])
def test_zip_dataset_opt_in_preserves_loader(tmp_path, monkeypatch, profile):
    loader = 'data/dataset.py'; image = 'data/fashion-iq/images/example.png'
    def fake_inventory(root, include_dataset=True):
        result = {loader: root / loader}
        if include_dataset:
            result[image] = root / image
        return result
    monkeypatch.setattr(zip_project, 'inventory', fake_inventory)
    assert set(zip_project.select_files(tmp_path, profile)) == {loader}
    assert set(zip_project.select_files(tmp_path, profile, include_dataset=True)) == {loader, image}


def test_inventory_without_dataset_does_not_require_dataset_directory(tmp_path, monkeypatch):
    from scripts.migrate_project import inventory
    required = [
        'data/dataset.py', 'ref/LAVIS/pretrained_weights/blip2_pretrained.pth',
        'word&md&pdf/Thucnghiem_tiep_CIR.docx',
        '.migration/cache/torch/hub/checkpoints/eva_vit_g.pth',
        '.migration/cache/huggingface/hub/models--bert-base-uncased/config.json',
    ] + [f'checkpoints/supervisor_protocol_v2/point/dress/seed{s}/checkpoint_best.pth'
         for s in (42, 7, 123)]
    for name in required:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.touch()
    for name in ['checkpoints/reliability_v2', 'results/reliability_v2']:
        (tmp_path / name).mkdir(parents=True)
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: b'data/dataset.py\0')
    files = inventory(tmp_path, include_dataset=False)
    assert 'data/dataset.py' in files
    assert not any(name.startswith('data/fashion-iq/') for name in files)
    # The opt-in still fails fast if the dataset really is missing.
    with pytest.raises(FileNotFoundError):
        inventory(tmp_path, include_dataset=True)
