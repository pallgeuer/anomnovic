#!/usr/bin/env python3
"""Download and verify the public NOVIC, OpenCLIP, and SAM2 inference assets."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
MANIFEST = ROOT / 'docs' / 'THIRD_PARTY_SHA256SUMS.txt'
NOVIC_RELEASE = 'https://github.com/pallgeuer/novic/releases/download/v1.0.0'
CLIP_REVISION = '01b771ed0d1395ca5ffdd279897d665ebe00dfd2'
SAM_REVISION = '665f8e2ad61cf5f53d65644ff27c8ee525124610'
CLIP_REPO = 'apple/DFN5B-CLIP-ViT-H-14-378'
SAM_REPO = 'facebook/sam2.1-hiera-large'
CLIP_FILES = (
    'open_clip_pytorch_model.bin', 'open_clip_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt',
)
SAM_FILES = ('sam2.1_hiera_large.pt',)
NOVIC_ARCHIVES = {
    'ft0': ('dfn5bl_ft0_ye2_ovod_20240628_142131.zip',
            'ovod_20240628_142131/ovod_chunk0433_20240630_235415.train'),
    'ft2': ('dfn5bl_ft2_ya2_ovod_20240620_162925.zip',
            'ovod_20240620_162925/ovod_chunk0899_20240621_202727.train'),
}


def snapshot_path(repo: str, revision: str, filename: str) -> Path:
    return Path('assets/huggingface/hub') / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision / filename


def expected_paths() -> set[Path]:
    paths = set()
    for model, (archive, _) in NOVIC_ARCHIVES.items():
        paths.add(Path('assets/archives') / archive)
        paths.add(Path('assets/checkpoints') / f'novic_{model}.train')
    for filename in CLIP_FILES:
        paths.add(snapshot_path(CLIP_REPO, CLIP_REVISION, filename))
    for filename in SAM_FILES:
        paths.add(snapshot_path(SAM_REPO, SAM_REVISION, filename))
    return paths


def load_manifest() -> dict[Path, str]:
    checksums = {}
    for line in MANIFEST.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        checksum, filename = line.split(maxsplit=1)
        path = Path(filename)
        if len(checksum) != 64 or any(char not in '0123456789abcdef' for char in checksum):
            raise ValueError(f'Invalid SHA-256 in {MANIFEST}: {line}')
        if path in checksums:
            raise ValueError(f'Duplicate manifest path: {path}')
        checksums[path] = checksum
    if set(checksums) != expected_paths():
        missing = sorted(str(path) for path in expected_paths() - set(checksums))
        extra = sorted(str(path) for path in set(checksums) - expected_paths())
        raise ValueError(f'Unexpected manifest entries. Missing {missing}, extra {extra}')
    return checksums


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, checksum: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'Missing inference asset: {path}')
    actual = sha256(path)
    if actual != checksum:
        raise ValueError(f'SHA-256 mismatch for {path}. Expected {checksum}, got {actual}')
    print(f'Verified {path}')


def download(url: str, destination: Path, checksum: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url}')
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.download-', delete=False) as temp:
        temporary = Path(temp.name)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, temp)
            temp.flush()
            verify(temporary, checksum)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def extract(archive: Path, member: str, destination: Path, checksum: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'Extracting {member} from {archive}')
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.extract-', delete=False) as temp:
        temporary = Path(temp.name)
        try:
            with zipfile.ZipFile(archive) as package, package.open(member) as source:
                shutil.copyfileobj(source, temp)
            temp.flush()
            verify(temporary, checksum)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def cache_ref(repo: str) -> Path:
    return ASSETS / 'huggingface' / 'hub' / ('models--' + repo.replace('/', '--')) / 'refs' / 'main'


def write_ref(repo: str, revision: str) -> None:
    ref = cache_ref(repo)
    ref.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=ref.parent,
                                     prefix='.ref-', delete=False) as temp:
        temporary = Path(temp.name)
        try:
            temp.write(revision)
            temp.flush()
            os.replace(temporary, ref)
        finally:
            temporary.unlink(missing_ok=True)


def verify_refs() -> None:
    for repo, revision in ((CLIP_REPO, CLIP_REVISION), (SAM_REPO, SAM_REVISION)):
        ref = cache_ref(repo)
        if not ref.is_file() or ref.read_text(encoding='utf-8').strip() != revision:
            raise ValueError(f'Hugging Face main reference is missing or incorrect: {ref}')


def prepare(checksums: dict[Path, str]) -> None:
    for model, (filename, member) in NOVIC_ARCHIVES.items():
        archive = ROOT / 'assets' / 'archives' / filename
        archive_hash = checksums[archive.relative_to(ROOT)]
        if not archive.exists():
            download(f'{NOVIC_RELEASE}/{filename}', archive, archive_hash)
        verify(archive, archive_hash)
        checkpoint = ASSETS / 'checkpoints' / f'novic_{model}.train'
        checkpoint_hash = checksums[checkpoint.relative_to(ROOT)]
        if not checkpoint.exists():
            extract(archive, member, checkpoint, checkpoint_hash)
        verify(checkpoint, checkpoint_hash)

    from huggingface_hub import hf_hub_download

    for repo, revision, filenames in (
        (CLIP_REPO, CLIP_REVISION, CLIP_FILES),
        (SAM_REPO, SAM_REVISION, SAM_FILES),
    ):
        for filename in filenames:
            relative = snapshot_path(repo, revision, filename)
            path = ROOT / relative
            if not path.exists():
                print(f'Downloading {repo}/{filename} at {revision}')
                fetched = Path(hf_hub_download(
                    repo_id=repo, filename=filename, revision=revision,
                    cache_dir=ASSETS / 'huggingface' / 'hub'))
                if fetched.absolute() != path.absolute():
                    raise ValueError(f'Hugging Face returned an unexpected path: {fetched}')
            verify(path, checksums[relative])
        write_ref(repo, revision)
    verify_refs()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Verify all prepared assets without downloading')
    parser.add_argument('--verify-file', nargs=2, metavar=('MANIFEST_PATH', 'FILE'),
                        help='Verify one file against a named manifest entry')
    args = parser.parse_args()
    checksums = load_manifest()
    if args.verify_file:
        relative = Path(args.verify_file[0])
        if relative not in checksums:
            parser.error(f'No manifest entry for {relative}')
        verify(Path(args.verify_file[1]), checksums[relative])
    elif args.verify_only:
        for relative, checksum in checksums.items():
            verify(ROOT / relative, checksum)
        verify_refs()
    else:
        prepare(checksums)
    print('Inference assets verified')


if __name__ == '__main__':
    main()
