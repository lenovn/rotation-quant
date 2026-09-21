"""Download and unpack Phase6 migration files. Does not install or run experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

BASE = 'https://github.com/lenovn/rotation-quant/releases/download/phase6/'
ROOT = Path(__file__).resolve().parents[1]


def download(name, size, directory):
    target = directory / name
    if target.exists() and target.stat().st_size == size:
        print('Already downloaded:', name, flush=True)
        return target
    partial = target.with_name(target.name + '.partial')
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == size:
                partial.replace(target)
                return target
            if offset > size:
                partial.unlink()
                offset = 0
            request = Request(BASE + name, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urlopen(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected response offset')
                with partial.open('ab' if append else 'wb') as output:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            if partial.stat().st_size != size:
                raise ValueError('Incomplete download')
            partial.replace(target)
            print('Downloaded:', name, flush=True)
            return target
        except Exception:
            if attempt == 3:
                raise
            print('Retrying:', name, flush=True)
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-dir', type=Path, default=ROOT.parent / 'phase6-downloads')
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    args.download_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urlopen(BASE + 'phase6-assets.json', timeout=120) as response:
                manifest = json.load(response)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5)
    entries = manifest['parts']
    expected = [f'phase6-data.tar.gz.part-{i:03d}' for i in range(len(entries))]
    if not entries or [entry['name'] for entry in entries] != expected:
        raise ValueError('Archive parts must be nonempty, unique and consecutively ordered')
    if any(not isinstance(entry['bytes'], int) or entry['bytes'] <= 0 for entry in entries):
        raise ValueError('Invalid archive part size')
    paths = []
    for entry in manifest['parts']:
        name = entry['name']
        if Path(name).name != name or not name.startswith('phase6-data.tar.gz.part-'):
            raise ValueError('Invalid archive part name')
        paths.append(download(name, entry['bytes'], args.download_dir))
    if args.download_only:
        return
    print('Restoring archived files into', ROOT, flush=True)
    # Restore into a fresh clone; existing experiment files would be overwritten.
    process = subprocess.Popen(['tar', '-xzf', '-', '-C', str(ROOT)], stdin=subprocess.PIPE)
    try:
        for path in paths:
            with path.open('rb') as source:
                while True:
                    block = source.read(8 * 1024 * 1024)
                    if not block:
                        break
                    process.stdin.write(block)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Archive extraction failed')
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    print('Files restored. No environment installation, training or evaluation was started.', flush=True)
    print('Read phase6/HANDOFF.md before continuing.', flush=True)


if __name__ == '__main__':
    main()
