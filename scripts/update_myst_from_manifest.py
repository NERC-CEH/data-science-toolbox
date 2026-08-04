#!/usr/bin/env python3
"""Update myst.yml TOC entries from a notebooks.yml manifest.

Usage:
  python scripts/update_myst_from_manifest.py --manifest notebooks.yml --myst myst.yml [--dry-run]

The script maps each manifest entry's 'path' to methods/<repo_dir>/<path> and replaces
any 'file' entries in myst.yml whose basename matches the notebook filename.

This script makes a best-effort update; review myst.yml after running.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
except Exception:
    print("Missing dependency: pyyaml. Install with `pip install pyyaml`.")
    raise

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def repo_dir_from_url(url: str) -> str:
    name = url.rstrip("/ ")
    if name.endswith('.git'):
        name = name[:-4]
    return Path(name).name


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    data = yaml.safe_load(manifest_path.read_text(encoding='utf-8')) or {}
    return data


def build_name_to_path_map(manifest: Dict[str, Any], dest_base: Path = Path('methods')) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in manifest.get('notebooks', []) or []:
        url = entry.get('url')
        subpath = entry.get('path') or entry.get('subpath') or entry.get('notebook')
        if not url or not subpath:
            logging.debug('Skipping manifest entry missing url or path: %s', entry)
            continue
        repo_dir = repo_dir_from_url(url)
        new_path = (dest_base / repo_dir / Path(subpath)).as_posix()
        name = Path(subpath).name
        mapping[name] = new_path
    return mapping


def load_myst(myst_path: Path) -> Dict[str, Any]:
    if not myst_path.exists():
        raise SystemExit(f"myst.yml not found: {myst_path}")
    data = yaml.safe_load(myst_path.read_text(encoding='utf-8')) or {}
    return data


def save_myst(data: Dict[str, Any], myst_path: Path):
    myst_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def find_toc_root(data: Dict[str, Any]):
    # prefer project.toc
    project = data.get('project')
    if isinstance(project, dict) and 'toc' in project:
        return project, project['toc'], 'project'
    # fallback to top-level toc
    if 'toc' in data:
        return data, data['toc'], 'root'
    return None, None, None


def update_toc_files(toc_node, name_to_path: Dict[str, str]) -> int:
    """Recursively walk toc_node (list/dict) and update 'file' entries. Returns count changed."""
    changed = 0

    def walk(node):
        nonlocal changed
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            # update file key
            f = node.get('file')
            if isinstance(f, str):
                fname = Path(f).name
                if fname in name_to_path:
                    node['file'] = name_to_path[fname]
                    changed += 1
            # recurse
            for v in list(node.values()):
                if isinstance(v, (list, dict)):
                    walk(v)

    walk(toc_node)
    return changed


def main(argv=None):
    p = argparse.ArgumentParser(description='Update myst.yml TOC from notebooks.yml manifest')
    p.add_argument('--manifest', '-m', type=Path, default=Path('notebooks.yml'))
    p.add_argument('--myst', type=Path, default=Path('myst.yml'))
    p.add_argument('--dest', type=Path, default=Path('methods'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verbose', '-v', action='store_true')
    args = p.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    manifest = load_manifest(args.manifest)
    name_to_path = build_name_to_path_map(manifest, dest_base=args.dest)
    if not name_to_path:
        logging.info('No notebook entries found in manifest; nothing to do')
        return

    logging.debug('Mapping of notebook basenames to new paths:\n%s', '\n'.join(f'{k} -> {v}' for k, v in name_to_path.items()))

    myst_data = load_myst(args.myst)
    parent, toc, where = find_toc_root(myst_data)
    if toc is None:
        raise SystemExit('Could not find a TOC (project.toc or top-level toc) in myst.yml')

    changed = update_toc_files(toc, name_to_path)
    if changed:
        if args.dry_run:
            logging.info('DRY-RUN: Would update %d entries in %s', changed, args.myst)
        else:
            # ensure we write back into the same structure (project or root)
            if where == 'project':
                parent['toc'] = toc
                myst_data['project'] = parent
            else:
                myst_data['toc'] = toc
            save_myst(myst_data, args.myst)
            logging.info('Updated %d entries in %s', changed, args.myst)
    else:
        logging.info('No toc entries required updating')


if __name__ == '__main__':
    main()
