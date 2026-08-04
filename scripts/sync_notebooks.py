#!/usr/bin/env python3
"""Simplified test script: download notebooks and listed file assets using raw.githubusercontent URLs only.

Behavior:
- Reads notebooks.yml and for each entry downloads:
  - the notebook file at 'path' via raw.githubusercontent (owner/repo + branch + path)
  - each asset listed in 'assets' if it's a file (assets ending with '/' are skipped)
- Dry-run by default; use --execute to actually write files. Destination not created in dry-run.
- Optional GITHUB_TOKEN environment variable is respected for higher rate limits.
"""
from __future__ import annotations
import argparse
import logging
import os
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
except Exception:
    print("Missing dependency: pyyaml. Install with `pip install pyyaml`.")
    raise

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data or "notebooks" not in data:
        raise SystemExit(f"Manifest {path} does not contain a 'notebooks' key")
    return data


def parse_github_owner_repo(url: str) -> Optional[str]:
    if url.startswith("https://github.com/"):
        part = url[len("https://github.com/"):]
        if part.endswith('.git'):
            part = part[:-4]
        return part.strip("/ ")
    if url.startswith("git@github.com:"):
        part = url[len("git@github.com:"):]
        if part.endswith('.git'):
            part = part[:-4]
        return part.strip("/ ")
    return None


def _get_request_headers() -> Dict[str, str]:
    headers = {"User-Agent": "sync-notebooks-test/1.0"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def github_api_list_files(owner_repo: str, path: str, ref: str):
    """Recursively list files under path using GitHub REST API and return dicts with 'path' and 'download_url'."""
    results = []
    base_api = f"https://api.github.com/repos/{owner_repo}/contents"
    def _list(p: str):
        url = f"{base_api}/{p.lstrip('/')}?ref={ref}"
        logging.debug("GitHub API request: %s", url)
        last_exc = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers=_get_request_headers())
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"GitHub API {url} returned {resp.status}")
                    data = json.load(resp)
                break
            except urllib.error.HTTPError as he:
                last_exc = he
                if he.code == 403:
                    reset = he.headers.get('X-RateLimit-Reset') if hasattr(he, 'headers') else None
                    remaining = he.headers.get('X-RateLimit-Remaining') if hasattr(he, 'headers') else None
                    logging.error("GitHub API HTTP 403: rate-limited or forbidden (remaining=%s reset=%s)", remaining, reset)
                    if not os.getenv('GITHUB_TOKEN'):
                        logging.error("Unauthenticated requests are limited to 60/hr. Set GITHUB_TOKEN to raise the limit.")
                    time.sleep(1 + attempt)
                else:
                    logging.debug("HTTPError when listing %s: %s (attempt %d)", url, he, attempt)
                    time.sleep(0.5 + attempt)
            except Exception as e:
                last_exc = e
                logging.debug("Error when listing %s: %s (attempt %d)", url, e, attempt)
                time.sleep(0.5 + attempt)
        else:
            raise last_exc
        if isinstance(data, dict) and data.get("type") == "file":
            if data.get("download_url"):
                results.append({"path": data["path"], "download_url": data["download_url"]})
            return
        for entry in data:
            if entry.get("type") == "file":
                results.append({"path": entry["path"], "download_url": entry.get("download_url")})
            elif entry.get("type") == "dir":
                _list(entry["path"])
            else:
                logging.debug("Skipping unknown type from API: %s", entry.get("type"))
    _list(path.rstrip('/'))
    return results


def download_url_to_path(url: str, out_path: Path, execute: bool):
    logging.info("Download: %s -> %s", url, out_path)
    if not execute:
        logging.info("Dry-run: skipping actual download")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_get_request_headers())
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to download {url}: HTTP {resp.status}")
            data = resp.read()
    except urllib.error.HTTPError as he:
        if he.code == 403 and not os.getenv('GITHUB_TOKEN'):
            logging.error("HTTP 403 downloading %s. You may have hit the GitHub API rate limit; set GITHUB_TOKEN to increase limits.", url)
        raise
    out_path.write_bytes(data)


def main(argv=None):
    p = argparse.ArgumentParser(description="Download notebooks and file assets via raw.githubusercontent")
    p.add_argument("--manifest", "-m", type=Path, default=Path("notebooks.yml"))
    p.add_argument("--dest", "-d", type=Path, default=Path("methods"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    manifest = load_manifest(args.manifest)
    entries = manifest.get("notebooks", [])
    if not entries:
        logging.info("No notebooks listed in manifest")
        return

    dest = args.dest

    for e in entries:
        url = e.get("url")
        branch = e.get("branch") or "main"
        nb_path = e.get("path")
        assets = e.get("assets") or []

        if not url:
            logging.error("Manifest entry missing url: %s", e)
            continue

        owner_repo = parse_github_owner_repo(url)
        if not owner_repo:
            logging.error("Non-GitHub URL; skipping entry: %s", url)
            continue

        repo_dir = Path(url.rstrip("/ ")).name
        if repo_dir.endswith('.git'):
            repo_dir = repo_dir[:-4]
        target_dir = dest / repo_dir

        # Notebook
        if nb_path:
            raw_nb_url = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{nb_path.lstrip('/')}"
            out_nb_path = target_dir / Path(nb_path)
            try:
                download_url_to_path(raw_nb_url, out_nb_path, args.execute)
            except Exception as exc:
                logging.error("Failed to download notebook %s: %s", raw_nb_url, exc)

        for a in assets:
            a = a.strip()
            if not a:
                continue
            if a.endswith('/'):
                # directory asset: list via GitHub API and download each file
                path_in_repo = a.rstrip('/')
                try:
                    files = github_api_list_files(owner_repo, path_in_repo, branch)
                except Exception as exc:
                    logging.error("Failed to list directory asset %s for %s: %s", a, url, exc)
                    continue
                for f in files:
                    # f['path'] is repo-relative (e.g. images/foo.png)
                    out_asset_path = target_dir / Path(f['path'])
                    try:
                        download_url_to_path(f.get('download_url'), out_asset_path, args.execute)
                    except Exception as exc:
                        logging.error("Failed to download file %s from %s: %s", f.get('path'), owner_repo, exc)
            else:
                raw_asset_url = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{a.lstrip('/')}"
                out_asset_path = target_dir / Path(a)
                try:
                    download_url_to_path(raw_asset_url, out_asset_path, args.execute)
                except Exception as exc:
                    logging.error("Failed to download asset %s: %s", raw_asset_url, exc)


if __name__ == '__main__':
    main()
