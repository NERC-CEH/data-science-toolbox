#!/usr/bin/env python3
"""Sync notebooks listed in a manifest (notebooks.yml) into a local methods/ directory.

This script prepares and optionally runs shallow, parallel clone commands for each
repository listed in the manifest. By default it performs a dry-run and prints the
commands it would run. Use --execute to actually perform the clone/fetch operations.

NOTE: The repository owner requested no actual cloning be performed in this change —
so CI or local testing can run the script with --dry-run to verify command generation.
"""

from __future__ import annotations
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
except Exception as e:
    print("Missing dependency: pyyaml. Install with `pip install pyyaml`.")
    raise

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data or "notebooks" not in data:
        raise SystemExit(f"Manifest {path} does not contain a 'notebooks' key")
    return data


def repo_dir_from_url(url: str) -> str:
    # simple heuristic to derive a directory name from repo url
    name = url.rstrip("/ ")
    if name.endswith(".git"):
        name = name[:-4]
    return Path(name).name


def prepare_commands(entry: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    url = entry.get("url")
    if not url:
        raise ValueError("Manifest entry missing 'url'")
    branch = entry.get("branch")
    sha = entry.get("sha")
    subpath = entry.get("path")
    repo_dir = repo_dir_from_url(url)
    target_dir = dest / repo_dir

    clone_cmd = ["git", "clone", "--depth", "1"]
    if branch:
        clone_cmd += ["--branch", branch]
    clone_cmd += [url, str(target_dir)]

    cmds = {
        "clone_cmd": clone_cmd,
        "post_checkout": []
    }
    if sha:
        # need to fetch that sha; for shallow clones this may fail — caller must decide
        cmds["post_checkout"].append(["git", "fetch", "--depth", "1", "origin", sha])
        cmds["post_checkout"].append(["git", "checkout", sha])
    if subpath:
        cmds["subpath"] = subpath
    return {"url": url, "repo_dir": repo_dir, "target_dir": str(target_dir), **cmds}


def run_cmd(cmd, cwd: Path | None = None):
    logging.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main(argv=None):
    p = argparse.ArgumentParser(description="Prepare or run shallow clones for notebook manifest")
    p.add_argument("--manifest", "-m", type=Path, default=Path("notebooks.yml"), help="Path to manifest YAML")
    p.add_argument("--dest", "-d", type=Path, default=Path("methods"), help="Destination base directory")
    p.add_argument("--execute", action="store_true", help="Actually perform clones (dangerous) — default: dry-run")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    manifest = load_manifest(args.manifest)
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    entries = manifest.get("notebooks", [])
    if not entries:
        logging.info("No notebooks listed in manifest")
        return

    plans = []
    for e in entries:
        try:
            plan = prepare_commands(e, dest)
            plans.append(plan)
        except Exception as exc:
            logging.error("Failed to prepare commands for entry %s: %s", e, exc)

    # Print summary
    logging.info("Prepared %d notebook plans", len(plans))
    for pinfo in plans:
        logging.info("Repo: %s -> %s", pinfo['url'], pinfo['target_dir'])
        logging.info("Clone command: %s", " ".join(map(str, pinfo['clone_cmd'])))
        if pinfo.get("post_checkout"):
            logging.info("Post-checkout steps: %s", pinfo['post_checkout'])

    if not args.execute:
        logging.info("Dry-run mode. Use --execute to actually run clones on this machine.")
        return

    # Execution path (only runs if --execute provided)
    for pinfo in plans:
        try:
            run_cmd(pinfo['clone_cmd'])
            for post in pinfo.get('post_checkout', []):
                run_cmd(post, cwd=Path(pinfo['target_dir']))
        except subprocess.CalledProcessError as exc:
            logging.error("Command failed: %s", exc)


if __name__ == '__main__':
    main()
