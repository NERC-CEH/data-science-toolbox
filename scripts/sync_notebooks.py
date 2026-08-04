#!/usr/bin/env python3
"""Sync notebooks listed in a manifest (notebooks.yml) into a local methods/ directory.

By default the script runs in dry-run mode and prints the actions it would take.
Two modes are supported:
- raw: download the single notebook file from raw.githubusercontent.com (recommended)
- git: perform shallow git clones (legacy behaviour)

Use --execute to actually perform downloads/clones. Default mode is 'raw'.
"""

from __future__ import annotations
import argparse
import logging
import subprocess
import sys
import urllib.request
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


def repo_dir_from_url(url: str) -> str:
    name = url.rstrip("/ ")
    if name.endswith(".git"):
        name = name[:-4]
    # for urls like https://github.com/owner/repo or git@github.com:owner/repo.git
    return Path(name).name


def parse_github_owner_repo(url: str) -> Optional[str]:
    """Return 'owner/repo' for common GitHub URL forms, else None."""
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


def prepare_plan(entry: Dict[str, Any], dest: Path, mode: str = "raw") -> Dict[str, Any]:
    url = entry.get("url")
    if not url:
        raise ValueError("Manifest entry missing 'url'")
    branch = entry.get("branch") or "main"
    sha = entry.get("sha")
    subpath = entry.get("path")
    assets = entry.get("assets") or []
    repo_dir = repo_dir_from_url(url)
    target_dir = dest / repo_dir

    plan: Dict[str, Any] = {"url": url, "repo_dir": repo_dir, "target_dir": str(target_dir), "branch": branch, "subpath": subpath, "assets": assets}

    if mode == "git":
        clone_cmd = ["git", "clone", "--depth", "1"]
        if branch:
            clone_cmd += ["--branch", branch]
        clone_cmd += [url, str(target_dir)]
        plan.update({"type": "git", "clone_cmd": clone_cmd, "post_checkout": []})
        if sha:
            plan["post_checkout"].append(["git", "fetch", "--depth", "1", "origin", sha])
            plan["post_checkout"].append(["git", "checkout", sha])
        if subpath:
            plan["subpath"] = subpath
        return plan

    # raw mode: require a path to the notebook file
    if not subpath:
        # fallback to cloning if no subpath provided
        plan.update({"type": "git", "reason": "no path provided for raw mode; falling back to git"})
        return prepare_plan(entry, dest, mode="git")

    owner_repo = parse_github_owner_repo(url)
    if not owner_repo:
        plan.update({"type": "git", "reason": "non-GitHub URL; falling back to git"})
        return prepare_plan(entry, dest, mode="git")

    raw_url = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{subpath}"
    # Preserve the notebook's path inside the repo so relative asset references resolve
    out_path = target_dir / Path(subpath)
    plan.update({"type": "raw", "raw_url": raw_url, "out_path": str(out_path), "owner_repo": owner_repo})
    return plan


def download_raw(raw_url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s -> %s", raw_url, out_path)
    with urllib.request.urlopen(raw_url) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Failed to download {raw_url}: HTTP {resp.status}")
        data = resp.read()
    out_path.write_bytes(data)


# Automatic asset detection is disabled. Specify assets explicitly in notebooks.yml
# Example manifest entry:
# - name: example
#   url: https://github.com/owner/repo.git
#   branch: main
#   path: notebooks/example.ipynb
#   assets:
#     - images/                # directory -> sparse-checkout
#     - images/thumbnail.png   # individual file -> raw download

def extract_local_paths_from_notebook(nb_path: Path) -> set:
    logging.debug("Automatic asset detection is disabled; specify assets in the manifest under 'assets'.")
    return set()


def run_cmd(cmd, cwd: Optional[Path] = None):
    logging.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main(argv=None):
    p = argparse.ArgumentParser(description="Prepare or run shallow clones/downloads for notebook manifest")
    p.add_argument("--manifest", "-m", type=Path, default=Path("notebooks.yml"), help="Path to manifest YAML")
    p.add_argument("--dest", "-d", type=Path, default=Path("methods"), help="Destination base directory")
    p.add_argument("--execute", action="store_true", help="Actually perform downloads/clones (default: dry-run)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--mode", choices=["raw", "git"], default="raw", help="How to fetch notebooks: 'raw' downloads single files from raw.githubusercontent (recommended), 'git' uses shallow clones")
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
            plan = prepare_plan(e, dest, mode=args.mode)
            plans.append(plan)
        except Exception as exc:
            logging.error("Failed to prepare plan for entry %s: %s", e, exc)

    logging.info("Prepared %d notebook plans", len(plans))
    for pinfo in plans:
        if pinfo["type"] == "raw":
            logging.info("Download: %s -> %s", pinfo["raw_url"], pinfo["out_path"]) 
            if not args.execute:
                logging.info("Dry-run: would download file using URL above")
        else:
            logging.info("Git fetch fallback for %s: %s", pinfo.get("url"), pinfo.get("clone_cmd", pinfo.get("reason")))

    if not args.execute:
        logging.info("Dry-run mode. Use --execute to actually perform downloads or clones.")
        return

    # Execution
    for pinfo in plans:
        try:
            if pinfo["type"] == "raw":
                download_raw(pinfo["raw_url"], Path(pinfo["out_path"]))
                # After downloading the notebook, parse it and download any locally-referenced assets
                nb_path = Path(pinfo["out_path"])
                # Explicit asset handling: assets must be listed in the manifest under the 'assets' key
                assets = pinfo.get("assets") or []
                if not assets:
                    logging.debug("No assets listed for %s; skipping asset download", pinfo.get("repo_dir"))
                for asset in assets:
                    asset = asset.strip()
                    if not asset:
                        continue
                    # directory asset -> attempt sparse-checkout of that directory
                    if asset.endswith("/"):
                        import tempfile, shutil
                        tmpdir = Path(tempfile.mkdtemp())
                        try:
                            run_cmd(["git", "clone", "--depth", "1", "--no-checkout", pinfo["url"], str(tmpdir)])
                            run_cmd(["git", "-C", str(tmpdir), "sparse-checkout", "init", "--cone"])
                            run_cmd(["git", "-C", str(tmpdir), "sparse-checkout", "set", asset.rstrip('/')])
                            run_cmd(["git", "-C", str(tmpdir), "checkout"])
                            srcdir = tmpdir / asset.rstrip('/')
                            destdir = Path(pinfo["target_dir"]) / asset.rstrip('/')
                            if srcdir.exists():
                                if destdir.exists():
                                    shutil.rmtree(destdir)
                                shutil.move(str(srcdir), str(destdir))
                            else:
                                logging.error("Sparse-checkout did not produce directory %s", srcdir)
                        finally:
                            try:
                                shutil.rmtree(tmpdir)
                            except Exception:
                                pass
                    else:
                        owner_repo = pinfo.get("owner_repo")
                        branch = pinfo.get("branch") or "main"
                        raw_asset_url = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{asset}"
                        out_asset_path = Path(pinfo["target_dir"]) / Path(asset)
                        try:
                            download_raw(raw_asset_url, out_asset_path)
                        except Exception as exc:
                            logging.error("Failed to download asset %s: %s", raw_asset_url, exc)
            else:
                # git flow
                run_cmd(pinfo["clone_cmd"])  # type: ignore[arg-type]
                for post in pinfo.get('post_checkout', []):
                    run_cmd(post, cwd=Path(pinfo['target_dir']))
        except Exception as exc:
            logging.error("Failed to fetch %s: %s", pinfo.get('url', pinfo.get('raw_url')), exc)
    return plans


def update_myst_paths(plans, myst_path: Path = Path('myst.yml')):
    """myst.yml updates are intentionally disabled.

    This script no longer modifies myst.yml. If you need TOC updates, edit myst.yml
    manually or use a separate tool.
    """
    logging.info("myst.yml update disabled; skipping TOC modification.")
    return


if __name__ == '__main__':
    plans = main()
    # myst.yml update disabled by request.
