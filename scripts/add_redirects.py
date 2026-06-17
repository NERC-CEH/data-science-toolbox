"""Write HTML redirect pages into the build directory based on .github/redirects.json."""

import json
import pathlib
import sys

REDIRECTS_FILE = pathlib.Path(".github/redirects.json")

REDIRECT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Redirecting...</title>
  <meta http-equiv="refresh" content="0; url={to}">
  <link rel="canonical" href="{canonical}">
  <script>window.location.replace("{to}");</script>
</head>
<body>
  <p>This page has moved. <a href="{to}">Click here if you are not redirected.</a></p>
</body>
</html>
"""


def load_redirects():
    if not REDIRECTS_FILE.exists():
        sys.exit(f"Error: redirects file not found: {REDIRECTS_FILE}")
    try:
        data = json.loads(REDIRECTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"Error: invalid JSON in {REDIRECTS_FILE}: {exc}")
    if "redirects" not in data or not isinstance(data["redirects"], list):
        sys.exit(f"Error: {REDIRECTS_FILE} must contain a top-level 'redirects' list")
    return data["redirects"]


def main():
    build_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("_build/html")

    for entry in load_redirects():
        dest = build_dir / entry["from"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            REDIRECT_TEMPLATE.format(to=entry["to"], canonical=entry["canonical"]),
            encoding="utf-8",
        )
        print(f"Created redirect: {entry['from']} -> {entry['to']}")


if __name__ == "__main__":
    main()
