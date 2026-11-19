import argparse
import html
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


INLINE_LINK = re.compile(r'!?\[[^\]]*\]\(\s*(<[^>]+>|(?:\\.|[^\\()\s]|\([^()]*\))+)(?:\s+"[^"]*")?\s*\)')
REFERENCE = re.compile(r'^ {0,3}\[([^\]]+)\]:\s*(<[^>]+>|\S+)')
GENERATED_SUFFIXES = {".pyc", ".pyo", ".o", ".obj", ".a", ".so", ".dylib", ".dll", ".exe", ".log", ".tmp"}
GENERATED_NAMES = {"CMakeCache.txt", "cmake_install.cmake", "CTestTestfile.cmake", "build.ninja"}
GENERATED_DIRS = {"__pycache__", ".pytest_cache", "CMakeFiles"}


def git_paths(root, *arguments, input=None):
    result = subprocess.run(["git", "-C", str(root), *arguments], input=input,
                            capture_output=True)
    if result.returncode not in (0, 1) or (result.returncode == 1 and arguments[0] != "check-ignore"):
        raise RuntimeError(result.stderr.decode().strip())
    return [Path(value.decode()) for value in result.stdout.split(b"\0") if value]


def markdown_lines(text):
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        elif fence is None:
            yield number, line


def heading_anchors(text):
    anchors = set()
    for _, line in markdown_lines(text):
        heading = re.match(r'^ {0,3}#{1,6}\s+(.+?)(?:\s+#+\s*)?$', line)
        if not heading:
            continue
        title = INLINE_LINK.sub(lambda match: re.search(r'\[([^\]]*)\]', match[0])[1], heading[1])
        title = html.unescape(re.sub(r'<[^>]*>', '', title)).lower()
        base = re.sub(r'[^\w\- ]', '', title).replace(' ', '-')
        slug, suffix = base, 0
        while slug in anchors:
            suffix += 1
            slug = f"{base}-{suffix}"
        anchors.add(slug)
    return anchors


def local_links(text):
    lines = list(markdown_lines(text))
    references = {}
    for _, line in lines:
        definition = REFERENCE.match(line)
        if definition:
            references[' '.join(definition[1].casefold().split())] = definition[2]
    for number, line in lines:
        if REFERENCE.match(line):
            continue
        line = re.sub(r'(`+).*?\1', '', line)
        for match in INLINE_LINK.finditer(line):
            yield number, match[1].strip('<>')
        line = INLINE_LINK.sub('', line)
        for match in re.finditer(r'\[([^\]]+)\](?:\[([^\]]*)\])?', line):
            key = ' '.join((match[2] or match[1]).casefold().split())
            if key in references:
                yield number, references[key].strip('<>')


def audit(root):
    root = root.resolve()
    tracked = {path for path in git_paths(root, "ls-files", "--cached", "-z") if (root / path).is_file()}
    ignored = set()
    if tracked:
        payload = b''.join(str(path).encode() + b'\0' for path in sorted(tracked))
        ignored = set(git_paths(root, "check-ignore", "--no-index", "--stdin", "-z", input=payload))
    errors = []
    for path in sorted(tracked):
        if (path in ignored or path.suffix.lower() in GENERATED_SUFFIXES or
                path.name in GENERATED_NAMES or GENERATED_DIRS.intersection(path.parts)):
            errors.append(f"{path}: generated or ignored file is tracked by Git")
    candidates = tracked | set(git_paths(root, "ls-files", "--others", "--exclude-standard", "-z"))
    documents = sorted(path for path in candidates if path.suffix.lower() == '.md' and (root / path).is_file())
    anchor_cache = {}
    links = 0
    for path in documents:
        for number, destination in local_links((root / path).read_text(encoding="utf-8")):
            target = urlsplit(html.unescape(destination))
            if target.scheme or target.netloc:
                continue
            links += 1
            name = unquote(re.sub(r'\\([() ])', r'\1', target.path))
            if not name:
                resolved = root / path
            elif name.startswith('/'):
                resolved = root / name.lstrip('/')
            else:
                resolved = (root / path).parent / name
            resolved = resolved.resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                errors.append(f"{path}:{number}: link escapes repository: {destination}")
                continue
            if not resolved.exists():
                errors.append(f"{path}:{number}: missing target: {destination}")
            elif target.fragment and resolved.suffix.lower() == '.md':
                if resolved not in anchor_cache:
                    anchor_cache[resolved] = heading_anchors(resolved.read_text(encoding="utf-8"))
                if unquote(target.fragment) not in anchor_cache[resolved]:
                    errors.append(f"{path}:{number}: missing heading: {destination}")
    return errors, len(documents), links


def main():
    parser = argparse.ArgumentParser(description="Check local Markdown links and tracked generated files offline.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        errors, documents, links = audit(args.root)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Repository check failed: {error}\n")
    for error in errors:
        print(error)
    print(f"Checked {documents} Markdown files and {links} local links; {len(errors)} issues.")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
