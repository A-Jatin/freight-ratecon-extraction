"""`ratecon demo` runs offline with no key. That is the default on purpose."""

import argparse
import json
import sys
from pathlib import Path

from ratecon.extract import OpenRouterClient, ResponseCache
from ratecon.pipeline import ExtractionResult, confirm_fields, extract, extract_file

# One fixture carries a deliberately wrong authored response, so that the corpus
# exercises the pipeline catching a *model* mistake rather than only a hard
# document. Recording over it would replace the mistake with whatever the model
# happens to do that day and quietly delete the only such case we have.
KEEP_AUTHORED = frozenset({"11_model_misreads_the_lane"})

# `--also` points at a directory of documents, and a directory of documents
# usually also contains a README. Billing a provider call to classify our own
# prose as `document_type: "other"` is a small waste and a confusing cache entry.
DOCUMENT_SUFFIXES = frozenset({".txt", ".text", ".pdf"})

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAILED = 3


def _repo_root() -> Path | None:
    """Walk up looking for the eval corpus rather than counting directories.

    `parents[2]` is only correct for a source checkout. Installed into
    site-packages it points at some unrelated directory, so `demo` found no
    fixtures and — worse — `extract` silently pointed the cache at a path that
    did not exist and called the provider for every document.
    """
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "evals" / "fixtures").is_dir():
            return candidate
    return None


ROOT = _repo_root()
FIXTURES = ROOT / "evals" / "fixtures" if ROOT else None
# Installed rather than checked out, `extract` still needs somewhere to cache.
# Returning None here instead made the one command that genuinely works anywhere
# refuse to run, contradicting the message printed beside it.
CACHE = (ROOT / "evals" / "cache") if ROOT else (Path.home() / ".cache" / "ratecon")


def _client(allow_network: bool, force: bool = False) -> OpenRouterClient:
    return OpenRouterClient(ResponseCache(CACHE), allow_network=allow_network, force=force)


def _no_corpus() -> int:
    print(
        "Could not locate evals/fixtures. `demo` and `record` need the repository "
        "checkout; `extract <file>` works anywhere.",
        file=sys.stderr,
    )
    return EXIT_USAGE


def _summarise(name: str, r: ExtractionResult) -> str:
    if r.status != "ok":
        return f"{name:<30} status={r.status:<8} reason={r.meta.get('reason')}"
    fields = confirm_fields(r)
    tail = f"  confirm: {', '.join(fields)}" if fields else ""
    codes = ", ".join(sorted({f.code for f in r.findings}))
    provenance = r.meta.get("recorded", "live")
    return (
        f"{name:<30} {r.confidence.value:<7} "
        f"{r.data.origin.city if r.data.origin else '?'} -> "
        f"{r.data.destination.city if r.data.destination else '?'}  "
        f"{r.data.pickup_date}  total={r.data.total_rate}{tail}\n"
        f"{'':<30} findings: {codes or 'none'}  [response: {provenance}]"
    )


def _log(path: str | None, name: str, r: ExtractionResult) -> None:
    """One JSON line per extraction. This is the monitoring substrate Part 2
    describes — per-field status, finding codes, confidence, provenance and the
    policy version — and it is deliberately the *envelope* rather than the data,
    so the file can be shipped to a dashboard without carrying document text.
    """
    if not path:
        return
    row = {
        "source": name,
        "status": r.status,
        "confidence": r.confidence.value,
        "field_status": r.field_status,
        "findings": [
            {"code": f.code, "severity": f.severity.value, "fields": list(f.fields)}
            for f in r.findings
        ],
        "meta": r.meta,
    }
    try:
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as e:
        # Monitoring is not the job. Losing a log line is a warning; losing the
        # extraction because the log path was wrong is an outage of our making.
        print(f"warning: could not write --log {path}: {e}", file=sys.stderr)


def cmd_demo(args: argparse.Namespace) -> int:
    """Run every fixture from the committed cache. No key, no network."""
    if FIXTURES is None:
        return _no_corpus()
    client = _client(allow_network=False)
    worst = EXIT_OK
    for path in sorted(FIXTURES.glob("*.txt")):
        result = extract(path.read_text(), client)
        print(_summarise(path.stem, result))
        _log(args.log, path.stem, result)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        if result.status != "ok":
            worst = EXIT_FAILED
    return worst


def cmd_extract(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return EXIT_USAGE
    client = _client(allow_network=not args.offline)
    result = extract_file(path, client)
    print(json.dumps(result.to_dict(), indent=2))
    _log(args.log, path.name, result)
    return EXIT_OK if result.status == "ok" else EXIT_FAILED


def cmd_record(args: argparse.Namespace) -> int:
    """Populate the cache with real provider responses. Needs OPENROUTER_API_KEY.

    `--force` re-records documents that are already cached, which is the only way
    to replace the committed authored responses; without it every fixture hits
    the cache and the command is a no-op.
    """
    if FIXTURES is None:
        return _no_corpus()
    client = _client(allow_network=True, force=args.force)
    if not client.api_key:
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return EXIT_USAGE

    paths = sorted(FIXTURES.glob("*.txt"))
    if args.also:
        # Through `extract_file`, so a PDF is parsed rather than read as text and
        # an unreadable file becomes one failed row instead of a traceback that
        # abandons the documents already paid for.
        paths += [
            p
            for p in sorted(Path(args.also).iterdir())
            if p.is_file() and p.suffix.lower() in DOCUMENT_SUFFIXES
        ]

    worst = EXIT_OK
    for path in paths:
        if path.stem in KEEP_AUTHORED:
            print(f"{path.stem:<30} skipped (authored on purpose)")
            continue
        result = extract_file(path, client)
        print(_summarise(path.stem, result))
        _log(args.log, path.name, result)
        if result.status != "ok":
            worst = EXIT_FAILED
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(prog="ratecon", description=__doc__)
    parser.add_argument("--log", metavar="PATH", help="append one JSON envelope per document")
    sub = parser.add_subparsers(dest="cmd")

    p_demo = sub.add_parser("demo", help="run all fixtures offline (default)")
    p_demo.add_argument("--json", action="store_true", help="print the full envelope")
    p_demo.set_defaults(func=cmd_demo)

    p_ex = sub.add_parser("extract", help="extract one file")
    p_ex.add_argument("path")
    p_ex.add_argument("--offline", action="store_true", help="cache only, never call out")
    p_ex.set_defaults(func=cmd_extract)

    p_rec = sub.add_parser("record", help="call the provider and fill the cache")
    p_rec.add_argument("--also", help="an extra directory of documents to record")
    p_rec.add_argument(
        "--force", action="store_true", help="re-record documents that are already cached"
    )
    p_rec.set_defaults(func=cmd_record)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        args = parser.parse_args([*(["--log", args.log] if args.log else []), "demo"])
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
