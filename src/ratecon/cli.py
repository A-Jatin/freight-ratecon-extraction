"""`ratecon demo` runs offline with no key. That is the default on purpose."""

import argparse
import json
import sys
from pathlib import Path

from ratecon.extract import OpenRouterClient, ResponseCache
from ratecon.pipeline import ExtractionResult, confirm_fields, extract, extract_file

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "evals" / "fixtures"
CACHE = ROOT / "evals" / "cache"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAILED = 3


def _client(allow_network: bool) -> OpenRouterClient:
    return OpenRouterClient(ResponseCache(CACHE), allow_network=allow_network)


def _summarise(name: str, r: ExtractionResult) -> str:
    if r.status != "ok":
        return f"{name:<26} status={r.status:<8} reason={r.meta.get('reason')}"
    fields = confirm_fields(r)
    tail = f"  confirm: {', '.join(fields)}" if fields else ""
    codes = ", ".join(sorted({f.code for f in r.findings}))
    return (
        f"{name:<26} {r.confidence.value:<7} "
        f"{r.data.origin.city if r.data.origin else '?'} -> "
        f"{r.data.destination.city if r.data.destination else '?'}  "
        f"{r.data.pickup_date}  total={r.data.total_rate}{tail}\n"
        f"{'':<26} findings: {codes or 'none'}"
    )


def cmd_demo(args: argparse.Namespace) -> int:
    """Run every fixture from the committed cache. No key, no network."""
    if not FIXTURES.exists():
        print(f"No fixtures at {FIXTURES}", file=sys.stderr)
        return EXIT_USAGE
    client = _client(allow_network=False)
    worst = EXIT_OK
    for path in sorted(FIXTURES.glob("*.txt")):
        result = extract(path.read_text(), client)
        print(_summarise(path.stem, result))
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
    result = extract_file(path, _client(allow_network=not args.offline))
    print(json.dumps(result.to_dict(), indent=2))
    return EXIT_OK if result.status == "ok" else EXIT_FAILED


def cmd_record(args: argparse.Namespace) -> int:
    """Populate the cache with real provider responses. Needs OPENROUTER_API_KEY."""
    client = _client(allow_network=True)
    if not client.api_key:
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return EXIT_USAGE
    paths = (
        sorted(FIXTURES.glob("*.txt")) + sorted(Path(args.also).glob("*"))
        if args.also
        else sorted(FIXTURES.glob("*.txt"))
    )
    for path in paths:
        result = extract(path.read_text(), client)
        print(_summarise(path.stem, result))
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(prog="ratecon", description=__doc__)
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
    p_rec.set_defaults(func=cmd_record)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        args = parser.parse_args(["demo"])
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
