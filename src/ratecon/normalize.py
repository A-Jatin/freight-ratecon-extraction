"""Deterministic interpretation. Pure functions, no LLM, no clock.

Nothing here calls `datetime.now()`: every function is a pure function of the
document, so reprocessing the same bytes in 2029 gives the same answer.
"""

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal, NamedTuple

from ratecon.schema import Charge, EquipmentType, Stop

# --------------------------------------------------------------------------
# Text normalisation, shared by grounding checks
# --------------------------------------------------------------------------


def norm_text(s: str) -> str:
    """NFKC-fold, collapse whitespace, lowercase. Used for both haystack and needle."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("­", "")  # soft hyphen
    return re.sub(r"\s+", " ", s).strip().lower()


def contains_token(haystack_norm: str, needle: str) -> bool:
    """Token-boundary containment.

    Bare substring matching is unsafe for short values: against ordinary rate-con
    prose, `AK`, `HI`, `ME`, `OH` and `PA` all match as substrings without ever
    appearing as states, and `50.00` is a substring of `$1,450.00`.
    """
    n = norm_text(needle)
    if not n:
        return False
    return re.search(rf"(?<!\w){re.escape(n)}(?!\w)", haystack_norm) is not None


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------

_MONEY_RE = re.compile(r"-?\(?\$?\s*-?\d[\d,]*\.?\d*\)?")


def parse_money(text: str | None) -> Decimal | None:
    """`$1,250.00` / `1250.00 USD` / `(200.00)` -> Decimal. Parentheses mean negative.

    Decimal is built from a string, never a float. Accessorials can legitimately be
    negative (advances, ComCheck fees, quick-pay and factoring deductions, claim
    chargebacks), so the sign is preserved rather than assumed positive.
    """
    if text is None:
        return None
    raw = text.strip()
    if not raw or raw in {"-", "--"}:
        return None
    negative = (raw.startswith("(") and raw.endswith(")")) or raw.lstrip().startswith("-")
    cleaned = re.sub(r"[^\d.]", "", raw)
    if not cleaned or cleaned == ".":
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -value if negative else value


def money_in_source(source_norm: str, amount: Decimal) -> bool:
    """Is this amount printed anywhere in the document?

    Compared numerically rather than as a string, so `1,250.00` in the source
    matches `1250.00` in the output. String matching would fail on the separator.
    """
    for token in _MONEY_RE.findall(source_norm):
        parsed = parse_money(token)
        if parsed is not None and abs(parsed) == abs(amount):
            return True
    return False


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

DateStatus = Literal["resolved", "ambiguous", "unparseable", "absent"]

_UNAMBIGUOUS_FORMATS = ("%Y-%m-%d", "%d-%b-%Y", "%b %d, %Y", "%B %d, %Y", "%d %B %Y", "%d %b %Y")
_NUMERIC_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})$")
_DATE_TOKEN_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}-[A-Za-z]{3}-\d{4}\b")


class DateReading(NamedTuple):
    status: DateStatus
    value: date | None
    alternative: date | None  # the other reading, when genuinely ambiguous


def _try_formats(raw: str) -> date | None:
    from datetime import datetime

    for fmt in _UNAMBIGUOUS_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _expand_year(yy: int, century_hint: int | None) -> int:
    """Two-digit year expansion.

    There is no universal rule: POSIX pivots at 68/69, dateutil slides a 50-year
    window, Excel uses 1930-2029. It is a policy, not a fact. Ours: prefer the
    century of other four-digit years in the same document; otherwise 2000+.
    """
    if century_hint is not None:
        return century_hint - (century_hint % 100) + yy
    return 2000 + yy


def _numeric_candidates(raw: str, century_hint: int | None) -> list[date]:
    m = _NUMERIC_RE.match(raw.strip())
    if not m:
        return []
    a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
    year = int(y) if len(y) == 4 else _expand_year(int(y), century_hint)
    out: list[date] = []
    for month, day in ((a, b), (b, a)):  # MDY then DMY
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate not in out:
            out.append(candidate)
    return out


def infer_date_order(source_text: str) -> Literal["MDY", "DMY", "unknown"]:
    """Infer the document's date order from tokens that can only be read one way.

    Sample A prints `07/30/2026`; 30 cannot be a month, so the document is MDY,
    and that resolves its sibling `08/01/2026` which on its own is genuinely
    ambiguous. Four of the seven stop dates across the provided samples are
    locally ambiguous, so this inference is doing real work.
    """
    hint = _century_hint(source_text)
    votes = {"MDY": 0, "DMY": 0}
    for token in _DATE_TOKEN_RE.findall(source_text):
        m = _NUMERIC_RE.match(token)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12 and b <= 12:
            votes["DMY"] += 1
        elif b > 12 and a <= 12:
            votes["MDY"] += 1
    if votes["MDY"] > votes["DMY"]:
        return "MDY"
    if votes["DMY"] > votes["MDY"]:
        return "DMY"
    _ = hint
    return "unknown"


def _century_hint(source_text: str) -> int | None:
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", source_text)]
    return years[0] if years else None


def resolve_date(raw: str | None, source_text: str) -> DateReading:
    """Resolve one date string against the document's inferred order.

    Deliberately *not* delegated to `dateutil`: it silently picks a reading and
    returns a `datetime` indistinguishable from a confident one, which is exactly
    the signal this pipeline exists to surface. `fuzzy=True` is worse still — it
    manufactures dates out of phone and MC numbers.
    """
    if raw is None or not raw.strip() or raw.strip() in {"-", "--"}:
        return DateReading("absent", None, None)

    direct = _try_formats(raw)
    if direct is not None:
        return DateReading("resolved", direct, None)

    candidates = _numeric_candidates(raw, _century_hint(source_text))
    if not candidates:
        return DateReading("unparseable", None, None)
    if len(candidates) == 1:
        return DateReading("resolved", candidates[0], None)

    order = infer_date_order(source_text)
    if order == "MDY":
        return DateReading("resolved", candidates[0], candidates[1])
    if order == "DMY":
        return DateReading("resolved", candidates[1], candidates[0])
    # Genuinely ambiguous. Return the US reading (these are US brokerage documents)
    # but say so: nulling it destroys what the reviewer needs, and guessing
    # silently is how a truck arrives a month late.
    return DateReading("ambiguous", candidates[0], candidates[1])


# --------------------------------------------------------------------------
# Charges
# --------------------------------------------------------------------------

# Order matters, and the deny-list runs first. "Fuel Advance" is a *negative
# deduction*, not a surcharge; a substring match on "fuel" books a $300 advance
# as a $300 fuel surcharge - wrong field and wrong sign.
_DENY_FUEL = (r"\bfuel\s+advance\b", r"\breefer\s+fuel\b")

_LINE_HAUL = (
    r"\bbase\s+carrier\s+rate\b",
    r"\bline\s*haul\b",
    r"\bline-haul\b",
    r"\bbase\s+rate\b",
    r"\bfreight\s+(charge|rate)\b",
    r"\bflat\s+rate\b",
    r"\ball[-\s]?in(\s+rate)?\b",
    r"\btransportation\s+charge\b",
    r"\bagreed\s+rate\b",
)
_FUEL = (r"\bfuel\s*surcharge\b", r"\bfsc\b", r"\bf/s\b", r"\bfuel\s*adj\w*\b", r"\bfuel\b")
_ACCESSORIAL = (
    r"\bdetention\b",
    r"\blayover\b",
    r"\blumper\b",
    r"\btonu\b",
    r"\btruck\s+order\s+not\s+used\b",
    r"\bstop[-\s]?off\b",
    r"\bstop\s+charge\b",
    r"\bextra\s+stop\b",
    r"\bdriver\s+assist\b",
    r"\btarp\w*\b",
    r"\bpallet\b",
    r"\badvance\b",
    r"\bcom\s?check\b",
    r"\befs\b",
    r"\bquick\s?pay\b",
    r"\bfactoring\b",
    r"\breweigh\b",
    r"\bchassis\b",
    r"\bunloading\b",
    r"\bescort\b",
    r"\bpermit\w*\b",
    r"\bhazmat\b",
    r"\bstorage\b",
    r"\bredelivery\b",
    r"\bwait\s+time\b",
    r"\bpre[-\s]?pull\b",
    r"\bper\s+diem\b",
)
_TOTAL = (r"\btotal\b", r"\bamount\s+due\b", r"\bgross\s+pay\b", r"\bcustomer\s+rate\b")

ChargeKind = Literal["line_haul", "fuel", "accessorial", "total", "unknown"]


def classify_charge(label: str | None) -> ChargeKind:
    """Map a printed charge label onto our taxonomy.

    Anchored patterns, never substring: "Base Carrier Rate" and "Carrier Charge"
    share the token *Carrier*, and a naive `"carrier" in label` maps both to
    line-haul. On the provided multi-charge sample that would make the residual
    zero, suppress the unmapped-charge finding, and score the one genuinely
    interesting document as clean.
    """
    if label is None:
        return "unknown"
    s = norm_text(label)
    if not s:
        return "unknown"
    if any(re.search(p, s) for p in _DENY_FUEL):
        return "accessorial"
    for patterns, kind in (
        (_LINE_HAUL, "line_haul"),
        (_FUEL, "fuel"),
        (_ACCESSORIAL, "accessorial"),
        (_TOTAL, "total"),
    ):
        if any(re.search(p, s) for p in patterns):
            return kind  # type: ignore[return-value]
    return "unknown"


class ChargeBreakdown(NamedTuple):
    line_haul: Decimal | None
    fuel: Decimal | None
    accessorials: Decimal
    unmapped: list[tuple[str, Decimal]]
    residual: Decimal | None  # stated_total - sum(parts); None when no stated total


def summarise_charges(charges: list[Charge], stated_total: Decimal | None) -> ChargeBreakdown:
    """Classify each line and reconcile against the printed total.

    Zero-amount lines are ignored: many TMSs print every accessorial type at
    $0.00, and without suppression the unmapped-charge finding fires ten times on
    a perfectly clean document.
    """
    line_haul: Decimal | None = None
    fuel: Decimal | None = None
    accessorials = Decimal(0)
    unmapped: list[tuple[str, Decimal]] = []

    for c in charges:
        amount = parse_money(c.amount_text)
        if amount is None or amount == 0:
            continue
        kind = classify_charge(c.label_text)
        if kind == "line_haul":
            line_haul = amount if line_haul is None else line_haul + amount
        elif kind == "fuel":
            fuel = amount if fuel is None else fuel + amount
        elif kind == "accessorial":
            accessorials += amount
        elif kind == "unknown":
            unmapped.append((c.label_text or "", amount))

    residual = None
    if stated_total is not None:
        parts = (line_haul or Decimal(0)) + (fuel or Decimal(0)) + accessorials
        parts += sum((a for _, a in unmapped), Decimal(0))
        residual = stated_total - parts
    return ChargeBreakdown(line_haul, fuel, accessorials, unmapped, residual)


# --------------------------------------------------------------------------
# Equipment
# --------------------------------------------------------------------------

# Longest phrase first. Substring matching is unsafe here: "Cargo Van" and
# "Sprinter Van" both contain "van", but in FTL "van" means a 53' dry van
# trailer, while a cargo van is a sub-26,000-GVW vehicle needing no CDL. Mapping
# one to the other books a 44,000 lb load onto a Sprinter.
_EQUIPMENT: tuple[tuple[str, EquipmentType | None], ...] = (
    ("cargo van", EquipmentType.OTHER),
    ("sprinter van", EquipmentType.OTHER),
    ("sprinter", EquipmentType.OTHER),
    ("straight truck", EquipmentType.OTHER),
    ("box truck", EquipmentType.OTHER),
    ("power only", EquipmentType.OTHER),
    ("hotshot", EquipmentType.OTHER),
    ("step deck", EquipmentType.OTHER),
    ("stepdeck", EquipmentType.OTHER),
    ("drop deck", EquipmentType.OTHER),
    ("double drop", EquipmentType.OTHER),
    ("removable gooseneck", EquipmentType.OTHER),
    ("lowboy", EquipmentType.OTHER),
    ("conestoga", EquipmentType.OTHER),
    ("curtainside", EquipmentType.OTHER),
    ("rgn", EquipmentType.OTHER),
    ("temperature controlled", EquipmentType.REEFER),
    ("temp control", EquipmentType.REEFER),
    ("refrigerated", EquipmentType.REEFER),
    ("reefer", EquipmentType.REEFER),
    ("flatbed", EquipmentType.FLATBED),
    ("flat bed", EquipmentType.FLATBED),
    ("dry van", EquipmentType.VAN),
    ("van", EquipmentType.VAN),
)


def classify_equipment(text: str | None) -> tuple[EquipmentType | None, bool]:
    """Return (equipment, is_confident). Unknown strings map to None, never a guess."""
    if text is None or not text.strip():
        return None, False
    s = norm_text(text)
    for phrase, mapped in _EQUIPMENT:
        if phrase in s:
            # An open-deck variant is not substitutable for a flatbed (deck
            # height differs; the load will not transfer), so it lands in
            # `other` and is flagged rather than silently coerced.
            confident = mapped in (EquipmentType.VAN, EquipmentType.REEFER, EquipmentType.FLATBED)
            return mapped, confident
    return None, False


# --------------------------------------------------------------------------
# Stops
# --------------------------------------------------------------------------


class StopSelection(NamedTuple):
    origin: Stop | None
    destination: Stop | None
    sequence_disagrees_with_dates: bool


def select_stops(stops: list[Stop], source_text: str) -> StopSelection:
    """Origin is the lowest-sequence pickup; destination the highest-sequence delivery.

    The printed sequence is authoritative, not the dates. Stops are printed in
    routed order because that is the order the driver executes them, whereas
    per-stop dates are hand-entered and are exactly what gets fat-fingered.
    Sorting by date would let one typo silently reorder the route, and it has no
    answer at all when two stops share a date. Dates are used only to cross-check.
    """

    def seq(s: Stop, fallback: int) -> int:
        return s.sequence if s.sequence is not None else fallback

    pickups = [s for s in stops if s.kind == "pickup"]
    deliveries = [s for s in stops if s.kind == "delivery"]

    origin = min(pickups, key=lambda s: seq(s, stops.index(s))) if pickups else None
    destination = max(deliveries, key=lambda s: seq(s, stops.index(s))) if deliveries else None

    disagrees = False
    if origin is not None and destination is not None:
        o = resolve_date(origin.date_text, source_text).value
        d = resolve_date(destination.date_text, source_text).value
        if o is not None and d is not None and o > d:
            disagrees = True
    return StopSelection(origin, destination, disagrees)


# --------------------------------------------------------------------------
# Weight
# --------------------------------------------------------------------------

_KG = re.compile(r"\bkgs?\b|\bkilograms?\b")
_G_ONLY = re.compile(r"(?<!k)\bg\b|\bgrams?\b")


def parse_weight_lbs(text: str | None) -> tuple[Decimal | None, bool]:
    """Return (pounds, unit_was_stated).

    On freight documents the Weight column is the total for the line and Quantity
    is handling units, so a bare number is a weight, not a per-unit figure. The
    field is named `weight_lbs`, so an unlabelled number is taken as pounds and
    the assumption is recorded — declining to read a labelled Weight column would
    be declining to do the extraction. Where a unit *is* printed we convert,
    because a template that can emit grams can emit kilograms, and a blanket
    pounds assumption would then be a 2.2x error.
    """
    if text is None:
        return None, False
    s = norm_text(text)
    number = parse_money(re.sub(r"[a-z]", "", s))
    if number is None or number == 0:
        return None, False
    if _KG.search(s):
        return (number * Decimal("2.20462")).quantize(Decimal("0.01")), True
    if _G_ONLY.search(s):
        return (number * Decimal("0.00220462")).quantize(Decimal("0.01")), True
    return number, False
