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

# One number, wherever it sits in the span. `findall` returning more than one is
# the signal that the span fused two columns together.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# A number that is printed *as money*: it carries a currency mark, a currency
# code, or exactly the two decimal places money is written with. Without this
# discrimination the grounding check below matches any digit run in the
# document — on the provided samples the total `50.00` would ground against
# "50 W 4th Street" in the delivery address, which is not a check at all.
_MONEY_TOKEN_RE = re.compile(
    r"(?<![\w.,$-])(?:"
    r"\$\s*-?\d[\d,]*(?:\.\d{1,2})?"  # $1,250 / $1,250.00
    # Accounting negatives. Cents or a currency mark are required, or a phone
    # area code — `(905) 670-6882` — grounds a $905 total.
    r"|\(\s*\$\s*\d[\d,]*(?:\.\d{1,2})?\s*\)"
    r"|\(\s*\d[\d,]*\.\d{2}\s*\)"
    r"|-?\d[\d,]*\.\d{2}(?=\s*(?:usd|cad|dollars)\b|(?![\w.,-]))"  # 1,250.00 [USD]
    r"|-?\d[\d,]*(?:\.\d{1,2})?(?=\s*(?:usd|cad|dollars)\b)"  # 1250 USD
    r")"
)

# No freight rate, accessorial or trailer weight is a trillion of anything. A
# value this large is a fused span or a hallucinated digit run, and letting it
# through means `float(Decimal(...))` can reach `inf`, which serialises to JSON
# no strict parser will accept.
_MAX_MAGNITUDE = Decimal(10) ** 12


def parse_money(text: str | None) -> Decimal | None:
    """`$1,250.00` / `1250.00 USD` / `(200.00)` -> Decimal. Parentheses mean negative.

    Decimal is built from a string, never a float. Accessorials can legitimately be
    negative (advances, ComCheck fees, quick-pay and factoring deductions, claim
    chargebacks), so the sign is preserved rather than assumed positive.

    A span containing *two* numbers returns `None` rather than a value. Stripping
    the separators instead — `re.sub(r"[^\\d.]", "", raw)` — silently concatenates
    them, and on these documents that is reachable: the commodity table prints
    `Weight 182` beside `Quantity 07`, and one line of layout-preserved text
    turns those two cells into `1827`. Refusing an ambiguous span is the whole
    point of the exercise; fusing it is the failure mode with no signal attached.
    """
    if text is None:
        return None
    raw = text.strip()
    if not raw or raw in {"-", "--"}:
        return None

    numbers = _NUMBER_RE.findall(raw)
    if len(numbers) != 1:
        return None

    body = numbers[0]
    escaped = re.escape(body)
    negative = bool(
        # Parentheses around the *number*, not around the whole span. Testing the
        # span meant `(200.00) USD` — which is how a deduction is actually
        # printed on a document that suffixes the currency — starts with `(` and
        # ends with `D`, so it parsed as +200: right magnitude, inverted sign,
        # on the one charge class where the sign is the whole point.
        re.search(rf"\(\s*\$?\s*{escaped}\s*\)", raw)
        # A minus outside the currency mark: `-$500.00`. `_NUMBER_RE` cannot
        # match across the `$`, so `body` has no leading minus of its own.
        or re.search(rf"-\s*\$?\s*{escaped}", raw)
        or body.startswith("-")
        or raw.rstrip().endswith("-")  # trailing minus, still used by older TMSs
    )
    cleaned = body.lstrip("-").replace(",", "")
    if not cleaned or cleaned == ".":
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not value.is_finite() or abs(value) >= _MAX_MAGNITUDE:
        return None
    return -value if negative else value


# A whole line that is a *money* label followed immediately by its amount, and
# nothing else: `Total  1800`, `RATE: 1,800`, `Amount Due 1,800`.
#
# Two constraints do the work, and both were learned the hard way. The `[^\d]`
# prefix means the label cannot itself contain digits, so a line carrying a
# phone number or a ZIP before the keyword does not qualify. And the keyword
# must sit *immediately* before the number — only whitespace, a colon or a dash
# between them. Allowing a few free characters instead let `Total Miles 1250`,
# `Total Weight 44000` and `Total Pieces 1800` ground a total of that value,
# which is worse than the check it replaced: a mileage figure read as the rate
# publishes at `high` with nothing to flag it.
_LABELLED_AMOUNT_RE = re.compile(
    r"^[^\d\n]{0,60}?"
    r"\b(?:total|subtotal|rate|amount\s+due|amount|carrier\s+pay|gross\s+pay|pay"
    r"|line\s*haul|linehaul|freight|fsc|fuel\s+surcharge)"
    r"\s*[:\-]?\s*"
    r"(\d[\d,]*(?:\.\d{1,2})?)\s*(?:usd|cad|dollars)?\s*$"
)


def money_in_source(source_text: str, amount: Decimal) -> bool:
    """Is this amount printed as an amount anywhere in the document?

    Compared numerically rather than as a string, so `1,250.00` in the source
    matches `1250.00` in the output. String matching would fail on the separator.

    A bare digit run on its own is not evidence: ZIPs, PO numbers, phone
    fragments, MC numbers, quantities and street numbers are all digit runs, and
    accepting them makes this check pass for almost any small value the model
    could invent — on the provided samples the total `50.00` would ground
    against "50 W 4th Street" in the delivery address.

    So a number counts when it is either *shaped* like money (`$`, a currency
    code, or the two decimal places money is written with) or *positioned* like
    money — alone on a line after a rate label, which is how `Total   1800` is
    printed on the documents that do not bother with cents. Requiring the shape
    alone blocked those, and blocking a correct total is the more expensive
    error of the two: it routes a clean load to a human every time.
    """
    for raw_line in source_text.splitlines() or [source_text]:
        line = norm_text(raw_line)
        if not line:
            continue
        for token in _MONEY_TOKEN_RE.findall(line):
            parsed = parse_money(token)
            if parsed is not None and abs(parsed) == abs(amount):
                return True
        labelled = _LABELLED_AMOUNT_RE.match(line)
        if labelled:
            parsed = parse_money(labelled.group(1))
            if parsed is not None and abs(parsed) == abs(amount):
                return True
    return False


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

DateStatus = Literal["resolved", "ambiguous", "unparseable", "absent"]

_UNAMBIGUOUS_FORMATS = ("%Y-%m-%d", "%d-%b-%Y", "%b %d, %Y", "%B %d, %Y", "%d %B %Y", "%d %b %Y")
_NUMERIC_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})$")
_DATE_TOKEN_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}-[A-Za-z]{3}-\d{4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b"
)


def date_tokens(text: str) -> list[str]:
    """The date-shaped tokens inside a span.

    Grounding uses this rather than the whole span, for the same reason the
    address branch does not require a contiguous address: a model that returns
    `"03/16/2026 08:00"` for a stop whose date and appointment time are printed
    in separate columns has read the page correctly, and the joined string is
    nowhere in the source. The token itself still has to be printed, so a
    fabricated date is caught either way.
    """
    return _DATE_TOKEN_RE.findall(text)


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

    A document that votes *both* ways returns `unknown` rather than a majority.
    A bare majority is the wrong aggregation here: mixed orders on one page mean
    the page was assembled from two sources, which is exactly when a confident
    reading is least safe, and the cost of `unknown` is a flagged field while the
    cost of a wrong majority is a truck on the wrong day.
    """
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
    if votes["MDY"] and votes["DMY"]:
        return "unknown"
    if votes["MDY"]:
        return "MDY"
    if votes["DMY"]:
        return "DMY"
    return "unknown"


_ISO_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})-\d{1,2}-\d{1,2}\b")
_ALPHA_MONTH_YEAR_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}-(\d{4})$")


def _century_hint(source_text: str) -> int | None:
    """The century to expand a two-digit year into, taken from a real year.

    Only years that appear *inside a date* count. A bare `re.findall` for
    `19xx|20xx` over the whole document reads the broker letterhead instead: a
    `1900 Market Street` address or a `60601-1948` ZIP+4 then anchors every
    two-digit year to the wrong century, and `03/16/26` resolves to 1926 with
    `status="resolved"` and nothing to flag.
    """
    for token in _DATE_TOKEN_RE.findall(source_text):
        numeric = _NUMERIC_RE.match(token)
        if numeric and len(numeric.group(3)) == 4:
            return int(numeric.group(3))
        alpha = _ALPHA_MONTH_YEAR_RE.match(token)
        if alpha:
            return int(alpha.group(1))
    iso = _ISO_YEAR_RE.search(source_text)
    return int(iso.group(1)) if iso else None


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

# Split because the two halves lose to "Total" differently. A label that *names*
# the line haul — "Total Line Haul", "Freight Total" — is a line-haul line
# whatever else it says. A label that only carries a *generic* rate word —
# "Total All-In Rate" — is the grand total, and reading it as line haul adds the
# whole load on top of the real line-haul figure. Collapsing the two either
# inflates `line_haul_rate` or nulls it, and both were reachable.
_LINE_HAUL_NAMED = (
    r"\bbase\s+carrier\s+rate\b",
    r"\bline\s*haul\b",
    r"\bline-haul\b",
    r"\bbase\s+rate\b",
    r"\bfreight\b",
    r"\btransportation\s+charge\b",
)
_LINE_HAUL_GENERIC = (
    r"\bflat\s+rate\b",
    r"\ball[-\s]?in(\s+rate)?\b",
    r"\bagreed\s+rate\b",
)
_LINE_HAUL = _LINE_HAUL_NAMED + _LINE_HAUL_GENERIC
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

# A subtotal is neither a component nor the total. Summing it double-counts the
# lines above it; recording it as a total makes `MULTIPLE_TOTALS` fire on the
# ordinary "Subtotal 1,500 / Total 1,800" layout. It is simply dropped.
_SUBTOTAL = (r"\bsub[\s-]?total\b",)

ChargeKind = Literal["line_haul", "fuel", "accessorial", "total", "subtotal", "unknown"]


def classify_charge(label: str | None) -> ChargeKind:
    """Map a printed charge label onto our taxonomy.

    Anchored patterns, never substring: "Base Carrier Rate" and "Carrier Charge"
    share the token *Carrier*, and a naive `"carrier" in label` maps both to
    line-haul. On the provided multi-charge sample that would make the residual
    zero, suppress the unmapped-charge finding, and score the one genuinely
    interesting document as clean.

    The family order is deliberate, and "most specific first" is what it
    encodes. A label that *names* a charge beats the word "Total" — "Total
    Detention" is a detention line and "Total Line Haul" is a line-haul line,
    neither is the grand total. A label carrying only a *generic* rate word
    loses to it — "Total All-In Rate" is the total, and reading it as line haul
    adds the whole load on top of the real line-haul figure.

    So: deny-list, subtotal, accessorial, fuel, *named* line haul, total,
    *generic* line haul.
    """
    if label is None:
        return "unknown"
    s = norm_text(label)
    if not s:
        return "unknown"
    if any(re.search(p, s) for p in _DENY_FUEL):
        return "accessorial"
    for patterns, kind in (
        (_SUBTOTAL, "subtotal"),
        (_ACCESSORIAL, "accessorial"),
        (_FUEL, "fuel"),
        (_LINE_HAUL_NAMED, "line_haul"),
        (_TOTAL, "total"),
        (_LINE_HAUL_GENERIC, "line_haul"),
    ):
        if any(re.search(p, s) for p in patterns):
            return kind  # type: ignore[return-value]
    return "unknown"


class ChargeBreakdown(NamedTuple):
    line_haul: Decimal | None
    fuel: Decimal | None
    accessorials: Decimal
    unmapped: list[tuple[str, Decimal]]
    residual: Decimal | None  # stated_total - sum(parts); None when there is nothing to reconcile
    totals: list[tuple[str, Decimal]]  # every total-like line, for the two-totals check


def summarise_charges(charges: list[Charge], stated_total: Decimal | None) -> ChargeBreakdown:
    """Classify each line and reconcile against the printed total.

    Zero-amount lines are ignored: many TMSs print every accessorial type at
    $0.00, and without suppression the unmapped-charge finding fires ten times on
    a perfectly clean document.

    `residual` is `None` — not the whole total — when the rate table has no
    component lines at all. An all-in rate con that prints one line reading
    `Total  $1,800.00` has nothing to reconcile against; treating the total as an
    unexplained residual blocks a perfectly ordinary document down to `low`, and
    all-in quoting is the norm on spot broker-to-carrier freight.
    """
    line_haul: Decimal | None = None
    fuel: Decimal | None = None
    accessorials = Decimal(0)
    unmapped: list[tuple[str, Decimal]] = []
    totals: list[tuple[str, Decimal]] = []

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
        elif kind == "total":
            totals.append((c.label_text or "", amount))
        elif kind == "unknown":
            unmapped.append((c.label_text or "", amount))

    has_components = (
        line_haul is not None or fuel is not None or accessorials != 0 or bool(unmapped)
    )
    residual = None
    if stated_total is not None and has_components:
        parts = (line_haul or Decimal(0)) + (fuel or Decimal(0)) + accessorials
        parts += sum((a for _, a in unmapped), Decimal(0))
        residual = stated_total - parts
    return ChargeBreakdown(line_haul, fuel, accessorials, unmapped, residual, totals)


# --------------------------------------------------------------------------
# Equipment
# --------------------------------------------------------------------------

# Substring matching is unsafe here: "Cargo Van" and "Sprinter Van" both contain
# "van", but in FTL "van" means a 53' dry van trailer, while a cargo van is a
# sub-26,000-GVW vehicle needing no CDL. Mapping one to the other books a
# 44,000 lb load onto a Sprinter.
#
# Written grouped by family for readability and sorted longest-first at import,
# because hand-ordering is the kind of invariant that silently stops being true:
# grouped by family, `reefer` (6 chars) sits above `dry van` (7), so
# "53' Dry Van - no reefer required" matched REEFER *confidently* and suppressed
# the very flag that exists to catch it.
_EQUIPMENT_BY_FAMILY: tuple[tuple[str, EquipmentType | None], ...] = (
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

_EQUIPMENT: tuple[tuple[str, EquipmentType | None], ...] = tuple(
    sorted(_EQUIPMENT_BY_FAMILY, key=lambda pair: -len(pair[0]))
)


_NEGATION_BEFORE = re.compile(r"\b(?:no|non|not|without|excluding|except)\b[\s\-]*$")
_NEGATION_AFTER = re.compile(r"^[\s\-]*(?:not\s+(?:required|needed)|n/?a\b)")


def _is_negated(s: str, phrase: str) -> bool:
    """Is every occurrence of this phrase preceded or followed by a negation?

    "53' Dry Van - no reefer required" and "53 ft Van, non-refrigerated" are both
    dry vans that mention a reefer, and without this they resolve as *contested*
    and publish `other` — which is a worse answer than either family.
    """
    for m in re.finditer(re.escape(phrase), s):
        before = s[max(0, m.start() - 24) : m.start()]
        after = s[m.end() : m.end() + 24]
        if not _NEGATION_BEFORE.search(before) and not _NEGATION_AFTER.match(after):
            return False
    return True


def classify_equipment(text: str | None) -> tuple[EquipmentType | None, bool]:
    """Return (equipment, is_confident). Unknown strings map to None, never a guess.

    Every phrase is matched, not just the first, because the phrases overlap and
    the string is free text off a document. Negated mentions are dropped first —
    "Dry Van, no reefer" names two families but means one. What is left is
    resolved by two rules: a reefer *is* an insulated van trailer, so
    `{reefer, van}` is a reefer rather than a conflict; anything else that spans
    two families ("Dry Van or Flatbed") is a genuine either/or the document does
    not settle, so it lands in `other` and `EQUIPMENT_UNMAPPED` asks a human.
    """
    if text is None or not text.strip():
        return None, False
    s = norm_text(text)
    matches: list[tuple[str, EquipmentType | None]] = [
        (phrase, mapped)
        for phrase, mapped in _EQUIPMENT
        if phrase in s and not _is_negated(s, phrase)
    ]
    if not matches:
        return None, False

    # Longest-first, so a match nested inside a longer one ("van" inside
    # "dry van") never competes with its own container.
    winner = matches[0]
    families = {mapped for phrase, mapped in matches if phrase not in winner[0]} | {winner[1]}
    if len(families) > 1:
        if families == {EquipmentType.REEFER, EquipmentType.VAN}:
            return EquipmentType.REEFER, True
        return EquipmentType.OTHER, False

    # An open-deck variant is not substitutable for a flatbed (deck height
    # differs; the load will not transfer), so it lands in `other` and is
    # flagged rather than silently coerced.
    mapped = winner[1]
    confident = mapped in (EquipmentType.VAN, EquipmentType.REEFER, EquipmentType.FLATBED)
    return mapped, confident


# --------------------------------------------------------------------------
# Stops
# --------------------------------------------------------------------------


class StopSelection(NamedTuple):
    origin: Stop | None
    destination: Stop | None


def select_stops(stops: list[Stop]) -> StopSelection:
    """Origin is the first pickup in route order; destination the last delivery.

    The printed order is authoritative, not the dates. Stops are printed in
    routed order because that is the order the driver executes them, whereas
    per-stop dates are hand-entered and are exactly what gets fat-fingered.
    Sorting by date would let one typo silently reorder the route, and it has no
    answer at all when two stops share a date. Dates only cross-check, in
    `date_order_invalid`.

    Route order comes from the printed `sequence` only when *every* stop has one
    and they are distinct; otherwise it falls back wholesale to the order the
    model listed them in. Mixing the two — a printed sequence for some stops, a
    list index for the rest — compares numbers from different scales, and on a
    document with eleven stops a missing sequence number outranks a real one.
    """
    position = {id(s): i for i, s in enumerate(stops)}
    sequences = [s.sequence for s in stops]
    use_printed = all(n is not None for n in sequences) and len(set(sequences)) == len(sequences)

    def order(s: Stop) -> int:
        return s.sequence if use_printed and s.sequence is not None else position[id(s)]

    pickups = [s for s in stops if s.kind == "pickup"]
    deliveries = [s for s in stops if s.kind == "delivery"]

    origin = min(pickups, key=order) if pickups else None
    destination = max(deliveries, key=order) if deliveries else None
    return StopSelection(origin, destination)


# --------------------------------------------------------------------------
# Weight
# --------------------------------------------------------------------------

_KG = re.compile(r"\bkgs?\b|\bkilograms?\b")
_G_ONLY = re.compile(r"(?<!k)\bg\b|\bgrams?\b")
_TONNE = re.compile(r"\b(?:metric\s+tons?|tonnes?|mt)\b")
_TON = re.compile(r"\btons?\b")

_UNIT_FACTORS: tuple[tuple[re.Pattern[str], Decimal], ...] = (
    (_TONNE, Decimal("2204.62")),  # checked before `ton`, which it contains
    (_TON, Decimal("2000")),  # US short ton
    (_KG, Decimal("2.20462")),
    (_G_ONLY, Decimal("0.00220462")),
)


def parse_weight_lbs(text: str | None) -> tuple[Decimal | None, bool]:
    """Return (pounds, unit_was_stated).

    On freight documents the Weight column is the total for the line and Quantity
    is handling units, so a bare number is a weight, not a per-unit figure. The
    field is named `weight_lbs`, so an unlabelled number is taken as pounds and
    the assumption is recorded — declining to read a labelled Weight column would
    be declining to do the extraction. Where a unit *is* printed we convert,
    because a template that can emit grams can emit kilograms, and a blanket
    pounds assumption would then be a 2.2x error.

    `quantize` is guarded. It raises `InvalidOperation` once the result needs
    more digits than the Decimal context allows, and a hallucinated digit run
    with a unit attached reaches it directly — which used to escape `extract()`
    as a crash. `parse_money` now caps the magnitude, so this is belt and
    braces; a converted value that still will not quantize is not a weight.
    """
    if text is None:
        return None, False
    s = norm_text(text)
    number = parse_money(re.sub(r"[a-z]", "", s))
    if number is None or number == 0:
        return None, False
    for pattern, factor in _UNIT_FACTORS:
        if pattern.search(s):
            try:
                return (number * factor).quantize(Decimal("0.01")), True
            except InvalidOperation:
                return None, False
    return number, False
