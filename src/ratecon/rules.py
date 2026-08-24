"""The audit. Every rule is a pure function over `Assembled`, so the whole
confidence engine unit-tests with no LLM and no network.

Three design rules hold throughout:

* A finding names the field it impugns. A rule that cannot name one is either
  record-scoped (`fields=()`) or relational, in which case it impugns every
  field it involves — a delivery-before-pickup check that blamed only the
  delivery date would let a clearly broken document through, because the
  delivery date alone does not decide the tier.
* A BLOCK never mutates the data. It marks the field unusable and leaves the
  value in place, so an inconsistency in one field can never destroy another
  field's perfectly good value.
* Every check is either a *precision* check (is this value printed on the
  document?) or a *recall* check (did we read everything the document printed?).
  Precision alone is not enough and the asymmetry is vicious: an omission
  removes the findings that would have flagged it, so without `stop_count`
  below a truncated extraction scores strictly better than a faithful one.
"""

import re
from dataclasses import dataclass
from decimal import Decimal

from ratecon import normalize
from ratecon.normalize import ChargeBreakdown, DateReading, StopSelection
from ratecon.schema import EquipmentType, Finding, LlmExtraction, Severity

# The fields that decide whether a load can be created without a human. Weight,
# commodity, load_id and equipment are recorded and flagged but do not gate:
# they are not what makes a booking wrong in a way that costs money on the day.
GATING_FIELDS = frozenset({"total_rate", "pickup_date", "delivery_date", "origin", "destination"})

# Deterministic order for record-scoped findings, which expand to every gating
# field. Iterating the frozenset directly makes `field_status` key order vary
# between processes, which is noise in a diff and in a monitoring payload.
GATING_ORDER = ("total_rate", "pickup_date", "delivery_date", "origin", "destination")

# Every published field, so `field_status` can say "checked, fine" rather than
# staying silent and leaving a consumer to guess whether a rule ever looked.
PUBLISHED_FIELDS = (
    "load_id",
    "origin",
    "destination",
    "pickup_date",
    "delivery_date",
    "equipment_type",
    "line_haul_rate",
    "fuel_surcharge",
    "total_rate",
    "weight_lbs",
    "commodity",
)

# Fields whose published value comes from a verbatim span and is therefore
# checkable against the source. `not_grounded` covers exactly this tuple and
# `test_every_groundable_field_is_actually_grounded` holds it to that, so the
# constant cannot quietly drift out of agreement with the code again.
#
# Derived values are deliberately absent: `line_haul_rate` may be a sum of two
# lines, `fuel_surcharge` is often a correct null, and `weight_lbs` may have
# been unit-converted. None of those has a span to find, so asserting one would
# fail on correct extractions.
GROUNDABLE = (
    "load_id",
    "commodity",
    "equipment_type",
    "total_rate",
    "origin",
    "destination",
    "pickup_date",
    "delivery_date",
)


@dataclass(frozen=True)
class Assembled:
    """Everything the rules need: raw model output plus the deterministic reading."""

    llm: LlmExtraction
    source_text: str
    source_norm: str
    stops: StopSelection
    charges: ChargeBreakdown
    stated_total: Decimal | None
    pickup: DateReading
    delivery: DateReading
    equipment: EquipmentType | None
    equipment_confident: bool
    commodity_text: str | None
    weight_text: str | None
    weight_lbs: Decimal | None
    weight_unit_stated: bool


def _f(code: str, sev: Severity, fields: tuple[str, ...], msg: str) -> Finding:
    return Finding(code=code, severity=sev, fields=fields, message=msg)


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


def not_a_rate_con(a: Assembled) -> list[Finding]:
    """Record-scoped gate. Brokers are emailed BOLs, insurance certs and invoices;
    without this, field-level confidence is meaningless because the model will
    confidently extract nothing-shaped-like-something.

    The model's own classification only corroborates — the deterministic half
    has to agree before we reject the document. "Agree" means one of two things:
    no rate breakdown at all (a BOL), or a document the model calls an invoice
    or a tender, which carries money but is not an agreement to pay *this*
    carrier. The second case was previously unreachable, so a customer-side
    tender extracted cleanly and published as a carrier rate con.
    """
    said_no = a.llm.document_type != "rate_confirmation"
    if not said_no:
        return []
    no_money = a.charges.line_haul is None and a.charges.fuel is None and a.stated_total is None
    if no_money:
        return [
            _f(
                "NOT_A_RATE_CON",
                Severity.BLOCK,
                (),
                f"Document looks like a {a.llm.document_type} and carries no rate breakdown.",
            )
        ]
    if a.llm.document_type in ("invoice", "other"):
        return [
            _f(
                "NOT_A_RATE_CON",
                Severity.BLOCK,
                (),
                f"Document carries a rate but is classified {a.llm.document_type}; "
                "the amount may not be what this carrier is owed.",
            )
        ]
    return []


def critical_field_unusable(a: Assembled) -> list[Finding]:
    out: list[Finding] = []
    if a.stops.origin is None or not _usable(a.stops.origin.city_text, a.stops.origin.state_text):
        out.append(
            _f("CRITICAL_FIELD_UNUSABLE", Severity.BLOCK, ("origin",), "No usable pickup stop.")
        )
    if a.stops.destination is None or not _usable(
        a.stops.destination.city_text, a.stops.destination.state_text
    ):
        out.append(
            _f(
                "CRITICAL_FIELD_UNUSABLE",
                Severity.BLOCK,
                ("destination",),
                "No usable delivery stop.",
            )
        )
    if a.stated_total is None:
        out.append(
            _f("CRITICAL_FIELD_UNUSABLE", Severity.BLOCK, ("total_rate",), "No total rate printed.")
        )
    elif a.stated_total <= 0:
        # A present-but-non-positive value is not "missing"; it gets the same
        # code but a distinct message so the two are still separable downstream.
        out.append(
            _f(
                "CRITICAL_FIELD_UNUSABLE",
                Severity.BLOCK,
                ("total_rate",),
                f"Total rate is not positive ({a.stated_total}).",
            )
        )
    return out


def _usable(city: str | None, state: str | None) -> bool:
    """A whitespace-only span is missing, not present. Without the strip, the
    schema's own invariant — city and state are non-null — is satisfied by
    `Address(city="", state="")`, which is worse than `None` because it publishes
    as a real address.
    """
    return bool(city and city.strip() and state and state.strip())


# A numbered stop row: the leading integer is what distinguishes a real row from
# the prose headings every rate con carries ("Shipper Instructions", "Carrier
# Instructions"), which is why the number is required rather than optional.
_STOP_ROW_RE = re.compile(
    r"^[^\S\n]*(\d{1,2})[^\S\n]*[.)\-]?[^\S\n]+"
    r"(?:pick\s?-?up|pickup|drop\s?-?off|dropoff|drop|deliver(?:y)?|stop)\b",
    re.IGNORECASE | re.MULTILINE,
)

# A line carrying an amount is a charge line, not a stop row. Without this,
# `2  Stop Off Charge   $100.00` in the rate table counts as a third stop and
# BLOCKs every gating field on a correct two-stop extraction.
_LINE_HAS_MONEY = re.compile(r"\$|\busd\b|\bcad\b|\d\.\d{2}\b", re.IGNORECASE)


def stop_count_mismatch(a: Assembled) -> list[Finding]:
    """Recall. Did the model return every stop the document printed?

    Every other check in this file asks whether a returned value is real. None of
    them asks whether a value is *missing*, and omission is the failure mode with
    the worst incentives attached: dropping a stop also drops `MULTI_STOP` and
    `ORIGIN_DATE_DISAGREEMENT`, so a two-of-three extraction publishes a wrong
    origin city and a wrong pickup date at `high` with no findings at all, while
    the faithful reading of the same document scores `medium`. Deleting data
    must never improve the tier.

    Conservative by construction, and every part of that is load-bearing:

    * The pattern demands a leading row number, so the prose headings every rate
      con carries are not miscounted.
    * Lines carrying an amount are dropped, because `2  Stop Off Charge $100.00`
      in the rate table is a charge line, not a third stop.
    * The count is of *distinct* row numbers, never of the largest one. A TMS
      that numbers stops 10/20/30 is describing a numbering scheme, not thirty
      stops; and a template that repeats `1  Pickup` in a summary and again in
      the detail block should count once rather than twice.

    All three failure modes push in the same direction — under-count rather than
    over-count — because a missed omission costs one undetected error while a
    false BLOCK on every gating field costs a human review on every clean
    document of that template.
    """
    rows = [
        n
        for line in a.source_text.splitlines()
        if not _LINE_HAS_MONEY.search(line)
        for n in _STOP_ROW_RE.findall(line + "\n")
    ]
    if not rows:
        return []
    printed = len({int(n) for n in rows})
    returned = len(a.llm.stops)
    if printed <= returned:
        return []
    return [
        _f(
            "STOP_COUNT_MISMATCH",
            Severity.BLOCK,
            ("origin", "destination", "pickup_date", "delivery_date"),
            f"Document prints {printed} numbered stop rows; the model returned {returned}. "
            "The published lane and dates may belong to the wrong stops.",
        )
    ]


def not_grounded(a: Assembled) -> list[Finding]:
    """Does each published value actually appear in the document?

    The check runs on the model's *verbatim span*, never on the published value:
    the published pickup date is `2026-07-30` and no rate confirmation contains
    that string — the document says `07/30/2026`. Asserting the normalised value
    would block every document ever processed.

    Addresses are grounded on the parts we *publish* — city and ZIP against the
    whole document, state against the stop's own address span. Two earlier
    designs were both wrong. Checking only the parts against `address_text`
    proves the model was internally self-consistent and nothing more, so a
    wholly invented lane passed: the parts of a fabricated address agree with
    each other perfectly. Checking the whole `address_text` against the document
    then failed on every real PDF, because layout-preserved text interleaves the
    address column with the date column — the provided samples print
    "Illinois State Police, 100 | Shipping Date & Time" on one line — so the
    address is never contiguous in the source even when it is entirely correct.
    City and ZIP are distinctive enough to be found wherever they wrapped to,
    and they are what actually reaches the consumer. The state stays scoped to
    the address span because two-letter codes collide with ordinary prose.

    Worth stating plainly in the README: this is strong against a hallucinated
    value and weak against the error that actually costs money — reading the
    customer rate instead of the carrier rate, or lifting the broker's own city
    out of the letterhead, where the wrong value is also verbatim in the
    document. `MULTIPLE_TOTALS` exists for the first of those.
    """
    out: list[Finding] = []

    # `span.strip()` throughout: a whitespace-only span publishes as `None`, so
    # blocking a field for it docks confidence over a value nobody will ever see.
    for field, span in (
        ("load_id", a.llm.load_id_text),
        ("commodity", a.commodity_text),
        ("equipment_type", a.llm.equipment_text),
    ):
        if span and span.strip() and not normalize.contains_token(a.source_norm, span):
            out.append(
                _f("NOT_GROUNDED", Severity.BLOCK, (field,), f"'{span}' is not in the document.")
            )

    if a.stated_total is not None and not normalize.money_in_source(a.source_text, a.stated_total):
        out.append(
            _f(
                "NOT_GROUNDED",
                Severity.BLOCK,
                ("total_rate",),
                f"{a.stated_total} is not printed as an amount anywhere.",
            )
        )

    for field, stop in (("origin", a.stops.origin), ("destination", a.stops.destination)):
        if stop is None:
            continue
        address_norm = normalize.norm_text(stop.address_text or "")
        checks = (
            (stop.city_text, a.source_norm, "the document"),
            (stop.zip_text, a.source_norm, "the document"),
            (stop.state_text, address_norm or a.source_norm, f"the {field} address"),
        )
        for part, haystack, where in checks:
            if part and part.strip() and not normalize.contains_token(haystack, part):
                out.append(
                    _f(
                        "NOT_GROUNDED",
                        Severity.BLOCK,
                        (field,),
                        f"The {field} '{part}' does not appear in {where}.",
                    )
                )
                break

    # The date *tokens* inside the span, not the span itself — for exactly the
    # reason the address branch above does not require a contiguous address. A
    # model returning "03/16/2026 08:00" for a stop whose date and appointment
    # time sit in different columns has read the page correctly, and the joined
    # string appears nowhere in it.
    for field, stop in (("pickup_date", a.stops.origin), ("delivery_date", a.stops.destination)):
        span = stop.date_text if stop is not None else None
        if not span or not span.strip():
            continue
        needles = normalize.date_tokens(span) or [span]
        missing = [n for n in needles if not normalize.contains_token(a.source_norm, n)]
        if missing:
            out.append(
                _f(
                    "NOT_GROUNDED",
                    Severity.BLOCK,
                    (field,),
                    f"The {field} '{missing[0]}' is not printed on the document.",
                )
            )

    return out


def multiple_totals(a: Assembled) -> list[Finding]:
    """TMS-generated rate confirmations frequently print both sides of the rate
    table — the sample's own label is *Base **Carrier** Rate*, a name that only
    exists because a customer rate lives in the same schema. Taking the customer
    total pays away the entire margin on the load, and nobody notices, because
    an error that overpays the carrier generates no phone call.

    Two total-like labels carrying the *same* amount are not a conflict; they are
    "Total" and "Amount Due" on the same line of the same table, which is an
    ordinary layout. Counting labels rather than distinct amounts blocked those
    documents down to `low` for no reason.
    """
    amounts = {amount for _, amount in a.charges.totals}
    if len(amounts) > 1:
        labels = ", ".join(f"{label} {amount}" for label, amount in a.charges.totals)
        return [
            _f(
                "MULTIPLE_TOTALS",
                Severity.BLOCK,
                ("total_rate",),
                f"Document prints {len(amounts)} different total amounts ({labels}); "
                "carrier-side total must be confirmed.",
            )
        ]
    return []


def charges_dont_reconcile(a: Assembled) -> list[Finding]:
    """Unexplained residual only. `line_haul + fuel != total` is *usually correct*
    in freight — rate confirmations routinely carry detention, lumper, layover,
    TONU and stop-off lines — so a naive "totals must reconcile" rule would flag
    a large share of real loads.

    The block lands on the decomposition, not on `total_rate`. The printed total
    is the most reliable number on the document and the contractual figure the
    carrier is paid; the most likely cause of a residual is a charge line we
    failed to read, in which case the total was right all along.

    An all-in document with no component lines has `residual is None` and never
    reaches here — see `summarise_charges`.
    """
    residual = a.charges.residual
    if residual is None or abs(residual) <= Decimal("0.01"):
        return []

    identified = (
        (a.charges.line_haul or Decimal(0))
        + (a.charges.fuel or Decimal(0))
        + a.charges.accessorials
        + sum((amount for _, amount in a.charges.unmapped), Decimal(0))
    )
    out = [
        _f(
            "CHARGES_DONT_RECONCILE",
            Severity.BLOCK,
            ("line_haul_rate", "fuel_surcharge"),
            f"{residual} of the stated total is unaccounted for by any charge line.",
        )
    ]

    # Magnitude decides whether the total itself survives. A small residual is a
    # charge line we failed to read, and the printed total is still the number
    # the carrier is paid. But once the unexplained part exceeds everything we
    # *could* identify, the breakdown no longer corroborates the total at all —
    # which is also what a total injected into the document looks like.
    if abs(residual) > abs(identified):
        out.append(
            _f(
                "CHARGES_DONT_RECONCILE",
                Severity.BLOCK,
                ("total_rate",),
                f"Stated total {a.stated_total} is not corroborated by the breakdown: "
                f"only {identified} is accounted for.",
            )
        )
    else:
        out.append(
            _f(
                "CHARGES_DONT_RECONCILE",
                Severity.FLAG,
                ("total_rate",),
                f"Printed total kept as-is; {residual} is unexplained.",
            )
        )
    return out


def component_exceeds_total(a: Assembled) -> list[Finding]:
    """A single component larger than the whole. Arithmetic that cannot be true.

    Cheap, and it catches a class the residual check cannot: a mislabelled total
    folded into the line-haul line reconciles *worse*, not better, but a
    document where the components merely sum wrong may still have a component
    the size of two loads. The block lands on the decomposition, for the same
    reason as above — the printed total is the contractual figure.
    """
    if a.stated_total is None or a.stated_total <= 0:
        return []
    # A quick-pay deduction, a fuel advance or a ComCheck fee is a negative line,
    # and any of them makes line haul legitimately exceed the total on a document
    # that is entirely correct. The invariant only holds when every line is
    # positive, so a single negative amount stands the rule down.
    if any((normalize.parse_money(c.amount_text) or Decimal(0)) < 0 for c in a.llm.charges):
        return []
    out: list[Finding] = []
    for field, value in (
        ("line_haul_rate", a.charges.line_haul),
        ("fuel_surcharge", a.charges.fuel),
    ):
        if value is not None and value > a.stated_total:
            out.append(
                _f(
                    "COMPONENT_EXCEEDS_TOTAL",
                    Severity.BLOCK,
                    (field,),
                    f"{field} {value} exceeds the stated total {a.stated_total}.",
                )
            )
    return out


def unmapped_charge(a: Assembled) -> list[Finding]:
    """A charge we read correctly but cannot classify into the three-slot schema.

    Arithmetic still closes, so the total is trustworthy and this is advisory —
    unlike an unexplained residual, which means money is missing entirely.
    """
    if not a.charges.unmapped:
        return []
    desc = ", ".join(f"{label} {amount}" for label, amount in a.charges.unmapped)
    return [
        _f(
            "UNMAPPED_CHARGE",
            Severity.FLAG,
            ("total_rate",),
            f"Unclassified charge line(s): {desc}. Not mapped to fuel — no fuel label present.",
        )
    ]


def date_unresolved(a: Assembled) -> list[Finding]:
    """Ambiguity and absence are separate codes.

    `DATE_MISSING` used to be emitted as `CRITICAL_FIELD_UNUSABLE` from here,
    which made the two indistinguishable downstream — and Part 2's whole
    monitoring design is per-(template, field, failure mode) slicing, which a
    shared code quietly breaks.
    """
    out: list[Finding] = []
    for field, reading in (("pickup_date", a.pickup), ("delivery_date", a.delivery)):
        if reading.status == "ambiguous":
            out.append(
                _f(
                    "DATE_UNRESOLVED",
                    Severity.BLOCK,
                    (field,),
                    f"Reads as {reading.value} or {reading.alternative}; "
                    "no other date in the document settles the order.",
                )
            )
        elif reading.status == "unparseable":
            out.append(
                _f("DATE_UNPARSEABLE", Severity.BLOCK, (field,), f"Cannot read a date for {field}.")
            )
        elif reading.status == "absent":
            out.append(_f("DATE_MISSING", Severity.BLOCK, (field,), f"No {field} printed."))
    return out


def date_order_invalid(a: Assembled) -> list[Finding]:
    """Relational, so it impugns both dates rather than picking one to blame."""
    if a.pickup.value and a.delivery.value and a.delivery.value < a.pickup.value:
        return [
            _f(
                "DATE_ORDER_INVALID",
                Severity.BLOCK,
                ("pickup_date", "delivery_date"),
                f"Delivery {a.delivery.value} precedes pickup {a.pickup.value}.",
            )
        ]
    return []


def multi_stop(a: Assembled) -> list[Finding]:
    """More than two stops. The extraction is correct; the *schema* is lossy —
    one origin and one destination cannot represent a two-pickup load. Worth
    saying in the README that the contract should grow a `stops[]` array.
    """
    if len(a.llm.stops) > 2:
        return [
            _f(
                "MULTI_STOP",
                Severity.FLAG,
                ("origin", "destination"),
                f"{len(a.llm.stops)} stops collapsed into one origin/destination pair.",
            )
        ]
    return []


def multi_commodity(a: Assembled) -> list[Finding]:
    """Same category as `MULTI_STOP`: the reading is right, the contract is lossy.

    The published `commodity` and `weight_lbs` are the first line's, and nothing
    else is published. Weights are deliberately *not* summed — commodity rows
    repeat per stop on these documents (the provided samples print the same
    182 lb line under both the pickup and the drop), so adding them would double
    the load rather than describe it.
    """
    descriptions = {
        normalize.norm_text(c.description_text)
        for c in a.llm.commodities
        if c.description_text and c.description_text.strip()
    }
    if len(descriptions) <= 1:
        return []
    return [
        _f(
            "MULTI_COMMODITY",
            Severity.FLAG,
            ("commodity", "weight_lbs"),
            f"{len(descriptions)} distinct commodities printed; published the first "
            f"('{a.commodity_text}') and its weight only.",
        )
    ]


def origin_date_disagreement(a: Assembled) -> list[Finding]:
    """The header pickup date names a different stop than the one we selected.

    Not treated as a contradiction when it matches *some* stop: on multi-pickup
    loads a header naming the primary pickup is normal. It only means the
    document is ambiguous about which stop is primary, which is exactly the kind
    of thing a human should confirm rather than the machine decide.

    The header span is grounded first. Without that, a hallucinated header date
    matches no stop and blocks a document that is otherwise perfect — the rule
    would be reporting the model's invention as a property of the document.
    """
    span = a.llm.header_pickup_date_text
    if span and not normalize.contains_token(a.source_norm, span):
        return [
            _f(
                "NOT_GROUNDED",
                Severity.FLAG,
                ("pickup_date",),
                f"Header pickup date '{span}' is not printed on the document; ignored.",
            )
        ]
    header = normalize.resolve_date(span, a.source_text)
    if header.value is None or a.pickup.value is None or header.value == a.pickup.value:
        return []
    stop_dates = {normalize.resolve_date(s.date_text, a.source_text).value for s in a.llm.stops}
    if header.value in stop_dates:
        return [
            _f(
                "ORIGIN_DATE_DISAGREEMENT",
                Severity.FLAG,
                ("pickup_date",),
                f"Header pickup date {header.value} belongs to a different stop; "
                f"published {a.pickup.value} from the first pickup.",
            )
        ]
    return [
        _f(
            "ORIGIN_DATE_DISAGREEMENT",
            Severity.BLOCK,
            ("pickup_date",),
            f"Header pickup date {header.value} matches no stop on the document.",
        )
    ]


def equipment_unmapped(a: Assembled) -> list[Finding]:
    if a.llm.equipment_text and not a.equipment_confident:
        return [
            _f(
                "EQUIPMENT_UNMAPPED",
                Severity.FLAG,
                ("equipment_type",),
                f"'{a.llm.equipment_text}' has no unambiguous mapping to "
                "van/reefer/flatbed; recorded as other.",
            )
        ]
    return []


def weight_implausible(a: Assembled) -> list[Finding]:
    """A free sanity band, no dataset required. US FTL legal payload is roughly
    42,000-45,000 lb against an 80,000 lb gross limit.

    The message carries whether a unit was printed, because that changes what the
    reviewer is being asked. With a unit, an out-of-band weight is a real
    property of the load — a permit job, or a partial. Without one we assumed
    pounds, and the same number reading as kilograms is a 2.2x different load,
    so the reviewer is being asked about the *unit* rather than the freight.
    """
    if a.weight_lbs is None:
        # A span carrying digits that we refused to read — most often because it
        # fused two table cells. Publishing `weight_lbs: null` with no finding
        # would make a refusal indistinguishable from a document that prints no
        # weight at all, which is the same conflation `status` vs `confidence`
        # exists to avoid.
        #
        # The digit test matters: the real model returns `"-"` for sample B,
        # because that is literally what the Weight column prints. A dash is the
        # document saying "not stated", not a span we failed to parse, and
        # flagging it made a correct read look like a refusal.
        if a.weight_text and any(ch.isdigit() for ch in a.weight_text):
            return [
                _f(
                    "WEIGHT_UNREADABLE",
                    Severity.FLAG,
                    ("weight_lbs",),
                    f"Weight span '{a.weight_text}' is not a single number; not published.",
                )
            ]
        return []
    if a.weight_lbs > Decimal(45000):
        msg = f"{a.weight_lbs} lb exceeds legal FTL payload; permit or overweight load."
    elif a.weight_lbs < Decimal(1000):
        msg = f"{a.weight_lbs} lb is implausibly light for a truckload."
    else:
        return []
    if not a.weight_unit_stated:
        msg += f" No unit was printed alongside '{a.weight_text}'; pounds assumed."
    return [_f("WEIGHT_IMPLAUSIBLE", Severity.FLAG, ("weight_lbs",), msg)]


ALL_RULES = (
    not_a_rate_con,
    critical_field_unusable,
    stop_count_mismatch,
    not_grounded,
    multiple_totals,
    charges_dont_reconcile,
    component_exceeds_total,
    unmapped_charge,
    date_unresolved,
    date_order_invalid,
    multi_stop,
    multi_commodity,
    origin_date_disagreement,
    equipment_unmapped,
    weight_implausible,
)


def audit(a: Assembled) -> list[Finding]:
    """Run every rule. `NOT_A_RATE_CON` short-circuits: once the document is the
    wrong kind, field-level findings about it are noise.
    """
    gate = not_a_rate_con(a)
    if gate:
        return gate
    out: list[Finding] = []
    for rule in ALL_RULES[1:]:
        out.extend(rule(a))
    return out
