"""The audit. Every rule is a pure function over `Assembled`, so the whole
confidence engine unit-tests with no LLM and no network.

Two design rules hold throughout:

* A finding names the field it impugns. A rule that cannot name one is either
  record-scoped (`fields=()`) or relational, in which case it impugns every
  field it involves — a delivery-before-pickup check that blamed only the
  delivery date would let a clearly broken document through, because the
  delivery date alone does not decide the tier.
* A BLOCK never mutates the data. It marks the field unusable and leaves the
  value in place, so an inconsistency in one field can never destroy another
  field's perfectly good value.
"""

from dataclasses import dataclass
from decimal import Decimal

from ratecon import normalize
from ratecon.normalize import ChargeBreakdown, DateReading, StopSelection
from ratecon.schema import EquipmentType, Finding, LlmExtraction, Severity

# The fields that decide whether a load can be created without a human. Weight,
# commodity, load_id and equipment are recorded and flagged but do not gate:
# they are not what makes a booking wrong in a way that costs money on the day.
GATING_FIELDS = frozenset({"total_rate", "pickup_date", "delivery_date", "origin", "destination"})

# Fields whose published value comes from a single verbatim span and can
# therefore be checked against the source. Derived values are deliberately
# absent: `line_haul_rate` may be a sum of two lines, `fuel_surcharge` is often
# a correct null, and `weight_lbs` may have been unit-converted. None of those
# has a span to find, so asserting one would fail on correct extractions.
GROUNDABLE = ("load_id", "total_rate", "origin", "destination", "commodity")


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
    (no charge lines at all) has to agree before we reject the document.
    """
    said_no = a.llm.document_type != "rate_confirmation"
    no_money = a.charges.line_haul is None and a.charges.fuel is None and a.stated_total is None
    if said_no and no_money:
        return [
            _f(
                "NOT_A_RATE_CON",
                Severity.BLOCK,
                (),
                f"Document looks like a {a.llm.document_type} and carries no rate breakdown.",
            )
        ]
    return []


def critical_field_unusable(a: Assembled) -> list[Finding]:
    out: list[Finding] = []
    if a.stops.origin is None or not (a.stops.origin.city_text and a.stops.origin.state_text):
        out.append(
            _f("CRITICAL_FIELD_UNUSABLE", Severity.BLOCK, ("origin",), "No usable pickup stop.")
        )
    if a.stops.destination is None or not (
        a.stops.destination.city_text and a.stops.destination.state_text
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


def not_grounded(a: Assembled) -> list[Finding]:
    """Does each published value actually appear in the document?

    The check runs on the model's *verbatim span*, never on the published value:
    the published pickup date is `2026-07-30` and no rate confirmation contains
    that string — the document says `07/30/2026`. Asserting the normalised value
    would block every document ever processed.

    Worth stating plainly in the README: this is strong against a hallucinated
    value and weak against the error that actually costs money — reading the
    customer rate instead of the carrier rate, where the wrong number is also
    verbatim in the source. `MULTIPLE_TOTALS` exists for that.
    """
    out: list[Finding] = []

    if a.llm.load_id_text and not normalize.contains_token(a.source_norm, a.llm.load_id_text):
        out.append(
            _f(
                "NOT_GROUNDED",
                Severity.BLOCK,
                ("load_id",),
                f"'{a.llm.load_id_text}' is not in the document.",
            )
        )

    if a.stated_total is not None and not normalize.money_in_source(a.source_norm, a.stated_total):
        out.append(
            _f(
                "NOT_GROUNDED",
                Severity.BLOCK,
                ("total_rate",),
                f"{a.stated_total} is not printed anywhere.",
            )
        )

    for field, stop in (("origin", a.stops.origin), ("destination", a.stops.destination)):
        if stop is None or not stop.address_text:
            continue
        address_norm = normalize.norm_text(stop.address_text)
        # City/state/zip are checked against the stop's own address span, not the
        # whole document: two-letter codes match by accident far too easily.
        for part in (stop.city_text, stop.state_text, stop.zip_text):
            if part and not normalize.contains_token(address_norm, part):
                out.append(
                    _f(
                        "NOT_GROUNDED",
                        Severity.BLOCK,
                        (field,),
                        f"'{part}' does not appear in the {field} address.",
                    )
                )
                break
    return out


def multiple_totals(a: Assembled) -> list[Finding]:
    """TMS-generated rate confirmations frequently print both sides of the rate
    table — the sample's own label is *Base **Carrier** Rate*, a name that only
    exists because a customer rate lives in the same schema. Taking the customer
    total pays away the entire margin on the load, and nobody notices, because
    an error that overpays the carrier generates no phone call.
    """
    labels = [
        c.label_text
        for c in a.llm.charges
        if c.label_text and normalize.classify_charge(c.label_text) == "total"
    ]
    if len(labels) > 1:
        return [
            _f(
                "MULTIPLE_TOTALS",
                Severity.BLOCK,
                ("total_rate",),
                f"Document prints {len(labels)} total-like lines ({', '.join(labels)}); "
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
        elif reading.status in ("unparseable", "absent"):
            out.append(
                _f("CRITICAL_FIELD_UNUSABLE", Severity.BLOCK, (field,), f"No usable {field}.")
            )
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


def origin_date_disagreement(a: Assembled) -> list[Finding]:
    """The header pickup date names a different stop than the one we selected.

    Not treated as a contradiction when it matches *some* stop: on multi-pickup
    loads a header naming the primary pickup is normal. It only means the
    document is ambiguous about which stop is primary, which is exactly the kind
    of thing a human should confirm rather than the machine decide.
    """
    header = normalize.resolve_date(a.llm.header_pickup_date_text, a.source_text)
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
    """
    if a.weight_lbs is None:
        return []
    if a.weight_lbs > Decimal(45000):
        msg = f"{a.weight_lbs} lb exceeds legal FTL payload; permit or overweight load."
    elif a.weight_lbs < Decimal(1000):
        msg = f"{a.weight_lbs} lb is implausibly light for a truckload."
    else:
        return []
    return [_f("WEIGHT_IMPLAUSIBLE", Severity.FLAG, ("weight_lbs",), msg)]


ALL_RULES = (
    not_a_rate_con,
    critical_field_unusable,
    not_grounded,
    multiple_totals,
    charges_dont_reconcile,
    unmapped_charge,
    date_unresolved,
    date_order_invalid,
    multi_stop,
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
