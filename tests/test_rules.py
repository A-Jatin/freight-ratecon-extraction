"""Every rule is a pure function, so none of this needs a model or a network."""

from decimal import Decimal

from factories import DEFAULT_SOURCE, DELIVERY, codes, make_assembled, make_stop
from ratecon import rules
from ratecon.pipeline import route
from ratecon.schema import Charge, Confidence, Severity


def test_clean_document_produces_no_findings():
    assert rules.audit(make_assembled()) == []


def test_missing_total_blocks_a_gating_field():
    a = make_assembled(stated_total_text=None, charges=[])
    assert "CRITICAL_FIELD_UNUSABLE" in codes(rules.audit(a))
    assert route(rules.audit(a))[0] is Confidence.LOW


def test_non_positive_total_is_distinct_from_a_missing_one():
    """A present-but-negative value is not 'missing'; the message has to differ
    or the Part 2 slicing by failure mode is lying."""
    a = make_assembled(
        stated_total_text="-100.00",
        charges=[Charge(label_text="Line Haul", amount_text="-100.00")],
    )
    found = [f for f in rules.audit(a) if f.code == "CRITICAL_FIELD_UNUSABLE"]
    assert any("not positive" in f.message for f in found)


def test_hallucinated_load_id_is_caught():
    a = make_assembled(load_id_text="ML-000000")
    assert "NOT_GROUNDED" in codes(rules.audit(a))


def test_grounding_runs_on_the_verbatim_span_not_the_published_value():
    """The published pickup date is `2026-03-16` and no rate confirmation on
    earth contains that string - the document says `03/16/2026`. Checking the
    normalised value would block every document ever processed."""
    a = make_assembled()
    assert a.pickup.value is not None
    assert a.pickup.value.isoformat() not in a.source_norm
    assert rules.not_grounded(a) == []


def test_multi_stop_flags_both_lane_ends_without_blocking():
    """The extraction is right; the assignment's schema is what cannot represent
    a two-pickup load. Docking confidence for a spec limitation would be a
    category error, so it flags rather than blocks."""
    a = make_assembled(
        stops=[
            make_stop(1),
            make_stop(2, when="03/17/2026"),
            DELIVERY.model_copy(update={"sequence": 3}),
        ]
    )
    findings = rules.multi_stop(a)
    assert findings[0].severity is Severity.FLAG
    assert set(findings[0].fields) == {"origin", "destination"}


def test_unexplained_residual_blocks_the_breakdown_but_keeps_the_printed_total():
    """The printed total is the contractual figure the carrier is paid, and the
    likeliest cause of a residual is a charge line we failed to read - in which
    case the total was right all along. So the block lands on the decomposition."""
    a = make_assembled(
        stated_total_text="1900.00",
        charges=[
            Charge(label_text="Line Haul", amount_text="1500.00"),
            Charge(label_text="Fuel Surcharge", amount_text="300.00"),
        ],
    )
    findings = rules.charges_dont_reconcile(a)
    blocked = {f for f in findings if f.severity is Severity.BLOCK}
    assert {f.fields for f in blocked} == {("line_haul_rate", "fuel_surcharge")}
    assert a.stated_total == Decimal("1900.00")  # untouched


def test_explained_residual_is_advisory_only():
    a = make_assembled(
        stated_total_text="3900.00",
        charges=[
            Charge(label_text="Base Carrier Rate", amount_text="3400.00"),
            Charge(label_text="Carrier Charge", amount_text="500.00"),
        ],
    )
    assert rules.charges_dont_reconcile(a) == []
    assert rules.unmapped_charge(a)[0].severity is Severity.FLAG


def test_fuel_surcharge_is_never_imputed():
    """Deriving it as total - line_haul is the same hallucination performed with
    a calculator, and it would always reconcile."""
    a = make_assembled(
        stated_total_text="3900.00",
        charges=[
            Charge(label_text="Base Carrier Rate", amount_text="3400.00"),
            Charge(label_text="Carrier Charge", amount_text="500.00"),
        ],
    )
    assert a.charges.fuel is None


def test_two_total_lines_block_the_rate():
    """TMS documents often print both the customer and the carrier side. Taking
    the customer total pays away the whole margin, silently."""
    a = make_assembled(
        charges=[
            Charge(label_text="Customer Rate", amount_text="2600.00"),
            Charge(label_text="Total", amount_text="2150.00"),
        ]
    )
    findings = rules.multiple_totals(a)
    assert findings[0].severity is Severity.BLOCK
    assert findings[0].fields == ("total_rate",)


def test_delivery_before_pickup_blocks_both_dates():
    """Relational, so it cannot name a single culprit. Blaming only the delivery
    date would leave the record at `high`, because one flagged non-gating field
    does not move the tier."""
    a = make_assembled(
        stops=[
            make_stop(1, when="03/18/2026"),
            DELIVERY.model_copy(update={"date_text": "03/16/2026"}),
        ]
    )
    findings = rules.date_order_invalid(a)
    assert set(findings[0].fields) == {"pickup_date", "delivery_date"}
    assert route(rules.audit(a))[0] is Confidence.LOW


def test_header_date_matching_another_stop_is_a_flag_not_a_contradiction():
    """On multi-pickup loads a header naming the primary pickup is normal."""
    a = make_assembled(
        header_pickup_date_text="17-Mar-2026",
        stops=[
            make_stop(1),
            make_stop(2, when="03/17/2026"),
            DELIVERY.model_copy(update={"sequence": 3}),
        ],
    )
    findings = rules.origin_date_disagreement(a)
    assert findings[0].severity is Severity.FLAG


def test_header_date_matching_no_stop_blocks():
    a = make_assembled(header_pickup_date_text="09-Sep-2026")
    findings = rules.origin_date_disagreement(a)
    assert findings[0].severity is Severity.BLOCK


def test_ambiguous_equipment_is_flagged_never_coerced():
    a = make_assembled(equipment_text="Sprinter Van")
    assert "EQUIPMENT_UNMAPPED" in codes(rules.audit(a))


def test_weight_band():
    assert rules.weight_implausible(make_assembled(weight_text="60000"))
    assert rules.weight_implausible(make_assembled(weight_text="182"))
    assert not rules.weight_implausible(make_assembled(weight_text="38400"))


def test_a_bol_is_rejected_and_short_circuits():
    a = make_assembled(document_type="bol", stated_total_text=None, charges=[])
    findings = rules.audit(a)
    assert codes(findings) == {"NOT_A_RATE_CON"}


def test_stop_order_follows_the_printed_sequence_not_the_dates():
    """Stops are printed in routed order because that is the order the driver
    executes them; the per-stop dates are hand-entered and are what gets
    fat-fingered. Sorting by date would let one typo reorder the route."""
    a = make_assembled(
        stops=[
            make_stop(1, when="03/20/2026"),  # sequence 1, but the later date
            make_stop(
                2,
                address="Ozark Fabrication 1250 S Industrial Dr, Springfield, MO 65802, USA",
                city="Springfield",
                state="MO",
                zipc="65802",
                when="03/17/2026",
            ),
            DELIVERY.model_copy(update={"sequence": 3}),
        ]
    )
    assert a.stops.origin is not None
    assert a.stops.origin.city_text == "Cleveland"


def test_pickup_date_comes_from_the_selected_origin_stop():
    """Reading it from the header on a multi-pickup load publishes an origin city
    paired with a different stop's appointment date - a truck dispatched on that
    is a thousand miles from where it should be."""
    a = make_assembled(
        header_pickup_date_text="20-Mar-2026",
        stops=[
            make_stop(1, when="03/17/2026"),
            make_stop(
                2,
                address="Ozark Fabrication 1250 S Industrial Dr, Springfield, MO 65802, USA",
                city="Springfield",
                state="MO",
                zipc="65802",
                when="03/20/2026",
            ),
            DELIVERY.model_copy(update={"sequence": 3, "date_text": "03/25/2026"}),
        ],
        source=DEFAULT_SOURCE + "\n2 Pickup Springfield, MO 65802  03/20/2026\n",
    )
    assert a.stops.origin is not None
    assert a.stops.origin.city_text == "Cleveland"
    assert a.pickup.value is not None
    # The origin stop's own date, not the header's 20-Mar (which belongs to the
    # second pickup) and not the second pickup's date either.
    assert a.pickup.value.isoformat() == "2026-03-17"


# --------------------------------------------------------------------------
# The routing ladder
# --------------------------------------------------------------------------


def test_routing_ladder():
    from ratecon.schema import Finding

    def f(sev: Severity, *fields: str) -> Finding:
        return Finding(code="X", severity=sev, fields=fields, message="")

    assert route([])[0] is Confidence.HIGH
    assert route([f(Severity.FLAG, "commodity")])[0] is Confidence.HIGH
    assert route([f(Severity.FLAG, "origin")])[0] is Confidence.MEDIUM
    # A block on a non-gating field still costs a tier: otherwise a record with a
    # blocked load_id could be published as fully trustworthy.
    assert route([f(Severity.BLOCK, "load_id")])[0] is Confidence.MEDIUM
    assert route([f(Severity.BLOCK, "total_rate")])[0] is Confidence.LOW
    # Record-scoped findings hit everything.
    assert route([f(Severity.BLOCK)])[0] is Confidence.LOW


def test_block_marks_the_field_without_erasing_the_value():
    a = make_assembled(load_id_text="ML-000000")
    findings = rules.audit(a)
    _, status = route(findings)
    assert status["load_id"] == "blocked"
