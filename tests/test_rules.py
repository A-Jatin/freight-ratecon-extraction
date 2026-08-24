"""Every rule is a pure function, so none of this needs a model or a network."""

from decimal import Decimal

import pytest

from factories import (
    DEFAULT_SOURCE,
    DELIVERY,
    codes,
    make_assembled,
    make_commodity,
    make_stop,
    severities,
)
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


@pytest.mark.parametrize("end", ["origin", "destination"])
def test_a_whitespace_only_city_is_missing_not_present(end):
    """Without the strip, the schema's own invariant — city and state are
    non-null — is satisfied by `Address(city="", state="")`, which is worse than
    `None` because it publishes as a real address.

    Parametrised over both ends because the destination branch is a separate
    block of code: with only the origin covered, deleting the destination check
    outright left the suite green.
    """
    stops = (
        [make_stop(city="   "), DELIVERY]
        if end == "origin"
        else [make_stop(), DELIVERY.model_copy(update={"city_text": "   "})]
    )
    found = rules.critical_field_unusable(make_assembled(stops=stops))
    assert ("CRITICAL_FIELD_UNUSABLE", end, Severity.BLOCK) in severities(found)


# --------------------------------------------------------------------------
# Recall
# --------------------------------------------------------------------------


def test_a_dropped_stop_is_caught_rather_than_rewarded():
    """The whole reason this rule exists. Every other check asks whether a
    returned value is real; none of them asks whether one is missing — and
    dropping a stop *also* drops MULTI_STOP and ORIGIN_DATE_DISAGREEMENT, so
    before this rule the truncated reading outscored the faithful one.
    """
    source = DEFAULT_SOURCE.replace(
        "2 Drop", "2 Pickup  Ozark Fabrication 1250 S Industrial Dr, Springfield, MO\n3 Drop"
    )
    faithful = make_assembled(
        source=source,
        stops=[
            make_stop(1),
            make_stop(2, when="03/17/2026"),
            DELIVERY.model_copy(update={"sequence": 3}),
        ],
    )
    truncated = make_assembled(source=source, stops=[make_stop(2, when="03/17/2026"), DELIVERY])

    assert rules.stop_count_mismatch(faithful) == []
    assert "STOP_COUNT_MISMATCH" in codes(rules.stop_count_mismatch(truncated))
    # The property that matters, stated as a property: deleting data must never
    # improve the tier. Before this rule the truncated reading scored `high`
    # with no findings while the faithful one scored `medium`.
    rank = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
    assert rank[route(rules.audit(truncated))[0]] < rank[route(rules.audit(faithful))[0]]


def test_stop_count_ignores_prose_headings():
    """ "Shipper Instructions" and "Carrier Instructions" appear at the start of a
    line on most rate cons. Counting them as stop rows would block ordinary
    documents, which is why the pattern demands a leading row number."""
    source = DEFAULT_SOURCE + "\nShipper Instructions (Sample)\nCarrier Instructions (Sample)\n"
    assert rules.stop_count_mismatch(make_assembled(source=source)) == []


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------


def test_hallucinated_load_id_is_caught():
    a = make_assembled(load_id_text="ML-000000")
    assert "NOT_GROUNDED" in codes(rules.audit(a))


def test_a_fabricated_lane_is_caught():
    """The parts of an invented address agree with each other perfectly, so
    checking city/state/zip against the model's own `address_text` proves only
    that the model was self-consistent. The address itself has to be found in
    the document."""
    a = make_assembled(
        stops=[
            make_stop(
                address="999 Nowhere Rd, Peoria, IL 61602", city="Peoria", state="IL", zipc="61602"
            ),
            DELIVERY.model_copy(
                update={
                    "address_text": "1 Fake Ave, Boise, ID 83702",
                    "city_text": "Boise",
                    "state_text": "ID",
                    "zip_text": "83702",
                }
            ),
        ]
    )
    found = rules.not_grounded(a)
    assert {f.fields for f in found} == {("origin",), ("destination",)}
    assert route(rules.audit(a))[0] is Confidence.LOW


def test_an_address_that_wraps_across_layout_columns_still_grounds():
    """The regression that made the first version of this rule unusable.
    Layout-preserved PDF text interleaves the address column with the date
    column, so the address is never contiguous in the source even when it is
    entirely correct — the provided samples print
    "Illinois State Police, 100 | Shipping Date & Time" on one line. Grounding
    the whole `address_text` blocked all three of them.
    """
    wrapped = (
        "Stops\n"
        "1  Pickup                       Appt Type\n"
        "   Illinois State Police, 100   Shipping Date & Time\n"
        "   W Randolph St, Chicago,      03/16/2026\n"
        "   IL 60601, USA\n"
        "2  Drop\n"
        "   Piedmont Grocery DC 2100 Westinghouse Blvd, Charlotte, NC 28273, USA\n"
        "   Delivery Date & Time 03/18/2026\n"
        "   S.No Commodity Weight\n"
        "   1    Canned Goods 38400\n"
        "Total 2150.00 USD\nEquipment Dry Van\nReference ID ML-884213\n"
    )
    a = make_assembled(
        source=wrapped,
        stops=[
            make_stop(
                address="Illinois State Police, 100 W Randolph St, Chicago, IL 60601, USA",
                city="Chicago",
                state="IL",
                zipc="60601",
            ),
            DELIVERY,
        ],
    )
    assert rules.not_grounded(a) == []


def test_a_fabricated_city_alone_is_caught():
    """The ZIP check covers the usual fabricated address, so this isolates the
    city: everything else about the stop is real and only the city is invented.
    Without a city check the lane publishes with a city nobody printed."""
    a = make_assembled(
        stops=[
            make_stop(
                address="Great Lakes Distribution 4400 W 130th St, Peoria, OH 44135, USA",
                city="Peoria",
            ),
            DELIVERY,
        ]
    )
    found = [f for f in rules.not_grounded(a) if f.fields == ("origin",)]
    assert found and "Peoria" in found[0].message


def test_a_fabricated_zip_alone_is_caught():
    """The mirror of the city case: real city, real state, invented ZIP. A ZIP
    is what a downstream system geocodes on, so it has to be checked in its own
    right rather than inheriting the city's verdict."""
    a = make_assembled(
        stops=[
            make_stop(
                address="Great Lakes Distribution 4400 W 130th St, Cleveland, OH 99999, USA",
                zipc="99999",
            ),
            DELIVERY,
        ]
    )
    found = [f for f in rules.not_grounded(a) if f.fields == ("origin",)]
    assert found and "99999" in found[0].message


def test_a_fabricated_date_is_caught():
    a = make_assembled(stops=[make_stop(when="12/25/2026"), DELIVERY])
    assert ("pickup_date",) in {f.fields for f in rules.not_grounded(a)}


@pytest.mark.parametrize("field", rules.GROUNDABLE)
def test_every_groundable_field_is_actually_grounded(field):
    """`GROUNDABLE` used to be a documented constant that nothing read, and two
    of the fields it named were never checked. This test is what stops it
    drifting back: each name has to be defended by a rule that fires."""
    hallucinated = {
        "load_id": {"load_id_text": "ML-000000"},
        "commodity": {"commodities": [make_commodity("Uranium Hexafluoride")]},
        "equipment_type": {"equipment_text": "Hovercraft"},
        "total_rate": {"stated_total_text": "9999.99 USD"},
        "origin": {"stops": [make_stop(address="1 Fake Ave, Nowhere, ZZ"), DELIVERY]},
        "destination": {
            "stops": [
                make_stop(),
                DELIVERY.model_copy(update={"address_text": "1 Fake Ave, Nowhere, ZZ"}),
            ]
        },
        "pickup_date": {"stops": [make_stop(when="12/25/2026"), DELIVERY]},
        "delivery_date": {
            "stops": [make_stop(), DELIVERY.model_copy(update={"date_text": "12/25/2026"})]
        },
    }[field]
    found = rules.not_grounded(make_assembled(**hallucinated))
    assert any(field in f.fields for f in found), f"{field} is in GROUNDABLE but nothing checks it"
    # BLOCK, not FLAG. A value that is not in the document is unusable, and the
    # severity is what decides the tier — asserting only the code let a
    # BLOCK-to-FLAG downgrade on `total_rate` pass the whole suite.
    assert ("NOT_GROUNDED", field, Severity.BLOCK) in severities(found)


def test_grounding_runs_on_the_verbatim_span_not_the_published_value():
    """The published pickup date is `2026-03-16` and no rate confirmation on
    earth contains that string - the document says `03/16/2026`. Checking the
    normalised value would block every document ever processed."""
    a = make_assembled()
    assert a.pickup.value is not None
    assert a.pickup.value.isoformat() not in a.source_norm
    assert rules.not_grounded(a) == []


def test_a_hallucinated_header_date_is_ignored_rather_than_blamed_on_the_document():
    """Without grounding the header span first, an invented header date matches
    no stop and blocks a document that is otherwise perfect — the rule would be
    reporting the model's invention as a property of the document."""
    a = make_assembled(header_pickup_date_text="09-Sep-2026")
    found = rules.origin_date_disagreement(a)
    assert codes(found) == {"NOT_GROUNDED"}
    assert found[0].severity is Severity.FLAG


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------


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
    assert (
        "CHARGES_DONT_RECONCILE",
        "line_haul_rate,fuel_surcharge",
        Severity.BLOCK,
    ) in severities(findings)
    assert ("CHARGES_DONT_RECONCILE", "total_rate", Severity.FLAG) in severities(findings)
    assert a.stated_total == Decimal("1900.00")  # untouched


def test_an_all_in_document_with_no_component_lines_reconciles():
    """A rate table printing one line reading `Total $1,800.00` has nothing to
    reconcile against. Treating the total as an unexplained residual blocked a
    perfectly ordinary document to `low`, and all-in quoting is the norm on spot
    broker-to-carrier freight."""
    a = make_assembled(
        stated_total_text="2150.00 USD",
        charges=[Charge(label_text="Total", amount_text="2150.00 USD")],
    )
    assert a.charges.residual is None
    assert rules.charges_dont_reconcile(a) == []
    assert route(rules.audit(a))[0] is Confidence.HIGH


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


def test_two_different_total_amounts_block_the_rate():
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


def test_two_total_labels_carrying_the_same_amount_are_not_a_conflict():
    """ "Total" and "Amount Due" on the same line of the same table is an ordinary
    layout. Counting labels rather than distinct amounts blocked those documents
    to `low` for no reason."""
    a = make_assembled(
        charges=[
            Charge(label_text="Total", amount_text="2150.00"),
            Charge(label_text="Amount Due", amount_text="2150.00"),
        ]
    )
    assert rules.multiple_totals(a) == []


def test_a_component_larger_than_the_total_is_blocked():
    a = make_assembled(
        stated_total_text="1800.00",
        charges=[
            Charge(label_text="Line Haul", amount_text="3300.00"),
            Charge(label_text="Total", amount_text="1800.00"),
        ],
    )
    assert "COMPONENT_EXCEEDS_TOTAL" in codes(rules.component_exceeds_total(a))


def test_a_negative_accessorial_can_legitimately_leave_a_component_above_the_total():
    """A quick-pay deduction makes line haul exceed the total on a document that
    is entirely correct, so the invariant only holds when every line is
    positive."""
    a = make_assembled(
        stated_total_text="1950.00",
        charges=[
            Charge(label_text="Line Haul", amount_text="2150.00"),
            Charge(label_text="Quick Pay Deduction", amount_text="(200.00)"),
            Charge(label_text="Total", amount_text="1950.00"),
        ],
    )
    assert rules.component_exceeds_total(a) == []


# --------------------------------------------------------------------------
# Dates, stops, equipment, weight
# --------------------------------------------------------------------------


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
    assert findings[0].severity is Severity.BLOCK
    assert route(rules.audit(a))[0] is Confidence.LOW


def test_a_missing_date_and_an_ambiguous_one_get_different_codes():
    """They used to share `CRITICAL_FIELD_UNUSABLE`, which makes them
    indistinguishable downstream — and Part 2's monitoring is per-(template,
    field, failure mode) slicing, which a shared code quietly breaks."""
    missing = make_assembled(stops=[make_stop(), DELIVERY.model_copy(update={"date_text": None})])
    assert "DATE_MISSING" in codes(rules.date_unresolved(missing))
    ambiguous = make_assembled(
        source="Rate Con Date 3/2/26 Shipping 3/4/26 Delivery 3/5/26",
        stops=[
            make_stop(when="3/4/26"),
            DELIVERY.model_copy(update={"date_text": "3/5/26"}),
        ],
    )
    assert "DATE_UNRESOLVED" in codes(rules.date_unresolved(ambiguous))


def test_header_date_matching_another_stop_is_a_flag_not_a_contradiction():
    """On multi-pickup loads a header naming the primary pickup is normal."""
    a = make_assembled(
        header_pickup_date_text="16-Mar-2026",
        stops=[
            make_stop(1, when="03/17/2026"),
            make_stop(2, when="03/16/2026"),
            DELIVERY.model_copy(update={"sequence": 3}),
        ],
    )
    findings = rules.origin_date_disagreement(a)
    assert findings[0].severity is Severity.FLAG


def test_ambiguous_equipment_is_flagged_never_coerced():
    a = make_assembled(equipment_text="Sprinter Van")
    assert "EQUIPMENT_UNMAPPED" in codes(rules.audit(a))


def test_weight_band():
    def with_weight(w):
        return make_assembled(commodities=[make_commodity("Canned Goods", w)])

    assert rules.weight_implausible(with_weight("60000"))
    assert rules.weight_implausible(with_weight("182"))
    assert not rules.weight_implausible(with_weight("38400"))


def test_the_weight_message_says_whether_a_unit_was_printed():
    """With a unit, an out-of-band weight is a property of the load. Without one
    we assumed pounds, and the same number read as kilograms is a 2.2x different
    load — so the reviewer is being asked a different question."""
    assumed = rules.weight_implausible(make_assembled(commodities=[make_commodity("X", "182")]))
    assert "pounds assumed" in assumed[0].message
    stated = rules.weight_implausible(make_assembled(commodities=[make_commodity("X", "82 kg")]))
    assert "pounds assumed" not in stated[0].message


def test_several_commodities_flag_the_collapse():
    """Same category as MULTI_STOP: the reading is right, the contract is lossy.
    Weights are deliberately not summed — the rows repeat per stop."""
    a = make_assembled(
        commodities=[make_commodity("Steel Coil", "21000"), make_commodity("Steel Plate", "18000")]
    )
    findings = rules.multi_commodity(a)
    assert findings[0].severity is Severity.FLAG
    assert a.weight_lbs == Decimal("21000")


def test_a_bol_is_rejected_and_short_circuits():
    a = make_assembled(document_type="bol", stated_total_text=None, charges=[])
    findings = rules.audit(a)
    assert codes(findings) == {"NOT_A_RATE_CON"}


def test_an_invoice_carrying_a_rate_is_still_rejected():
    """The conjunction used to require *no money at all*, so a customer-side
    tender or a carrier invoice extracted cleanly and published as a rate con."""
    a = make_assembled(document_type="invoice")
    assert codes(rules.audit(a)) == {"NOT_A_RATE_CON"}


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


def test_route_order_falls_back_wholesale_when_sequences_are_incomplete():
    """Mixing a printed sequence for some stops with a list index for the rest
    compares numbers from different scales: on a long document a missing
    sequence number outranked a real one."""
    stops = [
        DELIVERY.model_copy(update={"sequence": 9, "address_text": "A", "city_text": "First"}),
        *[make_stop(None) for _ in range(9)],
        DELIVERY.model_copy(update={"sequence": None, "address_text": "B", "city_text": "Last"}),
    ]
    from ratecon import normalize

    selection = normalize.select_stops(stops)
    assert selection.destination is not None
    assert selection.destination.city_text == "Last"


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
        source=DEFAULT_SOURCE
        + "\n2 Pickup Ozark Fabrication 1250 S Industrial Dr, Springfield, MO 65802, USA"
        "\n   Shipping Date & Time 03/20/2026\n   Delivery Date & Time 03/25/2026\n"
        "   Header Pickup Date 20-Mar-2026\n",
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


def test_field_status_says_ok_rather_than_staying_silent():
    """An empty `field_status` cannot distinguish "we checked and it is fine"
    from "no rule ever looked at this"."""
    _, status = route([])
    assert set(status) == set(rules.PUBLISHED_FIELDS)
    assert set(status.values()) == {"ok"}


def test_field_status_key_order_is_deterministic():
    """Record-scoped findings used to expand through a frozenset, so the key
    order varied between processes — noise in a diff and in a monitoring row."""
    from ratecon.schema import Finding

    blocked = [Finding(code="X", severity=Severity.BLOCK, fields=(), message="")]
    assert list(route(blocked)[1]) == list(route(blocked)[1])
    assert list(route(blocked)[1]) == list(rules.PUBLISHED_FIELDS)


def test_block_marks_the_field_without_erasing_the_value():
    a = make_assembled(load_id_text="ML-000000")
    findings = rules.audit(a)
    _, status = route(findings)
    assert status["load_id"] == "blocked"


def test_every_rule_is_reachable():
    """A rule nothing can trigger is a comment with a function signature."""
    assert len(rules.ALL_RULES) == 15
    assert len({r.__name__ for r in rules.ALL_RULES}) == len(rules.ALL_RULES)


# --------------------------------------------------------------------------
# Regressions introduced by the first round of fixes, and their guards
# --------------------------------------------------------------------------


def test_a_numbered_accessorial_line_is_not_counted_as_a_stop():
    """`3  Stop Off Charge  $100.00` in the rate table matched the stop-row
    pattern, giving three rows on a two-stop load and BLOCKing every gating
    field on a correct extraction. Numbered above the real stops on purpose, so
    the assertion fails if the money filter is removed."""
    source = DEFAULT_SOURCE + "\n3  Stop Off Charge   $100.00 USD\n4  Stop Charge  $50.00 USD\n"
    assert rules.stop_count_mismatch(make_assembled(source=source)) == []


def test_a_ten_twenty_thirty_numbering_scheme_is_not_thirty_stops():
    """Taking `max(int(n))` as the count read a TMS numbering scheme as a stop
    count, so a perfect two-stop extraction went to `low`."""
    source = (
        "Stops\n10  Pickup  Great Lakes Distribution 4400 W 130th St, Cleveland, OH 44135, USA\n"
        "20  Drop  Piedmont Grocery DC 2100 Westinghouse Blvd, Charlotte, NC 28273, USA\n"
    )
    assert rules.stop_count_mismatch(make_assembled(source=source)) == []


def test_a_repeated_summary_row_is_counted_once():
    source = DEFAULT_SOURCE + "\n1 Pickup  Great Lakes Distribution (summary)\n"
    assert rules.stop_count_mismatch(make_assembled(source=source)) == []


def test_a_date_span_carrying_an_appointment_time_still_grounds():
    """The address branch of this rule exists because layout-preserved text is
    not contiguous. The date branch was subject to exactly the same
    interleaving, and blocked both gating dates for it."""
    a = make_assembled(
        stops=[
            make_stop(when="03/16/2026 08:00"),
            DELIVERY.model_copy(update={"date_text": "03/18/2026 06:00"}),
        ]
    )
    assert [f for f in rules.not_grounded(a) if "date" in "".join(f.fields)] == []


def test_a_whitespace_only_span_is_not_blocked_for_being_ungrounded():
    """`publish()` drops it, so blocking the field docks confidence over a value
    nobody will ever see."""
    a = make_assembled(load_id_text="   ")
    assert [f for f in rules.not_grounded(a) if f.fields == ("load_id",)] == []


def test_a_refused_weight_span_is_flagged_rather_than_silently_nulled():
    """A refusal must not look like a document that prints no weight — the same
    conflation `status` vs `confidence` exists to avoid."""
    a = make_assembled(commodities=[make_commodity("Canned Goods", "38400 22")])
    assert a.weight_lbs is None
    assert "WEIGHT_UNREADABLE" in codes(rules.weight_implausible(a))


def test_the_documents_own_null_marker_is_not_a_refused_span():
    """Found by the real model, not by me. `openai/gpt-5.6-luna` returns
    `weight_text: "-"` for the provided sample B, because that is literally what
    the Weight column prints. A dash is the document saying "not stated", not a
    span we failed to parse, and flagging it made a correct read look like a
    refusal."""
    for marker in ("-", "--", "  -  "):
        a = make_assembled(commodities=[make_commodity("Ceramics", marker)])
        assert a.weight_lbs is None
        assert rules.weight_implausible(a) == []
    # ...while a span that does carry digits and still will not parse is a
    # refusal, and says so.
    fused = make_assembled(commodities=[make_commodity("Ceramics", "182 07")])
    assert "WEIGHT_UNREADABLE" in codes(rules.weight_implausible(fused))
