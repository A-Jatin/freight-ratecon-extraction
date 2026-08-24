from datetime import date
from decimal import Decimal

import pytest

from ratecon import normalize
from ratecon.schema import Charge, EquipmentType

# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

MDY_DOC = "Rate Con Date 12-Mar-2026 ... Shipping Date 07/30/2026 ... Delivery 08/01/2026"
NO_HINT_DOC = "Rate Con Date 3/2/26 ... Shipping 3/4/26 ... Delivery 3/5/26"
DMY_DOC = "Shipping 25/03/2026 ... Delivery 04/04/2026"


@pytest.mark.parametrize(
    ("raw", "doc", "status", "value"),
    [
        # Unambiguous formats resolve without needing the document at all.
        ("2026-03-16", "", "resolved", date(2026, 3, 16)),
        ("30-Jul-2026", "", "resolved", date(2026, 7, 30)),
        # Day > 12 pins the order by itself.
        ("07/30/2026", "", "resolved", date(2026, 7, 30)),
        ("25/03/2026", "", "resolved", date(2026, 3, 25)),
        # Both components <= 12: only the document settles it. This is the case
        # that matters - four of the seven stop dates in the provided samples
        # look exactly like this.
        ("08/01/2026", MDY_DOC, "resolved", date(2026, 8, 1)),
        ("04/04/2026", DMY_DOC, "resolved", date(2026, 4, 4)),
        # ... and when nothing settles it, we say so rather than guessing quietly.
        ("3/4/26", NO_HINT_DOC, "ambiguous", date(2026, 3, 4)),
        ("", "", "absent", None),
        ("-", "", "absent", None),
        ("not a date", "", "unparseable", None),
    ],
)
def test_resolve_date(raw, doc, status, value):
    reading = normalize.resolve_date(raw, doc)
    assert reading.status == status
    assert reading.value == value


def test_the_dmy_branch_is_actually_entered():
    """`04/04/2026` in a DMY document reads the same either way, so the
    parametrised case above never distinguishes the branches. This one does: on
    a DMY document `03/04/2026` must be 3 April, not 4 March."""
    assert normalize.resolve_date("03/04/2026", DMY_DOC).value == date(2026, 4, 3)
    assert normalize.resolve_date("03/04/2026", MDY_DOC).value == date(2026, 3, 4)


def test_ambiguous_date_carries_the_other_reading():
    """Nulling it destroys what the reviewer needs; guessing silently is how a
    truck shows up a month late. So: publish one, attach the other, flag it."""
    reading = normalize.resolve_date("3/4/26", NO_HINT_DOC)
    assert reading.value == date(2026, 3, 4)
    assert reading.alternative == date(2026, 4, 3)


def test_document_order_inference():
    assert normalize.infer_date_order(MDY_DOC) == "MDY"
    assert normalize.infer_date_order(DMY_DOC) == "DMY"
    assert normalize.infer_date_order(NO_HINT_DOC) == "unknown"


def test_a_document_that_votes_both_ways_is_contested_not_a_majority():
    """Mixed orders on one page mean the page was assembled from two sources,
    which is exactly when a confident reading is least safe. A bare majority
    would let two stray tokens silently flip every date on the document."""
    mixed = "Shipping 07/30/2026 ... Delivery 25/03/2026 ... Appt 3/4/26"
    assert normalize.infer_date_order(mixed) == "unknown"
    assert normalize.resolve_date("3/4/26", mixed).status == "ambiguous"


def test_the_century_hint_comes_from_a_year_not_a_street_number():
    """A bare search for `19xx|20xx` reads the broker letterhead: a
    `1900 Market Street` address anchors every two-digit year to the wrong
    century, and `03/16/26` resolves to 1926 marked `resolved` with nothing to
    flag."""
    letterhead = "Midwest Logistics, 1900 Market Street, Philadelphia, PA 19103\nPickup 03/16/2026"
    assert normalize._century_hint(letterhead) == 2026
    assert normalize.resolve_date("03/17/26", letterhead).value == date(2026, 3, 17)


def test_resolution_is_a_pure_function_of_the_document():
    """Not a tautology: the same span in two documents with different order
    evidence must resolve differently, and neither may consult the clock."""
    assert normalize.resolve_date("08/01/2026", MDY_DOC).value == date(2026, 8, 1)
    assert normalize.resolve_date("08/01/2026", DMY_DOC).value == date(2026, 1, 8)


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,250.00", Decimal("1250.00")),
        ("1250.00 USD", Decimal("1250.00")),
        ("50.00", Decimal("50.00")),
        # Accessorials can be deductions: advances, ComCheck fees, quick-pay,
        # factoring, claim chargebacks. The sign has to survive.
        ("(200.00)", Decimal("-200.00")),
        ("-200.00", Decimal("-200.00")),
        ("200.00-", Decimal("-200.00")),  # trailing minus, older TMS output
        ("-", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_money(raw, expected):
    assert normalize.parse_money(raw) == expected


@pytest.mark.parametrize(
    "fused",
    [
        "$1,250.00 USD 53",  # amount fused with the trailer size column
        "FSC 18% $225.00",  # a percentage fused with the amount
        "182 7",  # Weight fused with Quantity - reachable on the real samples
        "44,000 lbs (26 pallets)",
    ],
)
def test_a_span_with_two_numbers_is_refused_not_concatenated(fused):
    """Stripping the separators instead silently produces `1827` from a Weight
    column beside a Quantity column, with no signal attached. Refusing the
    ambiguous span is the whole point of the exercise."""
    assert normalize.parse_money(fused) is None


def test_absurd_magnitudes_are_refused():
    """A 400-digit run reaches `float(Decimal(...))` as `inf`, which serialises
    to JSON no strict parser accepts."""
    assert normalize.parse_money("1" + "0" * 400) is None


def test_money_in_source_ignores_separators():
    """String matching would fail on the thousands separator; compare numerically."""
    src = normalize.norm_text("Total 1,250.00 USD")
    assert normalize.money_in_source(src, Decimal("1250.00"))
    assert not normalize.money_in_source(src, Decimal("1150.00"))


def test_grounding_a_total_rejects_incidental_digit_runs():
    """Any digit run used to count, so on the provided samples the total `50.00`
    grounded against "50 W 4th Street" in the delivery address, and a phone area
    code grounded a $905 total. That is not a check."""
    src = (
        "Phone (905) 670-6882\n"
        "New York University, 50 W 4th Street, New York, NY 10012\n"
        "ZIP 60601\nPO/Container No 799917\nTotal 50.00 USD\n"
    )
    assert normalize.money_in_source(src, Decimal("50.00"))  # the printed total
    assert not normalize.money_in_source(src, Decimal("905"))
    assert not normalize.money_in_source(src, Decimal("60601"))
    assert not normalize.money_in_source(src, Decimal("799917"))
    assert not normalize.money_in_source(src, Decimal("10012"))


@pytest.mark.parametrize(
    "line", ["Total 1800", "TOTAL: 1800", "Rate 1800", "RATE  1,800", "Total 1800 USD"]
)
def test_a_total_printed_without_cents_still_grounds(line):
    """Requiring money *shape* alone blocked every document that prints
    `Total   1800` without a currency mark — and blocking a correct total is the
    more expensive error, because it routes a clean load to a human every time.
    A number alone on a line after a rate label counts as positioned like money.
    """
    assert normalize.money_in_source(line, Decimal("1800"))


@pytest.mark.parametrize("line", ["PO/Container No 1800", "Suite 1800", "Trailer 1800"])
def test_the_labelled_fallback_does_not_ground_a_non_amount(line):
    assert not normalize.money_in_source(line, Decimal("1800"))


def test_contains_token_rejects_accidental_substrings():
    """Bare substring matching lets a hallucinated value pass: `50.00` is inside
    `$1,450.00`, and two-letter state codes match inside ordinary words."""
    src = normalize.norm_text("Total $1,450.00 for the mainline haul")
    assert not normalize.contains_token(src, "50.00")
    assert normalize.contains_token(src, "1,450.00")
    assert not normalize.contains_token(normalize.norm_text("Illinois State Police"), "IL")


# --------------------------------------------------------------------------
# Charges
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "kind"),
    [
        ("Base Carrier Rate", "line_haul"),
        ("Line Haul", "line_haul"),
        ("Linehaul", "line_haul"),
        ("Flat Rate", "line_haul"),
        ("All-In Rate", "line_haul"),
        ("Fuel Surcharge", "fuel"),
        ("FSC", "fuel"),
        ("F/S", "fuel"),
        ("Detention @ $50/hr", "accessorial"),
        ("Lumper", "accessorial"),
        ("TONU", "accessorial"),
        ("Stop-Off", "accessorial"),
        ("Quick Pay Deduction", "accessorial"),
        ("Total", "total"),
        # Most specific family wins, which cuts both ways. A *generic* rate word
        # loses to "Total": classifying "Total All-In Rate" as line haul added
        # the whole load on top of the real line-haul line.
        ("Total All-In Rate", "total"),
        ("Total Flat Rate", "total"),
        ("Amount Due", "total"),
        # ...but a label that *names* a charge beats it, or the running total of
        # one accessorial is booked as the grand total of the load — and a line
        # haul labelled "Total Line Haul" vanishes from the breakdown entirely,
        # publishing `line_haul_rate: null` on a document that prints it.
        ("Total Detention", "accessorial"),
        ("Total Fuel Surcharge", "fuel"),
        ("Total Line Haul", "line_haul"),
        ("Total Freight Charge", "line_haul"),
        ("Freight Total", "line_haul"),
        # A subtotal is neither. Summing it double-counts the lines above it;
        # recording it as a total makes MULTIPLE_TOTALS fire on the ordinary
        # "Subtotal 1,500 / Total 1,800" layout.
        ("Subtotal", "subtotal"),
        ("Sub Total", "subtotal"),
        # The pair the whole multi-charge demo turns on. Both contain the token
        # "Carrier"; a naive `"carrier" in label` maps both to line haul, the
        # residual goes to zero, and the one interesting document scores clean.
        ("Carrier Charge", "unknown"),
        ("Other", "unknown"),
        ("Additional Charge", "unknown"),
        # "Fuel Advance" is a negative deduction, not a surcharge. A substring
        # match on "fuel" books a $300 advance as a $300 fuel surcharge - wrong
        # field and wrong sign.
        ("Fuel Advance", "accessorial"),
        ("Reefer Fuel", "accessorial"),
    ],
)
def test_classify_charge(label, kind):
    assert normalize.classify_charge(label) == kind


def test_summarise_reconciles_when_an_unmapped_line_explains_the_gap():
    """line_haul + fuel != total is usually *correct* in freight - detention,
    lumper, layover, TONU and stop-off are separate lines. Only an unexplained
    residual is a problem."""
    breakdown = normalize.summarise_charges(
        [
            Charge(label_text="Base Carrier Rate", amount_text="3400.00"),
            Charge(label_text="Carrier Charge", amount_text="500.00"),
        ],
        Decimal("3900.00"),
    )
    assert breakdown.line_haul == Decimal("3400.00")
    assert breakdown.fuel is None  # never imputed from an unlabelled accessorial
    assert breakdown.unmapped == [("Carrier Charge", Decimal("500.00"))]
    assert breakdown.residual == 0


def test_summarise_surfaces_an_unexplained_residual():
    breakdown = normalize.summarise_charges(
        [
            Charge(label_text="Line Haul", amount_text="1500.00"),
            Charge(label_text="Fuel Surcharge", amount_text="300.00"),
        ],
        Decimal("1900.00"),
    )
    assert breakdown.residual == Decimal("100.00")


def test_an_all_in_table_has_nothing_to_reconcile():
    """A rate table printing one line reading `Total $1,800.00` has no component
    lines. Treating the total as an unexplained residual blocked ordinary
    all-in documents to `low`."""
    breakdown = normalize.summarise_charges(
        [Charge(label_text="Total", amount_text="1800.00")], Decimal("1800.00")
    )
    assert breakdown.residual is None
    assert breakdown.totals == [("Total", Decimal("1800.00"))]


def test_a_subtotal_line_is_neither_summed_nor_counted_as_a_total():
    """Summing it double-counts the lines above it, and recording it as a total
    makes `MULTIPLE_TOTALS` fire on an ordinary two-line rate table."""
    breakdown = normalize.summarise_charges(
        [
            Charge(label_text="Line Haul", amount_text="1500.00"),
            Charge(label_text="Fuel Surcharge", amount_text="300.00"),
            Charge(label_text="Subtotal", amount_text="1800.00"),
            Charge(label_text="Total", amount_text="1800.00"),
        ],
        Decimal("1800.00"),
    )
    assert breakdown.residual == 0
    assert breakdown.totals == [("Total", Decimal("1800.00"))]


def test_zero_amount_lines_are_ignored():
    """Many TMSs print every accessorial type at $0.00; without suppression the
    unmapped-charge finding fires ten times on a clean document."""
    breakdown = normalize.summarise_charges(
        [
            Charge(label_text="Line Haul", amount_text="1500.00"),
            Charge(label_text="Detention", amount_text="0.00"),
            Charge(label_text="Mystery Fee", amount_text="0.00"),
        ],
        Decimal("1500.00"),
    )
    assert breakdown.unmapped == []
    assert breakdown.residual == 0


# --------------------------------------------------------------------------
# Equipment
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "equipment", "confident"),
    [
        ("Dry Van", EquipmentType.VAN, True),
        ("53' Van", EquipmentType.VAN, True),
        ("Reefer", EquipmentType.REEFER, True),
        ("Temp Control 34F", EquipmentType.REEFER, True),
        ("Flatbed", EquipmentType.FLATBED, True),
        # In FTL, "van" is a 53' dry van trailer. A cargo van is a sub-26,000-GVW
        # vehicle needing no CDL. Substring matching books a 44,000 lb load onto
        # a Sprinter.
        ("Cargo Van", EquipmentType.OTHER, False),
        ("Sprinter Van", EquipmentType.OTHER, False),
        # Open deck, but not substitutable for a flatbed: the deck height
        # differs and the load will not transfer.
        ("Step Deck", EquipmentType.OTHER, False),
        ("RGN", EquipmentType.OTHER, False),
        ("Power Only", EquipmentType.OTHER, False),
        ("Something Unheard Of", None, False),
        (None, None, False),
    ],
)
def test_classify_equipment(raw, equipment, confident):
    assert normalize.classify_equipment(raw) == (equipment, confident)


@pytest.mark.parametrize(
    "raw",
    [
        "53' Dry Van - Reefer Not Required",
        "Dry Van (no reefer)",
        "53 ft Van, non-refrigerated",
    ],
)
def test_a_negated_family_does_not_win(raw):
    """The table used to be hand-ordered by family, so `reefer` (6 chars) sat
    above `dry van` (7) and the first hit won: a dry-van load whose note said
    "no reefer" published `reefer` *confidently*, suppressing the very flag that
    exists to catch it. These are dry vans that mention a reefer."""
    assert normalize.classify_equipment(raw) == (EquipmentType.VAN, True)


@pytest.mark.parametrize("raw", ["Refrigerated Van", "Temperature Controlled Van", "Reefer Van"])
def test_a_reefer_is_a_van_so_the_pair_is_not_a_conflict(raw):
    """Returning `other` for these was over-correction. A reefer *is* an
    insulated van trailer, so the two families do not actually disagree, and
    publishing `other` for an unambiguous reefer is worse data than either."""
    assert normalize.classify_equipment(raw) == (EquipmentType.REEFER, True)


@pytest.mark.parametrize("raw", ["Dry Van or Flatbed", "Flatbed / Reefer", "Conestoga Flatbed"])
def test_a_string_naming_two_real_families_is_never_confident(raw):
    """A genuine either/or the document does not settle. `other` plus
    EQUIPMENT_UNMAPPED asks a human, which is the same call the table already
    makes for a Step Deck."""
    equipment, confident = normalize.classify_equipment(raw)
    assert not confident
    assert equipment is EquipmentType.OTHER


def test_the_equipment_table_is_sorted_longest_first():
    """Stated as an invariant rather than trusted to hand-ordering, because
    hand-ordering is exactly what silently stopped being true."""
    lengths = [len(phrase) for phrase, _ in normalize._EQUIPMENT]
    assert lengths == sorted(lengths, reverse=True)


# --------------------------------------------------------------------------
# Weight
# --------------------------------------------------------------------------


def test_weight_defaults_to_pounds_and_records_the_assumption():
    """The field is named `weight_lbs` and the Weight column on a freight
    document is the line total, so a bare number is pounds."""
    assert normalize.parse_weight_lbs("38400") == (Decimal("38400"), False)


@pytest.mark.parametrize(
    ("raw", "pounds"),
    [
        ("1000 kg", Decimal("2204.62")),
        ("20 tons", Decimal("40000")),
        ("20 metric tons", Decimal("44092.40")),
    ],
)
def test_weight_converts_when_a_unit_is_printed(raw, pounds):
    """A template that can emit grams can emit kilograms, and a blanket pounds
    assumption would then be a 2.2x error."""
    value, stated = normalize.parse_weight_lbs(raw)
    assert stated
    assert value == pounds


def test_a_hallucinated_digit_run_with_a_unit_does_not_explode():
    """`Decimal.quantize` raises `InvalidOperation` past the context precision,
    and this input reaches it directly from model output."""
    assert normalize.parse_weight_lbs("1" + "0" * 40 + " kg") == (None, False)


# --------------------------------------------------------------------------
# Stops
# --------------------------------------------------------------------------


def test_route_order_uses_printed_sequences_when_they_are_complete():
    from factories import DELIVERY, make_stop

    selection = normalize.select_stops(
        [DELIVERY.model_copy(update={"sequence": 3}), make_stop(1), make_stop(2)]
    )
    assert selection.origin is not None and selection.origin.sequence == 1
    assert selection.destination is not None and selection.destination.sequence == 3


# --------------------------------------------------------------------------
# Regressions introduced by the first round of fixes, and their guards
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line", ["Total Miles 1250", "Total Weight 44000", "TOTAL PIECES 1800", "Total Stops 3"]
)
def test_a_counted_quantity_after_the_word_total_is_not_an_amount(line):
    """The labelled-amount fallback originally allowed a few free characters
    between the keyword and the number, so a mileage or piece count grounded a
    total of the same value — and on an all-in document nothing else would have
    noticed. The keyword now has to sit immediately before the number."""
    amount = Decimal(line.split()[-1])
    assert not normalize.money_in_source(line, amount)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The format this corpus actually prints: currency suffixed, so testing
        # the whole span for parentheses saw `(` ... `D` and dropped the sign.
        ("(200.00) USD", Decimal("-200.00")),
        ("USD (200.00)", Decimal("-200.00")),
        # `_NUMBER_RE` cannot match across the `$`, so the minus is outside the
        # number and `body.startswith("-")` never sees it.
        ("-$500.00", Decimal("-500.00")),
        ("$500.00", Decimal("500.00")),
    ],
)
def test_the_sign_survives_a_currency_mark(raw, expected):
    assert normalize.parse_money(raw) == expected
