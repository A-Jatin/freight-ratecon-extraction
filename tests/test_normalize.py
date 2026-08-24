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


def test_resolution_does_not_depend_on_the_clock():
    """Pure function of the document, so the same bytes give the same answer in 2029."""
    assert normalize.resolve_date("08/01/2026", MDY_DOC) == normalize.resolve_date(
        "08/01/2026", MDY_DOC
    )


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
        ("-", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_money(raw, expected):
    assert normalize.parse_money(raw) == expected


def test_money_in_source_ignores_separators():
    """String matching would fail on the thousands separator; compare numerically."""
    src = normalize.norm_text("Total 1,250.00 USD")
    assert normalize.money_in_source(src, Decimal("1250.00"))
    assert not normalize.money_in_source(src, Decimal("1150.00"))


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


# --------------------------------------------------------------------------
# Weight
# --------------------------------------------------------------------------


def test_weight_defaults_to_pounds_and_records_the_assumption():
    """The field is named `weight_lbs` and the Weight column on a freight
    document is the line total, so a bare number is pounds."""
    assert normalize.parse_weight_lbs("38400") == (Decimal("38400"), False)


def test_weight_converts_when_a_unit_is_printed():
    """A template that can emit grams can emit kilograms, and a blanket pounds
    assumption would then be a 2.2x error."""
    pounds, stated = normalize.parse_weight_lbs("1000 kg")
    assert stated
    assert pounds == Decimal("2204.62")
