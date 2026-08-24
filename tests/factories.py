"""Builders, so a rule test is three lines instead of thirty."""

from typing import Any

from ratecon.pipeline import assemble
from ratecon.rules import Assembled
from ratecon.schema import Charge, CommodityLine, LlmExtraction, Severity, Stop

DEFAULT_SOURCE = """\
CARRIER RATE & LOAD CONFIRMATION
Reference ID ML-884213
Rate Con Date 12-Mar-2026
Pickup Date 16-Mar-2026
Equipment Dry Van
1 Pickup  Great Lakes Distribution 4400 W 130th St, Cleveland, OH 44135, USA
   Shipping Date & Time 03/16/2026
   S.No  Commodity     Weight   Quantity
   1     Canned Goods  38400    22
2 Drop    Piedmont Grocery DC 2100 Westinghouse Blvd, Charlotte, NC 28273, USA
   Delivery Date & Time 03/18/2026
   S.No  Commodity     Weight   Quantity
   1     Canned Goods  38400    22
Base Carrier Rate 2150.00 USD
Total 2150.00 USD
"""


def make_stop(
    sequence: int = 1,
    kind: str = "pickup",
    address: str = "Great Lakes Distribution 4400 W 130th St, Cleveland, OH 44135, USA",
    city: str | None = "Cleveland",
    state: str | None = "OH",
    zipc: str | None = "44135",
    when: str | None = "03/16/2026",
) -> Stop:
    return Stop(
        sequence=sequence,
        kind=kind,  # type: ignore[arg-type]
        address_text=address,
        city_text=city,
        state_text=state,
        zip_text=zipc,
        date_text=when,
    )


DELIVERY = make_stop(
    2,
    "delivery",
    "Piedmont Grocery DC 2100 Westinghouse Blvd, Charlotte, NC 28273, USA",
    "Charlotte",
    "NC",
    "28273",
    "03/18/2026",
)


def make_commodity(
    description: str = "Canned Goods", weight: str | None = "38400"
) -> CommodityLine:
    return CommodityLine(description_text=description, weight_text=weight)


def make_extraction(**overrides: Any) -> LlmExtraction:
    base: dict[str, Any] = {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884213",
        "equipment_text": "Dry Van",
        "rate_con_date_text": "12-Mar-2026",
        "header_pickup_date_text": "16-Mar-2026",
        "stated_total_text": "2150.00 USD",
        "stops": [make_stop(), DELIVERY],
        "charges": [
            Charge(label_text="Base Carrier Rate", amount_text="2150.00 USD"),
            Charge(label_text="Total", amount_text="2150.00 USD"),
        ],
        "commodities": [make_commodity()],
    }
    base.update(overrides)
    return LlmExtraction.model_validate(base)


def make_assembled(source: str = DEFAULT_SOURCE, **overrides: Any) -> Assembled:
    return assemble(make_extraction(**overrides), source)


def codes(findings: list[Any]) -> set[str]:
    return {f.code for f in findings}


def severities(findings: list[Any]) -> set[tuple[str, str, Severity]]:
    """`Finding.__eq__` compares only `(code, fields)` so that rewording a message
    does not break every rule test — which also means no equality-based assertion
    can see a wrong severity. Four severity mutations survived the whole suite
    because of it. Assert on this instead wherever the severity is the point.
    """
    return {(f.code, ",".join(f.fields), f.severity) for f in findings}
