"""Builders, so a rule test is three lines instead of thirty."""

from typing import Any

from ratecon.pipeline import assemble
from ratecon.rules import Assembled
from ratecon.schema import Charge, LlmExtraction, Stop

DEFAULT_SOURCE = """\
CARRIER RATE & LOAD CONFIRMATION
Reference ID ML-884213
Rate Con Date 12-Mar-2026
Pickup Date 16-Mar-2026
Equipment Dry Van
1 Pickup  Great Lakes Distribution 4400 W 130th St, Cleveland, OH 44135, USA
   Shipping Date & Time 03/16/2026     Weight 38400
2 Drop    Piedmont Grocery DC 2100 Westinghouse Blvd, Charlotte, NC 28273, USA
   Delivery Date & Time 03/18/2026     Weight 38400
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


def make_extraction(**overrides: Any) -> LlmExtraction:
    base: dict[str, Any] = {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884213",
        "commodity_text": "Canned Goods",
        "weight_text": "38400",
        "equipment_text": "Dry Van",
        "rate_con_date_text": "12-Mar-2026",
        "header_pickup_date_text": "16-Mar-2026",
        "stated_total_text": "2150.00 USD",
        "stops": [make_stop(), DELIVERY],
        "charges": [
            Charge(label_text="Base Carrier Rate", amount_text="2150.00 USD"),
            Charge(label_text="Total", amount_text="2150.00 USD"),
        ],
    }
    base.update(overrides)
    return LlmExtraction.model_validate(base)


def make_assembled(source: str = DEFAULT_SOURCE, **overrides: Any) -> Assembled:
    return assemble(make_extraction(**overrides), source)


def codes(findings: list[Any]) -> set[str]:
    return {f.code for f in findings}
