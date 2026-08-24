"""Author the offline cache so `ratecon demo` runs with no key and no network.

READ THIS BEFORE TRUSTING THE DEMO OUTPUT.

These are *authored* model responses, not recorded ones. They are what a correct
model should return for each fixture. They exist so that the deterministic half
of the pipeline — normalisation, reconciliation, the rules and the routing
ladder, which is the part actually being demonstrated — can be run and inspected
by anyone who clones the repo.

They are NOT evidence about the model. Every entry is stamped
`meta.recorded = "authored"`, and `ratecon demo` prints that. Running
`ratecon record` with an `OPENROUTER_API_KEY` overwrites them with real provider
responses stamped `"provider"`.

Written this way because the build environment could not reach openrouter.ai.
"""

import json
from pathlib import Path

from ratecon.extract import MODEL, OpenRouterClient, RawCompletion, ResponseCache, cache_key

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
CACHE = HERE / "cache"


def stop(
    seq: int, kind: str, address: str, city: str, state: str, zipc: str | None, when: str
) -> dict:
    return {
        "sequence": seq,
        "kind": kind,
        "address_text": address,
        "city_text": city,
        "state_text": state,
        "zip_text": zipc,
        "date_text": when,
    }


def charge(label: str, amount: str) -> dict:
    return {"label_text": label, "amount_text": amount}


EXTRACTIONS: dict[str, dict] = {
    "01_clean_single_pickup": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884213",
        "commodity_text": "Canned Goods",
        "weight_text": "38400",
        "equipment_text": "Dry Van",
        "rate_con_date_text": "12-Mar-2026",
        "header_pickup_date_text": "16-Mar-2026",
        "stated_total_text": "2150.00 USD",
        "stops": [
            stop(
                1,
                "pickup",
                "Great Lakes Distribution 4400 W 130th St, Cleveland, OH 44135, USA",
                "Cleveland",
                "OH",
                "44135",
                "03/16/2026",
            ),
            stop(
                2,
                "delivery",
                "Piedmont Grocery DC 2100 Westinghouse Blvd, Charlotte, NC 28273, USA",
                "Charlotte",
                "NC",
                "28273",
                "03/18/2026",
            ),
        ],
        "charges": [charge("Base Carrier Rate", "2150.00 USD"), charge("Total", "2150.00 USD")],
    },
    "02_multi_stop_unmapped_charge": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884390",
        "commodity_text": "Steel Coil",
        "weight_text": "21000",
        "equipment_text": "Flatbed",
        "rate_con_date_text": "12-Mar-2026",
        "header_pickup_date_text": "20-Mar-2026",
        "stated_total_text": "3900.00 USD",
        "stops": [
            stop(
                1,
                "pickup",
                "Gulf Steel Supply 8800 Clinton Dr, Houston, TX, USA",
                "Houston",
                "TX",
                None,
                "03/17/2026",
            ),
            stop(
                2,
                "pickup",
                "Ozark Fabrication 1250 S Industrial Dr, Springfield, MO 65802, USA",
                "Springfield",
                "MO",
                "65802",
                "03/20/2026",
            ),
            stop(
                3,
                "delivery",
                "Cascade Structural Yard 4700 E Marginal Way S, Seattle, WA, USA",
                "Seattle",
                "WA",
                None,
                "03/25/2026",
            ),
        ],
        "charges": [
            charge("Base Carrier Rate", "3400.00 USD"),
            charge("Carrier Charge", "500.00 USD"),
            charge("Total", "3900.00 USD"),
        ],
    },
    "03_clean_reefer": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884455",
        "commodity_text": "Lettuce",
        "weight_text": "41200 lbs",
        "equipment_text": "Temp Control 34F",
        "rate_con_date_text": "12-Mar-2026",
        "header_pickup_date_text": "19-Mar-2026",
        "stated_total_text": "2875.00 USD",
        "stops": [
            stop(
                1,
                "pickup",
                "Valley Fresh Packing 901 Airport Blvd, Salinas, CA 93901, USA",
                "Salinas",
                "CA",
                "93901",
                "03/19/2026",
            ),
            stop(
                2,
                "delivery",
                "Rocky Mountain Produce 5600 Washington St, Denver, CO 80216, USA",
                "Denver",
                "CO",
                "80216",
                "03/21/2026",
            ),
        ],
        "charges": [charge("Line Haul", "2875.00 USD"), charge("Total", "2875.00 USD")],
    },
    "04_ambiguous_date": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884501",
        "commodity_text": "Paper Goods",
        "weight_text": "24000",
        "equipment_text": "Dry Van",
        "rate_con_date_text": "3/2/26",
        "header_pickup_date_text": None,
        "stated_total_text": "1480.00 USD",
        "stops": [
            stop(
                1,
                "pickup",
                "Keystone Warehouse 700 Enterprise Dr, Allentown, PA 18109, USA",
                "Allentown",
                "PA",
                "18109",
                "3/4/26",
            ),
            stop(
                2,
                "delivery",
                "Tri-State Supply 145 Bergen Ave, Newark, NJ 07103, USA",
                "Newark",
                "NJ",
                "07103",
                "3/5/26",
            ),
        ],
        "charges": [charge("Line Haul", "1480.00 USD"), charge("Total", "1480.00 USD")],
    },
    "05_unexplained_residual": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884612",
        "commodity_text": "Rolled Paper",
        "weight_text": "43000",
        "equipment_text": "Dry Van",
        "rate_con_date_text": "12-Mar-2026",
        "header_pickup_date_text": "23-Mar-2026",
        "stated_total_text": "1900.00 USD",
        "stops": [
            stop(
                1,
                "pickup",
                "Midwest Paper Mill 2200 River Rd, Green Bay, WI 54303, USA",
                "Green Bay",
                "WI",
                "54303",
                "03/23/2026",
            ),
            stop(
                2,
                "delivery",
                "Prairie Print Works 880 Stockyards Expy, Kansas City, KS 66105, USA",
                "Kansas City",
                "KS",
                "66105",
                "03/25/2026",
            ),
        ],
        "charges": [
            charge("Line Haul", "1500.00 USD"),
            charge("Fuel Surcharge", "300.00 USD"),
            charge("Total", "1900.00 USD"),
        ],
    },
    "06_fuel_surcharge_reconciles": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884700",
        "commodity_text": "Bottled Water",
        "weight_text": "44000",
        "equipment_text": "Dry Van",
        "rate_con_date_text": "12-Mar-2026",
        "header_pickup_date_text": "24-Mar-2026",
        "stated_total_text": "1800.00 USD",
        "stops": [
            stop(
                1,
                "pickup",
                "Lone Star Beverage 1500 Fulton St, San Antonio, TX 78201, USA",
                "San Antonio",
                "TX",
                "78201",
                "03/24/2026",
            ),
            stop(
                2,
                "delivery",
                "Delta Foods Depot 3300 Airline Dr, Metairie, LA 70001, USA",
                "Metairie",
                "LA",
                "70001",
                "03/26/2026",
            ),
        ],
        "charges": [
            charge("Line Haul", "1500.00 USD"),
            charge("Fuel Surcharge", "300.00 USD"),
            charge("Total", "1800.00 USD"),
        ],
    },
    "07_bill_of_lading": {
        "document_type": "bol",
        "load_id_text": "BOL-7741209",
        "commodity_text": "Canned Goods, foodstuffs NOI",
        "weight_text": "38400",
        "equipment_text": None,
        "rate_con_date_text": None,
        "header_pickup_date_text": None,
        "stated_total_text": None,
        "stops": [
            stop(
                1,
                "pickup",
                "Great Lakes Distribution 4400 W 130th St, Cleveland, OH 44135, USA",
                "Cleveland",
                "OH",
                "44135",
                "03/16/2026",
            ),
            stop(
                2,
                "delivery",
                "Piedmont Grocery DC 2100 Westinghouse Blvd, Charlotte, NC 28273, USA",
                "Charlotte",
                "NC",
                "28273",
                None,
            ),
        ],
        "charges": [],
    },
}


def main() -> None:
    client = OpenRouterClient(ResponseCache(CACHE), allow_network=False)
    for name, extraction in EXTRACTIONS.items():
        text = (FIXTURES / f"{name}.txt").read_text()
        request = client._request(text, None)
        key = cache_key(text, request)
        ResponseCache(CACHE).put(
            key,
            RawCompletion(
                json.dumps(extraction),
                "tool_calls",
                {"recorded": "authored", "model": MODEL},
            ),
        )
        print(f"authored {name} -> {key[:12]}")


if __name__ == "__main__":
    main()
