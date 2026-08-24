"""Author offline-cache entries for documents that have no recorded response.

MOST OF THE CACHE IS NOT AUTHORED. Thirteen of the fourteen committed entries
are real responses from `openai/gpt-5.6-luna`, stamped `meta.recorded =
"provider"` by `OpenRouterClient._call` and printed on every `ratecon demo` row.
Refresh them with:

    OPENROUTER_API_KEY=... uv run ratecon record --force --also evals/provided

This script is the fallback for documents that have no recording, and the source
of the one entry that must never have one. It never overwrites a `"provider"`
entry — a real response cost money and is evidence about the model, so replacing
it with a guess because a prompt changed would destroy that. Delete the file to
re-author one deliberately.

`11_model_misreads_the_lane` is authored and deliberately WRONG, and is the only
entry that is. It drops a stop and invents a load reference so that
`stop_count_mismatch` and `not_grounded` have a document to fire on; otherwise
the corpus would only ever demonstrate the pipeline on model output that
happened to be correct. `cli.KEEP_AUTHORED` makes `record` skip it.

The stop/charge/commodity dicts below double as the expected reading of each
document. Re-run after changing a fixture, the system prompt or the wire schema
— the cache key hashes the whole request, so any of those invalidates an entry.
"""

import json
from pathlib import Path
from typing import Any

from ratecon.extract import MODEL, OpenRouterClient, RawCompletion, ResponseCache, cache_key

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
PROVIDED = HERE / "provided"
CACHE = HERE / "cache"


def stop(
    seq: int,
    kind: str,
    address: str,
    city: str | None,
    state: str | None,
    zipc: str | None,
    when: str | None,
) -> dict[str, Any]:
    return {
        "sequence": seq,
        "kind": kind,
        "address_text": address,
        "city_text": city,
        "state_text": state,
        "zip_text": zipc,
        "date_text": when,
    }


def charge(label: str, amount: str) -> dict[str, Any]:
    return {"label_text": label, "amount_text": amount}


def commodity(description: str, weight: str | None) -> dict[str, Any]:
    return {"description_text": description, "weight_text": weight}


EXTRACTIONS: dict[str, dict[str, Any]] = {
    "01_clean_single_pickup": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884213",
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
        "commodities": [commodity("Canned Goods", "38400")],
    },
    "02_multi_stop_unmapped_charge": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884390",
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
        "commodities": [commodity("Steel Coil", "21000"), commodity("Steel Plate", "18000")],
    },
    "03_clean_reefer": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884455",
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
        "commodities": [commodity("Lettuce", "41200 lbs")],
    },
    "04_ambiguous_date": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884501",
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
        "commodities": [commodity("Paper Goods", "24000")],
    },
    "05_unexplained_residual": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884612",
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
        "commodities": [commodity("Rolled Paper", "43000")],
    },
    "06_fuel_surcharge_reconciles": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884700",
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
        "commodities": [commodity("Bottled Water", "44000")],
    },
    "07_bill_of_lading": {
        "document_type": "bol",
        "load_id_text": "BOL-7741209",
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
        "commodities": [commodity("Canned Goods, foodstuffs NOI", "38400")],
    },
    # Requirement 4's "missing fields": no printed total, no delivery date. The
    # weight is also over legal payload, so one document exercises three rules.
    "08_missing_total_and_delivery_date": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884815",
        "equipment_text": "Dry Van",
        "rate_con_date_text": "12-Mar-2026",
        "header_pickup_date_text": "27-Mar-2026",
        "stated_total_text": None,
        "stops": [
            stop(
                1,
                "pickup",
                "Cornbelt Grain Co-op 4100 N Industrial Ave, Davenport, IA 52806, USA",
                "Davenport",
                "IA",
                "52806",
                "03/27/2026",
            ),
            stop(
                2,
                "delivery",
                "Tri-County Feed & Supply 225 Depot St, Sioux Falls, SD 57104, USA",
                "Sioux Falls",
                "SD",
                "57104",
                None,
            ),
        ],
        "charges": [charge("Line Haul", "2050.00 USD")],
        "commodities": [commodity("Bagged Feed", "52000")],
    },
    # The customer rate printed beside the carrier rate — the error that pays
    # away the whole margin — plus a Step Deck, which is open-deck but not a
    # flatbed substitute.
    "09_customer_rate_step_deck": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-884902",
        "equipment_text": "Step Deck",
        "rate_con_date_text": "12-Mar-2026",
        "header_pickup_date_text": "29-Mar-2026",
        "stated_total_text": "2150.00 USD",
        "stops": [
            stop(
                1,
                "pickup",
                "Allegheny Machine Works 1900 Tech Center Dr, Pittsburgh, PA 15219, USA",
                "Pittsburgh",
                "PA",
                "15219",
                "03/29/2026",
            ),
            stop(
                2,
                "delivery",
                "Great Plains Tooling 6600 Stockyards Blvd, Omaha, NE 68107, USA",
                "Omaha",
                "NE",
                "68107",
                "03/31/2026",
            ),
        ],
        "charges": [
            charge("Line Haul", "2150.00 USD"),
            charge("Customer Rate", "2600.00 USD"),
            charge("Total", "2150.00 USD"),
        ],
        "commodities": [commodity("CNC Lathe", "38000")],
    },
    # A fat-fingered delivery date, read faithfully. The relational check is the
    # only thing that can catch it: both dates are individually plausible and
    # both are printed on the document.
    "10_transposed_dates": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-885044",
        "equipment_text": "Dry Van",
        "rate_con_date_text": "12-Mar-2026",
        "header_pickup_date_text": "30-Mar-2026",
        "stated_total_text": "1325.00 USD",
        "stops": [
            stop(
                1,
                "pickup",
                "Buckeye Packaging 880 Innovation Way, Dayton, OH 45402, USA",
                "Dayton",
                "OH",
                "45402",
                "03/30/2026",
            ),
            stop(
                2,
                "delivery",
                "Bluegrass Distribution 1440 Newtown Pike, Lexington, KY 40511, USA",
                "Lexington",
                "KY",
                "40511",
                "03/28/2026",
            ),
        ],
        "charges": [charge("Line Haul", "1325.00 USD"), charge("Total", "1325.00 USD")],
        "commodities": [commodity("Corrugate", "22000")],
    },
    # The only deliberately wrong response in the corpus. The model drops the
    # second pickup entirely and invents a load reference. Without a recall
    # check the omission would *raise* the tier, because dropping a stop also
    # drops MULTI_STOP and ORIGIN_DATE_DISAGREEMENT.
    "11_model_misreads_the_lane": {
        "document_type": "rate_confirmation",
        "load_id_text": "ML-000000",
        "equipment_text": "Dry Van",
        "rate_con_date_text": "12-Mar-2026",
        "header_pickup_date_text": "13-Apr-2026",
        "stated_total_text": "2780.00 USD",
        "stops": [
            stop(
                2,
                "pickup",
                "Carolina Glassworks 615 Textile Rd, Greenville, SC 29607, USA",
                "Greenville",
                "SC",
                "29607",
                "04/14/2026",
            ),
            stop(
                3,
                "delivery",
                "Chesapeake Grocery DC 9100 Pulaski Hwy, Baltimore, MD 21237, USA",
                "Baltimore",
                "MD",
                "21237",
                "04/16/2026",
            ),
        ],
        "charges": [charge("Line Haul", "2780.00 USD"), charge("Total", "2780.00 USD")],
        "commodities": [commodity("Bottled Tea", "41000")],
    },
}


# --------------------------------------------------------------------------
# The three rate confirmations supplied with the exercise
# --------------------------------------------------------------------------
#
# Separate from EXTRACTIONS because these are not fixtures I wrote — they are the
# assignment's own documents, and the point of running them is that I did not get
# to choose what is on them. `tests/test_provided_samples.py` imports these dicts
# rather than restating them, so the cache and the assertions cannot disagree.

_NYU = (
    "New York University Jeffrey S. Gould Welcome Center, 50 W 4th Street, New York, NY 10012, USA"
)
_CHICAGO = "Illinois State Police, 100 W Randolph St, Chicago, IL 60601, USA"

PROVIDED_EXTRACTIONS: dict[str, dict[str, Any]] = {
    "sampleA_LD64392": {
        "document_type": "rate_confirmation",
        "load_id_text": "LD64392",
        "equipment_text": "Flatbed",
        "rate_con_date_text": "28-Jul-2026",
        "header_pickup_date_text": "30-Jul-2026",
        "stated_total_text": "50.00 USD",
        "stops": [
            stop(1, "pickup", _CHICAGO, "Chicago", "IL", "60601", "07/30/2026"),
            stop(2, "delivery", _NYU, "New York", "NY", "10012", "08/01/2026"),
        ],
        "charges": [charge("Base Carrier Rate", "50.00 USD"), charge("Total", "50.00 USD")],
        "commodities": [commodity("Ceramics", "182")],
    },
    "sampleB_LD64408": {
        "document_type": "rate_confirmation",
        "load_id_text": "LD64408",
        "equipment_text": "Flatbed",
        "rate_con_date_text": "28-Jul-2026",
        "header_pickup_date_text": "03-Aug-2026",
        "stated_total_text": "700.00 USD",
        "stops": [
            stop(
                1,
                "pickup",
                "Miami International Airport (MIA), Northwest 42nd Avenue, Miami, FL, USA",
                "Miami",
                "FL",
                None,
                "07/28/2026",
            ),
            stop(2, "pickup", _CHICAGO, "Chicago", "IL", "60601", "08/03/2026"),
            stop(
                3,
                "delivery",
                "Hertz Car Rental - San Jose - San Jose Mineta International Airport (SJC), "
                "Airport Boulevard, San Jose, CA, USA",
                "San Jose",
                "CA",
                None,
                "08/05/2026",
            ),
        ],
        "charges": [
            charge("Base Carrier Rate", "500.00 USD"),
            charge("Carrier Charge", "200.00 USD"),
            charge("Total", "700.00 USD"),
        ],
        "commodities": [commodity("Ceramics", None), commodity("Commodity_t", None)],
    },
    "sampleC_LD64407": {
        "document_type": "rate_confirmation",
        "load_id_text": "LD64407",
        "equipment_text": "Flatbed",
        "rate_con_date_text": "28-Jul-2026",
        "header_pickup_date_text": "31-Jul-2026",
        "stated_total_text": "50.00 USD",
        "stops": [
            stop(1, "pickup", _CHICAGO, "Chicago", "IL", "60601", "07/31/2026"),
            stop(2, "delivery", _NYU, "New York", "NY", "10012", "08/02/2026"),
        ],
        "charges": [charge("Base Carrier Rate", "50.00 USD"), charge("Total", "50.00 USD")],
        "commodities": [commodity("Ceramics", "422")],
    },
}


def main() -> None:
    """Author every entry, then prune. Recorded entries are never overwritten —
    a real provider response is evidence, and re-authoring over it would replace
    that evidence with a guess. Delete the file to re-author one deliberately.
    """
    client = OpenRouterClient(ResponseCache(CACHE), allow_network=False)
    written = set()
    for directory, corpus in ((FIXTURES, EXTRACTIONS), (PROVIDED, PROVIDED_EXTRACTIONS)):
        written |= _author(client, directory, corpus)

    # Prune, or a changed prompt leaves the old entries behind and the directory
    # slowly fills with responses nothing can serve.
    for stale in sorted(CACHE.glob("*.json")):
        if stale.name in written:
            continue
        # Only authored entries are ours to delete. A real recorded response
        # cost money and is evidence about the model; pruning it because the
        # prompt changed would throw that away silently.
        if json.loads(stale.read_text()).get("meta", {}).get("recorded") != "authored":
            print(f"kept     {stale.name[:12]} (recorded from a provider)")
            continue
        stale.unlink()
        print(f"pruned   {stale.name[:12]}")


def _author(
    client: OpenRouterClient, directory: Path, corpus: dict[str, dict[str, Any]]
) -> set[str]:
    written: set[str] = set()
    for name, extraction in corpus.items():
        text = (directory / f"{name}.txt").read_text()
        request = client._request(text, None)
        key = cache_key(text, request)
        written.add(f"{key}.json")
        existing = CACHE / f"{key}.json"
        if existing.exists():
            meta = json.loads(existing.read_text()).get("meta", {})
            if meta.get("recorded") == "provider":
                print(f"kept     {name} -> {key[:12]} (recorded)")
                continue
        ResponseCache(CACHE).put(
            key,
            RawCompletion(
                json.dumps(extraction),
                "tool_calls",
                {"recorded": "authored", "model": MODEL},
            ),
        )
        written.add(f"{key}.json")
        print(f"authored {name} -> {key[:12]}")
    return written


if __name__ == "__main__":
    main()
