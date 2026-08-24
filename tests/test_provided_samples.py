"""The three rate confirmations supplied with the exercise, end to end, on the
model's own output.

Everything else in this suite runs on documents I wrote and on responses I
authored, which together can only demonstrate that the pipeline handles the
cases I thought of. This file runs the *recorded* responses — what
`openai/gpt-5.6-luna` actually returned for these three documents — through the
whole pipeline, and asserts the published values rather than the finding codes.
A green tick that does not say what lane and rate came out is not evidence.

It reads the committed cache with `allow_network=False`, so it is offline and
deterministic like the rest of the suite, and it fails if a recorded response is
missing or if the prompt drifts away from the one that produced it.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from ratecon.extract import OpenRouterClient, ResponseCache
from ratecon.pipeline import extract
from ratecon.schema import Confidence

ROOT = Path(__file__).resolve().parents[1]
PROVIDED = ROOT / "evals" / "provided"
CACHE = ROOT / "evals" / "cache"


def run(name: str):
    client = OpenRouterClient(ResponseCache(CACHE), allow_network=False)
    result = extract((PROVIDED / f"{name}.txt").read_text(), client)
    assert result.meta.get("recorded") == "provider", (
        f"{name} is not backed by a recorded provider response — "
        "re-run `ratecon record --force --also evals/provided`"
    )
    return result


def test_sample_a():
    r = run("sampleA_LD64392")
    assert r.status == "ok"
    assert r.data.load_id == "LD64392"
    assert (r.data.origin.city, r.data.origin.state) == ("Chicago", "IL")
    assert (r.data.destination.city, r.data.destination.state) == ("New York", "NY")
    assert r.data.pickup_date == "2026-07-30"
    # The one that matters: `08/01/2026` is locally ambiguous, and only the
    # sibling `07/30/2026` elsewhere on the page settles that the document is
    # MDY. Without that inference this is 1 August or 8 January.
    assert r.data.delivery_date == "2026-08-01"
    assert r.data.total_rate == Decimal("50.00")
    assert r.data.equipment_type == "flatbed"
    # 182 lb on an FTL flatbed is a property of the test document, not a
    # misread — and the flag is on a non-gating field, so it says so without
    # holding up the load.
    assert codes(r) == {"WEIGHT_IMPLAUSIBLE"}
    assert r.confidence is Confidence.HIGH


def test_sample_b_is_the_interesting_one():
    """Three stops, two of them pickups, two commodities, an unclassifiable
    charge line, and a header pickup date belonging to the second stop rather
    than the first. Every one of those is flagged, and none of them is silently
    resolved."""
    r = run("sampleB_LD64408")
    assert r.confidence is Confidence.MEDIUM
    assert codes(r) == {
        "MULTI_STOP",
        "MULTI_COMMODITY",
        "UNMAPPED_CHARGE",
        "ORIGIN_DATE_DISAGREEMENT",
    }
    # Two commodity descriptions, one `commodity` field. The first is published
    # and the collapse is named rather than hidden.
    assert r.data.commodity == "Ceramics"

    # The lane is the first pickup to the last drop, in printed order.
    assert (r.data.origin.city, r.data.destination.city) == ("Miami", "San Jose")
    # ...and the pickup date is that stop's own, not the header's 03-Aug, which
    # belongs to the Chicago pickup. Pairing Miami with 03-Aug sends a truck
    # 1,200 miles wrong.
    assert r.data.pickup_date == "2026-07-28"
    assert r.data.delivery_date == "2026-08-05"

    # "Base Carrier Rate" is line haul; "Carrier Charge" is not — they share the
    # token *Carrier*, and a substring match maps both to line haul, drives the
    # residual to zero and scores this document clean.
    assert r.data.line_haul_rate == Decimal("500.00")
    assert r.data.fuel_surcharge is None  # never imputed as total - line_haul
    assert r.data.total_rate == Decimal("700.00")
    assert any("Carrier Charge" in f.message for f in r.findings)


def test_sample_c():
    r = run("sampleC_LD64407")
    assert r.data.load_id == "LD64407"
    assert r.data.pickup_date == "2026-07-31"
    assert r.data.delivery_date == "2026-08-02"  # ambiguous alone, settled by 07/31
    assert r.data.total_rate == Decimal("50.00")
    assert r.confidence is Confidence.HIGH


@pytest.mark.parametrize("name", ["sampleA_LD64392", "sampleB_LD64408", "sampleC_LD64407"])
def test_no_provided_sample_trips_a_false_grounding_failure(name):
    """Layout-preserved text interleaves the address column with the date
    column, so an address is never contiguous in the source. An earlier version
    of the grounding rule blocked all three of these documents for it."""
    r = run(name)
    assert [f for f in r.findings if f.code == "NOT_GROUNDED"] == []


def test_four_of_the_seven_stop_dates_are_locally_ambiguous():
    """The claim the whole date design rests on, checked against what the model
    actually returned rather than asserted in prose. A stop date with both
    components <= 12 cannot be read without evidence from elsewhere on the page,
    and `infer_date_order` is the only thing that supplies it."""
    import json

    client = OpenRouterClient(ResponseCache(CACHE), allow_network=False)
    stop_dates = [
        s["date_text"]
        for name in ("sampleA_LD64392", "sampleB_LD64408", "sampleC_LD64407")
        for s in json.loads(client.complete((PROVIDED / f"{name}.txt").read_text()).content)[
            "stops"
        ]
        if s.get("date_text")
    ]
    assert len(stop_dates) == 7
    ambiguous = [d for d in stop_dates if all(int(p) <= 12 for p in d.split("/")[:2])]
    assert len(ambiguous) == 4, ambiguous


def codes(result) -> set[str]:
    return {f.code for f in result.findings}
