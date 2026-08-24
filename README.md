# ratecon

Extracts a validated, typed load record from the raw text of a freight rate confirmation, with a
per-field confidence signal and explicit routing to human review.

**The idea the rest of this follows from:**

> The model does span selection and light typing. Python does interpretation, arithmetic,
> normalisation, and trust.

The model is never shown a date type, a number type, or the words *line haul*. It returns verbatim
substrings and nothing else, so it is structurally unable to silently resolve `3/4/26`, decide that
a charge labelled "Carrier Charge" is a fuel surcharge, or quietly overwrite a printed total. Every
one of those decisions is made in plain Python that can be printed, unit-tested, and changed
without touching a prompt.

## Real output

A three-stop load with an unclassifiable charge line, straight from `ratecon extract`:

```json
{
  "data": {
    "load_id": "ML-884390",
    "origin":      { "city": "Houston", "state": "TX", "zip": null },
    "destination": { "city": "Seattle", "state": "WA", "zip": null },
    "pickup_date": "2026-03-17",
    "delivery_date": "2026-03-25",
    "equipment_type": "flatbed",
    "line_haul_rate": 3400.0,
    "fuel_surcharge": null,
    "total_rate": 3900.0,
    "weight_lbs": 21000.0,
    "commodity": "Steel Coil",
    "confidence": "medium"
  },
  "field_status": {
    "total_rate": "flagged", "origin": "flagged",
    "destination": "flagged", "pickup_date": "flagged"
  },
  "findings": [
    { "code": "UNMAPPED_CHARGE", "severity": "flag", "fields": ["total_rate"],
      "message": "Unclassified charge line(s): Carrier Charge 500.00. Not mapped to fuel — no fuel label present." },
    { "code": "MULTI_STOP", "severity": "flag", "fields": ["origin", "destination"],
      "message": "3 stops collapsed into one origin/destination pair." },
    { "code": "ORIGIN_DATE_DISAGREEMENT", "severity": "flag", "fields": ["pickup_date"],
      "message": "Header pickup date 2026-03-20 belongs to a different stop; published 2026-03-17 from the first pickup." }
  ],
  "confidence": "medium",
  "status": "ok"
}
```

`fuel_surcharge` is `null` and $500 sits outside the three-slot schema — because the document says
"Carrier Charge", which is not a fuel line. The arithmetic still closes (3400 + 500 = 3900), so the
total is trustworthy and this is advisory rather than fatal.

## Quickstart

```bash
uv sync
uv run ratecon demo        # every fixture, offline, no API key
uv run pytest              # 91 tests, all offline
```

```
01_clean_single_pickup        high    Cleveland -> Charlotte    2026-03-16  total=2150.00
02_multi_stop_unmapped_charge medium  Houston -> Seattle        2026-03-17  total=3900.00
                              confirm: destination, origin, pickup_date, total_rate
03_clean_reefer               high    Salinas -> Denver         2026-03-19  total=2875.00
04_ambiguous_date             low     Allentown -> Newark       2026-03-04  total=1480.00
                              confirm: delivery_date, pickup_date
05_unexplained_residual       medium  Green Bay -> Kansas City  2026-03-23  total=1900.00
                              confirm: fuel_surcharge, line_haul_rate, total_rate
06_fuel_surcharge_reconciles  high    San Antonio -> Metairie   2026-03-24  total=1800.00
07_bill_of_lading             low     — NOT_A_RATE_CON
```

## How it works

```mermaid
flowchart LR
    A["raw text"] --> B["LLM: verbatim spans only"]
    B --> C["validate + 1 repair retry"]
    C --> D["assemble: dates, money, stops, charges"]
    D --> E["audit: 12 field-scoped rules"]
    E --> F["route: high / medium / low"]
```

| Module | Does |
|---|---|
| `schema.py` | `LlmExtraction` (wire) and `RateConfirmation` (the published contract), plus wire-schema shaping |
| `extract.py` | `LLMClient` protocol, OpenRouter client, cache, bounded repair retry, `FakeClient` |
| `normalize.py` | date resolver, `Decimal` money, stop selection, charge classifier, equipment map |
| `rules.py` | the 12 rules, each a pure function |
| `pipeline.py` | assembly, the routing ladder, the total-function boundary |

### Confidence is a decision, not a score

Each finding names the field it impugns and is either **BLOCK** (that field is unusable) or
**FLAG** (usable, confirm it). Gating fields are `total_rate`, `pickup_date`, `delivery_date`,
`origin`, `destination`.

```
BLOCK on a gating field      → low
BLOCK on a non-gating field  → medium
FLAG  on a gating field      → medium
otherwise                    → high
```

Three tiers because a consumer has exactly three behaviours: use it, confirm the marked fields, or
look at the whole thing. There are no counting thresholds to defend — no "two majors makes a
minor" — because every rule has to justify itself by naming a field, which is a better forcing
function on rule design than a tally.

**A BLOCK never erases the value.** It marks the field and leaves the data in place, so a
disagreement in one field can never destroy another field's good value.

**Why not a continuous score?** The tier is a *decision*, not a probability. The right object is
`P(field correct | finding set)`, fit by logistic regression on the finding indicators using broker
corrections as labels. I don't have those labels, so I shipped the rules that make the decision and
the logging that would produce them. The rule set is the prior; corrections replace it inside a
month.

**Why not model-derived confidence?** Three independent reasons. Anthropic exposes no logprobs at
all, and on OpenRouter support is per-endpoint and shifts when routing shifts. Under constrained
decoding the token distribution is renormalised over grammar-legal tokens, so an enum field reads
~0.99 because only four tokens were ever legal — that is the grammar's confidence, not the model's,
and it is weakest precisely on the free-text spans we care about. And verbalised confidence
("rate yourself 1–10") is the vibes the brief asks us not to ship. Self-consistency doesn't rescue
it either: models sharing pretraining tend to fail the *same way* on the same ambiguous span, so
agreement is not correctness.

## The judgment calls

**Dates are resolved from the document, never by the model.** The model returns `"07/30/2026"`
verbatim. Python then infers the document's date order from tokens that can only be read one way —
`07/30/2026` has a 30, so the document is MDY — and applies that to its siblings. In the provided
samples four of seven stop dates are locally ambiguous and only this inference settles them. When
nothing settles it, the likelier reading is published *with the alternative attached and the field
flagged*: nulling it destroys what the reviewer needs, and guessing silently is how a truck arrives
a month late. `python-dateutil` is deliberately not a dependency — it silently picks a reading and
returns a date indistinguishable from a confident one, which is the exact signal this pipeline
exists to surface.

**`fuel_surcharge` is never imputed.** Only a line explicitly labelled fuel/FSC/F-S fills it, and
it is never derived as `total − line_haul` — that is the same hallucination performed with a
calculator, and it would always reconcile. A null fuel surcharge is the *modal* outcome on spot
broker→carrier rate confirmations, which are quoted all-in; separate FSC lines are characteristic
of contract freight.

**Charge labels are matched with anchored patterns, never substrings.** "Base Carrier Rate" and
"Carrier Charge" share the token *Carrier*; a naive `"carrier" in label` maps both to line haul,
drives the residual to zero, and scores the one genuinely interesting document as clean. A deny-list
runs first, because "Fuel Advance" is a negative deduction, not a surcharge.

**`line_haul + fuel ≠ total` is usually correct.** Detention, lumper, layover, TONU and stop-off are
separate lines, so a naive "totals must reconcile" rule would flag a large share of real freight.
Only an *unexplained* residual is a finding, and the block lands on the decomposition, not on the
printed total — the total is the contractual figure the carrier is paid, and the likeliest cause of
a residual is a charge line we failed to read. Once the unexplained part exceeds everything we could
identify, the total stops being corroborated and is blocked too.

**Stop order follows the printed sequence, not the dates.** Stops are printed in routed order
because that is the order the driver executes them; the per-stop dates are hand-entered and are what
gets fat-fingered. Sorting by date would let one typo silently reorder the route. Dates cross-check
only. And `pickup_date` comes from the stop selected as the origin, never from the header — on a
multi-pickup load the header names the primary pickup, and pairing that date with the first stop's
city dispatches a truck a thousand miles from where it should be.

**Equipment is matched longest-token-first.** "Cargo Van" must not become `van`: in FTL, *van* means
a 53' dry van trailer, while a cargo van is a sub-26,000-GVW vehicle needing no CDL. Step Deck is
open-deck but not substitutable for a flatbed — the deck height differs and the load will not
transfer — so it lands in `other` and is flagged rather than coerced.

**Grounding is honest about what it catches.** Every published value that comes from a single span
is checked against the source, with token boundaries and per-type comparators (bare substring
matching passes `50.00` inside `$1,450.00`, and two-letter state codes match inside ordinary words).
This is strong against a hallucinated value and **weak against the error that actually costs money**
— reading the customer rate instead of the carrier rate, where the wrong number is also verbatim in
the document. `MULTIPLE_TOTALS` exists for that case.

**Prompt injection is handled structurally, not with a keyword detector.** Rate confirmations arrive
by email from counterparties, into a system that creates money events. But real rate cons are
wall-to-wall imperatives — *"Do not break seal"*, *"Driver must call dispatch"* — so an
instruction-detector would reject most production volume. The defence is that an injected total
still has to survive arithmetic reconciliation and grounding. There is a test for exactly that.

**`status` is separate from `confidence`.** A 429, a refusal, a truncation or a cache miss produces
`status: failed` with a reason, never `confidence: low`. Collapsing them would encode "we read the
document and found nothing", which is false — and it would make a provider outage look exactly like
model drift on a dashboard.

## On the offline cache

`ratecon demo` reads committed responses so it runs with no key and no network. **Those responses
are authored, not recorded** — every entry is stamped `meta.recorded = "authored"`, and
`evals/author_offline_cache.py` says so at the top. They exist so the deterministic half of the
pipeline — which is the part being demonstrated — can be run and inspected by anyone who clones
this. They are not evidence about the model. Running `ratecon record` with an `OPENROUTER_API_KEY`
overwrites them with real provider responses.

## On evaluation

The repo ships a per-fixture findings table, not an accuracy number. Bounding the high tier's error
rate at 1% needs roughly 300 documents; at n=7 a coverage curve would be theatre. What the fixtures
demonstrate is that each rule fires on the document it was written for and stays quiet elsewhere.

`docs/part2-evaluation.md` covers what a real eval set and production monitoring look like, and
`docs/part3-carrier-match.md` is the Carrier Match design.

## What I deliberately did not build

Each of these is a three-line design note instead of an afternoon, which is the trade the brief's
"do not over-polish" is asking for.

- **A `$/mile` plausibility band.** It needs billing miles — PC\*MILER is the industry standard and
  is licensed; car-routing APIs are not truck-legal; zip-centroid great-circle runs 15–25% under
  road miles. It also detects an unusual *deal*, not an extraction *error*, so it belongs in a
  pricing anomaly monitor rather than here.
- **Transit feasibility.** Great-circle × 1.2 against a ~600 mi/day solo bound would catch
  transposed and wrong-year dates that survive the `pickup ≤ delivery` check. Cheap, and the one
  place coarse mileage is good enough.
- **ZIP↔state validation.** Every dataset option is a bad trade (10 MB of SQLite, or a runtime
  download that kills the offline story), and it would not catch the realistic error anyway:
  `60601` → `60610` are both valid Chicago ZIPs.
- **Per-field calibration from corrections.** The logistic regression described above, once there
  are labels.
- **Template fingerprinting and embedding-based clustering** for new-shipper detection.
- **Layout-aware table reconstruction and OCR.** `pdf.py` is the largest uncontrolled variance in
  this system, not the smallest — naive extraction on a table-heavy document can reattach a number
  to the wrong label. A scanned PDF currently fails loudly with `no_text_layer_likely_scanned`
  rather than pretending to be empty.
- **Appointment times and FCFS windows.** The schema is date-only; dispatch cannot act on a date
  without a time, and a 23:00 pickup with a timezone creates a real off-by-one-day.
- **Async batch processing** with a bounded semaphore and 429 backoff.

## Notes on the stack

Python 3.12+, `uv`, Pydantic v2, `ruff` (which replaces black and isort — carrying all three would
just be three configs that can disagree), `mypy --strict`, `pytest`. OpenRouter via the `openai`
SDK's `base_url`, so the same code points at any OpenAI-compatible endpoint by changing one
variable.

`pdfplumber` rather than PyMuPDF: PyMuPDF is AGPL-3.0, which is viral and auto-flagged by many
corporate OSS policies.

Tool calling is the schema path, with `response_format` as a per-model upgrade selected at runtime
from the models API. Validation is not optional, and OpenRouter's own documentation is why: the
structured-outputs page says enforcement "varies by provider … exact compliance is not guaranteed
on every endpoint", while the provider-routing page says that if no provider supports the parameter
"the request is still routed to that model and the parameter is ignored" — and the
structured-outputs page separately claims such a request "will fail with an error". Two official
pages disagree about the failure mode, so the pipeline depends on neither.
