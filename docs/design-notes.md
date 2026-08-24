# Design notes

Why the code makes the choices it does. `README.md` is the short version; this is the argument.

## The judgment calls

**Dates are resolved from the document, never by the model.** The model returns `"07/30/2026"`
verbatim; Python infers the document's order from tokens that can only be read one way and applies
it to the siblings. A document voting *both* ways returns `unknown` rather than a majority — mixed
orders mean the page came from two sources, which is when a confident reading is least safe. When
nothing settles it, the likelier reading is published *with the alternative attached and the field
blocked*. Two-digit years expand from a year found *inside a date*, never from any `19xx`/`20xx`
token, or a `1900 Market Street` letterhead anchors every date to 1926 marked `resolved`.
`python-dateutil` is deliberately absent: it picks a reading and returns a date indistinguishable
from a confident one, which is the exact signal this pipeline exists to surface.

**`fuel_surcharge` is never imputed.** Only a line labelled fuel/FSC/F-S fills it, and never
`total − line_haul` — that is the same hallucination performed with a calculator, and it would
always reconcile. A null fuel surcharge is the *modal* outcome on spot broker→carrier rate cons,
which are quoted all-in.

**Charge labels are matched with anchored patterns, most-specific-first.** "Base Carrier Rate" and
"Carrier Charge" share the token *Carrier*; a naive `"carrier" in label` maps both to line haul and
scores the one interesting document clean. A deny-list runs first ("Fuel Advance" is a negative
deduction). Then a label that *names* a charge beats the word Total — "Total Detention" is a
detention line, "Total Line Haul" is a line haul — while a label carrying only a *generic* rate word
loses to it, because reading "Total All-In Rate" as line haul adds the whole load on top of the real
line-haul figure. Getting that order wrong is reachable in both directions, and each direction
publishes wrong money.

**`line_haul + fuel ≠ total` is usually correct.** Detention, lumper, layover, TONU and stop-off are
separate lines, so a naive reconciliation rule would flag a large share of real freight. Only an
*unexplained* residual is a finding, and the block lands on the decomposition, not the printed
total — the total is the contractual figure, and the likeliest cause of a residual is a charge line
we failed to read. Once the unexplained part exceeds everything we could identify, the total stops
being corroborated and is blocked too. An all-in document with no component lines has nothing to
reconcile against and reconciles trivially, rather than reading as one enormous residual.

**Stop order follows the printed sequence, not the dates.** Stops are printed in routed order
because that is the order the driver executes them; per-stop dates are hand-entered and are what
gets fat-fingered. The printed sequence is used only when *every* stop has one and they are
distinct — otherwise route order falls back wholesale to list order, because mixing sequences with
list positions compares numbers from different scales. `pickup_date` comes from the stop selected as
the origin, never the header: on a multi-pickup load the header names the primary pickup, and
pairing that date with the first stop's city dispatches a truck a thousand miles wrong.

**Equipment: longest phrase first, negations dropped, conflicts flagged.** The table is written
grouped by family and sorted by length at import, because hand-ordering silently stopped being
true — `reefer` sat above `dry van`, so `"53' Dry Van - no reefer required"` published REEFER
*confidently*. Negated mentions are now dropped, so that string is a van; a reefer *is* an insulated
van trailer, so `{reefer, van}` resolves to reefer rather than a conflict; anything else spanning
two families ("Dry Van or Flatbed") lands in `other` and is flagged. "Cargo Van" must not become
`van` — in FTL *van* means a 53' dry van trailer, while a cargo van needs no CDL — and Step Deck is
open-deck but not flatbed-substitutable, so both are `other`.

**An ambiguous span is refused, not repaired.** `parse_money` returns `None` when a span holds two
numbers rather than stripping separators and concatenating. Not hypothetical here: the commodity
table prints `Weight 182` beside `Quantity 07`, and one line of layout-preserved text turns those
cells into `18207`. `WEIGHT_UNREADABLE` says a span was refused, so a refusal never looks like a
document that prints no weight.

## What the model still owns

- **Which printed figure is `stated_total_text`** — the highest-value decision in the system.
  `MULTIPLE_TOTALS` only sees labels the model chose to return, so if it returns the customer total
  and omits the carrier line, nothing downstream can tell. The prompt asks for the carrier-side
  total and for competing totals to come back as labelled charge lines, which makes the choice
  visible but does not remove it.
- **Which rows are stops**, and whether each is a pickup or a delivery. `STOP_COUNT_MISMATCH` covers
  the count, not a misclassified `kind`.
- **Where the address splits.** Genuine interpretation, delegated because US address parsing is
  `usaddress`/`libpostal` territory and a regex would be wrong — but delegated it is.
- **`document_type`**, which gates the record. A wrong `rate_confirmation` is caught by nothing.
- **Which span is the load reference** rather than a per-stop PO or seal number.
- **Which commodity cell is the weight**, and which rows are charge lines at all. A *clean* read of
  the wrong cell is invisible, and a charge line the model never returned leaves a residual that
  looks like an unread accessorial.

Grounding is honest about its reach. Eight published values are checked against the source —
`load_id`, `commodity`, `equipment_type`, `total_rate`, both stop dates, and each address's city and
ZIP — with token boundaries and per-type comparators. Amounts must be printed *as* money or
positioned as money on a labelled line; a bare digit run is not evidence, or the total `50.00` on
sample A would ground against "50 W 4th Street". Addresses ground on city and ZIP rather than the
whole address string, because layout-preserved text interleaves columns and the address is never
contiguous even when correct; dates ground on the date *token* inside the span for the same reason.
Three published values are **not** grounded — `weight_lbs` may be unit-converted, and
`line_haul_rate` and `fuel_surcharge` may each be a sum — so none has a single span to find; those
rest on reconciliation instead. All of it is strong against a fabricated value and **weak against a
value that is verbatim but wrong**: the customer rate instead of the carrier rate, or the broker's
own city lifted from the letterhead.

**Prompt injection is handled structurally, not with a keyword detector.** Rate confirmations arrive
by email from counterparties into a system that creates money events, but real ones are wall-to-wall
imperatives — *"Do not break seal"* — so an instruction-detector would reject most production
volume. The defence is that an injected total must survive arithmetic reconciliation and grounding,
and that the closing `</document>` tag is neutered before interpolation. There are tests for all
three, including an injection that prints a whole internally-consistent fake rate table — caught
because the document then prints two different totals. The honest limit: an attacker who prints a
*complete* replacement table with no competing total left visible defeats all of it. Every check is
a consistency check against a document the attacker controls. What stops that is provenance —
verifying the sending domain against the carrier record — which is a mail-ingest concern and is not
built here.

**`status` is separate from `confidence`.** A 429, a refusal, a truncation, an over-long input or a
cache miss produces `status: failed` with a reason, never `confidence: low`. Collapsing them would
encode "we read it and found nothing", which is false, and would make a provider outage look exactly
like model drift on a dashboard. A failure in our own normalisation gets its own reason
(`assembly_error`) for the same reason.

## On the offline cache

`ratecon demo` replays committed responses, so it runs with no key and no network. Thirteen of the
fourteen entries are **real recorded responses** from `openai/gpt-5.6-luna`, stamped
`meta.recorded = "provider"` and printed on every demo row as `[response: provider]`. Re-record them
with:

```bash
OPENROUTER_API_KEY=... uv run ratecon record --force --also evals/provided
```

The `--force` is load-bearing: `complete()` checks the cache before the network, so without it every
document hits and the command is a no-op. That was a real bug — the README described a remedy that
could not run.

Two entries are deliberately *not* recorded, and both exclusions are enforced in code rather than by
discipline:

- **`11_model_misreads_the_lane` is authored and deliberately wrong.** It drops a stop and invents a
  load reference, so `STOP_COUNT_MISMATCH` and `NOT_GROUNDED` both fire. Without it the corpus would
  only ever show the pipeline on model output that happened to be correct, and the recall check —
  the most important rule in the file — would have no document to demonstrate on. `cli.KEEP_AUTHORED`
  makes `record` skip it, because a re-record would replace the mistake with whatever the model did
  that day and silently delete the only case of its kind.
- **Recorded entries are never re-authored.** `author_offline_cache.py` skips any entry already
  stamped `"provider"`, and its prune step keeps them. A real response cost money and is evidence
  about the model; overwriting it with a guess because a prompt changed would destroy that. Delete
  the file to re-author one on purpose.

Recording is what a fabricated demo should be replaced by as soon as it can be, and it paid for
itself immediately: the real model returned `weight_text: "-"` on the provided sample B — the
document's own null marker — and exposed `WEIGHT_UNREADABLE` firing on a correct read. That bug
survived two adversarial review passes over authored responses, because no authored response ever
contained a dash.

## On evaluation

The repo ships a per-fixture findings table, not an accuracy number: bounding the high tier's error
rate at 1% needs roughly 300 documents, and at n=11 a coverage curve would be theatre. Fourteen of
the fifteen rules fire on at least one fixture — checkable from the `ratecon demo` transcript in
the README — and
`COMPONENT_EXCEEDS_TOTAL` is unit-test only, because every document I could write for it was
contrived.

The suite is mutation-checked rather than counted. Swapping origin with destination, swapping the
two dates, publishing `line_haul` as `total_rate`, downgrading a `NOT_GROUNDED` block to a flag, or
deleting the destination-unusable branch each left all tests green at one point; each is now killed
by a named test.

`docs/part2-evaluation.md` covers a real eval set and production monitoring;
`docs/part3-carrier-match.md` is the Carrier Match design.

## What I deliberately did not build

- **A `$/mile` plausibility band.** Needs billing miles: PC\*MILER is licensed, car-routing APIs are
  not truck-legal, zip-centroid great-circle runs 15–25% under road miles. It also detects an
  unusual *deal*, not an extraction *error*.
- **Transit feasibility.** Great-circle × 1.2 against ~600 mi/day would catch transposed and
  wrong-year dates that survive `pickup ≤ delivery`. The one place coarse mileage is good enough.
- **ZIP↔state validation.** Every dataset option is a bad trade, and it would not catch the
  realistic error anyway: `60601` → `60610` are both valid Chicago ZIPs.
- **Per-field calibration from corrections**, once there are labels. `--log` exists to produce them.
- **Template fingerprinting** for new-shipper detection — also the missing axis in Part 2's drift
  slicing.
- **Layout-aware table reconstruction and OCR.** `pdf.py` is the largest uncontrolled variance in
  the system, not the smallest. A scanned PDF fails loudly with `no_text_layer_likely_scanned`.
- **Appointment times and FCFS windows.** Dispatch cannot act on a date without a time, and a 23:00
  pickup with a timezone creates a real off-by-one-day.
- **A `stops[]` array in the published contract.** `MULTI_STOP` and `MULTI_COMMODITY` exist because
  the assignment's schema cannot hold a two-pickup or two-commodity load. The right fix is the
  schema; the flag is what you ship when the schema is someone else's.
- **Async batch processing** with a bounded semaphore and 429 backoff.

## Notes on the stack

Python 3.12+, `uv`, Pydantic v2, `ruff` (replacing black and isort — three configs that can disagree
is worse than one), `mypy --strict`, `pytest`. OpenRouter via the `openai` SDK's `base_url`.
`pdfplumber` rather than PyMuPDF, which is AGPL-3.0 and auto-flagged by many corporate OSS policies.

Tool calling is the only schema path, with `provider.require_parameters` set. No `response_format`
fallback on purpose: it would be a second path with different failure modes, exercised on a minority
of requests, while validation has to run either way. Validation is not optional, and OpenRouter's
own docs are why — the structured-outputs page says enforcement "varies by provider … exact
compliance is not guaranteed on every endpoint" *and* that an unsupported request "will fail with an
error", while the provider-routing page says the parameter "is ignored". Two official pages disagree
about the failure mode, so the pipeline depends on neither.

The wire schema carries no `description` fields. Pydantic derives them from class docstrings, so
otherwise the model is sent our design rationale — and since the cache key hashes the whole request,
rewording one comment would invalidate every committed response. Model guidance lives in the system
prompt, where it reads as prose.
