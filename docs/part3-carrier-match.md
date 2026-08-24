# Part 3 — Carrier Match

**Carrier Match is a calibrated probability-of-coverage problem over a few hundred candidates, not a
search problem over millions.** So the shape is: retrieve with SQL → score with a GBDT → gate with
rules → explain with a template. The LLM manufactures features from text offline and drafts outreach
afterwards. It never ranks.

## The label, and the bias that comes with it

The label is **`P(books | carrier, offered rate, hours-to-pickup)`** — not "the load was covered",
which conflates carrier behaviour with dispatcher follow-through, and not margin, which teaches the
model to rank cheap unreliable carriers first. Margin is a second head; we rank on expected value.

**Rate is a treatment variable, not an output.** Any broker can cover anything at $4/mile — coverage
is a price problem, so a model that scores carriers without conditioning on the number you are about
to say out loud is answering the wrong question. The decision is the *(carrier, rate)* pair that
covers before the deadline at least cost. Hours-to-pickup matters too: the carrier who books three
days out is a different population from the 6am day-of market.

The obvious objection is that this looks circular — the offered rate is the thing we are choosing,
so how can it be an input? It is not circular, it is an inner loop. At scoring time we sweep a small
rate grid per candidate (the DAT/Truckstop band for the lane, say seven points from the 25th to the
90th percentile), score each *(carrier, rate)* cell, and rank cells by expected value:
`p(books | carrier, rate, hours) × (target_rate − rate)`. The panel then shows five carriers with
the rate to open at, which is the artefact a broker actually wants — "call Dependable at $2,450" is
a usable instruction and "Dependable: 0.71" is not. Scoring 200 candidates × 7 rate points is 1,400
GBDT rows, which is still under 5ms.

**The selection bias cannot be reweighted away.** We only observe outcomes for carriers a human
contacted — i.e. the ones we showed. Exposure propensity is exactly *zero* for everyone else, so
IPW has nothing to divide by. The fixes are structural: a stochastic logging policy and an explicit
exploration slot in every slate.

And **instrument the negative**, or the model has no negatives and degenerates into a popularity
ranker over five incumbents. "Called → no, and why" will not be captured by hand at a twelve-person
brokerage, so design negatives to come from things that log themselves: inbound contacts on the
posting, the email thread, and the load going to someone else.

## Architecture

```
Broker posts a load
 │
 ├─ 1. GATE (deterministic, indexed columns, ~5ms)
 │     authority · insurance ≥ required · identity verified ·
 │     not on do-not-use · equipment capable
 │
 ├─ 2. RETRIEVE  ~200-2,000 candidates            [Postgres WHERE, ~20ms]
 │     lane history · domiciled near origin · recent inbound in this market
 │
 ├─ 3. FEATURES                                   [feature store, p99 ~15ms]
 │
 ├─ 4. SCORE  LightGBM → P(books | carrier, rate, hours), isotonic-calibrated
 │     swept over a 7-point rate grid per candidate   [~5ms / 1,400 rows]
 │     rank cells by  p x (target_rate - offered_rate)
 │
 └─ 5. PRESENT  4 exploit + 1 explore, propensity logged
       each with a reason string, a suggested rate band, and one-click
       outcome capture
```

**Gates run before scoring** — they are cheap predicates, and scoring candidates we cannot legally
use collapses a five-slate to two. Re-gate at tender (insurance expires, authority lapses), gate the
training labels, and gate the exploration arm too. **Retrieval is a WHERE clause deliberately**: a
two-tower retriever over 8,000 carriers solves a problem this brokerage does not have.

Calibration is not optional — LambdaMART-style objectives optimise order, not probability, and we
need a real one to compute expected value, set an auto-tender threshold, and say "all five of these
are bad, go to the load board."

## Features

Five families, ~30 features. The four that carry the most:

| Feature | Source | Why it predicts coverage |
|---|---|---|
| Loads run *into* the destination market, 90d | our TMS | What makes a carrier *willing* — they need the backhaul |
| Inbound contacts on this posting, last 60 min | load board | At this segment, the strongest real-time capacity signal there is |
| Domicile distance to origin | FMCSA MCS-150 | Free for every carrier in the country, dense where history is not |
| Fall-off / no-show rate | our TMS | The metric that actually kills the feature — see trust decay |

Plus ~20 more: lane recency and frequency, acceptance rate, tracking-compliance rate (brokers deduct
for refusing Macropoint/P44, and it correlates with fraud), on-time percentage, claims, power units,
capability flags (hazmat, TWIC, food-grade, tarps), QuickBooks payment history, dispatcher
answer-rate by hour.

The four integrations named in the brief land in different places, and it is worth being explicit
about which is doing what. **DAT and Truckstop** supply the *rate band* the grid is swept over and
the *inbound-contact* signal on a posting — the strongest real-time capacity feature we have — and
Truckstop's carrier monitoring is a second identity source that disagrees with Highway often enough
to be worth reconciling rather than deduplicating. **Highway** is the identity gate in step 1, never
a feature. **QuickBooks** contributes days-to-pay and dispute history: a carrier who has been paid
fast twice answers the phone, and one sitting on a 60-day invoice does not.

Two I would *not* headline: DAT truck posts (stale, duplicated, posted by dispatch services on
behalf of many MCs so they rarely join back to a carrier record, and per-seat licensed), and
deadhead-from-last-known-empty (which exists only for carriers we tracked recently — sparse for
exactly the carriers we are trying to discover).

Cross-part trap: RateView reports broker-to-carrier rates **all-in**, so differencing a
linehaul-only `line_haul_rate` against it biases every load negative.

## Where the LLM earns its place

| Use it for | Don't |
|---|---|
| Text → features, **offline**: carrier emails, call notes, "we run Chicago→Atlanta Tuesdays" | **The ranking itself** |
| Rendering the *why* string over facts already retrieved | Emitting any number — retrieve, don't generate |
| Conversational search: "reefer out of Laredo tomorrow" | Anything in the hot path |
| Drafting outreach | Compliance as a soft feature — it is a hard gate |

A GBDT over 200 rows is ~2ms and a fraction of a cent. Five LLM calls per posting is seconds and
real money for a *worse* ranking — gradient boosting on tabular features is calibrated,
monotonic-constrainable, auditable and retrainable nightly, none of which an LLM ranker is. Cost per
posting is effectively zero **because there is no LLM in the path**.

The opposite error is dismissing LLMs to look rigorous. The four uses above are genuinely correct,
and the first — turning a decade of unstructured carrier correspondence into features — is probably
worth more than any ranking improvement.

## Fraud

"Fraud is a filter, not a feature" is half right. *Identity* is binary and belongs in the gate — no
probability of coverage justifies ranking a carrier who might not be who they say they are.

*Behavioural* double-brokering risk is contextual and load-specific: below-market acceptance far from
domicile, a new dispatch domain or phone, a **remit-to or factoring change** (payment-redirect fraud
is a live vector), authority under six months, recent ownership change, refusal to accept tracking.
That is a **score triggering step-up verification**, not a block — blocking a legitimate small
carrier is lost capacity, and most carriers are small. And identity checks alone are not enough: a
verified, legitimate carrier can still re-broker your load, so expect FMCSA monitoring, an
onboarding vendor, theft intel, a do-not-use list, and out-of-band verification of remit-to changes.

## Cold start

No history is not no data. The **free FMCSA census file** gives domicile, fleet size and equipment
class for every carrier in the country, and inspection records reveal which lanes they actually run.
Add load-board inbound and a global cross-tenant prior with partial pooling, so the brokerage's own
signal takes over as it accrues.

So months 0–3 are a **content model over free public data plus load-board inbound**, with the ranker
near-100% exploration — the right call, not an embarrassment. Worth stating plainly: a GBDT ranker
assumes a history many customers structurally do not have, so the design must degrade gracefully for
a brokerage with 300 loads, because that is the median customer rather than the edge case.

## Latency, cost, evaluation

p95 under 300ms: carrier × lane aggregates precomputed nightly into a feature store, candidate sets
cached per (lane, equipment), scoring in-process. Explanations render from a template over the top
SHAP features; LLM polish, if any, is async and cached, never in the path.

Offline AUC and calibration gate a release but do not prove the feature works. Online it is
**lane-level switchback randomisation**, with time-to-coverage and margin per load as the primary
metrics — not click-through, which measures whether the panel is interesting rather than right.
Screen candidate rankers on logged data with off-policy estimators (SNIPS, doubly robust) first.

The real adoption risk is not AUC. **If the panel recommends a carrier who no-shows, the broker
stops opening it after about three events.** Trust decay is what kills this feature — which is why
fall-off rate is a headline feature and the exploration slot must be visibly labelled as one.
