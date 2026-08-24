# Part 2 — How we know it works, and how we'll know when it breaks

**Accuracy is the wrong metric for a system that can abstain.** The number to run the business on is
**straight-through rate at a fixed error budget on the auto-processed slice** — "auto-post 70% of
rate cons at under 0.5% wrong-rate." The binding constraint is precision on `total_rate` *within
that slice*; coverage is what we maximise subject to it. One blended accuracy number throws away the
`confidence` field we just built.

**The eval set.** ~50 documents to start — enough to catch large regressions and per-field failure
modes, and **nowhere near enough to certify the 0.5% budget above**, which needs closer to a
thousand. Those are two different instruments and the plan needs both: 50 as the pre-merge gate from
day one, growing toward a thousand out of production corrections before anyone is allowed to quote a
wrong-rate number. At n=50 a 90% pass rate carries a 95% interval of roughly ±8pp on the normal
approximation (±14pp near 50%, and the exact interval is asymmetric — 45/50 is [78.2%, 96.7%]), so it
can tell 90% from 60% and cannot tell 99.5% from 98%.

Stratify on three axes: source, *shipper template* (5–8 layouts — failures are per-customer, not
global), and edge case, ≥3 per tag. Double-annotated and adjudicated. In production the labels are
free: the load **as finally booked** is the answer key. Keep the adversarial and naturally-sampled
slices separate — the adversarial set covers failure modes, so its aggregate accuracy estimates
nothing. Score `MISSED` (null) and `WRONG` (bad value) separately: two orders of magnitude apart in
cost, and plain F1 collapses them. Never fuzzy-match money, dates or IDs.

**What an error costs depends on its sign.** The usual decomposition is `P(silent) × margin loss +
P(disputed) × rework + P(service failure) × account damage`, and what gets missed is that
**`P(silent)` is not a scalar**: overpaying the carrier is absorbed in silence, while the same error
underpaying them produces a phone call within a day. The policy should be sign-aware. Detection lag
compounds it — caught at dispatch the exposure is a TONU ($150–350); caught at settlement 30–45 days
later the customer is invoiced and the margin recognised, so the fix is a credit memo and a rebill.
Plausibly $50–800, and the operating point is insensitive across that range because the confidence
distribution is bimodal. The asymmetry establishes the *direction*, not a cut point.

**Drift, without labels.** Watch proxies, and slice every one by **(template × field)** — a global
metric hides exactly the case we care about, since one new shipper at 3% of volume running at 60%
accuracy barely moves an aggregate. Signals: broker override rate per field, per-field null rate,
unexplained-residual rate, schema-repair rate, confidence distribution — as effect sizes, not
p-value tests, which at production volume flag significant-but-irrelevant shifts daily until
everyone mutes the alert. A new-template alert must be a *conjunction*: unseen fingerprint **and**
volume **and** degraded confidence. For *provider* drift, pin the model and shadow any version
change before switching.

**Override rate is measured only on the reviewed low-confidence tail**, so it systematically
under-reports the confidently-wrong case — precisely the case the cost asymmetry cares about.
Monitoring that blind spot needs a deliberate 1–2% audit sample of the *high*-confidence tier.
Without it, the healthiest-looking metric is the one measuring nothing.

All of this needs a substrate, so `ratecon --log run.jsonl` writes one JSON envelope per document:
`status`, `confidence`, per-field `field_status`, the finding codes with their severities, the
document hash, and `prompt_version` / `schema_version` / `policy_version`. Deliberately the envelope
and not the data — it can be shipped to a dashboard without carrying document text. Three of the five signals above are a `GROUP BY` over that file as it stands; override rate
needs the reviewer's correction written back, and the template axis needs a layout fingerprint —
neither exists yet, and both are the next thing to build rather than something already shipped. `policy_version` is on it for the same reason: change a rule
and historical rows stop being comparable, so without it the monitoring quietly lies.

**The unit economics are not the constraint, and it is worth saying so.** Recording the corpus
against `openai/gpt-5.6-luna` cost $0.0072 for 13 documents — about **$0.55 per 1,000 rate
confirmations**. Against a per-load margin measured in tens or hundreds of dollars, inference cost
does not enter the decision: the operating point is set entirely by the cost of a wrong rate versus
the cost of a human review, which is why every number above is about error rates and review load
rather than tokens. It also means a shadow deployment of a candidate model is effectively free, and
that is the cheapest drift control available.

**Where the human goes.** A broker's TMS *generates* broker→carrier rate cons; it doesn't parse
them. What arrives is co-brokered freight, customer tenders, and carrier invoices at settlement, and
at this segment the reviewer is usually a back-office team rather than the broker. So the artefact
is a **work queue**: SLA, per-item timer, keyboard-only, one row per flagged field with the source
span beside it, Enter to accept. Brokers live on the phone and will not tab through a form.

One correction to the obvious design: **do not refuse to create the load on low confidence.** A
broker with a truck waiting will key it manually and the workflow is lost permanently. Create the
load, mark the contested fields, block the downstream *money* event — carrier tender and payment —
until they are confirmed. Every correction writes a golden-set row, which is how the rule set gets
replaced by a calibrated model within a couple of months.
