"""Assembly, routing, and the one public entry point."""

import hashlib
import traceback
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from ratecon import normalize, rules
from ratecon.extract import ExtractionError, LLMClient, RawCompletion, parse_and_repair
from ratecon.rules import GATING_FIELDS, Assembled
from ratecon.schema import (
    Address,
    Confidence,
    Finding,
    LlmExtraction,
    RateConfirmation,
    Severity,
    Stop,
)

Status = Literal["ok", "degraded", "failed"]
FieldStatus = Literal["ok", "flagged", "blocked"]

PROMPT_VERSION = "1"
SCHEMA_VERSION = "1"
POLICY_VERSION = "1"  # bump when a rule or the routing ladder changes


@dataclass
class ExtractionResult:
    """The envelope.

    `status` describes the *run*; `confidence` describes the *document*. A
    provider outage must never masquerade as "we read it and found nothing" —
    they are different facts, and conflating them would make a 429 storm look
    exactly like model drift on the monitoring dashboard.
    """

    data: RateConfirmation
    field_status: dict[str, FieldStatus]
    findings: list[Finding]
    confidence: Confidence
    status: Status
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data.model_dump(mode="json"),
            "field_status": self.field_status,
            "findings": [
                {
                    "code": f.code,
                    "severity": f.severity.value,
                    "fields": list(f.fields),
                    "message": f.message,
                }
                for f in self.findings
            ],
            "confidence": self.confidence.value,
            "status": self.status,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def route(findings: list[Finding]) -> tuple[Confidence, dict[str, FieldStatus]]:
    """Findings -> tier. No counting, no thresholds.

    Three tiers because the consumer has exactly three behaviours: use it,
    confirm the marked fields, or look at the whole thing. A BLOCK on a
    non-gating field still costs a tier rather than vanishing — otherwise a
    record with a blocked load_id and a blocked commodity could still be
    published as fully trustworthy.

    Every published field starts at `"ok"` rather than being absent. An empty
    `field_status` cannot distinguish "we checked and it is fine" from "no rule
    ever looked at this", and those are different facts to a consumer deciding
    what to put in front of a human.
    """
    status: dict[str, FieldStatus] = dict.fromkeys(rules.PUBLISHED_FIELDS, "ok")
    worst_gating = None
    worst_other = None

    for f in findings:
        targets = f.fields or rules.GATING_ORDER  # record-scoped hits every gating field
        for name in targets:
            if f.severity is Severity.BLOCK:
                status[name] = "blocked"
            elif status.get(name) != "blocked":
                status[name] = "flagged"

            gating = name in GATING_FIELDS
            if gating:
                worst_gating = _worse(worst_gating, f.severity)
            else:
                worst_other = _worse(worst_other, f.severity)

    if worst_gating is Severity.BLOCK:
        return Confidence.LOW, status
    if worst_other is Severity.BLOCK or worst_gating is Severity.FLAG:
        return Confidence.MEDIUM, status
    return Confidence.HIGH, status


def _worse(current: Severity | None, candidate: Severity) -> Severity:
    if current is Severity.BLOCK or candidate is Severity.BLOCK:
        return Severity.BLOCK
    return Severity.FLAG


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def assemble(llm: LlmExtraction, source_text: str) -> Assembled:
    source_norm = normalize.norm_text(source_text)
    stops = normalize.select_stops(llm.stops)
    stated_total = normalize.parse_money(llm.stated_total_text)
    charges = normalize.summarise_charges(llm.charges, stated_total)
    equipment, confident = normalize.classify_equipment(llm.equipment_text)

    # The first printed commodity row is the published one. `multi_commodity`
    # flags the collapse; nothing here sums weights across rows, because the
    # same line repeats under each stop on these documents.
    first = llm.commodities[0] if llm.commodities else None
    commodity_text = first.description_text if first else None
    weight_text = first.weight_text if first else None
    weight, unit_stated = normalize.parse_weight_lbs(weight_text)

    # The pickup date comes from the stop we selected as the origin, never from
    # the header. Reading it from the header on a multi-pickup load publishes an
    # origin city paired with a different stop's appointment date, and a truck
    # dispatched on that will be a thousand miles from where it should be.
    pickup = normalize.resolve_date(stops.origin.date_text if stops.origin else None, source_text)
    delivery = normalize.resolve_date(
        stops.destination.date_text if stops.destination else None, source_text
    )
    return Assembled(
        llm=llm,
        source_text=source_text,
        source_norm=source_norm,
        stops=stops,
        charges=charges,
        stated_total=stated_total,
        pickup=pickup,
        delivery=delivery,
        equipment=equipment,
        equipment_confident=confident,
        commodity_text=commodity_text,
        weight_text=weight_text,
        weight_lbs=weight,
        weight_unit_stated=unit_stated,
    )


def _address(stop: Stop | None) -> Address | None:
    """A whitespace-only city or state is missing, not present.

    Every part is stripped, not just city and state: an un-stripped `zip` puts
    leading whitespace into a field a downstream system will match on.
    """
    if stop is None or not (stop.city_text or "").strip() or not (stop.state_text or "").strip():
        return None
    zip_text = (stop.zip_text or "").strip()
    return Address(
        city=(stop.city_text or "").strip(),
        state=(stop.state_text or "").strip(),
        zip=zip_text or None,
    )


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d else None


def publish(a: Assembled, confidence: Confidence) -> RateConfirmation:
    return RateConfirmation(
        load_id=_clean_str(a.llm.load_id_text),
        origin=_address(a.stops.origin),
        destination=_address(a.stops.destination),
        pickup_date=_iso(a.pickup.value),
        delivery_date=_iso(a.delivery.value),
        equipment_type=a.equipment,
        line_haul_rate=a.charges.line_haul,
        # Never imputed from an unlabelled accessorial, and never derived as
        # total - line_haul: that is the same hallucination performed with a
        # calculator, and it would always reconcile.
        fuel_surcharge=a.charges.fuel,
        total_rate=a.stated_total,
        weight_lbs=a.weight_lbs,
        commodity=_clean_str(a.commodity_text),
        confidence=confidence,
    )


def _clean_str(s: str | None) -> str | None:
    """Strip, and treat whitespace-only as absent — the same rule `_address`
    applies, applied to every published string rather than only to two of them.
    """
    if s is None:
        return None
    stripped = s.strip()
    return stripped or None


def _empty(confidence: Confidence = Confidence.LOW) -> RateConfirmation:
    return RateConfirmation(
        load_id=None,
        origin=None,
        destination=None,
        pickup_date=None,
        delivery_date=None,
        equipment_type=None,
        line_haul_rate=None,
        fuel_surcharge=None,
        total_rate=None,
        weight_lbs=None,
        commodity=None,
        confidence=confidence,
    )


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def extract(text: str, client: LLMClient) -> ExtractionResult:
    """Total with respect to `Exception`. `KeyboardInterrupt` and `SystemExit`
    propagate, because swallowing those is worse than dying.

    The deterministic half runs *inside* the try, which is the whole point. It
    used to sit outside — on the reasoning that normalisation is pure Python and
    therefore safe — and that reasoning was wrong: every input to it is model
    output, and `Decimal.quantize` on a hallucinated digit run raised straight
    through the guarantee. A boundary that only holds for the code you happened
    to think about is not a boundary.
    """
    meta: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "document_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
    }
    if not text.strip():
        return _failed(meta, "empty_input")
    try:
        completion: RawCompletion = client.complete(text)
        meta.update(completion.meta)
        llm, repairs = parse_and_repair(completion, client, text)
        meta["repairs"] = repairs
    except ExtractionError as e:
        return _failed(meta, e.reason)
    except Exception as e:
        return _failed(meta, "unexpected_error", e)

    try:
        a = assemble(llm, text)
        findings = rules.audit(a)
        confidence, field_status = route(findings)
        data = publish(a, confidence)
    except Exception as e:
        # Distinct from `unexpected_error` above: that reason means the provider
        # call failed, this one means our own reading of a valid payload did.
        # Collapsing them would make a normaliser bug indistinguishable from an
        # outage on the dashboard, which is the same mistake as collapsing
        # `status` into `confidence`.
        return _failed(meta, "assembly_error", e)

    return ExtractionResult(data, field_status, findings, confidence, "ok", meta)


def _failed(meta: dict[str, Any], reason: str, error: Exception | None = None) -> ExtractionResult:
    detail: dict[str, Any] = {**meta, "reason": reason}
    if error is not None:
        detail["error_type"] = type(error).__name__
        # A digest rather than the traceback: enough to group identical crashes
        # in the logs without putting document text into them.
        detail["traceback_digest"] = hashlib.sha256(traceback.format_exc().encode()).hexdigest()[
            :12
        ]
    return ExtractionResult(_empty(), {}, [], Confidence.LOW, "failed", detail)


def extract_file(path: Path, client: LLMClient) -> ExtractionResult:
    """The CLI's total boundary. PDF parsing lives outside `extract()`, so
    without this a malformed PDF would crash the actual user entry point while
    the guarantee technically held.
    """
    from ratecon.pdf import PdfError, pdf_to_text

    try:
        text = pdf_to_text(path) if path.suffix.lower() == ".pdf" else path.read_text()
    except PdfError as e:
        return ExtractionResult(_empty(), {}, [], Confidence.LOW, "failed", {"reason": e.reason})
    except Exception as e:
        return ExtractionResult(
            _empty(),
            {},
            [],
            Confidence.LOW,
            "failed",
            {"reason": "read_error", "error_type": type(e).__name__},
        )
    result = extract(text, client)
    result.meta["source"] = path.name
    return result


def confirm_fields(result: ExtractionResult) -> list[str]:
    """The fields a human is being asked to look at, for the CLI summary."""
    return sorted(k for k, v in result.field_status.items() if v in ("flagged", "blocked"))


__all__ = [
    "Decimal",
    "ExtractionResult",
    "assemble",
    "confirm_fields",
    "extract",
    "extract_file",
    "publish",
    "route",
]
