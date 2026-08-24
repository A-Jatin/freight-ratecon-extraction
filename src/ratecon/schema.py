"""The two schemas, and the gap between them.

`LlmExtraction` is what we ask the model for: verbatim spans, nothing interpreted.
`RateConfirmation` is what we publish: the assignment's exact contract.

Everything interesting happens in between, in plain Python — see `normalize.py`
and `rules.py`. The model never sees a date type, a number type, or the words
"line haul", so it cannot silently resolve `3/4/26` or decide that a charge
labelled "Carrier Charge" is a fuel surcharge.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_serializer

# --------------------------------------------------------------------------
# What we ask the model for. Every value is a verbatim span from the document.
# --------------------------------------------------------------------------


class Stop(BaseModel):
    """One stop, as printed. `city/state/zip` are verbatim substrings of `address_text`.

    We ask the model to split the address rather than doing it in Python because
    US address parsing is `usaddress`/`libpostal` territory and a regex would be
    wrong. Splitting is span selection, which is the model's job here; it also
    means each part is independently checkable against the source, and it scopes
    the two-character state match to the address instead of the whole document
    (`OH` and `PA` both occur by accident in ordinary rate-con prose).
    """

    model_config = ConfigDict(extra="forbid")

    sequence: int | None
    kind: Literal["pickup", "delivery", "unknown"]
    address_text: str | None
    city_text: str | None
    state_text: str | None
    zip_text: str | None
    date_text: str | None


class Charge(BaseModel):
    """One rate-breakdown line, exactly as labelled. Classification happens in Python."""

    model_config = ConfigDict(extra="forbid")

    label_text: str | None
    amount_text: str | None


class LlmExtraction(BaseModel):
    """The wire contract.

    Note what is *absent*: `line_haul_rate` and `fuel_surcharge`. Those are derived
    from `charges[]` by an auditable keyword policy. Asking the model for them too
    would create two sources of truth with no precedence rule.
    """

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["rate_confirmation", "bol", "invoice", "other"]
    load_id_text: str | None
    commodity_text: str | None
    weight_text: str | None
    equipment_text: str | None
    rate_con_date_text: str | None
    header_pickup_date_text: str | None
    stated_total_text: str | None
    stops: list[Stop]
    charges: list[Charge]


# --------------------------------------------------------------------------
# What we publish. This is the assignment's schema, verbatim.
# --------------------------------------------------------------------------


class EquipmentType(StrEnum):
    VAN = "van"
    REEFER = "reefer"
    FLATBED = "flatbed"
    OTHER = "other"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Address(BaseModel):
    """The assignment types `city` and `state` as non-null and `zip` as nullable.

    We honour that: if city or state is unavailable the whole address is `None`
    rather than an `Address` with a fabricated field.
    """

    model_config = ConfigDict(frozen=True)

    city: str
    state: str
    zip: str | None


class RateConfirmation(BaseModel):
    """The assignment's exact JSON contract.

    Money is `Decimal` internally and serialised as a JSON number to match the
    spec. A string would be safer against float round-tripping in a consumer,
    but the spec says `number`, so we follow the spec and say so out loud.
    """

    model_config = ConfigDict(frozen=True)

    load_id: str | None
    origin: Address | None
    destination: Address | None
    pickup_date: str | None
    delivery_date: str | None
    equipment_type: EquipmentType | None
    line_haul_rate: Decimal | None
    fuel_surcharge: Decimal | None
    total_rate: Decimal | None
    weight_lbs: Decimal | None
    commodity: str | None
    confidence: Confidence

    @field_serializer("line_haul_rate", "fuel_surcharge", "total_rate", "weight_lbs")
    def _money_as_number(self, v: Decimal | None) -> float | None:
        return None if v is None else float(v)


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


class Severity(StrEnum):
    BLOCK = "block"  # the field is unusable
    FLAG = "flag"  # the field is usable, but confirm it


class Finding(BaseModel):
    """A scoped, machine-readable reason.

    `fields` is a tuple so a relational check (delivery before pickup) can impugn
    both dates; `()` means the finding is record-scoped.

    Equality deliberately ignores `message`: Pydantic's `__eq__` compares every
    field, so without this every reworded message would break every rule test.
    """

    model_config = ConfigDict(frozen=True)

    code: str
    severity: Severity
    fields: tuple[str, ...]
    message: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Finding):
            return NotImplemented
        return (self.code, self.fields) == (other.code, other.fields)

    def __hash__(self) -> int:
        return hash((self.code, self.fields))


# --------------------------------------------------------------------------
# Pydantic JSON Schema -> provider wire schema
# --------------------------------------------------------------------------


type Json = dict[str, Json] | list[Json] | str | int | float | bool | None


def to_wire_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Shape `model_json_schema()` into the subset strict providers accept.

    Pydantic emits `$defs`/`$ref` (rejected by Google's dialect), `title` and
    `default` keys (rejected in some strict paths), and only marks a field
    required if it has no default. Providers additionally require
    `additionalProperties: false` on every object and every key listed in
    `required`, with optionality expressed as a nullable type rather than by
    omission.
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    shaped = _clean(_inline(raw, defs))
    if not isinstance(shaped, dict):
        raise TypeError("a model schema must shape to an object")
    return shaped


def _inline(node: Json, defs: dict[str, Any]) -> Json:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = defs[ref.removeprefix("#/$defs/")]
            merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
            return _inline(merged, defs)
        return {k: _inline(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline(v, defs) for v in node]
    return node


def _clean(node: Json) -> Json:
    if isinstance(node, dict):
        out: dict[str, Json] = {
            k: _clean(v) for k, v in node.items() if k not in ("title", "default")
        }
        properties = out.get("properties")
        if out.get("type") == "object" and isinstance(properties, dict):
            out["additionalProperties"] = False
            out["required"] = list(properties)
        return out
    if isinstance(node, list):
        return [_clean(v) for v in node]
    return node
