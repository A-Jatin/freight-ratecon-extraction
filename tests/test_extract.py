"""The LLM boundary: the cache, the wire schema, and the request we actually send."""

import json

import pytest
from pydantic import BaseModel, ConfigDict

from ratecon.extract import ExtractionError, OpenRouterClient, RawCompletion, ResponseCache
from ratecon.schema import LlmExtraction, to_wire_schema


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "cache")


def _client(cache, **kwargs):
    return OpenRouterClient(cache, allow_network=False, **kwargs)


def test_a_cached_response_is_served_without_the_network(cache):
    client = _client(cache)
    key = client._key_for("DOC")
    cache.put(key, RawCompletion('{"ok": true}', "tool_calls", {"recorded": "authored"}))
    hit = client.complete("DOC")
    assert hit.meta["cache"] == "hit"
    assert hit.meta["recorded"] == "authored"


def test_force_bypasses_the_cache_so_record_can_replace_authored_responses(cache):
    """Without this, `ratecon record` could never overwrite the committed
    responses — every document hit the cache, so the one documented way to swap
    authored answers for real ones was unreachable. The proof is that a forced
    call reaches the network guard instead of returning the entry.
    """
    client = _client(cache, force=True)
    cache.put(client._key_for("DOC"), RawCompletion('{"ok": true}', "tool_calls"))
    with pytest.raises(ExtractionError) as raised:
        client.complete("DOC")
    assert raised.value.reason == "cache_miss_offline"


def test_cache_writes_are_atomic(cache, tmp_path):
    """An interrupted `record` must not leave a half-written JSON that later
    deserialises into a plausible-looking truncated model response."""
    client = _client(cache)
    cache.put(client._key_for("DOC"), RawCompletion("{}", "tool_calls"))
    assert not list((tmp_path / "cache").glob(".*.tmp"))


# --------------------------------------------------------------------------
# The wire schema
# --------------------------------------------------------------------------


def test_the_wire_schema_carries_no_descriptions(cache):
    """Pydantic derives `description` from the class docstring, so without
    stripping it the model is sent our design rationale — `libpostal`, "two
    sources of truth", the reason `line_haul_rate` is absent. It also made the
    response cache brittle: the key hashes the request, so rewording one comment
    invalidated every committed entry.
    """
    schema = to_wire_schema(LlmExtraction)

    def keywords(node, inside_properties=False):
        if isinstance(node, dict):
            for k, v in node.items():
                if not inside_properties:
                    yield k
                yield from keywords(v, inside_properties=(k == "properties"))
        elif isinstance(node, list):
            for v in node:
                yield from keywords(v)

    assert "description" not in set(keywords(schema))
    wire = json.dumps(schema)
    assert "libpostal" not in wire
    assert "two sources of truth" not in wire


def test_the_wire_schema_is_shaped_for_strict_mode():
    wire = to_wire_schema(LlmExtraction)
    assert "$defs" not in wire and "$ref" not in json.dumps(wire)
    stops = wire["properties"]["stops"]["items"]
    assert stops["additionalProperties"] is False
    assert set(stops["required"]) == set(stops["properties"])
    # Optionality as a nullable type, never by omission from `required`.
    assert stops["properties"]["zip_text"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}


def test_a_field_named_title_is_not_deleted_from_the_schema():
    """`title` and `default` are schema keywords at one level and field *names*
    at another. Stripping them blindly deleted the field and then failed
    validation for a reason nothing in the payload explained."""

    class Awkward(BaseModel):
        model_config = ConfigDict(extra="forbid")
        title: str
        default: int

    shaped = to_wire_schema(Awkward)
    assert set(shaped["properties"]) == {"title", "default"}


# --------------------------------------------------------------------------
# The request
# --------------------------------------------------------------------------


def test_the_request_pins_a_model_and_forces_the_tool_call(cache):
    request = _client(cache)._request("DOC", None)
    assert request["model"].count("/") == 1 and "latest" not in request["model"]
    assert request["tool_choice"]["function"]["name"] == "record_rate_confirmation"
    assert request["tools"][0]["function"]["strict"] is True
    assert request["extra_body"]["provider"]["require_parameters"] is True
    # Reasoning-model families reject any non-default temperature outright, and a
    # request that 400s on the only network path is worse than one a shade less
    # deterministic.
    assert "temperature" not in request


def test_the_repair_request_feeds_back_structured_errors(cache):
    request = _client(cache)._request("DOC", '[{"loc": ["stops"], "msg": "field required"}]')
    assert len(request["messages"]) == 3
    assert "field required" in request["messages"][-1]["content"]


def test_the_cache_key_covers_the_whole_request(cache):
    """Keying on a hand-maintained version string instead would let an edited
    prompt serve a stale response while `meta` claimed the new one — and those
    rows are the monitoring substrate."""
    client = _client(cache)
    assert client._key_for("DOC A") != client._key_for("DOC B")
    other_model = OpenRouterClient(cache, model="anthropic/claude-3", allow_network=False)
    assert client._key_for("DOC A") != other_model._key_for("DOC A")


def test_record_only_takes_documents_from_an_also_directory(tmp_path):
    """A directory of documents usually also contains a README. Billing a
    provider call to classify our own prose as `document_type: "other"` is a
    small waste and a confusing cache entry — and it happened."""
    from ratecon.cli import DOCUMENT_SUFFIXES, KEEP_AUTHORED

    (tmp_path / "a_rate_con.txt").write_text("RATE CONFIRMATION")
    (tmp_path / "README.md").write_text("# notes")
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4")
    taken = [p.name for p in sorted(tmp_path.iterdir()) if p.suffix.lower() in DOCUMENT_SUFFIXES]
    assert taken == ["a_rate_con.txt", "scan.pdf"]
    # And the one fixture whose wrongness is the point is never re-recorded.
    assert "11_model_misreads_the_lane" in KEEP_AUTHORED
