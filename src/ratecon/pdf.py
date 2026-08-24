"""A thin, honest PDF adapter.

Thin because the assignment hands us raw text; honest because this layer is the
largest uncontrolled variance in the whole system, not the smallest. Naive text
extraction on a table-heavy rate confirmation interleaves columns and can
reattach `50.00` to the wrong label — which is how a silently wrong rate gets
created before the model is even involved. `layout=True` keeps column spacing,
which is the cheapest defence available.

What a production version needs, and this deliberately does not have: table
reconstruction via `extract_tables()`, and OCR for scanned or faxed documents.
"""

from pathlib import Path

PAGE_SEPARATOR = "\n\n[[page-break]]\n\n"  # cannot collide with a document value


class PdfError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def pdf_to_text(path: Path, max_pages: int = 25) -> str:
    """Extract layout-preserving text, or fail loudly with a reason.

    A PDF with no text layer is a scan, and a large share of real rate
    confirmations arrive as faxes. That must surface as its own run-level
    failure rather than as a document that merely looks empty — otherwise the
    monitoring dashboard cannot tell "we cannot read scans" from "this shipper's
    format broke the model".
    """
    try:
        import pdfplumber
    except ImportError as e:  # pragma: no cover
        raise PdfError("pdfplumber_missing") from e

    try:
        with pdfplumber.open(path) as pdf:
            pages = [
                page.extract_text(layout=True, x_tolerance=1.5, y_tolerance=3) or ""
                for page in pdf.pages[:max_pages]
            ]
            truncated = len(pdf.pages) > max_pages
    except Exception as e:
        raise PdfError(f"pdf_parse_error:{type(e).__name__}") from e

    text = PAGE_SEPARATOR.join(pages)
    if not text.strip():
        raise PdfError("no_text_layer_likely_scanned")
    if truncated:
        # Silent truncation is exactly the data loss this project exists to
        # prevent, so it is never silent.
        raise PdfError("too_many_pages")
    return text
