"""Shared drawing and prose primitives for the generated demo pages under `docs/demos/`.

The demos commit their figures, and a committed figure has to be *regenerable* — which is the whole
reason none of them is hand-drawn. This module is the other half of that rule: two demos drawing the
same axis labels from two copies of the same helper is how two pages in one repository come to
disagree about what a tick mark means.

**Why hand-written SVG rather than a plotting library.** The repository has no plotting dependency,
these figures are straight lines and text, and a reader has to be able to diff the committed output
against the numbers it claims. A library's SVG output is neither diffable nor readable.

**Why the palette is explicit.** GitHub renders a committed SVG as an `<img>`, where the reader's
dark theme never reaches the document's own CSS. A figure that leaves its background transparent and
draws dark text disappears for half its audience, so every page paints its own light ground.
"""

from __future__ import annotations

BACKGROUND = "#ffffff"
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#dc2626"

FONT_STACK = "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"


def escape(text: str) -> str:
    """XML-escape a label. Mostly names and numbers, but the figure is committed, so be exact."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text_element(
    x: float,
    y: float,
    content: str,
    *,
    size: float = 12,
    fill: str = INK,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    """One `<text>` node, with the family fixed so the committed file renders the same everywhere."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" '
        f'font-family="{FONT_STACK}"'
        f">{escape(content)}</text>"
    )


def thousands(value: float) -> str:
    """Group digits so a six-figure count reads as one at a glance."""
    return f"{value:,.0f}"


def decade_label(exponent: int) -> str:
    """Axis label for a power of ten, as a reader says it rather than as a formula."""
    if exponent < 3:
        return thousands(10**exponent)
    if exponent < 6:
        return f"{10 ** (exponent - 3)}K"
    return f"{10 ** (exponent - 6)}M"


def wrap(text: str, width: int) -> list[str]:
    """Greedy wrap by word count. SVG has no flow layout, so a caption wraps itself or overruns."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


MARKDOWN_WIDTH = 100
LIST_MARKER_PREFIXES = ("- ", "* ", "+ ")


def _is_list_item(stripped: str) -> bool:
    """A markdown list opener. `**bold` opens a paragraph, so the marker test needs the space."""
    if stripped.startswith(LIST_MARKER_PREFIXES):
        return True
    head, _, rest = stripped.partition(". ")
    return bool(rest) and head.isdigit()


def reflow(markdown: str, width: int = MARKDOWN_WIDTH) -> str:
    """Re-wrap prose paragraphs to the repo's 100 columns, leaving structure alone.

    These pages are assembled from f-strings whose interpolated numbers have no fixed width, so
    without this the committed file is ragged in a way that reads as carelessness and makes every
    regeneration a noisy diff. Tables, headings, fences, quotes, images and list items pass through
    untouched.
    """
    out: list[str] = []
    paragraph: list[str] = []
    in_fence = False

    def flush() -> None:
        if paragraph:
            out.extend(wrap(" ".join(paragraph), width))
            paragraph.clear()

    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        structural = (
            in_fence
            or not stripped
            or stripped.startswith(("|", "#", ">", "![", "```"))
            or _is_list_item(stripped)
        )
        if structural:
            flush()
            out.append(line)
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(out)
