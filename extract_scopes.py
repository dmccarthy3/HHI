#!/usr/bin/env python3
"""
PDF Scope Description Extractor
Reads multiple PDF documents, extracts Scope Descriptions using Claude AI,
and organizes them into a single HTML document.
"""

import os
import sys
import glob
import time
import base64
import argparse
import html
import fitz  # PyMuPDF
import anthropic
from datetime import datetime
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-4-6"
REQUEST_DELAY = 3   # seconds between API calls to stay under rate limits
MAX_RETRIES = 4     # retry attempts on rate limit errors


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    text_parts = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text and text.strip():
                text_parts.append(f"[Page {page_num}]\n{text}")
        doc.close()
    except Exception as e:
        print(f"  Warning: Could not fully read {pdf_path}: {e}")
    return "\n\n".join(text_parts)


def _call_with_retry(client, model, max_tokens, messages):
    """Call the Claude API with exponential backoff on rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=messages
            ) as stream:
                final = stream.get_final_message()
            result = ""
            for block in final.content:
                if block.type == "text":
                    result = block.text.strip()
                    break
            return result
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)  # 2, 4, 8, 16 seconds
                print(f"  Rate limit hit. Waiting {wait}s before retry ({attempt + 2}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise


def extract_scope_description(client: anthropic.Anthropic, pdf_text: str, filename: str, model: str = DEFAULT_MODEL) -> str:
    """Use Claude to extract the Scope Description from PDF text."""
    print(f"  Extracting scope description from {filename}...")

    prompt = f"""You are reviewing a document called "{filename}". Your task is to find and extract the Scope Description (or Scope of Work) from this document.

Look for sections labeled:
- "Scope Description"
- "Scope of Work"
- "Scope"
- "Project Scope"
- "Work Scope"
- Or any similar heading that describes the scope of the project or work

Here is the document text:

{pdf_text}

Please extract ONLY the scope description content. If there are multiple scope-related sections, include all of them. Format your response as follows:
- Start directly with the scope content (no preamble like "Here is the scope...")
- Preserve the original wording as closely as possible
- If no scope description is found, respond with exactly: "NO SCOPE DESCRIPTION FOUND"
"""

    result = _call_with_retry(client, model, 2048, [{"role": "user", "content": prompt}])
    return result if result else "NO SCOPE DESCRIPTION FOUND"


def extract_scope_via_vision(client: anthropic.Anthropic, pdf_path: str, filename: str, model: str = DEFAULT_MODEL) -> str:
    """Use Claude vision to extract scope from a scanned/image-only PDF."""
    print(f"  Using vision extraction for scanned PDF: {filename}...")
    doc = fitz.open(pdf_path)
    content = []

    max_pages = min(len(doc), 10)  # cap at 10 pages to manage token cost
    for page_num in range(max_pages):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_data = base64.standard_b64encode(pix.tobytes("png")).decode()
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_data}
        })
    doc.close()

    content.append({
        "type": "text",
        "text": f"""You are reviewing a document called "{filename}". Find and extract the Scope Description (or Scope of Work) from these PDF page images.

Look for sections labeled "Scope Description", "Scope of Work", "Scope", "Project Scope", "Work Scope", or similar.

Extract ONLY the scope description content. If multiple scope sections exist, include all. Start directly with the content (no preamble). Preserve the original wording. If no scope description is found, respond with exactly: "NO SCOPE DESCRIPTION FOUND"."""
    })

    result = _call_with_retry(client, model, 2048, [{"role": "user", "content": content}])
    return result if result else "NO SCOPE DESCRIPTION FOUND"


def scope_to_html(scope_text: str) -> str:
    """Convert plain scope text to HTML paragraphs."""
    lines = scope_text.split("\n")
    parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        escaped = html.escape(line)
        # Detect sub-headings: short line, no trailing period, ALL CAPS or starts with a scope keyword
        if (len(line) < 80 and not line.endswith(".") and
                (line.isupper() or line.startswith("#") or
                 any(line.startswith(p) for p in ["Scope", "SCOPE", "Work", "WORK", "Project", "PROJECT"]))):
            parts.append(f'<h3 class="scope-subheading">{escaped.lstrip("#").strip()}</h3>')
        else:
            parts.append(f'<p>{escaped}</p>')
    return "\n".join(parts)


def build_html_document(results: list[dict], output_path: str) -> None:
    """Build an HTML document with all extracted scope descriptions."""
    found_count = sum(1 for r in results if r["scope"] != "NO SCOPE DESCRIPTION FOUND")
    not_found_count = len(results) - found_count
    generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # Build table-of-contents entries
    toc_items = []
    for i, result in enumerate(results, 1):
        anchor = f"doc-{i}"
        found = result["scope"] != "NO SCOPE DESCRIPTION FOUND"
        badge = '<span class="badge found">Found</span>' if found else '<span class="badge not-found">Not Found</span>'
        toc_items.append(
            f'<li><a href="#{anchor}">{html.escape(result["filename"])}</a> {badge}</li>'
        )
    toc_html = "\n".join(toc_items)

    # Build document sections
    sections = []
    for i, result in enumerate(results, 1):
        anchor = f"doc-{i}"
        filename_escaped = html.escape(result["filename"])
        path_escaped = html.escape(result["path"])
        scope_text = result["scope"]

        if scope_text == "NO SCOPE DESCRIPTION FOUND":
            content_html = '<p class="not-found-msg">⚠ No Scope Description was found in this document.</p>'
        else:
            content_html = scope_to_html(scope_text)

        sections.append(f"""
        <section class="document-section" id="{anchor}">
            <div class="section-header">
                <span class="doc-number">{i}</span>
                <h2>{filename_escaped}</h2>
            </div>
            <p class="source-path">Source: {path_escaped}</p>
            <div class="scope-content">
                {content_html}
            </div>
        </section>""")

    sections_html = "\n".join(sections)

    html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scope Descriptions Summary</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 15px;
            line-height: 1.7;
            color: #2c2c2c;
            background: #f4f6f9;
        }}

        /* ── Header ── */
        header {{
            background: linear-gradient(135deg, #1f397d 0%, #2e74b5 100%);
            color: white;
            padding: 48px 40px 36px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1rem;
            opacity: 0.85;
        }}

        /* ── Stats bar ── */
        .stats-bar {{
            display: flex;
            justify-content: center;
            gap: 32px;
            background: #fff;
            border-bottom: 1px solid #dde3ed;
            padding: 16px 40px;
            flex-wrap: wrap;
        }}

        .stat {{
            text-align: center;
        }}

        .stat .value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: #1f397d;
        }}

        .stat .label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #666;
        }}

        /* ── Layout ── */
        .container {{
            max-width: 960px;
            margin: 0 auto;
            padding: 32px 24px 60px;
        }}

        /* ── Table of contents ── */
        .toc {{
            background: #fff;
            border: 1px solid #dde3ed;
            border-radius: 8px;
            padding: 24px 28px;
            margin-bottom: 36px;
        }}

        .toc h2 {{
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #1f397d;
            margin-bottom: 14px;
        }}

        .toc ol {{
            padding-left: 20px;
        }}

        .toc li {{
            margin-bottom: 6px;
        }}

        .toc a {{
            color: #2e74b5;
            text-decoration: none;
            font-weight: 500;
        }}

        .toc a:hover {{
            text-decoration: underline;
        }}

        /* ── Badges ── */
        .badge {{
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 12px;
            vertical-align: middle;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .badge.found {{
            background: #e6f4ea;
            color: #1e7e34;
        }}

        .badge.not-found {{
            background: #fff3e0;
            color: #c65a00;
        }}

        /* ── Document sections ── */
        .document-section {{
            background: #fff;
            border: 1px solid #dde3ed;
            border-radius: 8px;
            margin-bottom: 28px;
            overflow: hidden;
        }}

        .section-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            background: #f0f4fb;
            border-bottom: 1px solid #dde3ed;
            padding: 16px 24px;
        }}

        .doc-number {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            background: #1f397d;
            color: white;
            border-radius: 50%;
            font-size: 0.85rem;
            font-weight: 700;
            flex-shrink: 0;
        }}

        .section-header h2 {{
            font-size: 1.05rem;
            font-weight: 600;
            color: #1f397d;
            word-break: break-word;
        }}

        .source-path {{
            font-size: 0.8rem;
            color: #888;
            font-style: italic;
            padding: 8px 24px 0;
        }}

        .scope-content {{
            padding: 20px 24px 24px;
        }}

        .scope-content p {{
            margin-bottom: 10px;
            color: #333;
        }}

        .scope-subheading {{
            font-size: 0.95rem;
            font-weight: 600;
            color: #2e74b5;
            margin: 18px 0 6px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}

        .not-found-msg {{
            color: #c65a00 !important;
            font-style: italic;
        }}

        /* ── Footer ── */
        footer {{
            text-align: center;
            font-size: 0.8rem;
            color: #999;
            padding: 20px;
            border-top: 1px solid #dde3ed;
            background: #fff;
        }}
    </style>
</head>
<body>

<header>
    <h1>Scope Descriptions Summary</h1>
    <p>Extracted from {len(results)} PDF document{'s' if len(results) != 1 else ''} using Claude AI</p>
</header>

<div class="stats-bar">
    <div class="stat">
        <div class="value">{len(results)}</div>
        <div class="label">Documents Processed</div>
    </div>
    <div class="stat">
        <div class="value">{found_count}</div>
        <div class="label">Scopes Found</div>
    </div>
    <div class="stat">
        <div class="value">{not_found_count}</div>
        <div class="label">Not Found</div>
    </div>
</div>

<div class="container">

    <nav class="toc">
        <h2>Documents</h2>
        <ol>
            {toc_html}
        </ol>
    </nav>

    {sections_html}

</div>

<footer>
    Generated on {generated_at} &nbsp;·&nbsp; PDF Scope Extractor powered by Claude AI
</footer>

</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_output)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Scope Descriptions from PDF documents and compile into an HTML file."
    )
    parser.add_argument(
        "pdf_paths",
        nargs="*",
        help="Path(s) to PDF file(s). Supports glob patterns (e.g., '*.pdf', 'docs/*.pdf')."
    )
    parser.add_argument(
        "--dir",
        help="Directory to search for PDF files (searches recursively)."
    )
    parser.add_argument(
        "--output", "-o",
        default="scope_descriptions.html",
        help="Output HTML file path (default: scope_descriptions.html)."
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY environment variable)."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})."
    )

    args = parser.parse_args()

    # Collect PDF files
    pdf_files = []

    if args.dir:
        pattern = os.path.join(args.dir, "**", "*.pdf")
        pdf_files.extend(glob.glob(pattern, recursive=True))

    for path_arg in args.pdf_paths:
        matches = glob.glob(path_arg, recursive=True)
        if matches:
            pdf_files.extend(matches)
        elif os.path.isfile(path_arg):
            pdf_files.append(path_arg)
        else:
            print(f"Warning: No files matched: {path_arg}")

    # Remove duplicates while preserving order
    seen = set()
    unique_pdfs = []
    for f in pdf_files:
        abs_f = os.path.abspath(f)
        if abs_f not in seen:
            seen.add(abs_f)
            unique_pdfs.append(f)
    pdf_files = unique_pdfs

    if not pdf_files:
        print("Error: No PDF files found. Provide PDF paths as arguments or use --dir.")
        print("\nUsage examples:")
        print("  python extract_scopes.py document1.pdf document2.pdf")
        print("  python extract_scopes.py *.pdf")
        print("  python extract_scopes.py --dir /path/to/pdfs")
        print("  python extract_scopes.py --dir /path/to/pdfs --output results.html")
        sys.exit(1)

    print(f"\nFound {len(pdf_files)} PDF file(s) to process:")
    for f in pdf_files:
        print(f"  - {f}")

    # Initialize Anthropic client
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nError: Anthropic API key required.")
        print("Set ANTHROPIC_API_KEY environment variable or use --api-key flag.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Process each PDF
    print(f"\nProcessing {len(pdf_files)} PDF(s) with Claude...\n")
    results = []

    for pdf_path in pdf_files:
        filename = Path(pdf_path).name
        print(f"[{len(results)+1}/{len(pdf_files)}] {filename}")

        print(f"  Reading PDF text...")
        pdf_text = extract_text_from_pdf(pdf_path)

        if not pdf_text.strip():
            print(f"  Warning: No text extracted. Trying vision-based extraction...")
            scope = extract_scope_via_vision(client, pdf_path, filename, model=args.model)
        else:
            scope = extract_scope_description(client, pdf_text, filename, model=args.model)

        time.sleep(REQUEST_DELAY)  # stay under rate limits

        if scope == "NO SCOPE DESCRIPTION FOUND":
            print(f"  Result: No scope description found.")
        else:
            preview = scope[:100].replace("\n", " ")
            print(f"  Result: Found scope description ({len(scope)} chars)")
            print(f"  Preview: {preview}...")

        results.append({
            "filename": filename,
            "path": os.path.abspath(pdf_path),
            "scope": scope
        })
        print()

    # Build HTML document
    print(f"Building HTML document: {args.output}")
    build_html_document(results, args.output)

    found = sum(1 for r in results if r["scope"] != "NO SCOPE DESCRIPTION FOUND")
    print(f"\nDone!")
    print(f"  Documents processed : {len(results)}")
    print(f"  Scopes found        : {found}")
    print(f"  Scopes not found    : {len(results) - found}")
    print(f"  Output saved to     : {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
