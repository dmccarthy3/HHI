#!/usr/bin/env python3
"""
PDF Scope Description Extractor
Reads multiple PDF documents, extracts Scope Descriptions using Claude AI,
and organizes them into a single Word document.
"""

import os
import sys
import glob
import argparse
import fitz  # PyMuPDF
import anthropic
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path


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


def extract_scope_description(client: anthropic.Anthropic, pdf_text: str, filename: str) -> str:
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

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        final = stream.get_final_message()

    # Extract text blocks (skip thinking blocks)
    result = ""
    for block in final.content:
        if block.type == "text":
            result = block.text.strip()
            break

    return result if result else "NO SCOPE DESCRIPTION FOUND"


def build_word_document(results: list[dict], output_path: str) -> None:
    """Build a Word document with all extracted scope descriptions."""
    doc = Document()

    # Title
    title = doc.add_heading("Scope Descriptions Summary", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.runs[0]
    title_run.font.color.rgb = RGBColor(0x1F, 0x39, 0x7D)

    # Subtitle / intro
    intro = doc.add_paragraph(
        f"This document compiles the Scope Descriptions extracted from {len(results)} PDF document(s)."
    )
    intro.alignment = WD_ALIGN_PARAGRAPH.CENTER
    intro.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    doc.add_paragraph()  # spacer

    found_count = sum(1 for r in results if r["scope"] != "NO SCOPE DESCRIPTION FOUND")
    not_found_count = len(results) - found_count

    # Table of Contents note
    toc_para = doc.add_paragraph()
    toc_run = toc_para.add_run(f"Documents processed: {len(results)}  |  Scopes found: {found_count}  |  Not found: {not_found_count}")
    toc_run.font.size = Pt(10)
    toc_run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    toc_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()  # spacer

    # Divider
    doc.add_paragraph("─" * 60)

    # Each document's scope
    for i, result in enumerate(results, 1):
        doc.add_paragraph()

        # Document heading
        heading = doc.add_heading(f"{i}. {result['filename']}", level=1)
        heading_run = heading.runs[0]
        heading_run.font.color.rgb = RGBColor(0x1F, 0x39, 0x7D)

        # Source path note
        path_para = doc.add_paragraph()
        path_run = path_para.add_run(f"Source: {result['path']}")
        path_run.font.size = Pt(9)
        path_run.font.italic = True
        path_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

        doc.add_paragraph()  # spacer before content

        scope_text = result["scope"]

        if scope_text == "NO SCOPE DESCRIPTION FOUND":
            no_scope = doc.add_paragraph()
            ns_run = no_scope.add_run("⚠ No Scope Description was found in this document.")
            ns_run.font.color.rgb = RGBColor(0xCC, 0x66, 0x00)
            ns_run.font.italic = True
        else:
            # Add scope content, preserving paragraph breaks
            paragraphs = scope_text.split("\n")
            for para_text in paragraphs:
                para_text = para_text.strip()
                if para_text:
                    # Detect sub-headings (lines that look like headings: short, no period at end, or ALL CAPS)
                    if (len(para_text) < 80 and not para_text.endswith(".") and
                            (para_text.isupper() or para_text.startswith("#") or
                             any(para_text.startswith(prefix) for prefix in
                                 ["Scope", "SCOPE", "Work", "WORK", "Project", "PROJECT"]))):
                        sub = doc.add_heading(para_text.lstrip("#").strip(), level=2)
                        sub.runs[0].font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
                    else:
                        p = doc.add_paragraph(para_text)
                        p.paragraph_format.space_after = Pt(4)
                else:
                    doc.add_paragraph()  # blank line

        # Divider between documents (except after last)
        if i < len(results):
            doc.add_paragraph()
            doc.add_paragraph("─" * 60)

    # Save
    doc.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Scope Descriptions from PDF documents and compile into a Word file."
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
        default="scope_descriptions.docx",
        help="Output Word document path (default: scope_descriptions.docx)."
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY environment variable)."
    )

    args = parser.parse_args()

    # Collect PDF files
    pdf_files = []

    if args.dir:
        pattern = os.path.join(args.dir, "**", "*.pdf")
        pdf_files.extend(glob.glob(pattern, recursive=True))

    for path_arg in args.pdf_paths:
        # Support glob patterns in arguments
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
        print("  python extract_scopes.py --dir /path/to/pdfs --output results.docx")
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

        # Extract raw text
        print(f"  Reading PDF text...")
        pdf_text = extract_text_from_pdf(pdf_path)

        if not pdf_text.strip():
            print(f"  Warning: No text extracted (may be a scanned/image PDF).")
            scope = "NO SCOPE DESCRIPTION FOUND — Document appears to contain no extractable text (possibly a scanned image PDF)."
        else:
            # Use Claude to find the scope description
            scope = extract_scope_description(client, pdf_text, filename)

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

    # Build Word document
    print(f"Building Word document: {args.output}")
    build_word_document(results, args.output)

    found = sum(1 for r in results if r["scope"] != "NO SCOPE DESCRIPTION FOUND")
    print(f"\nDone!")
    print(f"  Documents processed : {len(results)}")
    print(f"  Scopes found        : {found}")
    print(f"  Scopes not found    : {len(results) - found}")
    print(f"  Output saved to     : {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
