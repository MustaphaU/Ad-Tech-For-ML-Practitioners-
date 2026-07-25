#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Building PDF from Markdown chapters ==="

# Step 1: Create a combined markdown with mermaid blocks rendered as images
echo "[1/3] Extracting and rendering Mermaid diagrams..."

rm -rf mermaid_images
mkdir -p mermaid_images

# Use Python to extract mermaid blocks, render them, and create a PDF-ready markdown
python3 << 'PYEOF'
import re
import subprocess
import sys
import os

input_file = "combined.md"
output_file = "combined_for_pdf.md"
img_dir = "mermaid_images"

with open(input_file, "r") as f:
    content = f.read()

# Find all mermaid code blocks
pattern = r'```mermaid\n(.*?)```'
matches = list(re.finditer(pattern, content, re.DOTALL))

print(f"  Found {len(matches)} Mermaid diagrams")

new_content = content
offset = 0
rendered = 0
failed = 0

for i, match in enumerate(matches):
    mermaid_code = match.group(1).strip()
    img_name = f"diagram_{i:03d}.png"
    img_path = os.path.join(img_dir, img_name)
    mmd_path = os.path.join(img_dir, f"diagram_{i:03d}.mmd")

    # Write mermaid source
    with open(mmd_path, "w") as f:
        f.write(mermaid_code)

    # Render with mmdc
    try:
        result = subprocess.run(
            ["mmdc", "-i", mmd_path, "-o", img_path, "-b", "white", "-w", "1200", "-s", "2"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(img_path):
            # Replace the mermaid block with an image reference
            replacement = f"![](mermaid_images/{img_name}){{ width=95% }}"
            start = match.start() + offset
            end = match.end() + offset
            new_content = new_content[:start] + replacement + new_content[end:]
            offset += len(replacement) - (match.end() - match.start())
            rendered += 1
        else:
            print(f"  WARNING: Failed to render diagram {i}: {result.stderr[:200]}")
            # Leave a placeholder
            replacement = f"*[Diagram {i}: see combined.md for Mermaid source]*"
            start = match.start() + offset
            end = match.end() + offset
            new_content = new_content[:start] + replacement + new_content[end:]
            offset += len(replacement) - (match.end() - match.start())
            failed += 1
    except subprocess.TimeoutExpired:
        print(f"  WARNING: Timeout rendering diagram {i}")
        replacement = f"*[Diagram {i}: see combined.md for Mermaid source]*"
        start = match.start() + offset
        end = match.end() + offset
        new_content = new_content[:start] + replacement + new_content[end:]
        offset += len(replacement) - (match.end() - match.start())
        failed += 1

print(f"  Rendered: {rendered}, Failed: {failed}")

with open(output_file, "w") as f:
    f.write(new_content)

print(f"  Wrote {output_file}")
PYEOF

# Step 2: Generate PDF with pandoc
echo "[2/3] Generating PDF with pandoc..."

pandoc combined_for_pdf.md \
  -o ML_RL_Bidding_Textbook.pdf \
  --pdf-engine=xelatex \
  --include-in-header=preamble.tex \
  -M author="Nima Rafiee" \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V mainfont="Palatino" \
  -V sansfont="Helvetica Neue" \
  -V monofont="Menlo" \
  -V documentclass=report \
  -V classoption=openany \
  --toc \
  --toc-depth=2 \
  -V toc-title="Table of Contents" \
  --highlight-style=tango \
  --top-level-division=chapter \
  -V colorlinks=true \
  -V urlcolor=linkcolor \
  -V toccolor=linkcolor \
  --resource-path=. \
  2>&1

# Step 3: Clean up
echo "[3/3] Cleaning up..."
rm -f combined_for_pdf.md

echo ""
echo "=== Done! ==="
ls -lh ML_RL_Bidding_Textbook.pdf
