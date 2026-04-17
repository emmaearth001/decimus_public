#!/usr/bin/env python3
"""Clean and split the Berlioz Treatise on Instrumentation into chapter files.

Removes OCR artifacts, Google boilerplate, and splits by instrument sections.
"""

import re
from pathlib import Path

RAW_FILE = Path(__file__).parent.parent / "data/knowledge_base/text/berlioz/berlioz_raw.txt"
OUT_DIR = Path(__file__).parent.parent / "data/knowledge_base/text/berlioz"

# Section markers to look for (approximate, OCR may vary)
SECTIONS = [
    ("violins", ["violin", "violins"]),
    ("violas", ["viola", "violas"]),
    ("violoncellos", ["violoncello", "violoncellos", "cello"]),
    ("double_bass", ["double bass", "double basses", "contra-bass"]),
    ("harp", ["harp", "harps"]),
    ("flute", ["flute", "flutes"]),
    ("oboe", ["oboe", "oboes", "hautbois"]),
    ("clarinet", ["clarinet", "clarinets"]),
    ("bassoon", ["bassoon", "bassoons"]),
    ("horn", ["horn", "horns"]),
    ("trumpet", ["trumpet", "trumpets", "cornet"]),
    ("trombone", ["trombone", "trombones"]),
    ("tuba", ["tuba", "ophicleide"]),
    ("timpani", ["timpani", "kettle-drum", "kettledrum"]),
    ("percussion", ["percussion", "cymbals", "triangle", "tambourine", "drum"]),
    ("organ", ["organ"]),
    ("voices", ["voice", "voices", "soprano", "tenor", "baritone"]),
    ("orchestra", ["orchestra", "orchestration", "combination"]),
]


def clean_text(text: str) -> str:
    """Remove OCR artifacts and boilerplate."""
    lines = text.split('\n')

    # Skip Google boilerplate (first ~50 lines)
    start_idx = 0
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ['treatise', 'instrumentation', 'berlioz', 'chapter', 'part i']):
            if i > 20:  # Past the boilerplate
                start_idx = i
                break

    lines = lines[start_idx:]

    cleaned = []
    for line in lines:
        # Remove page numbers (standalone numbers)
        stripped = line.strip()
        if stripped.isdigit():
            continue
        # Remove very short OCR garbage lines (single chars, symbols)
        if len(stripped) <= 2 and not stripped.isalpha():
            continue
        # Remove Google watermark references
        if 'google' in stripped.lower() and len(stripped) < 100:
            continue
        if 'digitized' in stripped.lower():
            continue

        cleaned.append(line)

    text = '\n'.join(cleaned)

    # Fix common OCR errors
    text = re.sub(r'tlie', 'the', text)
    text = re.sub(r'liave', 'have', text)
    text = re.sub(r'wliich', 'which', text)
    text = re.sub(r'tliis', 'this', text)
    text = re.sub(r'witli', 'with', text)
    text = re.sub(r'mucli', 'much', text)
    text = re.sub(r'sucli', 'such', text)
    text = re.sub(r'eacli', 'each', text)
    text = re.sub(r'ricli', 'rich', text)

    # Collapse multiple blank lines
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text.strip()


def split_into_chapters(text: str) -> dict[str, str]:
    """Split text into rough chapter sections by instrument keywords."""
    lines = text.split('\n')
    chapters = {}
    current_section = "introduction"
    current_lines = []

    for line in lines:
        stripped = line.strip().upper()

        # Check if this line is a section header
        # Look for lines that are mostly uppercase and contain instrument names
        if len(stripped) > 3 and stripped == line.strip().upper():
            for section_name, keywords in SECTIONS:
                if any(kw.upper() in stripped for kw in keywords):
                    # Save current section
                    if current_lines:
                        content = '\n'.join(current_lines).strip()
                        if len(content) > 200:  # Skip tiny fragments
                            if current_section in chapters:
                                chapters[current_section] += '\n\n' + content
                            else:
                                chapters[current_section] = content

                    current_section = section_name
                    current_lines = [line]
                    break
            else:
                current_lines.append(line)
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        content = '\n'.join(current_lines).strip()
        if len(content) > 200:
            if current_section in chapters:
                chapters[current_section] += '\n\n' + content
            else:
                chapters[current_section] = content

    return chapters


def main():
    print(f"Reading {RAW_FILE}...")
    raw = RAW_FILE.read_text(encoding='utf-8', errors='replace')
    print(f"Raw: {len(raw)} chars, {len(raw.split())} words")

    print("Cleaning...")
    cleaned = clean_text(raw)
    print(f"Cleaned: {len(cleaned)} chars, {len(cleaned.split())} words")

    # Save the complete cleaned version
    complete_path = OUT_DIR / "berlioz_complete.txt"
    complete_path.write_text(cleaned, encoding='utf-8')
    print(f"Saved complete text: {complete_path}")

    # Split into chapters
    print("Splitting into sections...")
    chapters = split_into_chapters(cleaned)

    for section_name, content in chapters.items():
        filepath = OUT_DIR / f"berlioz_{section_name}.txt"
        filepath.write_text(content, encoding='utf-8')
        words = len(content.split())
        print(f"  {section_name}: {words} words -> {filepath.name}")

    print(f"\nTotal sections: {len(chapters)}")
    print("Done!")


if __name__ == "__main__":
    main()
