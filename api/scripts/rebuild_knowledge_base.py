#!/usr/bin/env python3
"""Extract text from PDFs and rebuild the ChromaDB knowledge base.

Usage:
    source venv312/bin/activate
    python scripts/rebuild_knowledge_base.py [--extract-only] [--ingest-only]
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
PDF_DIR = KB_DIR / "pdf"
TEXT_DIR = KB_DIR / "text"
DB_DIR = KB_DIR / "chromadb"
COLLECTION_NAME = "orchestration_rules"

# PDF metadata for proper source attribution
PDF_SOURCES = {
    "beethoven_piccolo_unt": {
        "composer": "beethoven",
        "title": "The Role of the Piccolo in Beethoven's Orchestration",
        "source_type": "dissertation",
        "institution": "University of North Texas",
    },
    "beethoven_5th_bilkent": {
        "composer": "beethoven",
        "title": "Conducting Beethoven's Fifth Symphony - Orchestration Analysis",
        "source_type": "thesis",
        "institution": "Bilkent University",
    },
    "beethoven_trombone": {
        "composer": "beethoven",
        "title": "Position and Function of Trombone in Beethoven's Symphonies",
        "source_type": "paper",
        "institution": "Academic Journal",
    },
    "mahler_9th_orchestration": {
        "composer": "mahler",
        "title": "Mahler's Symphony No. 9 - Analysis of Orchestration",
        "source_type": "paper",
        "institution": "Atlantis Press",
    },
    "mahler_6th_cuny": {
        "composer": "mahler",
        "title": "Analysis of Mahler's Sixth Symphony",
        "source_type": "thesis",
        "institution": "CUNY",
    },
    "mahler_1st_orchestration": {
        "composer": "mahler",
        "title": "Orchestration Techniques in Mahler's First Symphony",
        "source_type": "paper",
        "institution": "Academic Journal",
    },
    "mahler_symphonies_bekker": {
        "composer": "mahler",
        "title": "Gustav Mahler's Symphonies - Paul Bekker",
        "source_type": "book",
        "institution": "Conservatorio Rossini",
    },
    "mahler_identity_umass": {
        "composer": "mahler",
        "title": "Gustav Mahler's Symphonies and the Search for Identity",
        "source_type": "thesis",
        "institution": "University of Massachusetts",
    },
    "orchestral_texture_corpus": {
        "composer": "various",
        "title": "A Corpus Describing Orchestral Texture (Haydn/Mozart/Beethoven)",
        "source_type": "paper",
        "institution": "HAL Science",
    },
    "mozart_symphonies_unt": {
        "composer": "mozart",
        "title": "Forms of Mozart's Symphonies",
        "source_type": "thesis",
        "institution": "University of North Texas",
    },
    "musical_analysis_absil": {
        "composer": "various",
        "title": "Musical Analysis - Visiting the Great Composers",
        "source_type": "book",
        "institution": "Frans Absil",
    },
    "dvorak_new_world_liberty": {
        "composer": "dvorak",
        "title": "Analysis of Dvorak's New World Symphony",
        "source_type": "thesis",
        "institution": "Liberty University",
    },
    "mahler_richmond": {
        "composer": "mahler",
        "title": "Gustav Mahler - Orchestration and Identity",
        "source_type": "thesis",
        "institution": "University of Richmond",
    },
    "mahler_das_lied_tarrh": {
        "composer": "mahler",
        "title": "Das Lied von der Erde - Orchestration and Analysis",
        "source_type": "paper",
        "institution": "Academic Analysis (Tarrh)",
    },
    "mahler_orchestration_rules": {
        "composer": "mahler",
        "title": "Mahler Orchestration Rules and Techniques",
        "source_type": "rules",
        "institution": "Compiled from academic sources",
    },
    # Tchaikovsky
    "tchaikovsky_horn_sym5_tennessee": {
        "composer": "tchaikovsky",
        "title": "The Horn in Tchaikovsky's Symphony No. 5",
        "source_type": "thesis",
        "institution": "University of Tennessee",
    },
    "tchaikovsky_sym1_sym6_umaine": {
        "composer": "tchaikovsky",
        "title": "Comparative Analysis of Tchaikovsky's First and Sixth Symphonies",
        "source_type": "thesis",
        "institution": "University of Maine",
    },
    "tchaikovsky_orchestration_rules": {
        "composer": "tchaikovsky",
        "title": "Tchaikovsky Orchestration Rules and Techniques",
        "source_type": "rules",
        "institution": "Compiled from academic sources",
    },
    # Brahms
    "brahms_brass_unt": {
        "composer": "brahms",
        "title": "Brass Instruments as Used by Brahms in His Symphonies",
        "source_type": "thesis",
        "institution": "University of North Texas",
    },
    "brahms_symphonies_miami": {
        "composer": "brahms",
        "title": "The Symphonies of Johannes Brahms",
        "source_type": "thesis",
        "institution": "University of Miami",
    },
    "brahms_orchestration_rules": {
        "composer": "brahms",
        "title": "Brahms Orchestration Rules and Techniques",
        "source_type": "rules",
        "institution": "Compiled from academic sources",
    },
    # Ravel
    "ravel_harmonic_wesleyan": {
        "composer": "ravel",
        "title": "Harmonic Techniques in Maurice Ravel",
        "source_type": "thesis",
        "institution": "Wesleyan University",
    },
    "ravel_timbre_late_works": {
        "composer": "ravel",
        "title": "Ravel's Sound: Timbre and Orchestration in Late Works",
        "source_type": "paper",
        "institution": "Music Theory Online",
    },
    "ravel_orchestration_rules": {
        "composer": "ravel",
        "title": "Ravel Orchestration Rules and Techniques",
        "source_type": "rules",
        "institution": "Compiled from academic sources",
    },
    # Berlioz
    "berlioz_treatise_imslp": {
        "composer": "berlioz",
        "title": "Treatise on Instrumentation and Orchestration",
        "source_type": "treatise",
        "institution": "IMSLP (public domain)",
    },
    "berlioz_orchestration_rules": {
        "composer": "berlioz",
        "title": "Berlioz Orchestration Rules and Techniques",
        "source_type": "rules",
        "institution": "Compiled from academic sources",
    },
    # Stravinsky
    "stravinsky_theory_dimond": {
        "composer": "stravinsky",
        "title": "Introduction to Stravinsky - Theory of Music",
        "source_type": "paper",
        "institution": "Jonathan Dimond",
    },
    "stravinsky_orchestration_rules": {
        "composer": "stravinsky",
        "title": "Stravinsky Orchestration Rules and Techniques",
        "source_type": "rules",
        "institution": "Compiled from academic sources",
    },
    # Debussy
    "debussy_spatialization_bu": {
        "composer": "debussy",
        "title": "Spatialization in the Works of Claude Debussy",
        "source_type": "thesis",
        "institution": "Boston University",
    },
    "debussy_tonality_tufts": {
        "composer": "debussy",
        "title": "Debussy and the Veil of Tonality",
        "source_type": "paper",
        "institution": "Tufts University",
    },
    "debussy_orchestration_rules": {
        "composer": "debussy",
        "title": "Debussy Orchestration Rules and Techniques",
        "source_type": "rules",
        "institution": "Compiled from academic sources",
    },
}


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            text_parts.append(text)
    doc.close()
    return "\n\n".join(text_parts)


def clean_text(text: str) -> str:
    """Clean extracted PDF text for knowledge base ingestion."""
    # Remove excessive whitespace but keep paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page numbers (standalone numbers on their own line)
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    # Remove header/footer artifacts (short lines that repeat)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip very short lines that are likely headers/footers
        if len(stripped) < 3 and not stripped:
            cleaned_lines.append('')
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_text(text: str, source_name: str, target_words: int = 400, overlap_words: int = 50) -> list[dict]:
    """Split text into overlapping chunks for ChromaDB ingestion.

    Uses paragraph boundaries when possible, with word-count targeting.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = len(para.split())

        # If this single paragraph is huge, split it by sentences
        if para_words > target_words * 1.5:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_words = len(sent.split())
                if current_word_count + sent_words > target_words and current_chunk:
                    chunk_text_str = ' '.join(current_chunk)
                    chunks.append({
                        "text": chunk_text_str,
                        "word_count": len(chunk_text_str.split()),
                    })
                    # Keep overlap from end of current chunk
                    overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) > 2 else ''
                    current_chunk = [overlap_text] if overlap_text else []
                    current_word_count = len(overlap_text.split()) if overlap_text else 0
                current_chunk.append(sent)
                current_word_count += sent_words
        elif current_word_count + para_words > target_words and current_chunk:
            chunk_text_str = '\n\n'.join(current_chunk)
            chunks.append({
                "text": chunk_text_str,
                "word_count": len(chunk_text_str.split()),
            })
            # Keep last paragraph as overlap
            overlap = current_chunk[-1] if current_chunk else ''
            current_chunk = [overlap, para] if overlap else [para]
            current_word_count = len(overlap.split()) + para_words if overlap else para_words
        else:
            current_chunk.append(para)
            current_word_count += para_words

    # Don't forget the last chunk
    if current_chunk:
        chunk_text_str = '\n\n'.join(current_chunk)
        if len(chunk_text_str.split()) > 20:  # Skip tiny trailing chunks
            chunks.append({
                "text": chunk_text_str,
                "word_count": len(chunk_text_str.split()),
            })

    return chunks


def extract_all_pdfs():
    """Extract text from all PDFs in the pdf directory."""
    if not PDF_DIR.exists():
        print(f"No PDF directory found at {PDF_DIR}")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDFs to extract.\n")

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        txt_path = TEXT_DIR / f"{stem}.txt"

        if txt_path.exists():
            print(f"  SKIP {stem} (text file already exists)")
            continue

        print(f"  Extracting {stem}...", end=" ", flush=True)
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            cleaned = clean_text(raw_text)
            word_count = len(cleaned.split())

            if word_count < 100:
                print(f"WARNING: only {word_count} words extracted (may be scanned/image PDF)")
                continue

            # Write with a header showing the source
            meta = PDF_SOURCES.get(stem, {})
            header = f"## {meta.get('title', stem)}\n"
            header += f"## Source: {meta.get('institution', 'Unknown')} ({meta.get('source_type', 'unknown')})\n"
            header += f"## Composer: {meta.get('composer', 'unknown')}\n\n"

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(header + cleaned)

            print(f"OK ({word_count:,} words)")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\nExtraction complete.")


def rebuild_chromadb():
    """Rebuild the ChromaDB collection from all text files."""
    import chromadb

    text_files = sorted(TEXT_DIR.glob("*.txt"))
    # Skip the 'complete.txt' concatenated file to avoid duplicates
    text_files = [f for f in text_files if f.stem != "complete"]

    if not text_files:
        print("No text files found to ingest.")
        return

    print(f"Found {len(text_files)} text files to ingest.\n")

    # Prepare all chunks
    all_ids = []
    all_documents = []
    all_metadatas = []

    for txt_path in text_files:
        stem = txt_path.stem
        print(f"  Chunking {stem}...", end=" ", flush=True)

        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        meta_info = PDF_SOURCES.get(stem, {})
        chunks = chunk_text(text, stem)
        print(f"{len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            doc_id = f"{stem}_{i}"
            all_ids.append(doc_id)
            all_documents.append(chunk["text"])
            all_metadatas.append({
                "chunk_index": i,
                "source": stem,
                "chapter": txt_path.name,
                "word_count": chunk["word_count"],
                "composer": meta_info.get("composer", "rimsky-korsakov"),
                "source_type": meta_info.get("source_type", "textbook"),
                "institution": meta_info.get("institution", ""),
            })

    print(f"\nTotal: {len(all_ids)} chunks to ingest.")

    # Delete old collection and create fresh
    print(f"\nRebuilding ChromaDB at {DB_DIR}...")
    client = chromadb.PersistentClient(path=str(DB_DIR.resolve()))

    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted old collection.")
    except Exception:
        print("  No existing collection to delete.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Orchestration knowledge base - Rimsky-Korsakov + symphony analysis papers"},
    )

    # ChromaDB has a batch size limit, insert in batches of 100
    batch_size = 100
    for start in range(0, len(all_ids), batch_size):
        end = min(start + batch_size, len(all_ids))
        collection.add(
            ids=all_ids[start:end],
            documents=all_documents[start:end],
            metadatas=all_metadatas[start:end],
        )
        print(f"  Ingested {end}/{len(all_ids)} chunks...")

    print(f"\nDone! Collection '{COLLECTION_NAME}' now has {collection.count()} documents.")


def verify_knowledge_base():
    """Run test queries to verify the knowledge base works."""
    import chromadb

    client = chromadb.PersistentClient(path=str(DB_DIR.resolve()))
    collection = client.get_collection(COLLECTION_NAME)
    count = collection.count()
    print(f"\nKnowledge base has {count} documents.\n")

    test_queries = [
        "Beethoven symphony orchestration trombone brass",
        "Mahler symphony instrumentation technique",
        "Mozart symphony classical orchestration transparent texture",
        "melody doubling for violin",
        "romantic orchestration full rich sound",
        "Dvorak New World symphony instrumentation",
    ]

    for q in test_queries:
        results = collection.query(query_texts=[q], n_results=3)
        print(f"Query: {q}")
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            source = meta.get('source', '?')
            composer = meta.get('composer', '?')
            print(f"  [{i+1}] source={source} composer={composer} dist={dist:.3f}")
            print(f"      {doc[:120]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Rebuild Decimus knowledge base")
    parser.add_argument("--extract-only", action="store_true", help="Only extract PDFs to text")
    parser.add_argument("--ingest-only", action="store_true", help="Only rebuild ChromaDB from text files")
    parser.add_argument("--verify", action="store_true", help="Only run verification queries")
    args = parser.parse_args()

    if args.verify:
        verify_knowledge_base()
        return

    if args.extract_only:
        extract_all_pdfs()
        return

    if args.ingest_only:
        rebuild_chromadb()
        verify_knowledge_base()
        return

    # Full pipeline
    extract_all_pdfs()
    rebuild_chromadb()
    verify_knowledge_base()


if __name__ == "__main__":
    main()
