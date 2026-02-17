#!/usr/bin/env python3
"""
WikiProject Medicine — Reference Density Analyzer
===================================================
Fetches ALL articles tagged under WikiProject Medicine on English Wikipedia
and computes reference density (refs per 1,000 words) for each one.

Output: CSV file with one row per article.

Requirements:
    pip install mwparserfromhell requests

Usage:
    python wikiproject_medicine_refdensity.py

    # Resume a previous interrupted run:
    python wikiproject_medicine_refdensity.py --resume

    # Limit to N articles (for testing):
    python wikiproject_medicine_refdensity.py --limit 200

    # Custom output filename:
    python wikiproject_medicine_refdensity.py --output my_results.csv

The script runs in two phases:
  1. DISCOVERY  — Enumerates all articles from WikiProject Medicine quality
                  categories via the Wikipedia API (~2-5 minutes).
  2. ANALYSIS   — Fetches wikitext in batches of 50 and parses references
                  (~30-60 minutes for all ~50,000 articles).

Progress is saved automatically. If interrupted, re-run with --resume to
pick up where you left off.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required. Install it with: pip install requests")
    sys.exit(1)

try:
    import mwparserfromhell
except ImportError:
    print("ERROR: 'mwparserfromhell' is required. Install it with: pip install mwparserfromhell")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

API_URL = "https://en.wikipedia.org/w/api.php"
BATCH_SIZE = 50          # Wikipedia API max for titles per request
RATE_LIMIT = 0.3         # Seconds between API requests (be polite)
SAVE_EVERY = 200         # Save progress every N articles
PROGRESS_FILE = "refdensity_progress.json"

QUALITY_CLASSES = [
    "FA", "GA", "A", "B", "C", "Start", "Stub", "List",
    "FL", "Unassessed"
]
CATEGORY_TEMPLATE = "{quality}-Class_medicine_articles"


# ─────────────────────────────────────────────────────────────────────
# HTTP Session
# ─────────────────────────────────────────────────────────────────────

def create_session():
    """Create a requests session with proper headers."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "WikiProjectMedicineRefDensity/2.0 "
            "(https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Medicine; "
            "research tool; polite bot)"
        )
    })
    return s


# ─────────────────────────────────────────────────────────────────────
# Phase 1: Article Discovery
# ─────────────────────────────────────────────────────────────────────

def discover_articles(session):
    """
    Enumerate all articles in WikiProject Medicine by iterating through
    the quality-class categories.

    Returns dict: {article_title: quality_class}
    """
    all_articles = {}
    quality_counts = {}

    for quality in QUALITY_CLASSES:
        category = CATEGORY_TEMPLATE.format(quality=quality)
        print(f"  Fetching Category:{category} ...", end=" ", flush=True)

        members = []
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": 1,      # Talk namespace (where WikiProject banners live)
            "cmlimit": 500,        # Max per request
            "format": "json"
        }

        while True:
            try:
                resp = session.get(API_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if "query" in data and "categorymembers" in data["query"]:
                    for m in data["query"]["categorymembers"]:
                        title = m["title"]
                        if title.startswith("Talk:"):
                            title = title[5:]
                        members.append(title)

                if "continue" in data:
                    params["cmcontinue"] = data["continue"]["cmcontinue"]
                    time.sleep(0.1)
                else:
                    break

            except requests.exceptions.RequestException as e:
                print(f"\n    WARNING: API error on {category}: {e}")
                print("    Retrying in 5 seconds...")
                time.sleep(5)
                continue

        count = len(members)
        quality_counts[quality] = count
        for title in members:
            if title not in all_articles:
                all_articles[title] = quality

        print(f"{count:,} articles")
        time.sleep(0.2)

    return all_articles, quality_counts


# ─────────────────────────────────────────────────────────────────────
# Phase 2: Fetch & Analyze
# ─────────────────────────────────────────────────────────────────────

def fetch_wikitext(session, titles):
    """
    Fetch wikitext for up to 50 article titles in a single API call.
    Returns dict: {title: wikitext_string}
    """
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json"
    }

    for attempt in range(3):
        try:
            resp = session.get(API_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            results = {}
            if "query" in data and "pages" in data["query"]:
                for page_id, page_data in data["query"]["pages"].items():
                    if int(page_id) < 0:
                        continue
                    title = page_data.get("title", "")
                    if "revisions" in page_data:
                        content = (
                            page_data["revisions"][0]
                            .get("slots", {})
                            .get("main", {})
                            .get("*", "")
                        )
                        results[title] = content
            return results

        except requests.exceptions.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"\n    API error (attempt {attempt + 1}/3): {e}")
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)

    return {}


def analyze_wikitext(title, wikitext):
    """
    Parse one article's wikitext and extract reference metrics.

    Returns a dict with all computed fields, or None on failure.
    """
    try:
        parsed = mwparserfromhell.parse(wikitext)

        # ── Reference counting ──
        # Total <ref> tags (including self-closing reuses)
        ref_total = wikitext.count("<ref")

        # Named references: <ref name=...>...</ref>
        named_refs = len(re.findall(r'<ref\s+name\s*=', wikitext, re.IGNORECASE))

        # Self-closing reuses: <ref name="x" />  (these reuse an earlier ref)
        self_closing = len(re.findall(
            r'<ref\s+name\s*=[^/]*/\s*>', wikitext, re.IGNORECASE
        ))

        # Unique references = total minus reuses
        unique_refs = ref_total - self_closing

        # ── Citation templates ──
        templates = parsed.filter_templates()
        cite_templates = 0
        for t in templates:
            tname = str(t.name).strip().lower()
            if any(k in tname for k in ("cite", "citation", "harvnb", "sfn")):
                cite_templates += 1

        # ── Word count (from plain text) ──
        plain_text = parsed.strip_code()
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()
        word_count = len(plain_text.split())

        # ── Other metrics ──
        section_count = len(list(parsed.filter_headings()))
        char_count = len(wikitext)

        # ── Reference density ──
        if word_count > 0:
            ref_density = round(ref_total / word_count * 1000, 2)
            unique_ref_density = round(unique_refs / word_count * 1000, 2)
        else:
            ref_density = 0.0
            unique_ref_density = 0.0

        return {
            "title": title,
            "word_count": word_count,
            "char_count": char_count,
            "ref_total": ref_total,
            "unique_refs": unique_refs,
            "named_refs": named_refs,
            "self_closing_refs": self_closing,
            "cite_templates": cite_templates,
            "section_count": section_count,
            "ref_density_per_1k": ref_density,
            "unique_ref_density_per_1k": unique_ref_density,
            "status": "ok",
            "error_reason": "",
        }

    except Exception as e:
        return {
            "title": title,
            "word_count": 0, "char_count": 0,
            "ref_total": 0, "unique_refs": 0,
            "named_refs": 0, "self_closing_refs": 0,
            "cite_templates": 0, "section_count": 0,
            "ref_density_per_1k": 0.0,
            "unique_ref_density_per_1k": 0.0,
            "status": "error",
            "error_reason": f"parse_failed: {str(e)[:120]}",
        }


# ─────────────────────────────────────────────────────────────────────
# Progress Management
# ─────────────────────────────────────────────────────────────────────

def load_progress(path):
    """Load saved progress from a JSON file."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_progress(path, data):
    """Save progress to a JSON file (atomic write)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────
# CSV Output
# ─────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "title",
    "quality_class",
    "status",
    "error_reason",
    "word_count",
    "char_count",
    "ref_total",
    "unique_refs",
    "named_refs",
    "self_closing_refs",
    "cite_templates",
    "section_count",
    "ref_density_per_1k",
    "unique_ref_density_per_1k",
]


def write_csv(output_path, results):
    """
    Write all results to a CSV file.
    results: list of dicts (one per article)
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in sorted(results, key=lambda r: r["title"]):
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute reference density for all WikiProject Medicine articles."
    )
    parser.add_argument(
        "--output", "-o",
        default="wikiproject_medicine_reference_density.csv",
        help="Output CSV file path (default: wikiproject_medicine_reference_density.csv)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous interrupted run"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N articles (for testing)"
    )
    parser.add_argument(
        "--progress-file",
        default=PROGRESS_FILE,
        help=f"Progress file path (default: {PROGRESS_FILE})"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("WikiProject Medicine — Reference Density Analyzer")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    session = create_session()
    progress_path = args.progress_file

    # ── Load or start fresh ──
    progress = None
    if args.resume:
        progress = load_progress(progress_path)
        if progress:
            print(f"\nResuming from saved progress:")
            print(f"  Articles discovered: {len(progress['articles']):,}")
            print(f"  Articles processed:  {len(progress['processed']):,}")
        else:
            print("\nNo saved progress found. Starting fresh.")

    if not progress:
        # Clear any old progress
        if os.path.exists(progress_path):
            os.remove(progress_path)

        progress = {
            "articles": {},       # {title: quality_class}
            "processed": {},      # {title: result_dict}
            "quality_counts": {},
            "started": datetime.now().isoformat(),
        }

    # ── Phase 1: Discovery ──
    if not progress["articles"]:
        print("\n[Phase 1] Discovering WikiProject Medicine articles...")
        articles, quality_counts = discover_articles(session)
        progress["articles"] = articles
        progress["quality_counts"] = quality_counts
        save_progress(progress_path, progress)

        total = len(articles)
        print(f"\n  Total unique articles found: {total:,}")
        print(f"  Quality distribution:")
        for q in QUALITY_CLASSES:
            if quality_counts.get(q, 0) > 0:
                print(f"    {q:12s}: {quality_counts[q]:>6,}")
    else:
        total = len(progress["articles"])
        print(f"\n[Phase 1] Using cached article list ({total:,} articles)")

    # ── Phase 2: Analysis ──
    print("\n[Phase 2] Fetching and analyzing articles...")

    articles = progress["articles"]
    processed = progress["processed"]

    # Determine what still needs processing
    to_process = [t for t in articles if t not in processed]

    if args.limit:
        to_process = to_process[:args.limit]

    remaining = len(to_process)
    already_done = len(processed)

    print(f"  Already processed: {already_done:,}")
    print(f"  Remaining:         {remaining:,}")

    if remaining == 0:
        print("  Nothing to process!")
    else:
        start_time = time.time()
        batch_count = 0
        new_count = 0
        error_count = 0

        for i in range(0, remaining, BATCH_SIZE):
            batch_titles = to_process[i:i + BATCH_SIZE]
            batch_count += 1

            # Fetch wikitext
            wikitext_map = fetch_wikitext(session, batch_titles)

            # Analyze each article
            for title in batch_titles:
                quality = articles[title]

                if title in wikitext_map:
                    result = analyze_wikitext(title, wikitext_map[title])
                    result["quality_class"] = quality
                    processed[title] = result
                    if result["status"] == "ok":
                        new_count += 1
                    else:
                        error_count += 1
                else:
                    # Article not found (may have been deleted/moved)
                    processed[title] = {
                        "title": title,
                        "quality_class": quality,
                        "status": "skipped",
                        "error_reason": "not_found_in_api (deleted/moved/redirect)",
                        "word_count": 0, "char_count": 0,
                        "ref_total": 0, "unique_refs": 0,
                        "named_refs": 0, "self_closing_refs": 0,
                        "cite_templates": 0, "section_count": 0,
                        "ref_density_per_1k": 0.0,
                        "unique_ref_density_per_1k": 0.0,
                    }
                    error_count += 1

            # ── Progress display ──
            total_done = already_done + new_count + error_count
            pct = (total_done / len(articles)) * 100

            elapsed = time.time() - start_time
            rate = (new_count + error_count) / elapsed if elapsed > 0 else 0
            items_left = remaining - (new_count + error_count)
            eta_secs = items_left / rate if rate > 0 else 0
            eta = str(timedelta(seconds=int(eta_secs)))

            if batch_count % 10 == 0 or batch_count <= 3 or (new_count + error_count) >= remaining:
                print(
                    f"  [{total_done:>6,} / {len(articles):,}] "
                    f"{pct:5.1f}%  |  "
                    f"{rate:.0f} art/s  |  "
                    f"ETA: {eta}  |  "
                    f"Errors: {error_count}"
                )

            # ── Save progress periodically ──
            if (new_count + error_count) % SAVE_EVERY == 0:
                progress["processed"] = processed
                save_progress(progress_path, progress)

            # Rate limiting
            time.sleep(RATE_LIMIT)

        # Final save
        progress["processed"] = processed
        progress["finished"] = datetime.now().isoformat()
        save_progress(progress_path, progress)

        elapsed = time.time() - start_time
        print(f"\n  Processed {new_count + error_count:,} articles in "
              f"{timedelta(seconds=int(elapsed))}")
        if error_count > 0:
            print(f"  ({error_count:,} errors/missing articles)")

    # ── Phase 3: Write CSV ──
    print(f"\n[Phase 3] Writing CSV to: {args.output}")

    results_list = list(processed.values())
    write_csv(args.output, results_list)

    print(f"  Wrote {len(results_list):,} rows")

    # ── Summary ──
    ok = [r for r in results_list if r.get("status") == "ok"]
    errored = [r for r in results_list if r.get("status") == "error"]
    skipped = [r for r in results_list if r.get("status") == "skipped"]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total articles in CSV: {len(results_list):,}")
    print(f"    Successful (ok):     {len(ok):,}")
    print(f"    Parse errors:        {len(errored):,}")
    print(f"    Skipped (not found): {len(skipped):,}")

    if ok:
        densities = [r["ref_density_per_1k"] for r in ok]
        densities.sort()
        n = len(densities)
        mean_d = sum(densities) / n
        median_d = densities[n // 2] if n % 2 == 1 else (densities[n // 2 - 1] + densities[n // 2]) / 2
        q25 = densities[int(n * 0.25)]
        q75 = densities[int(n * 0.75)]
        total_refs = sum(r["ref_total"] for r in ok)
        total_words = sum(r["word_count"] for r in ok)
        zero_refs = sum(1 for r in ok if r["ref_total"] == 0)

        print(f"\n  Successful articles ({n:,}):")
        print(f"    Total references:    {total_refs:,}")
        print(f"    Total words:         {total_words:,}")
        print(f"    Zero-reference:      {zero_refs:,} ({zero_refs / n * 100:.1f}%)")
        print()
        print(f"  Reference Density (per 1,000 words):")
        print(f"    Mean:     {mean_d:.2f}")
        print(f"    Median:   {median_d:.2f}")
        print(f"    Q25:      {q25:.2f}")
        print(f"    Q75:      {q75:.2f}")
        print(f"    Min:      {densities[0]:.2f}")
        print(f"    Max:      {densities[-1]:.2f}")

        # By quality class
        print()
        print(f"  By Quality Class (successful only):")
        by_q = defaultdict(list)
        for r in ok:
            by_q[r["quality_class"]].append(r["ref_density_per_1k"])
        for q in QUALITY_CLASSES:
            if q in by_q:
                ds = by_q[q]
                ds.sort()
                qn = len(ds)
                qmean = sum(ds) / qn
                qmed = ds[qn // 2]
                print(f"    {q:12s}: n={qn:>6,}  mean={qmean:6.1f}  median={qmed:6.1f}")

    if errored or skipped:
        print(f"\n  Failed/Skipped articles ({len(errored) + len(skipped):,}):")
        if errored:
            print(f"    Parse errors (status='error'):")
            for r in errored[:5]:
                print(f"      - {r['title']}: {r.get('error_reason', 'unknown')}")
            if len(errored) > 5:
                print(f"      ... and {len(errored) - 5:,} more")
        if skipped:
            print(f"    Not found (status='skipped'):")
            for r in skipped[:5]:
                print(f"      - {r['title']}")
            if len(skipped) > 5:
                print(f"      ... and {len(skipped) - 5:,} more")

    print(f"\n  Output file: {os.path.abspath(args.output)}")
    print(f"  Progress:    {os.path.abspath(progress_path)}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
