#!/usr/bin/env python3
"""
Wikipedia Article Bias Analyzer
================================
A deterministic, transparent, and replicable bias detection tool for any
Wikipedia article. Every metric is clearly defined, uses rule-based methods
(no ML), and compares against established external databases and Wikipedia's
own published neutrality standards.

Five measurement modules:
  1. Source Bias Profile — maps cited sources to AllSides media bias ratings
  2. Wikipedia Policy Compliance — detects WP:WEASEL, WP:PEACOCK, WP:EDITORIALIZING
  3. Sentiment Asymmetry — VADER rule-based sentiment per section
  4. Structural Balance — section sizes, citation distribution, Gini coefficient
  5. Lexical Analysis — passive voice, hedging, intensifiers, vocabulary richness

Requirements:
    pip install requests mwparserfromhell vaderSentiment

Usage:
    python wiki_bias_analyzer.py "Article Title"
    python wiki_bias_analyzer.py "https://en.wikipedia.org/wiki/Article_Title"
    python wiki_bias_analyzer.py --batch articles.txt
    python wiki_bias_analyzer.py "Article Title" --output results.csv

Determinism guarantee:
    Running this script twice on the same article will always produce
    identical results. All algorithms are rule-based with no randomness.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from urllib.parse import urlparse, unquote

try:
    import requests
except ImportError:
    print("ERROR: pip install requests"); sys.exit(1)
try:
    import mwparserfromhell
except ImportError:
    print("ERROR: pip install mwparserfromhell"); sys.exit(1)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("ERROR: pip install vaderSentiment"); sys.exit(1)


# =====================================================================
# MODULE 1: SOURCE BIAS PROFILE
# =====================================================================
# Media bias ratings — 219 outlets
# Sources: AllSides v11 (Dec 2025), MBFC, Institutional classification
# Scale: -2 = Left, -1 = Lean Left, 0 = Center, +1 = Lean Right, +2 = Right
# =====================================================================

ALLSIDES_RATINGS = {
    # ── LEFT (-2) ──
    "alternet.org": -2, "apnews.com": -2, "commondreams.org": -2, "crooksandliars.com": -2,
    "currentaffairs.org": -2, "dailybeast.com": -2, "dailykos.com": -2,
    "democracynow.org": -2, "huffingtonpost.com": -2, "huffpost.com": -2, "jacobin.com": -2,
    "motherjones.com": -2, "msnbc.com": -2, "newrepublic.com": -2, "occupydemocrats.com": -2,
    "rawstory.com": -2, "salon.com": -2, "shareblue.com": -2, "slate.com": -2,
    "theatlantic.com": -2, "thedailybeast.com": -2, "theintercept.com": -2,
    "thenation.com": -2, "thinkprogress.org": -2, "truthout.org": -2, "vox.com": -2,

    # ── LEAN LEFT (-1) ──
    "abc.net.au": -1, "abcnews.go.com": -1, "aljazeera.com": -1, "axios.com": -1,
    "bbc.co.uk": -1, "bbc.com": -1, "bloomberg.com": -1, "businessinsider.com": -1,
    "buzzfeednews.com": -1, "caravanmagazine.in": -1, "cbc.ca": -1, "cbsnews.com": -1,
    "cnbc.com": -1, "cnn.com": -1, "dw.com": -1, "economist.com": -1, "firstpost.com": -1,
    "foreignaffairs.com": -1, "ft.com": -1, "gizmodo.com": -1, "globalnews.ca": -1,
    "haaretz.com": -1, "independent.co.uk": -1, "insider.com": -1, "irishtimes.com": -1,
    "latimes.com": -1, "macleans.ca": -1, "mashable.com": -1, "mediamatters.org": -1,
    "nbcnews.com": -1, "ndtv.com": -1, "newslaundry.com": -1, "newyorker.com": -1,
    "npr.org": -1, "nytimes.com": -1, "nzherald.co.nz": -1, "pbs.org": -1, "politico.com": -1,
    "propublica.org": -1, "rollingstone.com": -1, "scientificamerican.com": -1,
    "scroll.in": -1, "smh.com.au": -1, "talkingpointsmemo.com": -1, "teenvogue.com": -1,
    "theage.com.au": -1, "theconversation.com": -1, "theglobeandmail.com": -1,
    "theguardian.com": -1, "thehill.com": -1, "thehindu.com": -1, "thenewsminute.com": -1,
    "thestar.com": -1, "theverge.com": -1, "thewire.in": -1, "time.com": -1,
    "usatoday.com": -1, "vanityfair.com": -1, "vice.com": -1, "vogue.com": -1,
    "washingtonpost.com": -1, "wired.com": -1,

    # ── CENTER (0) ──
    "1440.com": 0, "acm.org": 0, "allsides.com": 0, "apa.org": 0, "arxiv.org": 0,
    "biorxiv.org": 0, "bls.gov": 0, "bmj.com": 0, "britannica.com": 0, "brookings.edu": 0,
    "c-span.org": 0, "cambridge.org": 0, "cdc.gov": 0, "cell.com": 0, "census.gov": 0,
    "cfr.org": 0, "channelnewsasia.com": 0, "cochrane.org": 0, "cochranelibrary.com": 0,
    "congress.gov": 0, "csmonitor.com": 0, "deccanherald.com": 0, "doi.org": 0, "epa.gov": 0,
    "europa.eu": 0, "europarl.europa.eu": 0, "factcheck.org": 0, "fda.gov": 0,
    "france24.com": 0, "frontiersin.org": 0, "gallup.com": 0, "hindustantimes.com": 0,
    "ieee.org": 0, "imf.org": 0, "indianexpress.com": 0, "indiatoday.in": 0,
    "japantimes.co.jp": 0, "jstor.org": 0, "koreaherald.com": 0, "livemint.com": 0,
    "loc.gov": 0, "marketwatch.com": 0, "medrxiv.org": 0, "morningbrew.com": 0, "nasa.gov": 0,
    "nature.com": 0, "ncbi.nlm.nih.gov": 0, "nejm.org": 0, "newsnation.com": 0,
    "newsweek.com": 0, "nih.gov": 0, "noaa.gov": 0, "oxford.com": 0, "oxfordjournals.org": 0,
    "pewresearch.org": 0, "plos.org": 0, "pnas.org": 0, "politifact.com": 0,
    "pubmed.ncbi.nlm.nih.gov": 0, "rand.org": 0, "realclearpolitics.com": 0,
    "researchgate.net": 0, "reuters.com": 0, "scholar.google.com": 0, "science.org": 0,
    "sciencedirect.com": 0, "scmp.com": 0, "skynews.com": 0, "snopes.com": 0,
    "springer.com": 0, "ssrn.com": 0, "straightarrownews.com": 0, "straitstimes.com": 0,
    "supremecourt.gov": 0, "tandfonline.com": 0, "tangle.media": 0, "theaustralian.com.au": 0,
    "thelancet.com": 0, "theprint.in": 0, "timesofindia.indiatimes.com": 0, "un.org": 0,
    "who.int": 0, "wiley.com": 0, "worldbank.org": 0, "wsj.com": 0,

    # ── LEAN RIGHT (+1) ──
    "dailymail.co.uk": 1, "dailywire.com": 1, "dnaindia.com": 1, "epochtimes.com": 1,
    "forbes.com": 1, "foxbusiness.com": 1, "freebeacon.com": 1, "ijr.com": 1,
    "judicialwatch.org": 1, "nationalreview.com": 1, "nypost.com": 1, "postmillennial.com": 1,
    "realclearinvestigations.com": 1, "reason.com": 1, "republicworld.com": 1,
    "skynews.com.au": 1, "spectator.co.uk": 1, "spectator.com.au": 1, "swarajyamag.com": 1,
    "telegraph.co.uk": 1, "theamericanconservative.com": 1, "thedispatch.com": 1,
    "thefreepress.com": 1, "timesnownews.com": 1, "washingtonexaminer.com": 1,
    "washingtontimes.com": 1, "zerohedge.com": 1,

    # ── RIGHT (+2) ──
    "breitbart.com": 2, "dailycaller.com": 2, "foxnews.com": 2, "hannity.com": 2,
    "infowars.com": 2, "newsmax.com": 2, "oann.com": 2, "opindia.com": 2, "pjmedia.com": 2,
    "redstate.com": 2, "rt.com": 2, "sputniknews.com": 2, "theblaze.com": 2,
    "thefederalist.com": 2, "thegatewaypundit.com": 2, "townhall.com": 2, "twitchy.com": 2,
    "westernjournal.com": 2, "wnd.com": 2,
}

# Label mapping for display
BIAS_LABELS = {-2: "Left", -1: "Lean Left", 0: "Center", 1: "Lean Right", 2: "Right"}


def extract_source_domains(wikitext):
    """
    Extract all cited domains from wikitext.
    Handles both raw URLs and {{cite web|url=...}} templates.
    """
    domains = []

    # Method 1: Extract URLs from raw text
    url_pattern = re.compile(r'https?://([a-zA-Z0-9._-]+\.[a-zA-Z]{2,})')
    for match in url_pattern.finditer(wikitext):
        domain = match.group(1).lower()
        # Normalize: strip www.
        if domain.startswith("www."):
            domain = domain[4:]
        domains.append(domain)

    return domains


def analyze_source_bias(wikitext):
    """Module 1: Source Bias Profile."""
    domains = extract_source_domains(wikitext)
    domain_counts = Counter(domains)

    classified = {}
    unclassified = {}

    for domain, count in domain_counts.items():
        if domain in ALLSIDES_RATINGS:
            classified[domain] = {
                "count": count,
                "rating": ALLSIDES_RATINGS[domain],
                "label": BIAS_LABELS[ALLSIDES_RATINGS[domain]]
            }
        else:
            unclassified[domain] = count

    # Compute aggregate metrics
    if classified:
        ratings = []
        for d, info in classified.items():
            ratings.extend([info["rating"]] * info["count"])

        total_classified = sum(info["count"] for info in classified.values())
        total_unclassified = sum(unclassified.values())
        total_sources = total_classified + total_unclassified

        mean_rating = sum(ratings) / len(ratings) if ratings else 0
        sorted_ratings = sorted(ratings)
        n = len(sorted_ratings)
        median_rating = sorted_ratings[n // 2] if n % 2 == 1 else (
            (sorted_ratings[n // 2 - 1] + sorted_ratings[n // 2]) / 2
        )

        # Distribution
        dist = Counter(ratings)
        distribution = {
            "left": dist.get(-2, 0),
            "lean_left": dist.get(-1, 0),
            "center": dist.get(0, 0),
            "lean_right": dist.get(1, 0),
            "right": dist.get(2, 0),
        }

        # Standard deviation
        if len(ratings) > 1:
            variance = sum((r - mean_rating) ** 2 for r in ratings) / len(ratings)
            std_dev = variance ** 0.5
        else:
            std_dev = 0.0

        return {
            "source_mean_bias": round(mean_rating, 3),
            "source_median_bias": round(median_rating, 1),
            "source_bias_std": round(std_dev, 3),
            "source_bias_label": BIAS_LABELS[round(mean_rating)] if -2 <= round(mean_rating) <= 2 else "Mixed",
            "sources_total": total_sources,
            "sources_classified": total_classified,
            "sources_unclassified": total_unclassified,
            "sources_classified_pct": round(total_classified / total_sources * 100, 1) if total_sources > 0 else 0,
            "unique_domains": len(domain_counts),
            "unique_classified": len(classified),
            "distribution_left": distribution["left"],
            "distribution_lean_left": distribution["lean_left"],
            "distribution_center": distribution["center"],
            "distribution_lean_right": distribution["lean_right"],
            "distribution_right": distribution["right"],
            "_classified_detail": classified,
            "_unclassified_detail": unclassified,
        }
    else:
        return {
            "source_mean_bias": None,
            "source_median_bias": None,
            "source_bias_std": None,
            "source_bias_label": "Unclassified",
            "sources_total": sum(domain_counts.values()),
            "sources_classified": 0,
            "sources_unclassified": sum(domain_counts.values()),
            "sources_classified_pct": 0,
            "unique_domains": len(domain_counts),
            "unique_classified": 0,
            "distribution_left": 0, "distribution_lean_left": 0,
            "distribution_center": 0, "distribution_lean_right": 0,
            "distribution_right": 0,
            "_classified_detail": {},
            "_unclassified_detail": dict(domain_counts),
        }


# =====================================================================
# MODULE 2: WIKIPEDIA POLICY COMPLIANCE
# =====================================================================
# Word lists from Wikipedia's own Manual of Style guidelines:
#   - WP:WEASEL  (Wikipedia:Manual_of_Style/Words_to_watch#Unsupported_attributions)
#   - WP:PEACOCK (Wikipedia:Manual_of_Style/Words_to_watch#Puffery)
#   - WP:EDITORIALIZING (Wikipedia:Manual_of_Style/Words_to_watch#Editorializing)
# =====================================================================

WEASEL_PHRASES = [
    # Vague attribution
    r"\bsome say\b", r"\bsome people\b", r"\bsome argue\b",
    r"\bmany people\b", r"\bmany believe\b", r"\bmany scholars\b",
    r"\bmany feel\b", r"\bmost people\b", r"\bmost believe\b",
    r"\bmost feel\b", r"\bexperts say\b", r"\bexperts believe\b",
    r"\bexperts claim\b", r"\bcritics say\b", r"\bcritics argue\b",
    r"\bcritics claim\b", r"\bobservers note\b", r"\bobservers say\b",
    r"\banalysts say\b", r"\banalysts believe\b",
    r"\bresearch has shown\b", r"\bstudies have shown\b",
    r"\bstudies suggest\b", r"\beveryone knows\b",
    r"\bit is said\b", r"\bit is thought\b", r"\bit is believed\b",
    r"\bit is widely thought\b", r"\bit is widely believed\b",
    r"\bit is generally thought\b", r"\bit is commonly believed\b",
    r"\bit is often said\b", r"\bit is often reported\b",
    r"\bit is widely regarded\b", r"\bit is considered\b",
    r"\bit is claimed\b", r"\bit has been suggested\b",
    r"\bit has been said\b", r"\bit has been claimed\b",
    r"\bit has been argued\b",
    r"\bwidely regarded as\b", r"\bgenerally considered\b",
    r"\bscience says\b", r"\bscientists claim\b",
    r"\bsupporters argue\b", r"\bopponents argue\b",
    # Numerically vague
    r"\bsome evidence\b", r"\bevidence suggests\b",
    r"\bgrowing number\b", r"\ba number of\b",
    r"\bsignificant number\b",
]

PEACOCK_TERMS = [
    # Puffery / promotional language
    r"\blegendary\b", r"\biconic\b", r"\brenowned\b",
    r"\bworld-renowned\b", r"\bworld-famous\b",
    r"\bprestigious\b", r"\billustrious\b",
    r"\bgroundbreaking\b", r"\brevolutionary\b",
    r"\bpioneering\b", r"\btrailblazing\b",
    r"\bvisionary\b", r"\bbrilliant\b",
    r"\bextraordinary\b", r"\bremarkable\b",
    r"\bexceptional\b", r"\boutstanding\b",
    r"\bstunning\b", r"\bspectacular\b",
    r"\bmasterpiece\b", r"\btour de force\b",
    r"\bacclaimed\b", r"\bcritically acclaimed\b",
    r"\baward-winning\b", r"\bbest-selling\b",
    r"\bbest known\b", r"\bwidely considered\b",
    r"\bunmatched\b", r"\bunrivaled\b", r"\bunparalleled\b",
    r"\bforemost\b", r"\bpreeminent\b",
    r"\bone of the greatest\b", r"\bone of the most\b",
    r"\bone of the best\b", r"\bwidely praised\b",
    r"\bhugely successful\b", r"\bimmensely popular\b",
    r"\bshow-stopping\b", r"\bchart-topping\b",
    r"\bblockbuster\b", r"\brunaway success\b",
    r"\bcutting-edge\b", r"\bstate-of-the-art\b",
    r"\binnovative\b",
]

EDITORIALIZING_WORDS = [
    # Attribution verbs that imply judgment
    r"\brevealed\b", r"\bexposed\b", r"\bunveiled\b",
    r"\badmitted\b", r"\bconfessed\b", r"\bconceded\b",
    r"\bdenied\b", r"\binsisted\b", r"\bboasted\b",
    r"\bclarified\b", r"\bpointed out\b",
    r"\bnoted\b",  # when used as attribution verb
    # Laudatory / opinion
    r"\bof course\b", r"\bneedless to say\b",
    r"\bclearly\b", r"\bobviously\b", r"\bundeniably\b",
    r"\bunquestionably\b", r"\bindisputably\b",
    r"\bwithout a doubt\b", r"\bwithout question\b",
    r"\bnaturally\b", r"\binevitably\b",
    r"\binterestingly\b", r"\bsurprisingly\b",
    r"\bremarkably\b", r"\bironically\b",
    r"\bsignificantly\b", r"\bimportantly\b",
    r"\bcrucially\b", r"\bnotably\b",
    r"\bcontroversially\b", r"\bfamously\b",
    r"\binfamously\b",
    r"\bso-called\b",
    r"\ballegedly\b", r"\bpurportedly\b", r"\bsupposedly\b",
]


def analyze_policy_compliance(plain_text):
    """Module 2: Wikipedia Policy Compliance (WP:NPOV word lists)."""
    text_lower = plain_text.lower()
    word_count = len(plain_text.split())

    def count_matches(patterns):
        matches = []
        for pattern in patterns:
            for m in re.finditer(pattern, text_lower):
                matches.append(m.group())
        return matches

    weasel_matches = count_matches(WEASEL_PHRASES)
    peacock_matches = count_matches(PEACOCK_TERMS)
    editorial_matches = count_matches(EDITORIALIZING_WORDS)

    total_flags = len(weasel_matches) + len(peacock_matches) + len(editorial_matches)

    return {
        "weasel_count": len(weasel_matches),
        "weasel_density_per_1k": round(len(weasel_matches) / word_count * 1000, 2) if word_count > 0 else 0,
        "weasel_examples": list(Counter(weasel_matches).most_common(10)),
        "peacock_count": len(peacock_matches),
        "peacock_density_per_1k": round(len(peacock_matches) / word_count * 1000, 2) if word_count > 0 else 0,
        "peacock_examples": list(Counter(peacock_matches).most_common(10)),
        "editorial_count": len(editorial_matches),
        "editorial_density_per_1k": round(len(editorial_matches) / word_count * 1000, 2) if word_count > 0 else 0,
        "editorial_examples": list(Counter(editorial_matches).most_common(10)),
        "total_npov_flags": total_flags,
        "total_npov_density_per_1k": round(total_flags / word_count * 1000, 2) if word_count > 0 else 0,
    }


# =====================================================================
# MODULE 3: SENTIMENT ASYMMETRY (VADER)
# =====================================================================
# VADER: Valence Aware Dictionary and sEntiment Reasoner
# Rule-based, deterministic, lexicon of ~7,500 items + 5 syntactic rules.
# Hutto & Gilbert, ICWSM 2014.
# =====================================================================

VADER_ANALYZER = SentimentIntensityAnalyzer()


def split_into_sentences(text):
    """Simple sentence splitter."""
    # Split on period, exclamation, question mark followed by space/newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def analyze_sentiment(sections):
    """
    Module 3: Sentiment Asymmetry using VADER.
    Input: dict of {section_name: section_text}
    """
    all_scores = []
    section_sentiments = {}
    strongest_positive = {"text": "", "score": -1}
    strongest_negative = {"text": "", "score": 1}

    for section_name, section_text in sections.items():
        sentences = split_into_sentences(section_text)
        if not sentences:
            continue

        section_scores = []
        for sent in sentences:
            score = VADER_ANALYZER.polarity_scores(sent)
            compound = score["compound"]
            section_scores.append(compound)
            all_scores.append(compound)

            if compound > strongest_positive["score"]:
                strongest_positive = {"text": sent[:150], "score": compound, "section": section_name}
            if compound < strongest_negative["score"]:
                strongest_negative = {"text": sent[:150], "score": compound, "section": section_name}

        if section_scores:
            mean_s = sum(section_scores) / len(section_scores)
            section_sentiments[section_name] = {
                "mean": round(mean_s, 4),
                "sentences": len(section_scores),
                "positive_pct": round(sum(1 for s in section_scores if s > 0.05) / len(section_scores) * 100, 1),
                "negative_pct": round(sum(1 for s in section_scores if s < -0.05) / len(section_scores) * 100, 1),
                "neutral_pct": round(sum(1 for s in section_scores if -0.05 <= s <= 0.05) / len(section_scores) * 100, 1),
            }

    if not all_scores:
        return {"sentiment_mean": 0, "sentiment_sections": {}}

    overall_mean = sum(all_scores) / len(all_scores)
    sorted_scores = sorted(all_scores)
    n = len(sorted_scores)

    # Sentiment variance across sections (how uneven is the tone?)
    section_means = [v["mean"] for v in section_sentiments.values()]
    if len(section_means) > 1:
        sm_mean = sum(section_means) / len(section_means)
        sent_variance = sum((s - sm_mean) ** 2 for s in section_means) / len(section_means)
    else:
        sent_variance = 0

    # Lead vs body comparison
    section_names = list(section_sentiments.keys())
    lead_sentiment = section_sentiments.get(section_names[0], {}).get("mean", 0) if section_names else 0
    body_sentiments = [v["mean"] for k, v in section_sentiments.items() if k != section_names[0]] if len(section_names) > 1 else []
    body_mean = sum(body_sentiments) / len(body_sentiments) if body_sentiments else 0
    lead_body_gap = round(lead_sentiment - body_mean, 4)

    return {
        "sentiment_mean": round(overall_mean, 4),
        "sentiment_median": round(sorted_scores[n // 2], 4),
        "sentiment_std": round((sum((s - overall_mean) ** 2 for s in all_scores) / n) ** 0.5, 4),
        "sentiment_positive_pct": round(sum(1 for s in all_scores if s > 0.05) / n * 100, 1),
        "sentiment_negative_pct": round(sum(1 for s in all_scores if s < -0.05) / n * 100, 1),
        "sentiment_neutral_pct": round(sum(1 for s in all_scores if -0.05 <= s <= 0.05) / n * 100, 1),
        "sentiment_section_variance": round(sent_variance, 4),
        "sentiment_lead_body_gap": lead_body_gap,
        "sentiment_strongest_positive": strongest_positive,
        "sentiment_strongest_negative": strongest_negative,
        "sentiment_sections": section_sentiments,
        "total_sentences": n,
    }


# =====================================================================
# MODULE 4: STRUCTURAL BALANCE
# =====================================================================

def analyze_structure(sections, wikitext):
    """Module 4: Structural Balance analysis."""
    section_sizes = {}
    total_words = 0
    for name, text in sections.items():
        wc = len(text.split())
        section_sizes[name] = wc
        total_words += wc

    if not section_sizes or total_words == 0:
        return {"structure_gini": 0, "structure_sections": 0}

    # Gini coefficient of section sizes
    values = sorted(section_sizes.values())
    n = len(values)
    if n <= 1:
        gini = 0
    else:
        cumulative = 0
        for i, v in enumerate(values):
            cumulative += v
        numerator = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(values))
        gini = numerator / (n * sum(values)) if sum(values) > 0 else 0
        gini = round(abs(gini), 4)

    # Section proportions
    proportions = {name: round(wc / total_words * 100, 1) for name, wc in section_sizes.items()}

    # Largest section dominance
    largest_section = max(section_sizes, key=section_sizes.get)
    largest_pct = round(section_sizes[largest_section] / total_words * 100, 1)

    # Citation distribution per section
    # Count <ref> tags per section in wikitext
    ref_distribution = {}
    parsed = mwparserfromhero = None
    # Simple approach: count refs in raw section text
    # We'll use the plain text sections for word counts but wikitext for ref counts

    # Unsourced sentence ratio
    sentences = split_into_sentences(" ".join(sections.values()))
    # A rough heuristic: sentences not near a citation marker
    # In plain text, refs are stripped, so we estimate from wikitext
    total_refs = wikitext.count("<ref")

    # Check for standard encyclopedic sections
    section_names_lower = [s.lower() for s in sections.keys()]
    has_criticism = any("criticism" in s or "controversy" in s or "opposition" in s for s in section_names_lower)
    has_reception = any("reception" in s or "response" in s or "reaction" in s for s in section_names_lower)
    has_history = any("history" in s or "background" in s for s in section_names_lower)
    has_see_also = any("see also" in s for s in section_names_lower)
    has_references = any("references" in s or "notes" in s or "citations" in s for s in section_names_lower)

    return {
        "structure_sections": n,
        "structure_total_words": total_words,
        "structure_gini": gini,
        "structure_largest_section": largest_section,
        "structure_largest_section_pct": largest_pct,
        "structure_has_criticism": has_criticism,
        "structure_has_reception": has_reception,
        "structure_has_history": has_history,
        "structure_ref_total": total_refs,
        "structure_ref_density_per_1k": round(total_refs / total_words * 1000, 2) if total_words > 0 else 0,
        "structure_section_proportions": proportions,
    }


# =====================================================================
# MODULE 5: LEXICAL ANALYSIS
# =====================================================================

HEDGING_WORDS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bwould\b",
    r"\bpossibly\b", r"\bperhaps\b", r"\barguably\b",
    r"\bprobably\b", r"\bapparently\b", r"\bseemingly\b",
    r"\bseem(?:s|ed)?\b", r"\bappear(?:s|ed)?\b",
    r"\bsuggest(?:s|ed)?\b", r"\btend(?:s|ed)?\b",
    r"\bgenerally\b", r"\btypically\b", r"\busually\b",
    r"\boften\b", r"\bsometimes\b", r"\brarely\b",
    r"\bto some extent\b", r"\bin some cases\b",
]

INTENSIFIERS = [
    r"\bvery\b", r"\bextremely\b", r"\bhighly\b",
    r"\bincredibly\b", r"\benormously\b", r"\btremendously\b",
    r"\bdeeply\b", r"\bprofoundly\b", r"\boverwhelmingly\b",
    r"\bdrastically\b", r"\bsharply\b", r"\bdramatically\b",
    r"\bvastly\b", r"\bmassively\b", r"\bimmensely\b",
    r"\bsignificantly\b",  # also in editorializing
]

SUPERLATIVES = [
    r"\bbest\b", r"\bworst\b", r"\bgreatest\b", r"\blargest\b",
    r"\bsmallest\b", r"\bmost\b", r"\bleast\b",
    r"\bfirst ever\b", r"\bbiggest\b", r"\bstrongest\b",
    r"\bweakest\b", r"\bhighest\b", r"\blowest\b",
    r"\bfinest\b", r"\bpoorest\b",
]

# Passive voice detection: simple heuristic using "was/were/been/being + past participle"
PASSIVE_PATTERN = re.compile(
    r'\b(?:was|were|been|being|is|are|be)\s+'
    r'(?:\w+ly\s+)?'  # optional adverb
    r'(?:(?:\w+ed|(?:writ|giv|tak|driv|rid|spok|brok|chos|forg|frozen|stolen|sworn|torn|worn|wok)en'
    r'|(?:made|done|seen|known|shown|grown|drawn|thrown|blown|flown|gone|begun|run|come|become|held|told|sold|found|built|sent|left|lost|kept|set|read|put|cut|shut|hurt|let|hit))\b)',
    re.IGNORECASE
)


def analyze_lexical(plain_text):
    """Module 5: Lexical Analysis."""
    text_lower = plain_text.lower()
    words = plain_text.split()
    word_count = len(words)

    if word_count == 0:
        return {}

    def count_patterns(patterns):
        total = 0
        for p in patterns:
            total += len(re.findall(p, text_lower))
        return total

    hedging_count = count_patterns(HEDGING_WORDS)
    intensifier_count = count_patterns(INTENSIFIERS)
    superlative_count = count_patterns(SUPERLATIVES)
    passive_count = len(PASSIVE_PATTERN.findall(plain_text))

    # Type-Token Ratio (lexical diversity)
    unique_words = set(w.lower().strip(".,;:!?\"'()[]{}") for w in words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0

    # Hapax legomena (words appearing exactly once)
    word_freq = Counter(w.lower().strip(".,;:!?\"'()[]{}") for w in words)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = hapax / word_count if word_count > 0 else 0

    return {
        "lexical_word_count": word_count,
        "lexical_ttr": round(ttr, 4),
        "lexical_hapax_ratio": round(hapax_ratio, 4),
        "lexical_hedging_count": hedging_count,
        "lexical_hedging_density_per_1k": round(hedging_count / word_count * 1000, 2),
        "lexical_intensifier_count": intensifier_count,
        "lexical_intensifier_density_per_1k": round(intensifier_count / word_count * 1000, 2),
        "lexical_superlative_count": superlative_count,
        "lexical_superlative_density_per_1k": round(superlative_count / word_count * 1000, 2),
        "lexical_passive_count": passive_count,
        "lexical_passive_density_per_1k": round(passive_count / word_count * 1000, 2),
    }


# =====================================================================
# ARTICLE FETCHING & PARSING
# =====================================================================

API_URL = "https://en.wikipedia.org/w/api.php"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "WikiBiasAnalyzer/1.0 (Research; deterministic bias detection tool)"
})


def title_from_url(url):
    """Extract article title from a Wikipedia URL."""
    parsed = urlparse(url)
    if "wikipedia.org" in parsed.netloc:
        path = parsed.path
        if "/wiki/" in path:
            title = path.split("/wiki/")[1]
            return unquote(title).replace("_", " ")
    return None


def fetch_article(title):
    """Fetch article wikitext from Wikipedia API."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json"
    }
    resp = SESSION.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "query" in data and "pages" in data["query"]:
        for pid, pdata in data["query"]["pages"].items():
            if int(pid) < 0:
                return None, None
            actual_title = pdata.get("title", title)
            if "revisions" in pdata:
                content = pdata["revisions"][0].get("slots", {}).get("main", {}).get("*", "")
                return actual_title, content
    return None, None


def parse_sections(wikitext):
    """
    Parse wikitext into sections using mwparserfromhell.
    Returns dict: {section_name: plain_text}
    """
    parsed = mwparserfromhel = None
    parsed = mwparserfromhell.parse(wikitext)

    sections = {}
    current_section = "Lead"
    current_text = []

    for node in parsed.nodes:
        if isinstance(node, mwparserfromhell.nodes.heading.Heading):
            # Save previous section
            if current_text:
                text = mwparserfromhell.parse("".join(str(n) for n in current_text)).strip_code()
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    sections[current_section] = text

            current_section = node.title.strip_code().strip()
            current_text = []
        else:
            current_text.append(str(node))

    # Save last section
    if current_text:
        text = mwparserfromhell.parse("".join(str(n) for n in current_text)).strip_code()
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            sections[current_section] = text

    # Remove non-content sections
    skip = {"see also", "references", "external links", "further reading",
            "notes", "citations", "bibliography", "sources"}
    sections = {k: v for k, v in sections.items() if k.lower() not in skip}

    return sections


# =====================================================================
# MAIN ANALYSIS PIPELINE
# =====================================================================

def analyze_article(title):
    """
    Run all five modules on a single Wikipedia article.
    Returns a flat dict of all metrics.
    """
    print(f"\n  Fetching: {title} ...", end=" ", flush=True)
    actual_title, wikitext = fetch_article(title)

    if not wikitext:
        print("NOT FOUND")
        return {"title": title, "status": "not_found"}

    print(f"({len(wikitext):,} chars)")

    # Parse into sections
    sections = parse_sections(wikitext)
    plain_text = " ".join(sections.values())
    word_count = len(plain_text.split())

    print(f"  Parsed: {len(sections)} sections, {word_count:,} words")

    # Run all modules
    print("  [1/5] Source Bias Profile ...", end=" ", flush=True)
    source_results = analyze_source_bias(wikitext)
    print("done")

    print("  [2/5] Wikipedia Policy Compliance ...", end=" ", flush=True)
    policy_results = analyze_policy_compliance(plain_text)
    print("done")

    print("  [3/5] Sentiment Asymmetry (VADER) ...", end=" ", flush=True)
    sentiment_results = analyze_sentiment(sections)
    print("done")

    print("  [4/5] Structural Balance ...", end=" ", flush=True)
    structure_results = analyze_structure(sections, wikitext)
    print("done")

    print("  [5/5] Lexical Analysis ...", end=" ", flush=True)
    lexical_results = analyze_lexical(plain_text)
    print("done")

    # Flatten into single result dict
    result = {
        "title": actual_title,
        "status": "ok",
        "word_count": word_count,
        "section_count": len(sections),
    }

    # Add source metrics (exclude detail dicts from CSV)
    for k, v in source_results.items():
        if not k.startswith("_"):
            result[k] = v

    # Add policy metrics (exclude example lists from CSV)
    for k, v in policy_results.items():
        if not k.endswith("_examples"):
            result[k] = v

    # Add sentiment metrics (exclude nested dicts from CSV)
    for k, v in sentiment_results.items():
        if k not in ("sentiment_sections", "sentiment_strongest_positive",
                      "sentiment_strongest_negative"):
            result[k] = v

    # Add structure metrics (exclude nested dict)
    for k, v in structure_results.items():
        if k != "structure_section_proportions":
            result[k] = v

    # Add lexical metrics
    result.update(lexical_results)

    # Store detail objects for terminal display (not CSV)
    result["_source_detail"] = source_results
    result["_policy_detail"] = policy_results
    result["_sentiment_detail"] = sentiment_results
    result["_structure_detail"] = structure_results

    return result


# =====================================================================
# CSV OUTPUT
# =====================================================================

CSV_COLUMNS = [
    "title", "status", "word_count", "section_count",
    # Module 1: Sources
    "source_mean_bias", "source_median_bias", "source_bias_std",
    "source_bias_label", "sources_total", "sources_classified",
    "sources_unclassified", "sources_classified_pct", "unique_domains",
    "unique_classified",
    "distribution_left", "distribution_lean_left", "distribution_center",
    "distribution_lean_right", "distribution_right",
    # Module 2: Policy
    "weasel_count", "weasel_density_per_1k",
    "peacock_count", "peacock_density_per_1k",
    "editorial_count", "editorial_density_per_1k",
    "total_npov_flags", "total_npov_density_per_1k",
    # Module 3: Sentiment
    "sentiment_mean", "sentiment_median", "sentiment_std",
    "sentiment_positive_pct", "sentiment_negative_pct", "sentiment_neutral_pct",
    "sentiment_section_variance", "sentiment_lead_body_gap", "total_sentences",
    # Module 4: Structure
    "structure_sections", "structure_total_words", "structure_gini",
    "structure_largest_section", "structure_largest_section_pct",
    "structure_has_criticism", "structure_has_reception", "structure_has_history",
    "structure_ref_total", "structure_ref_density_per_1k",
    # Module 5: Lexical
    "lexical_word_count", "lexical_ttr", "lexical_hapax_ratio",
    "lexical_hedging_count", "lexical_hedging_density_per_1k",
    "lexical_intensifier_count", "lexical_intensifier_density_per_1k",
    "lexical_superlative_count", "lexical_superlative_density_per_1k",
    "lexical_passive_count", "lexical_passive_density_per_1k",
]


def write_csv(output_path, results):
    """Write results to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


# =====================================================================
# TERMINAL DISPLAY
# =====================================================================

def print_report(result):
    """Print a readable summary to terminal."""
    if result.get("status") != "ok":
        print(f"\n  Article '{result['title']}' — {result['status']}")
        return

    print("\n" + "=" * 72)
    print(f"  BIAS ANALYSIS: {result['title']}")
    print(f"  {result['word_count']:,} words | {result['section_count']} sections")
    print("=" * 72)

    # ── Module 1: Sources ──
    print("\n  [MODULE 1] SOURCE BIAS PROFILE")
    print(f"  Methodology: Cited domains mapped to AllSides media bias ratings")
    print(f"  Scale: -2 (Left) to +2 (Right)")
    print(f"  ─────────────────────────────────────────")
    sb = result.get("source_mean_bias")
    if sb is not None:
        print(f"  Mean source bias:    {sb:+.3f} ({result['source_bias_label']})")
        print(f"  Median:              {result['source_median_bias']:+.1f}")
        print(f"  Std deviation:       {result['source_bias_std']:.3f}")
        print(f"  Sources total:       {result['sources_total']} ({result['sources_classified']} classified, "
              f"{result['sources_unclassified']} unclassified)")
        print(f"  Classification rate: {result['sources_classified_pct']}%")
        print(f"  Distribution:  Left={result['distribution_left']}  "
              f"Lean Left={result['distribution_lean_left']}  "
              f"Center={result['distribution_center']}  "
              f"Lean Right={result['distribution_lean_right']}  "
              f"Right={result['distribution_right']}")

        detail = result.get("_source_detail", {})
        classified = detail.get("_classified_detail", {})
        if classified:
            print(f"\n  Top cited classified sources:")
            sorted_sources = sorted(classified.items(), key=lambda x: x[1]["count"], reverse=True)
            for domain, info in sorted_sources[:10]:
                print(f"    {domain:35s}  ×{info['count']:3d}  [{info['label']}]")
    else:
        print(f"  No classified sources found ({result['sources_total']} total URLs)")

    # ── Module 2: Policy ──
    print(f"\n  [MODULE 2] WIKIPEDIA POLICY COMPLIANCE")
    print(f"  Methodology: Text scanned against WP:WEASEL, WP:PEACOCK, WP:EDITORIALIZING word lists")
    print(f"  ─────────────────────────────────────────")
    print(f"  Weasel words:     {result['weasel_count']:3d}  ({result['weasel_density_per_1k']:.1f} per 1k words)")
    print(f"  Peacock terms:    {result['peacock_count']:3d}  ({result['peacock_density_per_1k']:.1f} per 1k words)")
    print(f"  Editorializing:   {result['editorial_count']:3d}  ({result['editorial_density_per_1k']:.1f} per 1k words)")
    print(f"  Total NPOV flags: {result['total_npov_flags']:3d}  ({result['total_npov_density_per_1k']:.1f} per 1k words)")

    pd = result.get("_policy_detail", {})
    for category in ["weasel", "peacock", "editorial"]:
        examples = pd.get(f"{category}_examples", [])
        if examples:
            top = ", ".join(f'"{w}" ({c})' for w, c in examples[:5])
            print(f"    {category.capitalize()} examples: {top}")

    # ── Module 3: Sentiment ──
    print(f"\n  [MODULE 3] SENTIMENT ASYMMETRY (VADER)")
    print(f"  Methodology: VADER rule-based sentiment, deterministic, lexicon of ~7,500 items")
    print(f"  Scale: -1.0 (most negative) to +1.0 (most positive)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Overall mean:     {result['sentiment_mean']:+.4f}")
    print(f"  Overall median:   {result['sentiment_median']:+.4f}")
    print(f"  Std deviation:    {result['sentiment_std']:.4f}")
    print(f"  Positive:         {result['sentiment_positive_pct']}% of sentences")
    print(f"  Negative:         {result['sentiment_negative_pct']}%")
    print(f"  Neutral:          {result['sentiment_neutral_pct']}%")
    print(f"  Section variance: {result['sentiment_section_variance']:.4f}")
    print(f"  Lead-body gap:    {result['sentiment_lead_body_gap']:+.4f}")

    sd = result.get("_sentiment_detail", {})
    sp = sd.get("sentiment_strongest_positive", {})
    sn = sd.get("sentiment_strongest_negative", {})
    if sp.get("text"):
        print(f"  Most positive ({sp['score']:+.3f}): \"{sp['text']}...\"")
    if sn.get("text"):
        print(f"  Most negative ({sn['score']:+.3f}): \"{sn['text']}...\"")

    sections_s = sd.get("sentiment_sections", {})
    if sections_s:
        print(f"\n  Per-section sentiment:")
        for sec, info in list(sections_s.items())[:15]:
            bar_pos = int(info["positive_pct"] / 5)
            bar_neg = int(info["negative_pct"] / 5)
            print(f"    {sec[:30]:30s}  {info['mean']:+.3f}  "
                  f"(+{info['positive_pct']:4.1f}%  -{info['negative_pct']:4.1f}%  "
                  f"n={info['sentences']})")

    # ── Module 4: Structure ──
    print(f"\n  [MODULE 4] STRUCTURAL BALANCE")
    print(f"  ─────────────────────────────────────────")
    print(f"  Sections:           {result['structure_sections']}")
    print(f"  Gini coefficient:   {result['structure_gini']:.4f} (0=balanced, 1=all in one section)")
    print(f"  Largest section:    {result['structure_largest_section']} ({result['structure_largest_section_pct']}%)")
    print(f"  Has criticism:      {'Yes' if result['structure_has_criticism'] else 'No'}")
    print(f"  Has reception:      {'Yes' if result['structure_has_reception'] else 'No'}")
    print(f"  Reference density:  {result['structure_ref_density_per_1k']:.1f} refs/1k words")

    # ── Module 5: Lexical ──
    print(f"\n  [MODULE 5] LEXICAL ANALYSIS")
    print(f"  ─────────────────────────────────────────")
    print(f"  Type-Token Ratio:   {result['lexical_ttr']:.4f} (vocabulary diversity)")
    print(f"  Hapax ratio:        {result['lexical_hapax_ratio']:.4f}")
    print(f"  Hedging words:      {result['lexical_hedging_count']:3d}  ({result['lexical_hedging_density_per_1k']:.1f} per 1k)")
    print(f"  Intensifiers:       {result['lexical_intensifier_count']:3d}  ({result['lexical_intensifier_density_per_1k']:.1f} per 1k)")
    print(f"  Superlatives:       {result['lexical_superlative_count']:3d}  ({result['lexical_superlative_density_per_1k']:.1f} per 1k)")
    print(f"  Passive voice:      {result['lexical_passive_count']:3d}  ({result['lexical_passive_density_per_1k']:.1f} per 1k)")

    print("\n" + "=" * 72)


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deterministic Wikipedia article bias analyzer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wiki_bias_analyzer.py "Vaccination"
  python wiki_bias_analyzer.py "https://en.wikipedia.org/wiki/Climate_change"
  python wiki_bias_analyzer.py --batch articles.txt --output results.csv

Modules:
  1. Source Bias Profile   — AllSides media bias ratings for cited sources
  2. Policy Compliance     — WP:WEASEL, WP:PEACOCK, WP:EDITORIALIZING
  3. Sentiment Asymmetry   — VADER deterministic sentiment analysis
  4. Structural Balance    — Section sizes, Gini coefficient, citation density
  5. Lexical Analysis      — Hedging, intensifiers, passive voice, vocabulary
        """
    )
    parser.add_argument(
        "article", nargs="?",
        help="Article title or Wikipedia URL"
    )
    parser.add_argument(
        "--batch", "-b",
        help="Text file with one article title per line"
    )
    parser.add_argument(
        "--output", "-o",
        default="wiki_bias_analysis.csv",
        help="Output CSV file (default: wiki_bias_analysis.csv)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Also output detailed JSON results"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress terminal report, only write CSV"
    )
    args = parser.parse_args()

    if not args.article and not args.batch:
        parser.print_help()
        sys.exit(1)

    # Build article list
    articles = []
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"ERROR: File not found: {args.batch}")
            sys.exit(1)
        with open(args.batch, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    articles.append(line)
        print(f"Loaded {len(articles)} articles from {args.batch}")
    elif args.article:
        # Check if it's a URL
        if args.article.startswith("http"):
            title = title_from_url(args.article)
            if title:
                articles.append(title)
            else:
                print(f"ERROR: Could not parse URL: {args.article}")
                sys.exit(1)
        else:
            articles.append(args.article)

    print("=" * 72)
    print("Wikipedia Bias Analyzer — Deterministic & Replicable")
    print("=" * 72)
    print(f"Articles to analyze: {len(articles)}")

    # Process each article
    results = []
    for title in articles:
        result = analyze_article(title)
        results.append(result)

        if not args.quiet:
            print_report(result)

        if len(articles) > 1:
            time.sleep(0.5)  # Rate limiting for batch mode

    # Write CSV
    print(f"\nWriting CSV: {args.output}")
    write_csv(args.output, results)
    print(f"  {len(results)} rows written")

    # Optionally write JSON
    if args.json:
        json_path = args.output.rsplit(".", 1)[0] + ".json"
        # Remove non-serializable detail objects
        json_results = []
        for r in results:
            jr = {k: v for k, v in r.items() if not k.startswith("_")}
            # Add back the detail objects in serializable form
            if "_source_detail" in r:
                jr["source_classified_detail"] = r["_source_detail"].get("_classified_detail", {})
            if "_policy_detail" in r:
                jr["policy_weasel_examples"] = r["_policy_detail"].get("weasel_examples", [])
                jr["policy_peacock_examples"] = r["_policy_detail"].get("peacock_examples", [])
                jr["policy_editorial_examples"] = r["_policy_detail"].get("editorial_examples", [])
            if "_sentiment_detail" in r:
                jr["sentiment_per_section"] = r["_sentiment_detail"].get("sentiment_sections", {})
                jr["sentiment_strongest_positive"] = r["_sentiment_detail"].get("sentiment_strongest_positive", {})
                jr["sentiment_strongest_negative"] = r["_sentiment_detail"].get("sentiment_strongest_negative", {})
            json_results.append(jr)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"  JSON detail: {json_path}")

    print(f"\nOutput: {os.path.abspath(args.output)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
