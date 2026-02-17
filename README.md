# Wikipedia Bias Analyzer

Deterministic, transparent, and replicable bias detection for any English Wikipedia article.

This toolkit provides three tools:

1. **Interactive Web App** (`wiki_bias_analyzer.html`) — Open in any browser, type an article name, get full bias analysis. No server needed.
2. **Python CLI** (`wiki_bias_analyzer.py`) — Command-line tool for single or batch article analysis with CSV/JSON output.
3. **Reference Density Scanner** (`wikiproject_medicine_refdensity.py`) — Compute reference density for all WikiProject Medicine articles (50,000+) as CSV.

---

## Quick Start

### Web App (zero install)

Open `wiki_bias_analyzer.html` in any modern browser. That's it. Type an article title, pick from the autocomplete dropdown, and view results.

### Python CLI

```bash
pip install requests
python wiki_bias_analyzer.py "Vaccination"
python wiki_bias_analyzer.py "Climate change" --json
python wiki_bias_analyzer.py --batch articles.txt --output results.csv
```

### Reference Density Scanner

```bash
pip install mwparserfromhell requests
python wikiproject_medicine_refdensity.py --limit 100          # test run
python wikiproject_medicine_refdensity.py --resume              # resume interrupted run
python wikiproject_medicine_refdensity.py --output results.csv  # full 50k+ run
```

---

## Five Analysis Modules

Every article is analyzed across five independent modules:

### Module 1: Source Bias Profile
Extracts all cited domains from `<ref>` tags and maps them to [AllSides Media Bias Ratings](https://www.allsides.com/media-bias/ratings). Covers ~150 outlets including US, UK, Indian, and international media plus academic/institutional sources (rated Center).

- **Scale:** -2 (Left) → -1 (Lean Left) → 0 (Center) → +1 (Lean Right) → +2 (Right)
- **Output:** Weighted mean bias, distribution histogram, top cited sources with ratings

### Module 2: Wikipedia Policy Compliance
Scans article text against Wikipedia's own published word lists:

- **WP:WEASEL** (~40 phrases) — Vague attribution ("some say", "many believe", "experts claim")
- **WP:PEACOCK** (~45 terms) — Promotional language ("legendary", "groundbreaking", "world-renowned")
- **WP:EDITORIALIZING** (~35 words) — Opinion markers ("obviously", "ironically", "so-called")

Reports counts, density per 1,000 words, and top flagged phrases.

### Module 3: Sentiment Asymmetry (VADER)
Uses the VADER sentiment lexicon (Hutto & Gilbert, 2014) — a rule-based, deterministic scorer with ~1,200 high-impact words (full Python version uses the complete ~7,500 word lexicon via `vaderSentiment`).

- Per-section sentiment means
- Lead vs. body gap detection
- Most positive/negative sentences highlighted

### Module 4: Structural Balance
Measures how evenly content is distributed across sections:

- **Gini coefficient** of section sizes (0 = perfectly balanced, 1 = all content in one section)
- Reference density (refs per 1,000 words)
- Presence/absence of Criticism, Controversy, Reception sections

### Module 5: Lexical Analysis
Quantifies writing style indicators:

- **Hedging** density (may, might, possibly, arguably...)
- **Intensifiers** density (very, extremely, dramatically...)
- **Superlatives** density (best, worst, greatest...)
- **Passive voice** frequency
- **Type-Token Ratio** (vocabulary diversity)
- **Hapax ratio** (words used only once)

---

## Determinism Guarantee

All analysis is deterministic and rule-based. The same article will always produce identical results regardless of who runs it, when, or where. No machine learning, no randomness, no API keys required.

Verify this yourself:
```bash
python wiki_bias_analyzer.py "Vaccination" --json > run1.json
python wiki_bias_analyzer.py "Vaccination" --json > run2.json
diff run1.json run2.json   # no output = identical
```

---

## Data Sources

| Source | Used In | Type |
|--------|---------|------|
| [AllSides Media Bias Ratings](https://www.allsides.com/media-bias/ratings) | Module 1 | External database (~150 outlets) |
| [Wikipedia WP:WEASEL](https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Words_to_watch#Unsupported_attributions) | Module 2 | Wikipedia's own policy |
| [Wikipedia WP:PEACOCK](https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Words_to_watch#Puffery) | Module 2 | Wikipedia's own policy |
| [Wikipedia WP:EDITORIALIZING](https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Words_to_watch#Editorializing) | Module 2 | Wikipedia's own policy |
| [VADER Sentiment Lexicon](https://github.com/cjhutto/vaderSentiment) (Hutto & Gilbert, 2014) | Module 3 | Peer-reviewed, deterministic |
| Gini coefficient | Module 4 | Standard statistical measure |

---

## File Overview

```
wikipedia-bias-analyzer/
├── wiki_bias_analyzer.html          # Interactive web app (open in browser)
├── wiki_bias_analyzer.py            # Python CLI tool
├── wikiproject_medicine_refdensity.py  # WikiProject Medicine ref density scanner
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## Limitations

- **AllSides coverage**: ~150 outlets. Many niche, regional, and non-English sources won't be classified.
- **VADER lexicon (web app)**: Trimmed to ~1,200 words for browser performance. The Python CLI uses the full lexicon via `vaderSentiment`.
- **Wikitext parsing**: The browser-based parser is simplified. Complex nested templates may not be fully stripped.
- **No content analysis**: These tools measure *how* something is written, not *whether* claims are true.
- **Snapshot in time**: Results reflect the current article revision. Wikipedia articles change constantly.

---

## License

MIT License. See [LICENSE](LICENSE).
