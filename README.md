# Wikipedia Bias Analyzer

Deterministic, transparent, and replicable bias detection for any English Wikipedia article.

<br>

<div align="center">

### [**Try it now — no install required**](https://nethahussain.github.io/wikipedia-bias-analyzer/)

<br>

<a href="https://nethahussain.github.io/wikipedia-bias-analyzer/">
<img src="https://img.shields.io/badge/Launch_App-blue?style=for-the-badge&logo=wikipedia&logoColor=white" alt="Launch App" />
</a>

<br><br>

*Type any Wikipedia article title, get instant bias analysis across 5 modules.*
*Runs entirely in your browser. No backend, no API keys, no data collection.*

</div>

<br>

---

## How It Works

The web app fetches any Wikipedia article's raw wikitext via the public API and runs five independent analysis modules entirely in your browser:

| Module | What It Measures | Source |
|--------|-----------------|--------|
| **Source Bias Profile** | Political leaning of cited sources | [AllSides Media Bias Ratings](https://www.allsides.com/media-bias/ratings) (~150 outlets) |
| **Policy Compliance** | Weasel words, peacock terms, editorializing | [Wikipedia's own WP:NPOV word lists](https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Words_to_watch) |
| **Sentiment Asymmetry** | Per-section emotional tone | [VADER Lexicon](https://github.com/cjhutto/vaderSentiment) (Hutto & Gilbert, 2014) |
| **Structural Balance** | Section size distribution, citation density | Gini coefficient |
| **Lexical Analysis** | Hedging, intensifiers, passive voice, vocab diversity | Standard NLP metrics |

**Determinism guarantee:** Same article always produces identical results. No ML, no randomness.

---

## Also Included

### Python CLI (`wiki_bias_analyzer.py`)

Full-featured command-line version with the complete VADER lexicon (~7,500 words) and batch processing:

```bash
pip install requests vaderSentiment
python wiki_bias_analyzer.py "Vaccination"
python wiki_bias_analyzer.py "Climate change" --json
python wiki_bias_analyzer.py --batch articles.txt --output results.csv
```

### Reference Density Scanner (`wikiproject_medicine_refdensity.py`)

Compute reference density for all 50,000+ WikiProject Medicine articles with resume support:

```bash
pip install mwparserfromhell requests
python wikiproject_medicine_refdensity.py --limit 100          # test run
python wikiproject_medicine_refdensity.py --resume              # resume interrupted run
python wikiproject_medicine_refdensity.py --output results.csv  # full run
```

---

## The Five Modules in Detail

### Module 1: Source Bias Profile
Extracts all cited domains from `<ref>` tags and maps them to [AllSides Media Bias Ratings](https://www.allsides.com/media-bias/ratings). Covers ~150 outlets including US, UK, Indian, and international media plus academic/institutional sources (rated Center).

- **Scale:** -2 (Left) to +2 (Right)
- **Output:** Weighted mean, distribution histogram, top sources with color-coded ratings

### Module 2: Wikipedia Policy Compliance
Scans text against Wikipedia's own published word lists:
- **WP:WEASEL** (~40 phrases) — "some say", "many believe", "experts claim"
- **WP:PEACOCK** (~45 terms) — "legendary", "groundbreaking", "world-renowned"
- **WP:EDITORIALIZING** (~35 words) — "obviously", "ironically", "so-called"

### Module 3: Sentiment Asymmetry (VADER)
Rule-based, deterministic sentiment scoring. Per-section analysis with lead-body gap detection and most positive/negative sentences highlighted.

### Module 4: Structural Balance
Gini coefficient of section sizes, reference density per 1,000 words, detection of Criticism/Controversy/Reception sections.

### Module 5: Lexical Analysis
Hedging density, intensifier density, superlatives, passive voice frequency, type-token ratio, hapax ratio.

---

## Verify Determinism

```bash
python wiki_bias_analyzer.py "Vaccination" --json > run1.json
python wiki_bias_analyzer.py "Vaccination" --json > run2.json
diff run1.json run2.json   # no output = identical
```

---

## Repository Structure

```
wikipedia-bias-analyzer/
├── index.html                         # Live web app (GitHub Pages)
├── wiki_bias_analyzer.html            # Same app (standalone copy)
├── wiki_bias_analyzer.py              # Python CLI tool
├── wikiproject_medicine_refdensity.py  # Reference density scanner
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md
```

---

## Limitations

- **AllSides coverage**: ~150 outlets. Niche, regional, and non-English sources won't be classified.
- **VADER lexicon (web app)**: Trimmed to ~1,200 words for browser performance. Python CLI uses the full ~7,500 word lexicon.
- **Wikitext parsing**: Browser parser is simplified — complex nested templates may not be fully stripped.
- **Not a truth detector**: These tools measure *how* something is written, not *whether* claims are true.
- **Point-in-time**: Results reflect the current article revision.

---

## License

MIT License. See [LICENSE](LICENSE).
