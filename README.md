# EDST landing page update — themed methods

Proposed update to the NC-UK Environmental Data Science Toolbox site
(https://github.com/NERC-CEH/data-science-toolbox). It groups the data science
methods into four themes, adds a methods overview page, and restyles the landing
page using the UKCEH brand palette (lighter tints). Built against `main` at
`4e33dbf` (after the Connected River merge, PR #114).

## What's in this folder

Copy everything into the repo root; paths match the repo layout.

| File | Status | What changed |
|---|---|---|
| `intro.md` | **modified** | New landing page (see below) |
| `data_science_methods.md` | **new** | Methods overview page: theme cards + every method grouped by theme |
| `myst.yml` | **modified** | Only the `toc` changed — everything else is as on `main` |
| `_static/custom.css` | **modified** | Existing rules kept unchanged; new rules appended at the end |
| `images/themes/*.svg` | **new** | Four theme illustrations (climate, biodiversity, pollution, data) |

## The four themes

| Theme | Methods |
|---|---|
| **Climate, land & Earth observation** | Bias Correction of Climate Models · Fine-Tuning a Geospatial Foundation Model for UK Crop Classification · Calculating Risk to Terrestrial Carbon Pool |
| **Biodiversity & species** | Joint Species Distribution Models with jsdmstan · Understanding the Error of Multispecies Biodiversity Indicators · Extracting Evidence-Linked Abundance Drivers from Species-Account Text |
| **Pollution & chemical exposure** | Non-target Analysis of Environmental Mass Spectrometry Data · Exploration of Chemical Pollution from a Place-Based Perspective · Integration of Metals Data across Environmental Compartments · Modelling Censored Log-normal Data in mgcv · Multivariate Modelling of Censored Chemicals using jsdmstan |
| **Data integration & FAIR** | RO-Crate Tutorial · Evaluating the Added Benefits of Data Integration using Bayesian Model Comparison · The Connected River (Parts 1 & 2) |

14 methods in total (The Connected River counts as one two-part method).
The groupings are a proposal — easy to change (see "Moving a method" below).

## Changes in detail

### Landing page (`intro.md`)
- **Original content kept**: title, hexagon logo (same light/dark files and
  `abstract-image--light/--dark` classes), the intro paragraph and the
  Feedback line, word for word. The logo now sits beside the text so the page
  is shorter.
- **Quick-links band**: method/theme counts, a pointer to the built-in search
  (Ctrl/⌘ K), and buttons for *Browse all methods* and *Contribute a method*.
- **Explore by theme**: four illustrated cards, each linking to its section
  on the methods page.
- **Get started**: cards for Utilising the Toolbox, Template Notebook,
  Get Involved and How to Cite.

### Methods page (`data_science_methods.md`, new)
- Opens when you click **Data Science Methods** in the sidebar.
- The four theme cards, then every method as a card with key concepts, key
  datasets and technique tags (e.g. Bayesian, AI & ML, Geospatial), grouped
  under its theme. Cards link straight to the notebooks.

### Sidebar (`myst.yml` → `project.toc`)
- *Data Science Methods* now has its own page (`file: data_science_methods.md`)
  and its notebooks are nested under the four theme headings.
- *About this Project*, the notebook file paths and all `site:` settings are
  unchanged.

### Styling (`_static/custom.css`)
New rules appended under `/* EDST landing page … */`; the existing logo-swap
rules are untouched.
- UKCEH brand colours (Brand Guidelines 2.0, pp. 40–42), using the lighter
  tints: Water `#D1DBF1` (climate), Land `#E1EAD9` (biodiversity),
  Earth `#F5EDEA` (pollution), Data `#F5F5D3` (data/FAIR), Air `#D6EAE6`
  (bands). Text stays black, per the guidelines; the lime `#DBFE52` is used
  only for the main button.
- Theme cards are tinted; method cards get a coloured top edge for their theme.
- **Sidebar**: the four theme rows under Data Science Methods get the same
  colours, with a matching guide line beside their methods.
- Dark mode: tints switch to dark surfaces / coloured edges so text stays
  readable.

## How to preview

```bash
python scripts/sync_notebooks.py --execute   # downloads the method notebooks into methods/ (git-ignored)
npx jupyter-book start                       # then open the local address it prints
```

The sync step matters: without `methods/`, MyST leaves the methods out of the
sidebar and the theme groups look empty.

Checked with mystmd 1.12: the build gives no new warnings and all links on the
landing page (12) and methods page (18) resolve. (The existing warning about `site.url` belonging under `site.options`
was already there before this change.)

## Maintaining it

**Adding a method**
1. Add the notebook to `notebooks.yml` as usual.
2. In `myst.yml`, add its `file:` under the right theme.
3. In `data_science_methods.md`, copy a `:::{card}` block into that theme's
   grid and edit the title, `:link:`, concepts, data and `:footer:` tags.
4. Update the counts: the theme card text (in both `intro.md` and
   `data_science_methods.md`) and "**14** methods" in the band in `intro.md`.

**Moving a method to another theme**: move its line in `myst.yml` and its card
in `data_science_methods.md`, then fix the counts.

**Reordering or renaming themes**: the sidebar colours follow theme *order*
(1st = blue, 2nd = green, 3rd = clay, 4th = yellow — see the last block of
`custom.css`). The landing/methods card colours follow the order of the cards
in each `edst-themes` grid. Keep all three in the same order.

## Notes and known limits

- The theme cards on the landing page link with a relative address
  (`data-science-methods#theme-climate`). This works because `intro.md` is the
  site root; if the landing page ever moves, update those four `:link:` lines.
- The sidebar colouring uses the CSS `:has()` selector (all current major
  browsers). Older browsers simply show the plain sidebar.
- The theme illustrations are simple placeholder SVGs drawn for this proposal;
  swap in photos or UKCEH imagery if preferred (keep a 320×150 ratio or adjust).
- Suisse International (the brand typeface) isn't used — the site keeps the
  MyST theme font.

## Undo

```bash
git checkout -- intro.md myst.yml _static/custom.css
rm -r data_science_methods.md images/themes
```
