# Relevance cutoff

A search hides weak matches. This page describes the rule, how its values were chosen, and how
well it does on the measured corpora. `--weak` always shows everything.

## Why scores differ by type

Scores are cosine similarities, and their range depends on what is compared: a text query scores lower
against images than against text, and image queries score higher against other images. A single
threshold for all types would either hide most image results or let through many unrelated text results.

## The rule

A result is shown only when it passes two tests.

1. A minimum score for its pair of query type and result type:

   | Query -> result | Minimum score (image budget 70 / 140 / 280) | Margin |
   |---|---|---|
   | text -> text | 0.62 | 0.10 |
   | text -> pdf | 0.64 | 0.11 |
   | text -> image | 0.63 / 0.62 / 0.62 | 0.06 |
   | image -> image | 0.76 / 0.75 / 0.70 | 0.09 |
   | image -> text | 0.62 / 0.62 / 0.55 | 0.04 |
   | image -> pdf | 0.62 / 0.62 / 0.61 | 0.13 |

2. A margin above the query's reference level. Short or meaningless queries ("asdf", "hello", "...")
   score a little higher against everything, so their best results can pass a fixed minimum. anygrep
   ships the embeddings of 64 varied texts and 12 pictures, written and drawn for this purpose, and
   compares each query with them: the median of those scores is how high "everything" is for this query,
   and a result must beat it by the margin. The reference does not depend on what you indexed, so a
   folder in which most files are about the query's topic still shows them.

When nothing passes, anygrep prints "No strong matches; showing the closest results" and lists the
closest three results, or `-m` of them (at most 10) when `-m` is given, marked as weak. See
[usage.md](usage.md#weak-results) for the output details.

## Calibration

The values come from `scripts/calibrate_thresholds.py`. It indexes the demo corpus plus 4 more notes, a
PDF and 8 more images (all synthetic) and runs 47 text queries and 9 image queries with known relevant
files, plus 36 unrelated text queries (15 off-topic questions and 21 nonsense or very short queries such
as "a", "hi", "1234", "あ") and 4 unrelated image queries (noise, flat gray, stripes, checkerboard).

Each minimum score is the highest value that keeps at least 95% of the relevant files for its pair,
rounded up to the next 0.01 above the best unrelated score if that is lower. Each margin is the highest
value that keeps 95% of the relevant files. The full output is in
[calibration-results.json](calibration-results.json).

## In-sample results

At the default image budget of 140:

| Query -> result | Recall | Precision without cutoff | Precision with cutoff |
|---|---|---|---|
| text -> text | 0.96 (27/28) | 0.02 | 0.87 |
| text -> pdf | 1.00 (10/10) | 0.03 | 0.91 |
| text -> image | 0.96 (22/23) | 0.02 | 0.50 |
| image -> image | 1.00 (9/9) | 0.04 | 1.00 |
| image -> text | 1.00 (2/2) | 0.01 | 0.15 |
| image -> pdf | 1.00 (1/1) | 0.02 | 1.00 |

None of the 40 unrelated queries shows a result; they get the "No strong matches" fallback. Precision
counts every file above the cutoff that was not labelled relevant, including related ones, such as a
Wi-Fi settings screenshot for "router admin page address". These numbers are in-sample: the values were
chosen on these queries.

## Held-out results

The same rule was then checked, unchanged, on held-out corpora (`scripts/heldout_corpus.py`, also
synthetic, on other topics):

- a mixed folder of notes, README-style project notes, Japanese notes, PDFs and 14 pictures
- a folder of 10 recipes
- a folder of 8 sourdough notes and 2 unrelated ones
- 6 Japanese notes on one topic

There, 67 of 72 relevant files are shown, and 2 of 34 unrelated queries show something: a question about
ice hockey rules and one about sumo rankings, both matching a picture of a chess board. In the sourdough
folder, "sourdough bread" shows 6 of the 8 sourdough notes; the other 2, about bulk fermentation and cold
proofing, do not use the word and score just under the minimum.

The rule used before this one, a margin above the median score of the indexed chunks, showed 64 of the
72 relevant files there, only 2 of the 8 sourdough notes, and let 8 of the 34 unrelated queries through.

## Caveats

- Image budget 280 spreads image scores, and unrelated image queries get through more often.
- The corpora are small, so treat these numbers as a rough guide, and use `--weak` when a result you
  expected is missing.
- Images are the most likely unrelated results: some pictures, such as a chess board or a chart, come up
  for many unrelated questions.

## Near-duplicates

Near-duplicate files are shown once (see [usage.md](usage.md#near-duplicates)). They are found from the
embeddings, which see layout and topic more than small details:

- Two versions of a chart that differ only in a few bar heights are collapsed.
- A note with a new paragraph, or a tiny thumbnail of a screenshot, can stay separate.
- Text and PDF files must also share almost all their words, so short monthly payslips or statements
  from one template stay separate. In a long template where only a few amounts and the month change
  (a 15-line payslip, say), about 90 percent of the words can be the same, as in an edited copy (the
  rule allows up to 18 percent changed word pairs, about 10 percent replaced words), and the months
  are collapsed into one result with a `(+N similar: ...)` note. Images have no such check.
- Piped output (`PATH:LINE:SNIPPET`) and `-l` never collapse: they list every matching file.

The rule and its measurements are in `scripts/measure_duplicates.py`,
[duplicate-results.json](duplicate-results.json) and [DESIGN.md](DESIGN.md) (in Japanese).
`--no-collapse` shows every file.
