# Benchmarks

## Machine

Measured in October 2026 with anygrep 0.1.0 and the int8 model on an AMD Ryzen 7 7700
(8 cores, 16 threads, 32 GB) under WSL2 on Windows 11, CPU only, at the lowest priority (`nice -n 19`),
on an otherwise idle machine unless noted. The default is half the logical CPU threads, at most 8, which
is 8 here. The numbers are medians of single runs or short samples; expect a few percent of noise.

## Speed and memory (default 8 threads)

| What | Time |
|---|---|
| Text | about 1,100 tokens/s, or 3.5 chunks/s for 384-token chunks |
| Image, 70 tokens | about 0.7 s per image |
| Image, 140 tokens (default) | about 1.2 s per image |
| Image, 280 tokens | about 2.6 s per image |
| One search command | about 1.5 s in total, mostly loading the model; the search itself takes tens of milliseconds |
| Re-running `index` with nothing changed | 0.2 s for the demo corpus |
| Memory | about 1.3 GB peak while indexing text, PDFs and images; about 1.0 GB for text and PDFs only; about 0.5 GB for a text search |

About 80% of the time per image is the vision encoder; reading and resizing the image is about 5%.
An image with the same bytes as one already indexed or embedded earlier in the run (`photo (1).jpg`)
is not embedded again: its vector is copied, so a folder with many copies indexes faster by about their
share. Images are embedded one at a time; running several same-size images through the vision encoder
together was not measurably faster and needed up to twice the memory at 280 tokens.

## Threads

Per image (140 tokens) and per text chunk, measured one setting at a time:

| Threads (`-j`) | Seconds per image | Text chunks per second |
|---|---|---|
| 4 | 1.52 | 2.75 |
| 6 | 1.30 | 3.25 |
| 8 (default here) | 1.21 | 3.52 |
| 12 | 1.00 | 4.15 |
| 16 | 1.11 | 3.72 |

On an idle machine `-j 12` (or `ANYGREP_THREADS=12`) is about 17% faster than 8; 16 is slower than 12
because the threads compete with each other and everything else. The default stays at 8 so the machine
stays usable. `index --background` uses half of the threads at a low priority.

## Demo corpus

The demo corpus (12 notes, 3 PDFs with 8 pages, 8 images; `scripts/make_demo_corpus.py`) indexes in
about 13 s with the default 8 threads (17 s with 4), including about 1 s to load the model. The GIF in
the README shows only searches of the already indexed corpus; the wait for each result is played 6x
faster. `scripts/record_demo.sh` records it.

Its index file is about 120 KB. Each chunk takes about 1.5 KB plus its text.

## Larger collections

Estimated from the per-item speeds above (8 threads), so expect about 15% either way:

- A notes folder of 1,000 notes at about 300 tokens each takes around 5 minutes.
- 10,000 images take around 2 hours at 70 tokens and 3.5 hours at 140 tokens.
- A synthetic Downloads-like folder (1,200 images, 100 PDFs, 50 notes; 1.5 GB) takes about 35
  minutes, about 60% of it for the images.

In a synthetic index of 100,000 chunks (16,000 files), the first search in a process takes about 0.4 s
after the model is loaded, most of it reading the vectors. A search of a folder with 64 of those files
reads only their vectors and takes about 0.01 s (Linux and macOS; on Windows it reads all of them).

## Image token budget

In an earlier benchmark on the same machine, a synthetic set of 50 images (shapes and 28 px text drawn
with Pillow, not real photos) was used to compare budgets. The top-1 text-to-image accuracy was:

| Budget | Top-1 accuracy |
|---|---|
| 70 | 0.78 |
| 140 | 0.90 |
| 280 | 0.88 |

The difference between 140 and 280 is one image of 50, which is noise. 140 is the default because it is
about twice as fast as 280 at similar accuracy on this set. It has not been measured on real photo
collections.
