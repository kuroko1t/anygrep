# How it works

The design decisions behind these steps are in [DESIGN.md](DESIGN.md) (in Japanese).

1. Walk the given folders, apply the ignore rules, and compare each file's size and
   modification time with the index. If only the time changed, a SHA-256 of the content decides whether
   the file is re-embedded. A moved or renamed file with the same content keeps its embedding: images
   always, text and PDFs when the file name, which is part of the embedded title, is unchanged. Files
   that disappeared are pruned. Each file also records the processing generation it was indexed with:
   when a newer anygrep changes how some kind of file is extracted, chunked or embedded, the next update
   indexes those files again, like changed files, without a reset. Text and PDFs are queued before images, and images newest first, so an
   interrupted run has already made the cheap and recent files searchable.
2. Extract the text (whole file or per PDF page) and cut it into token-aware chunks with line or
   page positions. Load images with EXIF orientation applied.
3. Embed with EmbeddingGemma 2 (int8 ONNX, CPU).
   - Text chunks get the document prompt `title: <file name> | text: ...`, and queries get
     `task: search result | query: ...`.
   - Images are resized, keeping the aspect ratio, to fit the token budget, then cut into 16x16 patches
     for the vision encoder. The resulting soft tokens go through the same text model, so images and
     text share one 768-dimensional space. Images are embedded one at a time. An image with the
     same bytes as one already embedded at the same budget, under any name, gets a copy of its vector.
4. Store the normalized vectors (float16) and chunk text in a single SQLite file. Finished files are
   committed every few seconds. That is what makes indexing resumable: after Ctrl-C or a crash, the next
   run starts from step 1 and finds only the files that are left. Large text files are committed in slices.
   `anygrep index --background` runs these steps in a detached, low-priority process that writes its
   progress to a small state file next to the index, which `anygrep status` and searches read.
5. A search first runs steps 1 to 4 for the folder it searches. On a terminal it indexes everything
   new with a progress bar; in scripts only when the estimated time is within a budget (30 seconds by
   default; the estimate is corrected by the speed measured in earlier runs), otherwise it indexes
   nothing and searches what is already indexed. It then embeds the query, reads the stored vectors of
   the files under its folders, and computes the cosine similarity with numpy.
   By default it shows the best chunk per file, ranked across all types, with near-duplicate files
   grouped under the best-ranked one. Results below the [relevance cutoff](relevance.md) are hidden.
   A text query of 3 or more characters is also looked up as a plain string in the chunk text (an
   SQLite FTS5 trigram index, kept in the same transactions as the chunks) and in the file paths; files
   that contain it come first, marked as exact, whatever their score.

The image preprocessing is a small numpy port of the reference processor. It matches the reference
implementation's embeddings with a cosine similarity of at least 0.9999.
