% Chunking Strategy
% Core Preprocessing Clarification

# Fixed Size + Overlap (Sliding Window)

`This pipeline uses fixed token windows with overlap, i.e., sliding-window chunking.`

## Core rule
- `chunk_size_tokens = W`
- `chunk_overlap_tokens = O`
- `stride = W - O`
- Next chunk starts `stride` tokens after current chunk start.

## Example used in your retrieval tuning runs
- `W = 280`, `O = 90`  ->  `stride = 190`
- Chunk starts: `0, 190, 380, 570, ...`
- This preserves context continuity across chunk boundaries.

## Where segment-aware fits
- Optional `segment_aware_chunking` runs **before** chunking.
- It first splits page text into logical segments (headings/entities),
  then applies the same sliding-window rule inside each segment.

## Practical takeaway
- Not non-overlapping fixed blocks.
- Not dynamic-length semantic-only chunks.
- It is fixed-window **overlapping** chunking, optionally preceded by segment splitting.
