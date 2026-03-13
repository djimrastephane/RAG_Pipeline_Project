% Generation Context Limits
% Small Explainer Diagram

```mermaid
flowchart TD
    A[Retrieved top-k chunks] --> B{Per chunk}
    B --> C[Trim chunk_text to max 2200 chars]
    C --> D[Build block:<br/>[chunk_id=... pages=...] + chunk_text]
    D --> E{Add block exceeds total 9000 chars?}
    E -- No --> F[Append block to CONTEXT]
    E -- Yes --> G[Stop adding more blocks]
    F --> H{Used 5 chunks already?}
    H -- No --> B
    H -- Yes --> I[Finalize CONTEXT]
    G --> I
    I --> J[Send prompt to LLM]
```

**What counts toward each limit**
- `2200` = max characters of `chunk_text` per retrieved chunk (text only).
- `9000` = total prompt context characters across appended blocks, including metadata labels like `[chunk_id=... pages=...]`.
- Default max blocks = `5` chunks.

**Plain-English summary**
- The model gets a compact evidence pack: up to 5 snippets, each shortened if needed, and all snippets together capped at about 9000 characters.
