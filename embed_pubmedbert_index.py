from __future__ import annotations

import argparse
import json
import os
import sys


def record_text(rec):
    # Prefer the corpus search_text (retrieval-optimized title+body); fall back to title + text.
    # Same field choice as embed_small_index.record_text so a record embeds the same text either way.
    st = rec.get("search_text")
    if isinstance(st, str) and st.strip():
        return st
    return f"{rec.get('title', '')}\n{rec.get('text', '')}".strip()


def build(corpus_path, index_path, model_name, device, batch_size):
    import faiss
    from sentence_transformers import SentenceTransformer

    # Load the serving embedding model and lock the index dimension to it up front.
    model = SentenceTransformer(model_name, device=device)
    dim = int(model.get_sentence_embedding_dimension())
    assert dim == 768, f"expected 768-dim embeddings but model {model_name} produced dim {dim}"

    index = faiss.IndexFlatIP(dim)  # inner product over L2-normalized vectors == cosine similarity

    lines_read = 0
    kept = 0
    skipped_bad = 0
    skipped_nondict = 0
    batch = []
    flushes = 0

    def flush():
        # Encode one buffered batch and append its vectors in file order (vector i == record i).
        nonlocal flushes
        if not batch:
            return
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        index.add(vecs)
        batch.clear()
        flushes += 1
        if flushes % 40 == 0:
            print(f"  progress: kept={kept} ntotal={index.ntotal}", flush=True)

    # Stream the corpus one line at a time; never load the 770 MB file whole.
    with open(corpus_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            lines_read += 1
            line = line.strip()
            if not line:
                # Blank line: the retriever skips it, so skip here to keep the record count aligned.
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Malformed JSON: the retriever drops it; drop it here so the vectors stay aligned.
                skipped_bad += 1
                continue
            if not isinstance(obj, dict):
                # Valid JSON that is not an object: the retriever drops it as a non-dict too.
                skipped_nondict += 1
                continue
            batch.append(record_text(obj))
            kept += 1
            if len(batch) >= batch_size:
                flush()

    flush()  # final partial batch

    # Parity guard: one vector per kept record or serving refuses the index and drops to lexical.
    assert index.ntotal == kept, f"ntotal {index.ntotal} != kept {kept}"

    faiss.write_index(index, index_path)
    meta_path = index_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"embedding_model": model_name, "dim": dim, "vectors": index.ntotal}, fh)

    print(f"lines_read={lines_read}")
    print(f"kept={kept}")
    print(f"skipped_bad={skipped_bad}")
    print(f"skipped_nondict={skipped_nondict}")
    print(f"ntotal={index.ntotal}")
    print(f"dim={dim}")
    print(f"wrote {index_path} and {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/rag_corpus_pubmedbert.jsonl")
    ap.add_argument("--index", default="data/rag_index_pubmedbert.faiss")
    ap.add_argument("--model", default="NeuML/pubmedbert-base-embeddings")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    # Never overwrite an existing index or its sidecar; the prior artifacts are frozen.
    meta_path = args.index + ".meta.json"
    if os.path.exists(args.index):
        print(f"refusing to overwrite existing index: {args.index}", file=sys.stderr)
        sys.exit(1)
    if os.path.exists(meta_path):
        print(f"refusing to overwrite existing sidecar: {meta_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.corpus):
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    build(args.corpus, args.index, args.model, args.device, args.batch_size)


if __name__ == "__main__":
    main()
