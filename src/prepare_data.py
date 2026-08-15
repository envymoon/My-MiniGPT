"""Normalize, deduplicate, group-split, tokenize, and pack local or HF data."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
EXCESS_NEWLINES = re.compile(r"\n{3,}")
GUTENBERG_START = re.compile(
    r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}",
    re.IGNORECASE,
)
GUTENBERG_END = re.compile(
    r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Document:
    source: str
    text: str
    group_id: str | None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARACTERS.sub("", text)
    text = HORIZONTAL_WHITESPACE.sub(" ", text)
    text = EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    """Hash the effective manifest so stale packed tokens are never reused."""
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return stable_hash(canonical)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove standard Gutenberg wrappers while leaving book structure intact."""
    start = GUTENBERG_START.search(text)
    if start:
        text = text[start.end() :]
    text = GUTENBERG_END.sub("", text)
    return text


def collapse_repeated_document(text: str) -> tuple[str, int]:
    """Collapse files made by concatenating the same complete book repeatedly.

    A 1024-character document prefix is sufficiently specific for books. The
    minimum span prevents ordinary refrains or short front matter from being
    mistaken for a repeated complete work.
    """
    marker_size = 1_024
    if len(text) < 2 * marker_size:
        return text, 0
    # Probe beyond the title/front matter too: malformed files sometimes alter
    # only a repeated copy's PART heading while duplicating the entire body.
    for marker_offset in (0, 2_048, 4_096, 8_192, 16_384):
        if marker_offset + marker_size > len(text):
            break
        marker = text[marker_offset : marker_offset + marker_size]
        second_marker = text.find(marker, marker_offset + marker_size)
        copy_span = second_marker - marker_offset
        if second_marker < 0 or copy_span < 50_000:
            continue
        positions = []
        search_from = marker_offset
        while True:
            position = text.find(marker, search_from)
            if position < 0:
                break
            positions.append(position)
            search_from = position + marker_size
        if len(positions) >= 2:
            # Keep the first complete copy. Small heading/separator differences
            # in later copies do not affect the inferred first-copy boundary.
            boundary = copy_span
            paragraph_boundary = text.rfind("\n\n", max(0, boundary - 64), boundary + 1)
            if paragraph_boundary >= 0:
                boundary = paragraph_boundary
            return text[:boundary].rstrip(), len(positions) - 1
    return text, 0


def read_text_file(path: Path, configured_encoding: str | None = None) -> str:
    """Decode common TXT encodings without silently discarding bad bytes."""
    data = path.read_bytes()
    encodings = [configured_encoding] if configured_encoding else []
    encodings.extend(["utf-8-sig", "cp1252", "gb18030"])
    attempted: set[str] = set()
    for encoding in encodings:
        if not encoding or encoding in attempted:
            continue
        attempted.add(encoding)
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(
        f"cannot decode {path}; set an explicit 'encoding' in the data manifest"
    )


def nested_value(record: dict[str, Any], field: str | None) -> Any:
    if not field:
        return None
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def record_matches_filters(record: dict[str, Any], source: dict[str, Any]) -> bool:
    """Apply small, explicit manifest filters before text preparation."""
    for rule in source.get("filters", []):
        actual = nested_value(record, rule["field"])
        expected = rule.get("value")
        operation = rule.get("op", "eq")
        if actual is None:
            return False
        try:
            if operation == "eq" and actual != expected:
                return False
            if operation == "ne" and actual == expected:
                return False
            if operation == "gte" and not actual >= expected:
                return False
            if operation == "gt" and not actual > expected:
                return False
            if operation == "lte" and not actual <= expected:
                return False
            if operation == "lt" and not actual < expected:
                return False
            if operation == "in" and actual not in expected:
                return False
            if operation not in {"eq", "ne", "gte", "gt", "lte", "lt", "in"}:
                raise ValueError(f"unsupported filter operation: {operation!r}")
        except TypeError:
            return False
    return True


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("sources"):
        raise ValueError("manifest must contain at least one source")
    return manifest


def iter_local_documents(source: dict[str, Any]) -> Iterator[Document]:
    pattern = os.path.expandvars(os.path.expanduser(source["path"]))
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        print(f"warning: no files matched {pattern!r}")
    for path in paths:
        file_path = Path(path)
        source_format = source.get("format", "auto").lower()
        is_plain_text = source_format == "text" or (
            source_format == "auto" and file_path.suffix.lower() in {".txt", ".text", ".md"}
        )
        if is_plain_text:
            try:
                text = read_text_file(file_path, source.get("encoding"))
            except (OSError, UnicodeError) as error:
                print(f"warning: skip {file_path}: {error}")
                continue
            if source.get("strip_gutenberg_boilerplate", False):
                text = strip_gutenberg_boilerplate(text)
            # A plain-text file is one work. Its filename is the stable split group.
            yield Document(source["name"], text, file_path.name)
            continue

        if source_format not in {"auto", "jsonl"}:
            raise ValueError(
                f"unsupported local format {source_format!r}; use 'text' or 'jsonl'"
            )
        with file_path.open("r", encoding=source.get("encoding", "utf-8")) as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                    text = nested_value(record, source.get("text_field", "text"))
                    group_id = nested_value(record, source.get("group_field"))
                except (json.JSONDecodeError, TypeError) as error:
                    print(f"warning: skip {file_path}:{line_number}: {error}")
                    continue
                if isinstance(text, str):
                    yield Document(source["name"], text, None if group_id is None else str(group_id))


def iter_hf_documents(source: dict[str, Any], seed: int) -> Iterator[Document]:
    # Imported lazily so purely local preparation does not require datasets.
    from datasets import load_dataset

    # Some large Hub datasets contain an occasional Parquet shard whose schema
    # differs from the dataset-level metadata.  `datasets` raises a CastError
    # lazily while iterating that shard, which used to abort tokenizer training
    # and leave the whole corpus unusable.  A remote source is optional by
    # design, so isolate its failure and continue with the remaining sources.
    try:
        dataset = load_dataset(
            source["hf_dataset"],
            name=source.get("hf_config"),
            split=source.get("split", "train"),
            streaming=True,
        )
        shuffle_buffer = int(source.get("shuffle_buffer", 10_000))
        if shuffle_buffer > 0:
            dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
        max_documents = source.get("max_documents")
        accepted_documents = 0
        for record in dataset:
            if not record_matches_filters(record, source):
                continue
            if max_documents is not None and accepted_documents >= int(max_documents):
                break
            text = nested_value(record, source.get("text_field", "text"))
            group_id = nested_value(record, source.get("group_field"))
            if isinstance(text, str):
                accepted_documents += 1
                yield Document(
                    source["name"], text, None if group_id is None else str(group_id)
                )
    except Exception as error:
        print(
            f"warning: skip HF source={source['name']!r} after remote/schema error: "
            f"{type(error).__name__}: {error}"
        )


def iter_source_documents(source: dict[str, Any], seed: int) -> Iterator[Document]:
    if source.get("enabled", True) is False:
        print(f"skipping disabled source={source.get('name')!r}")
        return
    if "hf_dataset" in source:
        yield from iter_hf_documents(source, seed)
    elif "path" in source:
        yield from iter_local_documents(source)
    else:
        raise ValueError(f"source {source.get('name')!r} needs 'path' or 'hf_dataset'")


def empty_stats() -> dict[str, int]:
    return {
        "seen": 0,
        "kept": 0,
        "duplicates": 0,
        "repeated_copies_removed": 0,
        "filtered": 0,
        "train_tokens": 0,
        "val_tokens": 0,
    }


def prepare(manifest_path: str, output_dir: str) -> None:
    manifest = load_manifest(manifest_path)
    tokenizer_name = manifest.get("tokenizer_name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    # Preprocessing whole documents is intentional; model context windows are
    # formed later by PackedTokenDataset, so tokenizer truncation is disabled.
    tokenizer.model_max_length = 10**30
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError(f"tokenizer {tokenizer_name!r} has no EOS token")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.bin"
    val_path = output / "val.bin"
    min_chars = int(manifest.get("min_chars", 200))
    max_chars = int(manifest.get("max_chars", 5_000_000))
    validation_fraction = float(manifest.get("validation_fraction", 0.005))
    seed = int(manifest.get("seed", 42))
    progress_every = int(manifest.get("progress_every", 1_000))
    seen: set[str] = set()
    stats = empty_stats()
    source_stats: dict[str, dict[str, int]] = {}
    recorded_groups: dict[str, dict[str, list[str]]] = {}

    with train_path.open("wb") as train_handle, val_path.open("wb") as val_handle:
        for source_index, source in enumerate(manifest["sources"]):
            source_name = source["name"]
            source_validation_fraction = float(
                source.get("validation_fraction", validation_fraction)
            )
            if not 0.0 <= source_validation_fraction < 1.0:
                raise ValueError(
                    f"source {source_name!r} validation_fraction must be in [0, 1)"
                )
            validation_cutoff = int(source_validation_fraction * 100_000)
            validation_groups = {
                str(group) for group in source.get("validation_groups", [])
            }
            current = empty_stats()
            source_stats[source_name] = current
            if source.get("record_groups", False):
                recorded_groups[source_name] = {"train": [], "validation": []}
            max_tokens = source.get("max_tokens")
            print(f"preparing source={source_name}")
            for document in iter_source_documents(source, seed + source_index):
                current["seen"] += 1
                stats["seen"] += 1
                text = normalize_text(document.text)
                if source.get("collapse_repeated_documents", False):
                    text, removed_copies = collapse_repeated_document(text)
                    current["repeated_copies_removed"] += removed_copies
                    stats["repeated_copies_removed"] += removed_copies
                    if removed_copies:
                        print(
                            f"source={source_name} group={document.group_id!r} "
                            f"removed_repeated_copies={removed_copies}"
                        )
                if not min_chars <= len(text) <= max_chars:
                    current["filtered"] += 1
                    stats["filtered"] += 1
                    continue
                digest = stable_hash(text)
                if digest in seen:
                    current["duplicates"] += 1
                    stats["duplicates"] += 1
                    continue
                seen.add(digest)
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                token_ids.append(eos_token_id)
                encoded = np.asarray(token_ids, dtype=np.uint32)

                # Split books/works/authors as a unit when group_field is supplied.
                # Never randomly split already-tokenized chunks from the same work.
                split_identity = (
                    f"{source_name}:{document.group_id}"
                    if document.group_id is not None
                    else digest
                )
                split_hash = stable_hash(split_identity)
                is_validation = (
                    document.group_id in validation_groups
                    or int(split_hash[:8], 16) % 100_000 < validation_cutoff
                )
                if source_name in recorded_groups:
                    group_label = document.group_id or digest[:16]
                    split_label = "validation" if is_validation else "train"
                    recorded_groups[source_name][split_label].append(group_label)
                handle = val_handle if is_validation else train_handle
                encoded.tofile(handle)
                key = "val_tokens" if is_validation else "train_tokens"
                current[key] += len(encoded)
                current["kept"] += 1
                stats[key] += len(encoded)
                stats["kept"] += 1

                source_tokens = current["train_tokens"] + current["val_tokens"]
                if progress_every and current["kept"] % progress_every == 0:
                    print(
                        f"source={source_name} documents={current['kept']:,} "
                        f"tokens={source_tokens:,}"
                    )
                if max_tokens is not None and source_tokens >= int(max_tokens):
                    print(f"source={source_name} reached token budget {source_tokens:,}")
                    break

    if stats["train_tokens"] == 0 or stats["val_tokens"] == 0:
        raise ValueError(
            "preparation produced an empty train or validation split; add data or "
            "increase validation_fraction"
        )
    metadata = {
        "tokenizer_name": tokenizer_name,
        "vocab_size": len(tokenizer),
        "dtype": "uint32",
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "stats": stats,
        "source_stats": source_stats,
        "recorded_groups": recorded_groups,
        "manifest": str(Path(manifest_path).resolve()),
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/data_sources.json")
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare(arguments.manifest, arguments.output_dir)
