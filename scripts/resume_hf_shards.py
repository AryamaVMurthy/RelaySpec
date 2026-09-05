from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int)
    parser.add_argument("--filename", action="append")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--connections-per-shard", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resume_one(
    *,
    repo: str,
    revision: str,
    filename: str,
    blob_dir: Path,
    connections_per_shard: int,
    attempts: int = 20,
) -> tuple[str, int]:
    url = hf_hub_url(repo, filename, revision=revision)
    metadata = get_hf_file_metadata(url)
    if metadata.size is None or metadata.etag is None:
        raise RuntimeError(f"missing immutable metadata for {filename}")
    expected_size = int(metadata.size)
    etag = metadata.etag.strip('"')
    if len(etag) != 64:
        raise RuntimeError(f"expected a SHA-256 LFS etag for {filename}: {etag}")
    destination = blob_dir / etag
    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise RuntimeError(f"cached blob has wrong size: {destination}")
        return filename, expected_size

    candidates = sorted(
        blob_dir.glob(f"{etag}.*.incomplete"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"no resumable partial found for {filename}")
    partial = candidates[0]
    prefix_size = partial.stat().st_size
    if prefix_size > expected_size:
        raise RuntimeError(f"partial exceeds expected size: {partial}")

    def download_range(start: int, end: int) -> tuple[int, Path]:
        chunk_path = partial.with_name(f"{partial.name}.range-{start}-{end}")
        expected_chunk_size = end - start + 1
        for attempt in range(1, attempts + 1):
            observed_chunk_size = (
                chunk_path.stat().st_size if chunk_path.exists() else 0
            )
            if observed_chunk_size == expected_chunk_size:
                return start, chunk_path
            if observed_chunk_size > expected_chunk_size:
                raise RuntimeError(f"range chunk exceeds expected size: {chunk_path}")
            request_start = start + observed_chunk_size
            try:
                with requests.get(
                    url,
                    headers={"Range": f"bytes={request_start}-{end}"},
                    stream=True,
                    timeout=(30, 90),
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(
                            f"range request returned HTTP {response.status_code}"
                        )
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {request_start}-{end}/"):
                        raise RuntimeError(f"unexpected Content-Range: {content_range}")
                    with chunk_path.open("ab") as stream:
                        for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                            if block:
                                stream.write(block)
                                stream.flush()
            except (requests.RequestException, RuntimeError) as error:
                if attempt == attempts:
                    raise RuntimeError(
                        f"failed range {start}-{end} after {attempts} attempts"
                    ) from error
                time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    if prefix_size < expected_size:
        remaining = expected_size - prefix_size
        chunk_size = (remaining + connections_per_shard - 1) // connections_per_shard
        ranges = []
        for index in range(connections_per_shard):
            start = prefix_size + index * chunk_size
            if start >= expected_size:
                break
            ranges.append((start, min(start + chunk_size - 1, expected_size - 1)))
        with ThreadPoolExecutor(max_workers=len(ranges)) as range_executor:
            range_futures = [
                range_executor.submit(download_range, start, end)
                for start, end in ranges
            ]
            completed_ranges = sorted(future.result() for future in range_futures)
        if partial.stat().st_size != prefix_size:
            raise RuntimeError(f"partial changed while ranges downloaded: {partial}")
        with partial.open("ab") as destination_stream:
            for _, chunk_path in completed_ranges:
                with chunk_path.open("rb") as chunk_stream:
                    shutil.copyfileobj(
                        chunk_stream,
                        destination_stream,
                        length=16 * 1024 * 1024,
                    )
        for _, chunk_path in completed_ranges:
            chunk_path.unlink()

    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"resumed shard has wrong size: {filename}")
    observed_hash = sha256(partial)
    if observed_hash != etag:
        raise RuntimeError(
            f"SHA-256 mismatch for {filename}: expected {etag}, found {observed_hash}"
        )
    os.replace(partial, destination)
    return filename, expected_size


def main() -> None:
    args = parse_args()
    repo_cache_name = f"models--{args.repo.replace('/', '--')}"
    blob_dir = args.cache_dir / repo_cache_name / "blobs"
    if not blob_dir.is_dir():
        raise RuntimeError(f"cache blob directory does not exist: {blob_dir}")
    if args.filename:
        filenames = list(args.filename)
    elif args.shards:
        filenames = [
            f"model-{index:05d}-of-{args.shards:05d}.safetensors"
            for index in range(1, args.shards + 1)
        ]
    else:
        raise ValueError("provide --filename or --shards")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                resume_one,
                repo=args.repo,
                revision=args.revision,
                filename=filename,
                blob_dir=blob_dir,
                connections_per_shard=args.connections_per_shard,
            ): filename
            for filename in filenames
        }
        for future in as_completed(futures):
            filename, size = future.result()
            print(f"complete {filename} {size}", flush=True)


if __name__ == "__main__":
    main()
