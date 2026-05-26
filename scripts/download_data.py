"""Download and serialize raw datasets for grammar correction experiments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, List, Optional, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DATA_DIR / "manifest.json"
CONLL_REFERENCE_URL = "https://www.comp.nus.edu.sg/~nlp/conll14st.html"
USER_AGENT = "GrammarAutocorrector/0.2.0"
MAX_RETRIES = 3
CHUNK_SIZE = 8192
T = TypeVar("T")


class LinkCollector(HTMLParser):
    """Collect anchor links from a remote HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        """Store href attributes found in anchor tags."""

        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def log(message: str) -> None:
    """Print a timestamped progress message."""

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def human_size(num_bytes: int) -> str:
    """Render a human-readable file size."""

    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} GB"


def sha256sum(path: Path) -> str:
    """Compute a SHA-256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_with_retries(action: Callable[[], T], description: str) -> T:
    """Execute an action with exponential-backoff retry handling."""

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return action()
        except Exception as exc:  # pragma: no cover
            last_error = exc
            wait_seconds = 2 ** (attempt - 1)
            log(
                f"{description} failed on attempt {attempt}/{MAX_RETRIES}: {exc}. "
                f"Retrying in {wait_seconds}s."
            )
            time.sleep(wait_seconds)

    if last_error is None:
        raise RuntimeError(f"{description} failed without an exception.")
    raise last_error


def fetch_text(url: str) -> str:
    """Fetch a remote text resource."""

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace")


def discover_conll_archive_url(reference_url: str) -> str:
    """Discover a downloadable CoNLL archive from the reference page."""

    html = fetch_text(reference_url)
    collector = LinkCollector()
    collector.feed(html)

    archive_links = [
        urljoin(reference_url, link)
        for link in collector.links
        if link.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))
    ]

    preferred = [
        link
        for link in archive_links
        if "conll" in link.casefold()
        and ("test" in link.casefold() or "data" in link.casefold())
    ]
    if preferred:
        return preferred[0]
    if archive_links:
        return archive_links[0]

    raise RuntimeError(
        "Could not discover a downloadable CoNLL-2014 archive from the reference page. "
        "Set CONLL2014_ARCHIVE_URL to a direct archive URL and rerun the script."
    )


def download_file(url: str, destination: Path) -> Path:
    """Download a file to disk while printing progress information."""

    request = Request(url, headers={"User-Agent": USER_AGENT})
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with (
            urlopen(request, timeout=120) as response,
            partial_path.open("wb") as output,
        ):  # noqa: S310
            total_bytes = int(response.headers.get("Content-Length", "0"))
            bytes_written = 0

            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                bytes_written += len(chunk)

            partial_path.replace(destination)
            size_display = human_size(total_bytes or bytes_written)
            log(f"Downloaded {destination.name} ({size_display})")
            return destination
    except (HTTPError, URLError, TimeoutError) as exc:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    """Extract a supported archive type."""

    extract_dir.mkdir(parents=True, exist_ok=True)
    lower_name = archive_path.name.casefold()

    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
    elif lower_name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path) as archive:
            archive.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    return extract_dir


def serialize_directory_to_json(
    dataset_name: str, source_url: str, directory: Path, output_path: Path
) -> Path:
    """Serialize extracted dataset files into JSON format."""

    serialized_files: List[Dict[str, object]] = []
    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(directory).as_posix()
        payload: Dict[str, object] = {
            "relative_path": relative_path,
            "size_bytes": file_path.stat().st_size,
        }

        if file_path.suffix.casefold() in {".txt", ".m2", ".sgml", ".xml"}:
            payload["content"] = file_path.read_text(encoding="utf-8", errors="replace")
        else:
            payload["content"] = None

        serialized_files.append(payload)

    output = {
        "dataset": dataset_name,
        "source_url": source_url,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": serialized_files,
    }

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(
        f"Wrote serialized dataset to {output_path} "
        f"({human_size(output_path.stat().st_size)})"
    )
    return output_path


def load_huggingface_dataset(dataset_name: str) -> Dict[str, object]:
    """Load a Hugging Face dataset and serialize its splits into plain Python types."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency managed externally
        raise ImportError(
            "datasets is required for Hugging Face downloads. Install it with "
            "`pip install datasets`."
        ) from exc

    dataset = load_dataset(dataset_name)
    serializable_splits: Dict[str, object] = {}
    for split_name, split_dataset in dataset.items():
        serializable_splits[split_name] = list(split_dataset)
    return serializable_splits


def save_dataset_json(
    dataset_alias: str,
    source_dataset: str,
    splits: Dict[str, object],
    output_path: Path,
) -> Path:
    """Persist a dataset dictionary as formatted JSON."""

    payload = {
        "dataset": dataset_alias,
        "source_dataset": source_dataset,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": splits,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(
        f"Wrote {dataset_alias} to {output_path} "
        f"({human_size(output_path.stat().st_size)})"
    )
    return output_path


def build_manifest(files: List[Path]) -> None:
    """Create a manifest of downloaded files and their checksums."""

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [
            {
                "path": file_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256sum(file_path),
            }
            for file_path in sorted(files)
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote manifest to {MANIFEST_PATH}")


def download_conll2014() -> List[Path]:
    """Download and serialize the CoNLL-2014 dataset archive."""

    log("Discovering CoNLL-2014 archive URL")
    archive_url = os.getenv("CONLL2014_ARCHIVE_URL") or run_with_retries(
        lambda: discover_conll_archive_url(CONLL_REFERENCE_URL),
        "CoNLL-2014 archive discovery",
    )

    archive_name = Path(archive_url).name or "conll2014_archive"
    archive_path = RAW_DATA_DIR / archive_name
    extracted_dir = RAW_DATA_DIR / "conll2014_extracted"
    serialized_json = RAW_DATA_DIR / "conll2014.json"

    downloaded_archive = run_with_retries(
        lambda: download_file(archive_url, archive_path),
        "CoNLL-2014 archive download",
    )

    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    run_with_retries(
        lambda: extract_archive(downloaded_archive, extracted_dir),
        "CoNLL-2014 archive extraction",
    )
    serialized_path = serialize_directory_to_json(
        dataset_name="conll2014",
        source_url=archive_url,
        directory=extracted_dir,
        output_path=serialized_json,
    )

    return [downloaded_archive, serialized_path]


def download_hf_proxy_dataset(dataset_alias: str, source_dataset: str) -> List[Path]:
    """Download a Hugging Face dataset and write it to JSON."""

    output_path = RAW_DATA_DIR / f"{dataset_alias}.json"
    splits = run_with_retries(
        lambda: load_huggingface_dataset(source_dataset),
        f"{dataset_alias} Hugging Face download",
    )
    saved_file = save_dataset_json(dataset_alias, source_dataset, splits, output_path)
    return [saved_file]


def main() -> int:
    """Run dataset download and serialization tasks."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []
    failures: List[str] = []

    dataset_steps = [
        ("CoNLL-2014", download_conll2014),
        ("BEA-2019 proxy", lambda: download_hf_proxy_dataset("bea2019_proxy", "jfleg")),
        ("JFLEG", lambda: download_hf_proxy_dataset("jfleg", "jfleg")),
    ]

    for dataset_name, step in dataset_steps:
        log(f"Starting {dataset_name} download")
        try:
            generated_files.extend(step())
        except Exception as exc:  # pragma: no cover - depends on runtime network access
            failures.append(f"{dataset_name}: {exc}")
            log(f"{dataset_name} failed: {exc}")

    if generated_files:
        build_manifest(generated_files)

    if failures:
        log("Completed with failures:")
        for failure in failures:
            log(f"  - {failure}")
        return 1

    log("All datasets downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
