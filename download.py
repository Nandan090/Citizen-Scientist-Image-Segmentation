"""
iNaturalist Data Acquisition Script
=====================================
Downloads observations and images from iNaturalist for specified plant species.
Organizes images into class folders compatible with torchvision.ImageFolder.

Usage:
    python download_inaturalist.py

Output structure:
    data/
    └── image/
        ├── Fallopia_japonica/
        │   ├── img_001.jpg
        │   └── ...
        ├── Species_B/
        └── ...
"""

import os
import time
import logging
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os 

# --- CONFIGURATION ---

# Output directory (matches DATA_ROOT in your pipeline script)


OUTPUT_DIR = os.getcwd()


# Species to download — edit this list as needed
# Format: "Genus species" (iNaturalist uses scientific names)
SPECIES_LIST = [
    "Fallopia japonica",         # Japanese Knotweed
    "Lupinus polyphyllus",       # Garden Lupin

]

# Download settings
MAX_IMAGES_PER_SPECIES = 8000  # Matches SAMPLES_PER_CLASS in pipeline
QUALITY_GRADE = "research"      # "research" = verified; "needs_id" = unverified; "any"
LICENSE = "cc-by,cc-by-nc,cc0"  # Only download open-license images
MIN_IMAGE_SIZE_BYTES = 10_000   # Skip images smaller than 10KB (likely corrupt/tiny)
MAX_WORKERS = 8                 # Parallel download threads
REQUESTS_PER_SECOND = 1.0       # iNaturalist API rate limit: be polite
PAGE_SIZE = 200                 # Max allowed by iNaturalist API

INAT_API_BASE = "https://api.inaturalist.org/v1"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --- UTILITIES ---

def sanitize_folder_name(name: str) -> str:
    """Convert 'Fallopia japonica' -> 'Fallopia_japonica'"""
    return name.strip().replace(" ", "_")


def get_taxon_id(species_name: str) -> int | None:
    """Look up iNaturalist taxon ID for a species name."""
    url = f"{INAT_API_BASE}/taxa"
    params = {"q": species_name, "rank": "species", "per_page": 1}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            taxon = results[0]
            logger.info(f"  Taxon found: {taxon['name']} (ID: {taxon['id']})")
            return taxon["id"]
        else:
            logger.warning(f"  No taxon found for '{species_name}'")
            return None
    except requests.RequestException as e:
        logger.error(f"  Taxon lookup failed for '{species_name}': {e}")
        return None


def fetch_observation_page(taxon_id: int, page: int, quality_grade: str) -> list[dict]:
    """Fetch one page of observations with photos."""
    url = f"{INAT_API_BASE}/observations"
    params = {
        "taxon_id": taxon_id,
        "quality_grade": quality_grade,
        "photos": "true",
        "license": LICENSE,
        "per_page": PAGE_SIZE,
        "page": page,
        "order": "desc",
        "order_by": "votes",  # Highest quality first
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])
    except requests.RequestException as e:
        logger.warning(f"  Page {page} fetch failed: {e}")
        return []


def collect_image_urls(taxon_id: int, max_images: int, quality_grade: str) -> list[str]:
    """Collect image URLs from iNaturalist observations."""
    urls = []
    page = 1
    logger.info(f"  Collecting image URLs (target: {max_images})...")

    while len(urls) < max_images:
        observations = fetch_observation_page(taxon_id, page, quality_grade)
        if not observations:
            logger.info(f"  No more observations at page {page}.")
            break

        for obs in observations:
            for photo in obs.get("photos", []):
                # Get medium resolution URL: replace 'square' with 'medium'
                url = photo.get("url", "")
                if url:
                    url = url.replace("square", "medium")
                    urls.append(url)
                    if len(urls) >= max_images:
                        break
            if len(urls) >= max_images:
                break

        logger.info(f"  Page {page}: collected {len(urls)} URLs so far...")
        page += 1
        time.sleep(1.0 / REQUESTS_PER_SECOND)

    return urls[:max_images]


def download_image(args: tuple) -> tuple[str, bool]:
    """Download a single image. Returns (url, success)."""
    url, save_path = args
    if save_path.exists():
        return url, True  # Already downloaded

    try:
        r = requests.get(url, timeout=15, stream=True)
        r.raise_for_status()

        content = b"".join(r.iter_content(chunk_size=8192))
        if len(content) < MIN_IMAGE_SIZE_BYTES:
            return url, False  # Skip tiny/corrupt images

        save_path.write_bytes(content)
        return url, True

    except requests.RequestException:
        return url, False


def download_species(species_name: str, output_dir: Path):
    """Full pipeline: taxon lookup -> URL collection -> parallel download."""
    folder_name = sanitize_folder_name(species_name)
    species_dir = output_dir / folder_name
    species_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*55}")
    logger.info(f"Species: {species_name}")
    logger.info(f"Output:  {species_dir}")

    # Check existing downloads
    existing = list(species_dir.glob("*.jpg"))
    if len(existing) >= MAX_IMAGES_PER_SPECIES:
        logger.info(f"  Already have {len(existing)} images. Skipping.")
        return

    # Step 1: Get taxon ID
    taxon_id = get_taxon_id(species_name)
    if taxon_id is None:
        return

    # Step 2: Collect URLs
    needed = MAX_IMAGES_PER_SPECIES - len(existing)
    urls = collect_image_urls(taxon_id, needed, QUALITY_GRADE)
    if not urls:
        logger.warning(f"  No images found for {species_name}.")
        return
    logger.info(f"  Found {len(urls)} image URLs.")

    # Step 3: Build download tasks (skip already-downloaded)
    tasks = []
    for i, url in enumerate(urls):
        filename = f"{folder_name}_{len(existing) + i:05d}.jpg"
        save_path = species_dir / filename
        tasks.append((url, save_path))

    # Step 4: Parallel download
    success, fail = 0, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_image, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks),
                           desc=f"  Downloading {folder_name}"):
            _, ok = future.result()
            if ok:
                success += 1
            else:
                fail += 1

    logger.info(f"  Done: {success} saved, {fail} failed.")


# --- MAIN ---

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("iNaturalist Data Acquisition")
    logger.info(f"Target species : {len(SPECIES_LIST)}")
    logger.info(f"Images per class: {MAX_IMAGES_PER_SPECIES}")
    logger.info(f"Quality grade  : {QUALITY_GRADE}")
    logger.info(f"Output dir     : {output_dir.resolve()}")

    for species in SPECIES_LIST:
        download_species(species, output_dir)

    # Final summary
    logger.info("\n" + "="*55)
    logger.info("DOWNLOAD SUMMARY")
    for species in SPECIES_LIST:
        folder = output_dir / sanitize_folder_name(species)
        count = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        status = "✓" if count >= MAX_IMAGES_PER_SPECIES else f"⚠ only {count}"
        logger.info(f"  {sanitize_folder_name(species):<35} {status}")
    logger.info("="*55)


if __name__ == "__main__":
    main()
