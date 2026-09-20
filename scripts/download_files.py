import tarfile
import requests

from pathlib import Path


def download_from_zenodo(record: int, target_dir: Path) -> list[Path]:
    base_url = "https://zenodo.org/api/records/"
    url = base_url + str(record)

    # Make a request to the Zenodo API
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses

    # Extract the files section from the response
    files = response.json().get("files", [])

    if not files:
        print("No files found for this record.")
        return []
    print(f"Found {len(files)} files")

    # Download each file
    downloaded_paths = []
    for file in files:
        file_url = file["links"]["self"]
        target_filepath = target_dir / file["key"]

        print(f"Downloading {file_url}...")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with target_filepath.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"{file_url} downloaded to {target_filepath}")
        downloaded_paths.append(target_filepath)

    return downloaded_paths


def unpack(fpath: Path) -> Path | None:
    """
    Unpack a .tar.gz file into a directory with the same name (without extensions).

    Args:
        fpath: Path to the .tar.gz file

    Returns:
        Path to the extracted directory, or None if file is not a .tar.gz
    """
    if not fpath.suffix == ".gz" or not fpath.stem.endswith(".tar"):
        print(f"Skipping {fpath.name} - not a .tar.gz file")
        return None

    # Determine output directory (remove .tar.gz extensions)
    output_dir = fpath.parent / fpath.stem.replace(".tar", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {fpath.name} to {output_dir}...")

    try:
        with tarfile.open(fpath, "r:gz") as tar:
            tar.extractall(path=output_dir)
        print(f"Successfully extracted to {output_dir}")
        return output_dir
    except tarfile.TarError as e:
        print(f"Error extracting {fpath.name}: {e}")
        return None


if __name__ == "__main__":
    target_dir = Path(__file__).resolve().parent.parent / "resources" / "output"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths_to_unpack = download_from_zenodo(18611418, target_dir)
    for path in paths_to_unpack:
        unpack(path)
