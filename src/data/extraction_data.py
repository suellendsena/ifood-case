import os
import shutil
import urllib.request
import tarfile
import logging
import argparse

class RawDataLoader:

    def __init__(self, url, download_path="tmp/ds-data.tar.gz", extract_path="tmp/ds-data", target_path="data/raw"):
        self.url = url
        self.download_path = download_path
        self.extract_path = extract_path
        self.target_path = target_path
        self.expected_files = {"offers.json", "profile.json", "transactions.json"}
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)

    def download_tarball(self):

        self.logger.info("Downloading the .tar.gz archive...")
        urllib.request.urlretrieve(self.url, self.download_path)
        self.logger.info("Download completed.")

    def extract_tarball(self):

        self.logger.info("Extracting contents of the .tar.gz archive...")
        with tarfile.open(self.download_path, "r:gz") as tar:
            tar.extractall(path=self.extract_path)
        self.logger.info("Extraction completed.")

    def list_json_files(self):

        self.logger.info("Listing all .json files found in the extracted content:")
        json_files = []
        for root, _, files in os.walk(self.extract_path):
            for file in files:
                if file.endswith(".json"):
                    relative_path = os.path.relpath(os.path.join(root, file), self.extract_path)
                    json_files.append(relative_path)
                    self.logger.info(f" - {relative_path}")
        return json_files

    def organize_files(self):

        self.logger.info("Organizing extracted files...")
        copied_files = set()
        all_json_files = self.list_json_files()

        for root, _, files in os.walk(self.extract_path):
            for file in files:
                if file in self.expected_files:
                    src = os.path.join(root, file)
                    dst_json = os.path.join(self.target_path, file)
                    shutil.copy(src, dst_json)
                    copied_files.add(file)
                    self.logger.info(f"Copied JSON file: {file}")


        missing = self.expected_files - copied_files
        if missing:
            self.logger.warning(f"Expected files not found: {missing}")
        else:
            self.logger.info("All expected files were successfully organized.")

def main():

    parser = argparse.ArgumentParser(description="Download, extract, and organize raw data files.")
    parser.add_argument("--url", required=True, help="URL of the .tar.gz archive to download")
    parser.add_argument("--download_path", default="tmp/ds-data.tar.gz", help="Path to save the downloaded .tar.gz archive")
    parser.add_argument("--extract_path", default="tmp/ds-data", help="Directory to extract the contents of the archive")
    parser.add_argument("--target_path", default="data/raw", help="Directory to store the organized JSON files")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.download_path), exist_ok=True)
    os.makedirs(args.extract_path, exist_ok=True)
    os.makedirs(args.target_path, exist_ok=True)

    loader = RawDataLoader(
        url=args.url,
        download_path=args.download_path,
        extract_path=args.extract_path,
        target_path=args.target_path
    )

    loader.download_tarball()
    loader.extract_tarball()
    loader.organize_files()

if __name__ == "__main__":
    main()
