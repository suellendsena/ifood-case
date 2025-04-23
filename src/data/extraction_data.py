# Databricks notebook source
import os
import shutil
import urllib.request
import tarfile
import logging

class RawDataLoader:
    def __init__(self, url, download_path="/tmp/ds-data.tar.gz", extract_path="/tmp/ds-data", target_path="dbfs:/FileStore/raw"):
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
        os.makedirs("/tmp/staging", exist_ok=True)

        for root, _, files in os.walk(self.extract_path):
            for file in files:
                if file in self.expected_files:
                    src = os.path.join(root, file)
                    tmp_dst = os.path.join("/tmp/staging", file)
                    shutil.copy(src, tmp_dst)

                    # Ensure it's visible to Spark
                    dbutils.fs.cp(f"file:{tmp_dst}", f"{self.target_path}/{file}")
                    copied_files.add(file)
                    self.logger.info(f"Copied JSON file: {file}")

        missing = self.expected_files - copied_files
        if missing:
            self.logger.warning(f"Expected files not found: {missing}")
        else:
            self.logger.info("All expected files were successfully organized.")
