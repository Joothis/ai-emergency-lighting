import os
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List
import pymongo
import logging

logger = logging.getLogger(__name__)

class Storage(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def get_pdf_processing_status(self, pdf_name: str) -> Dict[str, Any] | None:
        pass

    @abstractmethod
    def update_pdf_processing_status(self, pdf_name: str, status: str, task_id: str | None = None, result: Dict[str, Any] | None = None):
        pass

    @abstractmethod
    def store_extracted_content(self, pdf_name: str, content_data: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def get_extracted_content(self, pdf_name: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def list_processed_pdfs(self) -> List[Dict[str, Any]]:
        pass

class MongoStorage(Storage):
    def __init__(self, mongo_url: str, db_name: str):
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.client = None
        self.db = None
        self.pdf_processing_collection = None
        self.extracted_content_collection = None

    def connect(self):
        try:
            self.client = pymongo.MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.db_name]
            self.pdf_processing_collection = self.db["pdf_processing"]
            self.extracted_content_collection = self.db["extracted_content"]
            self.client.admin.command('ping')
            logger.info("Connected to MongoDB successfully.")
            return True
        except pymongo.errors.ServerSelectionTimeoutError as err:
            logger.error(f"MongoDB connection failed: {err}")
            return False

    def get_pdf_processing_status(self, pdf_name: str) -> Dict[str, Any] | None:
        return self.pdf_processing_collection.find_one({"pdf_name": pdf_name})

    def update_pdf_processing_status(self, pdf_name: str, status: str, task_id: str | None = None, result: Dict[str, Any] | None = None):
        update_data = {"status": status, "updated_at": datetime.now().isoformat()}
        if task_id:
            update_data["task_id"] = task_id
        if result:
            update_data["result"] = json.dumps(result) # Store as string in MongoDB
        self.pdf_processing_collection.update_one(
            {"pdf_name": pdf_name},
            {"$set": update_data},
            upsert=True
        )

    def store_extracted_content(self, pdf_name: str, content_data: List[Dict[str, Any]]):
        self.extracted_content_collection.delete_many({"pdf_name": pdf_name})
        for entry in content_data:
            self.extracted_content_collection.insert_one({
                "pdf_name": pdf_name,
                "content_type": entry.get("type"),
                "symbol": entry.get("symbol", ""),
                "description": entry.get("description", ""),
                "content": entry.get("text", entry.get("description", "")),
                "source_sheet": entry.get("source_sheet", ""),
                "created_at": datetime.now().isoformat()
            })

    def get_extracted_content(self, pdf_name: str) -> List[Dict[str, Any]]:
        rows = self.extracted_content_collection.find({"pdf_name": pdf_name}).sort("created_at", pymongo.ASCENDING)
        content = []
        for row in rows:
            content.append({
                "type": row.get("content_type"),
                "symbol": row.get("symbol"),
                "description": row.get("description"), 
                "content": row.get("content"),
                "source_sheet": row.get("source_sheet"),
                "created_at": row.get("created_at")
            })
        return content

    def list_processed_pdfs(self) -> List[Dict[str, Any]]:
        rows = self.pdf_processing_collection.find({}).sort("updated_at", pymongo.DESCENDING)
        pdfs = []
        for row in rows:
            pdfs.append({
                "pdf_name": row.get("pdf_name"),
                "status": row.get("status"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at")
            })
        return pdfs

class FileStorage(Storage):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.processing_dir = os.path.join(base_dir, "processing_status")
        self.content_dir = os.path.join(base_dir, "extracted_content")
        os.makedirs(self.processing_dir, exist_ok=True)
        os.makedirs(self.content_dir, exist_ok=True)

    def connect(self):
        # File storage doesn't need a "connection" in the same way a DB does
        return True

    def _get_processing_file_path(self, pdf_name: str) -> str:
        return os.path.join(self.processing_dir, f"{pdf_name}.json")

    def _get_content_file_path(self, pdf_name: str) -> str:
        return os.path.join(self.content_dir, f"{pdf_name}.json")

    def get_pdf_processing_status(self, pdf_name: str) -> Dict[str, Any] | None:
        file_path = self._get_processing_file_path(pdf_name)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return None

    def update_pdf_processing_status(self, pdf_name: str, status: str, task_id: str | None = None, result: Dict[str, Any] | None = None):
        file_path = self._get_processing_file_path(pdf_name)
        data = self.get_pdf_processing_status(pdf_name) or {"pdf_name": pdf_name}
        data["status"] = status
        data["updated_at"] = datetime.now().isoformat()
        if task_id:
            data["task_id"] = task_id
        if result:
            data["result"] = result # Store as dict in file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def store_extracted_content(self, pdf_name: str, content_data: List[Dict[str, Any]]):
        file_path = self._get_content_file_path(pdf_name)
        with open(file_path, "w") as f:
            json.dump(content_data, f, indent=4)

    def get_extracted_content(self, pdf_name: str) -> List[Dict[str, Any]]:
        file_path = self._get_content_file_path(pdf_name)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return []

    def list_processed_pdfs(self) -> List[Dict[str, Any]]:
        pdfs = []
        for filename in os.listdir(self.processing_dir):
            if filename.endswith(".json"):
                pdf_name = filename[:-5] # Remove .json
                status_data = self.get_pdf_processing_status(pdf_name)
                if status_data:
                    pdfs.append({
                        "pdf_name": status_data.get("pdf_name"),
                        "status": status_data.get("status"),
                        "created_at": status_data.get("created_at", ""),
                        "updated_at": status_data.get("updated_at", "")
                    })
        # Sort by updated_at, newest first
        pdfs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return pdfs
