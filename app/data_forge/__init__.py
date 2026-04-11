"""
Data Forge — multi-format ingestion → normalised records → training datasets.

Entry points:
    from app.data_forge.ingest import DataForge
    forge = DataForge()
    records = forge.ingest("./data/uploads/my_file.pdf")
    dataset = forge.build_dataset(records, task="sft", base_model="Qwen/Qwen2.5-7B-Instruct")
"""
from .ingest import DataForge, IngestedRecord
from .dataset_builder import DatasetBuilder

__all__ = ["DataForge", "IngestedRecord", "DatasetBuilder"]
