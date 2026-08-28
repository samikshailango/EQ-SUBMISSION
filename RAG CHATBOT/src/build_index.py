from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

from vectorstore import build_vectorstore

if __name__ == "__main__":
    logger.info("Building vector index...")
    vs = build_vectorstore()
    logger.info("Vector store built and persisted successfully.")
    print("Vector store built and persisted successfully.")
