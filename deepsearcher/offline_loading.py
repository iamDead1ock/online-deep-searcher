import os
import json
import requests
from typing import List, Union, Callable, Optional

from tqdm import tqdm

# from deepsearcher.configuration import embedding_model, vector_db, file_loader
from deepsearcher import configuration
from deepsearcher.loader.splitter import split_docs_to_chunks
from deepsearcher.utils import log


QUERY_GENERATION_PROMPT = """
Based on the following research prompt, generate {max_queries} specific search queries that would help gather comprehensive information to answer this prompt. Each query should focus on different aspects of the topic.

Research Prompt: {prompt}

Please provide {max_queries} search queries, one per line, without numbering or bullet points:
"""

def load_from_local_files(
    paths_or_directory: Union[str, List[str]],
    collection_name: str = None,
    collection_description: str = None,
    force_new_collection: bool = False,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
    batch_size: int = 256,
):
    """
    Load knowledge from local files or directories into the vector database.

    This function processes files from the specified paths or directories,
    splits them into chunks, embeds the chunks, and stores them in the vector database.

    Args:
        paths_or_directory: A single path or a list of paths to files or directories to load.
        collection_name: Name of the collection to store the data in. If None, uses the default collection.
        collection_description: Description of the collection. If None, no description is set.
        force_new_collection: If True, drops the existing collection and creates a new one.
        chunk_size: Size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        batch_size: Number of chunks to process at once during embedding.

    Raises:
        FileNotFoundError: If any of the specified paths do not exist.
    """
    vector_db = configuration.vector_db
    if collection_name is None:
        collection_name = vector_db.default_collection
    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    embedding_model = configuration.embedding_model
    file_loader = configuration.file_loader
    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection,
    )
    if isinstance(paths_or_directory, str):
        paths_or_directory = [paths_or_directory]
    all_docs = []
    for path in tqdm(paths_or_directory, desc="Loading files"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: File or directory '{path}' does not exist.")
        if os.path.isdir(path):
            docs = file_loader.load_directory(path)
        else:
            docs = file_loader.load_file(path)
        all_docs.extend(docs)
    # print("Splitting docs to chunks...")
    chunks = split_docs_to_chunks(
        all_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
    vector_db.insert_data(collection=collection_name, chunks=chunks)

def load_specific_file(
        file_path: str,
        collection_name: str = None,
        collection_description: str = None,
        force_new_collection: bool = False,
        chunk_size: int = 1500,
        chunk_overlap: int = 100,
        batch_size: int = 256,
):
    """
    Load knowledge from a specific file into the vector database.

    This function processes a single file, splits it into chunks, embeds the chunks,
    and stores them in the vector database.

    Args:
        file_path: Path to the specific file to load.
        collection_name: Name of the collection to store the data in. If None, uses the default collection.
        collection_description: Description of the collection. If None, no description is set.
        force_new_collection: If True, drops the existing collection and creates a new one.
        chunk_size: Size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        batch_size: Number of chunks to process at once during embedding.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IsADirectoryError: If the specified path is a directory instead of a file.
    """
    vector_db = configuration.vector_db
    if collection_name is None:
        collection_name = vector_db.default_collection
    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    embedding_model = configuration.embedding_model
    file_loader = configuration.file_loader

    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection,
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' does not exist.")

    if os.path.isdir(file_path):
        raise IsADirectoryError(f"Error: '{file_path}' is a directory, not a file.")

    print(f"Loading file: {file_path}")

    docs = file_loader.load_file(file_path)

    print("Splitting document to chunks...")
    chunks = split_docs_to_chunks(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
    vector_db.insert_data(collection=collection_name, chunks=chunks)


def load_multiple_specific_files(
        file_paths: List[str],
        collection_name: str = None,
        collection_description: str = None,
        force_new_collection: bool = False,
        chunk_size: int = 1500,
        chunk_overlap: int = 100,
        batch_size: int = 256,
):
    """
    Load knowledge from multiple specific files into the vector database.

    This function processes multiple specified files, splits them into chunks,
    embeds the chunks, and stores them in the vector database.

    Args:
        file_paths: List of paths to specific files to load.
        collection_name: Name of the collection to store the data in. If None, uses the default collection.
        collection_description: Description of the collection. If None, no description is set.
        force_new_collection: If True, drops the existing collection and creates a new one.
        chunk_size: Size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        batch_size: Number of chunks to process at once during embedding.

    Raises:
        FileNotFoundError: If any of the specified files do not exist.
        IsADirectoryError: If any of the specified paths is a directory instead of a file.
    """
    vector_db = configuration.vector_db
    if collection_name is None:
        collection_name = vector_db.default_collection
    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    embedding_model = configuration.embedding_model
    file_loader = configuration.file_loader

    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection,
    )

    all_docs = []

    for file_path in tqdm(file_paths, desc="Loading files"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: File '{file_path}' does not exist.")

        if os.path.isdir(file_path):
            raise IsADirectoryError(f"Error: '{file_path}' is a directory, not a file.")

        docs = file_loader.load_file(file_path)
        all_docs.extend(docs)

    print("Splitting documents to chunks...")
    chunks = split_docs_to_chunks(
        all_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
    vector_db.insert_data(collection=collection_name, chunks=chunks)

def create_safe_metadata(original_metadata: dict, max_length: int = 60000) -> dict:
    """
    Create secure metadata to ensure you don’t exceed Milvus limits

    Args:
        original_metadata: raw metadata
        max_length: Maximum allowed length (number of characters)

    Returns:
        Processed security metadata
    """
    if not original_metadata:
        return {"source_type": "web", "source": "unknown"}

    # Basic security metadata
    safe_metadata = {
        "source_type": original_metadata.get("source_type", "web")
    }

    # Field priority and length limits
    field_limits = {
        "source": 200,
        "title": 300,
        "url": 200,
        "domain": 100,
        "content_type": 50,
        "language": 20,
        "author": 100,
        "description": 500,
        "keywords": 200,
        "publish_date": 50,
    }

    try:
        current_length = len(json.dumps(safe_metadata, ensure_ascii=False))

        for field, limit in field_limits.items():
            if field in original_metadata and current_length < max_length - 1000:
                value = original_metadata[field]

                if isinstance(value, str):
                    # Truncate field values that are too long
                    if len(value) > limit:
                        value = value[:limit - 3] + "..."

                    # Test the length after adding the field
                    test_metadata = safe_metadata.copy()
                    test_metadata[field] = value
                    test_length = len(json.dumps(test_metadata, ensure_ascii=False))

                    # If adding will not exceed the limit, add
                    if test_length < max_length:
                        safe_metadata[field] = value
                        current_length = test_length
                    else:
                        log.color_print(
                            f"<metadata_safe> Skipping field '{field}' to avoid metadata overflow </metadata_safe>\n")
                        break
                elif isinstance(value, (int, float, bool)):
                    # For simple types, simply add
                    test_metadata = safe_metadata.copy()
                    test_metadata[field] = value
                    test_length = len(json.dumps(test_metadata, ensure_ascii=False))

                    if test_length < max_length:
                        safe_metadata[field] = value
                        current_length = test_length

        # final verification
        final_metadata_str = json.dumps(safe_metadata, ensure_ascii=False)
        if len(final_metadata_str) > max_length:
            # If it is still too long, keep only the most basic information
            safe_metadata = {
                "source_type": "web",
                "source": original_metadata.get("source", "")[:150] if original_metadata.get("source") else "unknown",
            }
            log.color_print(f"<metadata_safe> Applied emergency truncation </metadata_safe>\n")

        final_length = len(json.dumps(safe_metadata, ensure_ascii=False))
        if final_length != current_length:
            log.color_print(
                f"<metadata_safe> Metadata processed: {len(json.dumps(original_metadata, ensure_ascii=False))} -> {final_length} chars </metadata_safe>\n")

        return safe_metadata

    except Exception as e:
        log.color_print(f"<metadata_safe> Error processing metadata: {e}, using minimal metadata </metadata_safe>\n")
        # If processing fails, use minimal metadata
        return {
            "source_type": "web",
            "source": original_metadata.get("source", "unknown")[:150] if original_metadata.get("source") else "unknown"
        }


def process_chunks_metadata(chunks: List, metadata_processor: Optional[Callable] = None):
    """
    Process the metadata of chunks to ensure compliance with vector database requirements

    Args:
        chunks: chunks list
        metadata_processor: Metadata processing function. If None, the default create_safe_metadata is used.

    Returns:
        List of processed chunks
    """
    if metadata_processor is None:
        metadata_processor = create_safe_metadata

    processed_chunks = []

    for chunk in chunks:
        try:
            # Processing chunk metadata
            if hasattr(chunk, 'metadata') and chunk.metadata:
                original_metadata = chunk.metadata
                chunk.metadata = metadata_processor(original_metadata)

                metadata_str = json.dumps(chunk.metadata, ensure_ascii=False)
                if len(metadata_str) > 65000:
                    log.color_print(
                        f"<metadata_warning> Chunk metadata still too large after processing: {len(metadata_str)} chars </metadata_warning>\n")
                    chunk.metadata = {
                        "source_type": "web",
                        "source": chunk.metadata.get("source", "unknown")[:100]
                    }
            else:
                # If there is no metadata, add the default
                chunk.metadata = {
                    "source_type": "web",
                    "source": "unknown"
                }

            processed_chunks.append(chunk)

        except Exception as e:
            log.color_print(f"<metadata_error> Error processing chunk metadata: {e} </metadata_error>\n")
            # Set minimum metadata
            chunk.metadata = {
                "source_type": "web",
                "source": "error_processing"
            }
            processed_chunks.append(chunk)

    return processed_chunks

async def load_from_website(
    urls: Union[str, List[str]],
    collection_name: str = None,
    collection_description: str = None,
    force_new_collection: bool = False,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
    batch_size: int = 256,
    metadata_processor: Optional[Callable] = None,
    **crawl_kwargs,
):
    """
    Load knowledge from websites into the vector database.

    Added metadata size control to prevent exceeding Milvus limit

    This function crawls the specified URLs, processes the content,
    splits it into chunks, embeds the chunks, and stores them in the vector database.

    Args:
        urls: A single URL or a list of URLs to crawl.
        collection_name: Name of the collection to store the data in. If None, uses the default collection.
        collection_description: Description of the collection. If None, no description is set.
        force_new_collection: If True, drops the existing collection and creates a new one.
        chunk_size: Size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        batch_size: Number of chunks to process at once during embedding.
        **crawl_kwargs: Additional keyword arguments to pass to the web crawler.
    """
    if isinstance(urls, str):
        urls = [urls]

    vector_db = configuration.vector_db
    embedding_model = configuration.embedding_model
    web_crawler = configuration.web_crawler

    log.color_print(
        f"<load_website> Starting to load {len(urls)} URLs into collection: {collection_name} </load_website>\n")

    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection,
    )

    try:
        log.color_print("<load_website> Crawling web content... </load_website>\n")
        all_docs = await web_crawler._async_crawl_many(urls, **crawl_kwargs)

        log.color_print(f"<load_website> Successfully crawled {len(all_docs)} documents </load_website>\n")
        log.color_print("<load_website> Splitting documents into chunks... </load_website>\n")
        chunks = split_docs_to_chunks(
            all_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        log.color_print(f"<load_website> Created {len(chunks)} chunks </load_website>\n")

        log.color_print("<load_website> Processing chunk metadata for Milvus compatibility... </load_website>\n")
        chunks = process_chunks_metadata(chunks, metadata_processor)

        log.color_print("<load_website> Generating embeddings... </load_website>\n")
        chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)

        log.color_print("<load_website> Final metadata validation before insertion... </load_website>\n")
        problematic_chunks = 0
        for i, chunk in enumerate(chunks):
            try:
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    metadata_str = json.dumps(chunk.metadata, ensure_ascii=False)
                    if len(metadata_str) > 65000:
                        problematic_chunks += 1
                        log.color_print(
                            f"<load_website> WARNING: Chunk {i} metadata still too large: {len(metadata_str)} chars </load_website>\n")
                        # Apply emergency fixes
                        chunk.metadata = {
                            "source_type": "web",
                            "source": "metadata_too_large"
                        }
            except Exception as e:
                problematic_chunks += 1
                log.color_print(f"<load_website> ERROR: Problem with chunk {i} metadata: {e} </load_website>\n")
                chunk.metadata = {
                    "source_type": "web",
                    "source": "metadata_error"
                }

        if problematic_chunks > 0:
            log.color_print(f"<load_website> Fixed {problematic_chunks} problematic chunks </load_website>\n")

        # Insert data into vector database
        log.color_print(f"<load_website> Inserting {len(chunks)} chunks into vector database... </load_website>\n")
        vector_db.insert_data(collection=collection_name, chunks=chunks)

        log.color_print(
            f"<load_website> Successfully loaded website content into collection: {collection_name} </load_website>\n")

    except Exception as e:
        log.color_print(f"<load_website> Error loading website content: {e} </load_website>\n")
        import traceback
        log.color_print(f"<load_website> Traceback: {traceback.format_exc()} </load_website>\n")
        raise


async def load_from_dynamic_search(
        query: str,
        collection_name: str = None,
        collection_description: str = None,
        force_new_collection: bool = False,
        chunk_size: int = 1500,
        chunk_overlap: int = 100,
        batch_size: int = 256,
        searxng_url: str = "http://localhost:8080",
        search_engines: List[str] = None,
        num_results: int = 10,
        **crawl_kwargs,
):
    """
        Load knowledge from dynamic search results into the vector database.

        This function uses SearxNG to search for relevant URLs based on the query,
        then crawls those URLs and processes the content.

        Args:
            query: Search query to find relevant content.
            collection_name: Name of the collection to store the data in.
            collection_description: Description of the collection.
            force_new_collection: If True, drops the existing collection and creates a new one.
            chunk_size: Size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            batch_size: Number of chunks to process at once during embedding.
            searxng_url: URL of the SearxNG instance.
            search_engines: List of search engines to use (e.g., ['google', 'bing', 'duckduckgo']).
            num_results: Maximum number of search results to process.
            **crawl_kwargs: Additional keyword arguments to pass to the web crawler.
        """
    if isinstance(query, str):
        queries = [query]
    else:
        queries = query

        vector_db = configuration.vector_db
        embedding_model = configuration.embedding_model
        web_crawler = configuration.web_crawler

        if collection_name is None:
            collection_name = vector_db.default_collection
        collection_name = collection_name.replace(" ", "_").replace("-", "_")

        vector_db.init_collection(
            dim=embedding_model.dimension,
            collection=collection_name,
            description=collection_description,
            force_new_collection=force_new_collection,
        )

        all_urls = []
        for q in queries:
            urls = search_with_searxng(
                query=q,
                searxng_url=searxng_url,
                search_engines=search_engines,
                num_results=num_results
            )
            all_urls.extend(urls)

        all_urls = list(set(all_urls))

        print(f"Found {len(all_urls)} unique URLs to crawl")

        all_docs = await web_crawler._async_crawl_many(all_urls, **crawl_kwargs)

        chunks = split_docs_to_chunks(
            all_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
        vector_db.insert_data(collection=collection_name, chunks=chunks)


def search_with_searxng(
        query: str,
        searxng_url: str = "http://localhost:8080",
        search_engines: List[str] = None,
        num_results: int = 10,
        timeout: int = 30
) -> List[str]:
    """
    Search for URLs using SearxNG API.

    Args:
        query: Search query string.
        searxng_url: URL of the SearxNG instance.
        search_engines: List of search engines to use.
        num_results: Maximum number of results to return.
        timeout: Request timeout in seconds.

    Returns:
        List of URLs from search results.
    """
    search_url = f"{searxng_url}/search"

    params = {
        'q': query,
        'format': 'json',
        'pageno': 1
    }

    if search_engines:
        params['engines'] = ','.join(search_engines)

    try:
        response = requests.get(search_url, params=params, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        urls = []

        for result in data.get('results', [])[:num_results]:
            if 'url' in result:
                urls.append(result['url'])

        return urls

    except requests.RequestException as e:
        print(f"Error searching with SearxNG: {e}")
        return []

def generate_search_queries_from_prompt(prompt: str, max_queries: int = 3) -> List[str]:
    """
    Generate search queries from the main prompt using LLM.

    Args:
        prompt: The main research prompt.
        max_queries: Maximum number of search queries to generate.

    Returns:
        List of search query strings.
    """
    llm = configuration.llm

    query_generation_prompt = QUERY_GENERATION_PROMPT.format(prompt=prompt, max_queries=max_queries)

    try:
        chat_response = llm.chat(
            messages=[
                {"role": "user", "content": query_generation_prompt}
            ]
        )
        response_content = chat_response.content
        queries = [q.strip() for q in response_content.split('\n') if q.strip()]
        return queries[:max_queries]
    except Exception as e:
        print(f"Error generating search queries: {e}")
        return [prompt[:100]]