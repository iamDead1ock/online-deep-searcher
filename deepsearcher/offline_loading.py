import os
import requests
from typing import List, Union

from tqdm import tqdm

# from deepsearcher.configuration import embedding_model, vector_db, file_loader
from deepsearcher import configuration
from deepsearcher.loader.splitter import split_docs_to_chunks


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


async def load_from_website(
    urls: Union[str, List[str]],
    collection_name: str = None,
    collection_description: str = None,
    force_new_collection: bool = False,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
    batch_size: int = 256,
    **crawl_kwargs,
):
    """
    Load knowledge from websites into the vector database.

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

    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection,
    )

    all_docs = await web_crawler._async_crawl_many(urls, **crawl_kwargs)

    chunks = split_docs_to_chunks(
        all_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
    vector_db.insert_data(collection=collection_name, chunks=chunks)


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