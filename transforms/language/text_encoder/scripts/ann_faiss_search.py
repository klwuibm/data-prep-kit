import numpy as np
import faiss
import pandas as pd
import lancedb, lance
import uuid, time, datetime
import ray
import os
import argparse
import pyarrow as pa
from pyarrow import fs
from io import BytesIO
from typing import List

STORAGE_DTYPE = 'float16'
# The data type required by Faiss for a flat index
FAISS_DTYPE = 'float32'
# Simulating a large dataset. In a real-world scenario, this would be your 62.5 million records.
EMBEDDING_DIM = 384      # The dimension of your embeddings, as specified
np.random.seed(42)
num_records_to_write = 5_000


# --- Quantization helpers ---
def quantize_to_int8(np_embeddings: np.ndarray):
    """
    Quantize float16/float32 embeddings to int8.
    Returns int8 array (flattened) and per-vector scale factors.
    """
    # Find absolute max per vector
    scales = np.max(np.abs(np_embeddings), axis=1, keepdims=True)
    # Avoid divide-by-zero
    scales[scales == 0] = 1.0
    q_emb = (np_embeddings / scales * 127).astype(np.int8)
    return q_emb, scales.squeeze()

def dequantize_from_int8(arr_int8, scale):
    return arr_int8.astype(np.float32) / 127 * scale

def load_npy_s3(s3_credentials: dict, cos_centroids_path):
    """Loads centroids from S3/COS stored in .npy format using PyArrow's S3FileSystem."""
    s3 = fs.S3FileSystem(
        access_key=s3_credentials['access_key'],
        secret_key=s3_credentials['secret_key'],
        request_timeout=60,
        connect_timeout=60,
        retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=30),
        endpoint_override=s3_credentials['endpoint_override'],
    )
    try:

        # Open the file for reading in the S3 bucket
        with s3.open_input_stream(cos_centroids_path) as source:
            # Read the entire stream into a BytesIO object
            buffer = BytesIO(source.readall())
            buffer.seek(0)

            # Load the NumPy array from the buffer
            centroids = np.load(buffer)
            print(f"Centroids loaded from {cos_centroids_path} in .npy format")
            return centroids
    except Exception as e:
        print(f"Error loading centroids from S3/COS using PyArrow: {e}")
        return None
    
def safe_extract_embeddings(emb_array: pa.Array, expected_dim: int):
    """
    Safely extract embeddings from PyArrow array into a 2D NumPy array.
    Replaces malformed/missing vectors with zeros.
    """
    emb_list = emb_array.to_pylist()
    cleaned = []
    bad_count = 0

    for emb in emb_list:
        if isinstance(emb, (list, tuple)) and len(emb) == expected_dim:
            cleaned.append(emb)
        else:
            cleaned.append([0.0] * expected_dim)
            bad_count += 1

    if bad_count > 0:
        print(f"⚠ Found {bad_count} malformed/missing embeddings in batch — replaced with zeros.")

    return np.array(cleaned, dtype=np.float16)

def load_partition_from_lancedb(db_uri: str, table_name: str, begin_index: int, total_records: int):
    """
    Loads a specific partition of records from a LanceDB table.

    Args:
        db_uri (str): The path to the LanceDB database (e.g., "./lancedb").
        table_name (str): The name of the table to query.
        begin_index (int): The starting index for the partition (inclusive).
        total_records (int): The total number of records to retrieve from the start index.

    Returns:
        pandas.DataFrame: A DataFrame containing the embeddings and other record data
                          for the specified partition.
    """
    try:
        # Connect to the LanceDB database
        db = lancedb.connect(db_uri)

        # Open the table
        table = db.open_table(table_name)
    
        print(f"Connected to table '{table_name}' at '{db_uri}'.")
        print(f"Attempting to load {total_records} records starting from index {begin_index}...")

        lance_dataset_uri = f"{db_uri}{table_name}.lance"
        ds = lance.dataset(lance_dataset_uri)
        scanner = ds.scanner(batch_size=8192, columns=['document_id', 'contents', 'embeddings'], limit=total_records, offset=begin_index)
        # scanner = ds.scanner(batch_size=32768, limit=total_records, offset=begin_index)
        
    
    
        # 3. Iterate over the scanner to get data in small batches.
        # This is the key change that prevents OOM errors.
        print("\nStarting scanner batch processing...")
        list_of_dfs: List[pd.DataFrame] = []
        total_rows = 0
        for i, batch in enumerate(scanner.to_batches()):

            num_rows = batch.num_rows
            total_rows += num_rows
            # print(f"Batch {i}: {num_rows} rows")

            embeddings_2d = safe_extract_embeddings(batch["embeddings"], EMBEDDING_DIM)

            # Quantize
            q_emb, scales = quantize_to_int8(embeddings_2d)

            # Guarantee q_emb is int8 (N, 384)
            q_emb = np.asarray(q_emb, dtype=np.int8).reshape(-1, EMBEDDING_DIM)

            # Flatten to 1D for Arrow
            flat_q = pa.array(q_emb.ravel(), type=pa.int8())

            # Build FixedSizeListArray (N lists, each length 384)
            q_emb_arr = pa.FixedSizeListArray.from_arrays(flat_q, EMBEDDING_DIM)

            # Build a batch DataFrame
            batch_df = pa.Table.from_arrays(
                [
                    batch["document_id"], 
                    batch["contents"], 
                    q_emb_arr,              # efficient int8 embedding column
                    pa.array(scales, type=pa.float32())
                ],
                names=["document_id", "contents", "embeddings_int8", "scale"]
            ).to_pandas()  # optional: convert to Pandas later
            # For demonstration, we'll append it to a list to show it works.
            list_of_dfs.append(batch_df)

        if list_of_dfs:
            df_partition = pd.concat(list_of_dfs, ignore_index=True)

        if not df_partition.empty:
            print(f"\nSuccessfully created a DataFrame with {len(df_partition)} records.")
            return df_partition
        else:
            print(f"\nNo records were found for the specified offset: {begin_index} and limit: {total_records}.")
            return pd.DataFrame()
         
    except Exception as e:
        print(f"An error occurred in loading partition from lancedb: {e}")
        for i, batch in enumerate(scanner.to_batches()):
            print(f"Batch {i} -> rows: {batch.num_rows}")
        print(f"{embeddings_2d=}")
        return pd.DataFrame()

@ray.remote(memory=90*1024*1024*1024, num_cpus=14.5)
def search_queries_in_partition(
        s3_credentials: dict, 
        queries: np.ndarray, 
        db_uri: str, 
        table_name: str, 
        begin_index:int, 
        total_records: int,
        top_K:int,
        sim_threshold:float,
        output_folder: str
    ):

    # load partition from lancedb
    start_time = time.time()
    task_id = ray.get_runtime_context().get_task_id()
    print(f"{task_id} is searching from {begin_index} for {total_records} records")
    df_partition = load_partition_from_lancedb(db_uri, table_name, begin_index, total_records)
    assert not df_partition.empty, f"df_partition should not be empty: {begin_index=}, {total_records=}"
    
    print(f"{task_id} completed loading df_partition from lancedb")
    embeddings_int8 = np.vstack(df_partition['embeddings_int8'].to_numpy())  # shape: (N, EMBEDDING_DIM)
    scales = df_partition['scale'].to_numpy().astype(np.float32) 

    # This is your full dataset of embeddings.
    full_embeddings = embeddings_int8.astype(np.float32) * scales[:, None]
    query_embeddings = queries.astype(np.float32)
    num_queries = query_embeddings.shape[0]

    print(f"{task_id} {total_records=} embeddings of dimension {EMBEDDING_DIM}.")
    print("-" * 60)

    # --- 2. Normalize All Vectors ---
    # This is the CRITICAL step for cosine similarity.
    # We normalize every vector in the dataset to have a unit L2 norm (magnitude of 1).
    # After this, a dot product is equivalent to cosine similarity.
    print("Normalizing all embeddings to unit vectors...")
    faiss.normalize_L2(full_embeddings)
    faiss.normalize_L2(query_embeddings)
    print(f"{task_id} Normalization complete.")
    print("-" * 60)

    # --- 3. Build the IVF Index (Training and Adding) ---
    # The core of the clustering approach.

    # Number of clusters (nlist), as specified.
    N_CLUSTERS = 1024

    # The quantizer is a sub-index used to find the nearest cluster centroid.
    quantizer = faiss.IndexFlatL2(full_embeddings.shape[1])
    # Create the IVF index using METRIC_INNER_PRODUCT for cosine similarity.
    index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, N_CLUSTERS, faiss.METRIC_INNER_PRODUCT)

    # --- TRAINING THE INDEX ---
    # We train the index on a subset of the normalized data to find the cluster centroids.
    TRAINING_SIZE = min(524_288, full_embeddings.shape[0])
    training_vectors = full_embeddings[:TRAINING_SIZE]
    print(f"Training the index with {TRAINING_SIZE} vectors to find {N_CLUSTERS} clusters...")
    index.train(training_vectors)
    print(f"{task_id} Training complete.")

    # --- ADDING ALL VECTORS TO THE INDEX ---
    # This step assigns every single normalized vector to its closest cluster.
    # No data is discarded.
    print(f"Adding all {total_records} vectors to the index...")
    index.add(full_embeddings)
    print(f"{task_id} All vectors added. Index contains {index.ntotal} vectors.")
    print("-" * 60)

    # --- 4. Perform the Search ---
    # Now, let's simulate a query and perform a search.

    # Number of candidates to retrieve
    K_CANDIDATES = top_K

    # Number of clusters to search (nprobe). This is a crucial parameter for performance vs. accuracy.
    # make it equals to K_CANDIDATES
    index.nprobe = top_K

    # *** IMPORTANT: Normalize the query vector before searching. ***
    faiss.normalize_L2(query_embeddings)

    print(f"Performing search for {num_queries} with K={K_CANDIDATES} and nprobe={index.nprobe}...")
    distances, candidate_indices = index.search(query_embeddings, K_CANDIDATES)

    print(f"{task_id} Search complete.")
    print("-" * 60)

    # --- 5. Process and Verify the Results ---
    # The distances here are the inner product values. Since vectors are
    # normalized, a higher inner product means a higher cosine similarity.
    print(f"Top {K_CANDIDATES} distances (inner product):\n{distances.flatten()}")
    print(f"Top {K_CANDIDATES} indices from the full dataset:\n{candidate_indices.flatten()}")

    # Retrieve the full records from the DataFrame
    retrieved_docs_df = df_partition.iloc[candidate_indices.flatten()]
    retrieved_docs_df = retrieved_docs_df.reset_index(drop=True)
    print("\nComplete retrieving documents from the full DataFrame:")
    # print the retrieved documents to parquet files with query_id and cosine distance added but drop embeddings
    retrieved_docs_df['cosine_similarity'] = distances.flatten()

    query_ids = np.repeat(np.arange(num_queries), K_CANDIDATES)
    retrieved_docs_df['query_id'] = query_ids

    topK_df = retrieved_docs_df.drop(columns=['embeddings_int8'])
    final_df = topK_df[topK_df['cosine_similarity'] >= sim_threshold].reset_index(drop=True)

    # set up s3_credentials
    s3_fs = fs.S3FileSystem(
        access_key=s3_credentials['access_key'],
        secret_key=s3_credentials['secret_key'],
        request_timeout=60,
        connect_timeout=60,
        retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=30),
        endpoint_override=s3_credentials['endpoint_override'],
    )
    for i in range(0, len(final_df), num_records_to_write):
        # Slice the DataFrame to get the current chunk.
        # The .iloc[] method is used for integer-based slicing.
        chunk_df = final_df.iloc[i : i + num_records_to_write]

        output_path = f"{output_folder}{str(uuid.uuid4())}.parquet"
        
        chunk_df.to_parquet(
            output_path, 
            engine='pyarrow', 
            index=False,
            filesystem=s3_fs
        )
        
        print(f"Wrote chunk {i // num_records_to_write} with {len(chunk_df)} records to '{output_path}'.")
    end_time = time.time()
    return (task_id, end_time - start_time)


def main(args):

    s3_credentials ={}
    s3_access_key = os.getenv("S3_ACCESS_KEY", None)
    assert s3_access_key is not None, f"Error: s3_access_key is None"
    s3_credentials["access_key"] = s3_access_key
    s3_secret_key = os.getenv("S3_SECRET_KEY", None)
    assert s3_secret_key is not None, f"s3_secret_key is None"
    s3_credentials["secret_key"] = s3_secret_key
    s3_endpoint = os.getenv("S3_ENDPOINT", None)
    assert s3_endpoint is not None, f"s3_endpoint is None"
    s3_credentials["endpoint_override"] = s3_endpoint
    
    db_uri = args.lancedb_uri
    assert db_uri.startswith("s3://"), f"lancedb_uri must start with s3:// for cos path"
    table_name = args.table_name
    
    output_folder = args.output_folder
    assert output_folder, f"missing output_folder"
    query_embeddings_path = args.query_embeddings_path
    assert query_embeddings_path, f"missing query_embeddings_path"

    top_K = int(args.top_K)
    sim_threshold = float(args.sim_threshold)
    print(f"{top_K=}")
    print(f"{sim_threshold=}")
    # enure outputt_folder ends with /
    output_folder = output_folder if output_folder.endswith("/") else output_folder + "/"
    num_parallel_searches = int(args.num_parallel_searches)
    print(f"{num_parallel_searches=}")
    partition_size = int(args.lance_partition_size)

    # read the query embeddings as numpy nd.array.
    queries = load_npy_s3(s3_credentials, query_embeddings_path)

    db = lancedb.connect(db_uri)
    lance_table = db.open_table(table_name)
    total_rows = lance_table.count_rows()
    print(f"{total_rows=}")

    ranges = []
    # Loop from 0 to K in steps of partition_size
    for i in range(0, total_rows, partition_size):
        begin = i
        # The end of the range is the smaller of `i + partition_size` and `K`
        end = min(i + partition_size, total_rows)
        ranges.append((begin, end))

    print(f"{len(ranges)=}")

    ray_futures = []
    for start, end in ranges:
        if len(ray_futures) < num_parallel_searches:
            current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
            print(f"{current_time=} initial creating remote task {len(ray_futures)}")
            ray_future = search_queries_in_partition.remote(
                s3_credentials=s3_credentials,
                queries=queries,
                db_uri=db_uri,
                table_name=table_name, 
                begin_index=start,
                total_records=end-start,
                top_K=top_K,
                sim_threshold=sim_threshold,
                output_folder=output_folder)
            ray_futures.append(ray_future)
            time.sleep(1)
        else:
            ready, ray_futures = ray.wait(ray_futures, num_returns=1)
            if ready:
                task_id, elapse_time = ray.get(ready[0])
                print(f"{task_id} completed its remote task in {elapse_time} seconds")
                current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
                print(f"{current_time=}, inserting another remote task, {len(ray_futures)=}")
                # centroid_document_counts = ray.get(ready[0])
                # centroid_document_counts_sum = [sum(x) for x in zip(centroid_document_counts_sum, centroid_document_counts)]
                ray_future = search_queries_in_partition.remote(
                    s3_credentials=s3_credentials,
                    queries=queries,
                    db_uri=db_uri,
                    table_name=table_name, 
                    begin_index=start,
                    total_records=end-start,
                    top_K=top_K,
                    sim_threshold=sim_threshold,
                    output_folder=output_folder)
                ray_futures.append(ray_future)


    while len(ray_futures) > 0:
        ready, ray_futures = ray.wait(ray_futures, num_returns=1)

    print(f" All {num_parallel_searches} parallel tasks completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for Top-K most similar docs from lancedb table")
    parser.add_argument(
        f"--query_embeddings_path",
        type=str,
        required=False,
        default="",
        help="cos path for the sample query embeddings"
    ) 
    parser.add_argument(
        f"--output_folder",
        type=str,
        required=False,
        default="",
        help="Output path for the found results of all table chunks as parquet files.",
    )
    parser.add_argument(
        f"--lance_partition_size",
        type=int,
        required=False,
        default=4_194_302,
        help="partition size so that each worker would not run into OOM"
    )
    parser.add_argument(
        f"--lancedb_uri",
        type=str,
        required=False,
        default="",
        help="URI of the lancedb table to search.",
    )
    parser.add_argument(
        f"--table_name",
        type=str,
        required=False,
        default="",
        help="Name of the lancedb table to search.",
    )
    parser.add_argument(
        f"--num_parallel_searches",
        type=int,
        required=False,
        default=200,
        help="Number of parallel searches to run.",
    )
    parser.add_argument(
        f"--top_K",
        type=int,
        required=False,
        default=10,
        help="top_K search for each query within a partition",
    )
    parser.add_argument(
        f"--sim_threshold",
        type=float,
        required=False,
        default=0.85,
        help="cosine similarity threshold for search results",
    )
    args = parser.parse_args()
    
    ray.init()
    main(args)
    ray.shutdown()
