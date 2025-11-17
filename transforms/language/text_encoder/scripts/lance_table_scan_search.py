import ray
import lancedb, lance
import pyarrow as pa
import pandas as pd
import numpy as np
import uuid
from io import BytesIO
import argparse
import time, datetime
from pyarrow import fs
import pyarrow.parquet as pq
import json
import os



table_name = "Gneissweb"
num_records_to_write = 5000
vector_column_name = "embeddings"
lance_table_scan_chunk_size = 8192
max_indices_size = 30000

s3_east = fs.S3FileSystem(
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    request_timeout=60,
    connect_timeout=60,
    retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=30),
    endpoint_override="s3.us-east.cloud-object-storage.appdomain.cloud",
)

s3_south = fs.S3FileSystem(
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    request_timeout=60,
    connect_timeout=60,
    retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=30),
    endpoint_override="s3.us-south.cloud-object-storage.appdomain.cloud",
)

s3=s3_east

# centroids_npy_paths = [
#     'cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/centroids/mainframe.npy',
# ]

centroids_npy_paths =[
    "cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/centroids/java_6K_sample_embeddings_r2.npy"
]

# centroids_npy_paths = [
#     'cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/centroids-1000/dclm_Nemotron_CC.npy',
#     'cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/centroids-1000/finemath4plus.npy',
#     'cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/centroids-1000/mmlu_arc_validation.npy',
#     'cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/centroids-1000/wikipedia_stackexchange.npy',
# ]

def load_centroids_pyarrow_npy_s3(cos_centroids_path):
    """Loads centroids from S3/COS stored in .npy format using PyArrow's S3FileSystem."""
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


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray):
    """
    Computes the cosine similarity between each row of matrix a and each row of matrix b.

    Args:
        a: A 2D numpy array (query embeddings).
        b: A 2D numpy array (chunk of table embeddings).

    Returns:
        A 2D numpy array where the element at [i, j] is the cosine similarity
        between the i-th row of a and the j-th row of b.
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    dot_product = np.dot(a, b.T)
    return dot_product / (a_norm * b_norm.T + 1e-8)

@ray.remote(memory=90*1024*1024*1024, num_cpus=14)
def process_table_chunk_with_multi_query_global_threshold(
    s3_credentials: dict,
    lancedb_uri: str, 
    table_name: str, 
    chunk_size: int,
    cos_output_parquet_folder: str,
    queries: np.ndarray, 
    start: int,
    end: int,
    vector_column: str, 
    threshold: float
) -> list:
    task_id = ray.get_runtime_context().get_task_id()
    
    # Each remote task will scan the entire table and return all the rows that match the query.
    # set up s3_connection
    s3_fs = fs.S3FileSystem(
        access_key=s3_credentials['access_key'],
        secret_key=s3_credentials['secret_key'],
        request_timeout=60,
        connect_timeout=60,
        retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=30),
        endpoint_override=s3_credentials['endpoint_override'],
    )
    
    try:
        db = lancedb.connect(lancedb_uri)
        lance_table = db.open_table(table_name)
        # total_rows = lance_table.count_rows()
        lance_dataset_uri = f"{lancedb_uri}{table_name}.lance"
        ds = lance.dataset(lance_dataset_uri)
        original_schema = lance_table.schema
        fields_without_embeddings = [
            field for field in original_schema
            if field.name != vector_column
        ]
        schema_without_embeddings = pa.schema(fields_without_embeddings)
        query_index_field = pa.field("query_index", pa.int32())
        similarity_field = pa.field("cosine_similarity", pa.float32())
        schema_with_query_index = schema_without_embeddings.append(query_index_field)
        schema_with_query_index_similarity = schema_with_query_index.append(similarity_field)
    except Exception as e:
        print(f"Error getting schema for table '{table_name}': {e}")
    print(f"Task {task_id} - Scanning table {table_name} from {start} to {end} with {len(queries)} centroids")
    centroid_document_counts = [0] * queries.shape[0]
    offset = start
    empty_batches = []
    buffered_results_table = pa.Table.from_batches(empty_batches, schema=schema_with_query_index_similarity)
    queries = queries.astype(np.float16)
    while offset < end:
        num_rows = min(chunk_size, end - offset)
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S') 
        # print(f"Task {task_id} - {current_time} begin scanning {num_rows=} {offset=}") 
        scanner_chunk_table = ds.scanner(limit=num_rows, offset=offset).to_table()
        chunk_embeddings = np.array(scanner_chunk_table.column(vector_column).to_pylist())
        if chunk_embeddings.ndim == 1:
            chunk_embeddings = np.expand_dims(chunk_embeddings, axis=1)

        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S') 
        # print(f"Task {task_id} - {current_time} begin computing similarities (queries, chunk_embeddings)") 

        similarities = cosine_similarity_matrix(queries, chunk_embeddings)
        matching_indices = []
        query_index_list = []
        cosine_similarity_list = []
        for i in range(scanner_chunk_table.num_rows):
            # Check if the similarity with ANY query exceeds the threshold
            found = False
            # also find the index of centroid that results in the highest similarity
            max_similarity = 0
            for j in range(len(centroid_document_counts)):
                if max_similarity <= similarities[j, i]:
                    max_similarity = similarities[j, i]
                    q_index = j
                if similarities[j, i] >= threshold:
                    found = True
            # if np.any(similarities[:, i] >= threshold):
            query_index_list.append(q_index)
            cosine_similarity_list.append(max_similarity)
            if found:
                matching_indices.append(i)
                centroid_document_counts[q_index] += 1
        
        # current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S') 
        # print(f"Task {task_id} - {current_time} finish computing similarities {len(matching_indices)=}")

        sub_results = []
        if len(matching_indices) > 0:
            # print(f"Task {task_id} - {len(matching_indices)=} from (queries, chunk_embeddings)")
            columns_to_drop = [vector_column]
            chunk_table_0 = scanner_chunk_table.drop(columns=columns_to_drop)
            # add two additional columns to chunk_table, one query_index and one similarity
            query_index_array = pa.array(query_index_list, type=pa.int32())
            similarity_array = pa.array(cosine_similarity_list, type=pa.float32())
            try:
                chunk_table_1 = chunk_table_0.append_column(query_index_field, query_index_array)
                chunk_table = chunk_table_1.append_column(similarity_field, similarity_array)
            except Exception as e:
                print(f"Task {task_id} - Error appending columns to chunk_table_0: {e}")
            num_matching_indices = len(matching_indices)
            if num_matching_indices <= max_indices_size:
                chunk_result_table = chunk_table.take(pa.array(matching_indices))
                sub_results.append(chunk_result_table)
            else:
                for j in range(0, num_matching_indices, max_indices_size):
                    end_index = min(j + max_indices_size, num_matching_indices)
                    sub_matching_indices = matching_indices[j:end_index]
                    # Extract data column by column
                    sub_columns = []
                    for col_name in chunk_table.column_names:
                        original_column = chunk_table.column(col_name)
                        sub_column_data = original_column.take(sub_matching_indices)
                        sub_columns.append(sub_column_data)
                    sub_result_table = pa.Table.from_arrays(sub_columns, schema=chunk_table.schema)
                    sub_results.append(sub_result_table)
        # print(f"Task {task_id} - {len(sub_results)=} - {buffered_results_table.num_rows=}")
        for sub_result_table in sub_results:
            # print("Schema of buffered_results_table:", buffered_results_table.schema)
            # print("Schema of sub_result_table:", sub_result_table.schema)
            buffered_results_table = pa.concat_tables([buffered_results_table, sub_result_table])
            # print(f"Task {task_id} - {sub_result_table.num_rows=} {buffered_results_table.num_rows=} after pa.concat_tables") 
            if buffered_results_table.num_rows > num_records_to_write:
                num_files = buffered_results_table.num_rows//num_records_to_write
                for i in range(num_files):
                    # write the buffered_results to output folder as parquet and reset the buffer
                    output_file_path = f"{cos_output_parquet_folder}{str(uuid.uuid4())}.parquet"
                    start_index = i * num_records_to_write
                    end_index = (i+1) * num_records_to_write
                    chunk_table_to_write = buffered_results_table[start_index:end_index]
                    try:
                        with s3_fs.open_output_stream(output_file_path) as sink:
                            pq.write_table(chunk_table_to_write, sink)
                        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S') 
                        print(f"{current_time} Task: {task_id} Wrote {chunk_table_to_write.num_rows} rows to {output_file_path}")
                    except Exception as e:
                        print(f"Error writing output parquet: {e}")
                buffered_results_table = buffered_results_table[num_files*num_records_to_write:]
                current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S') 
                print(f"{current_time} Task {task_id} - After writing {num_files} files {buffered_results_table.num_rows=}")
        offset += num_rows
    # write the final residual buffered_results_table to output folder as parquet
    if buffered_results_table.num_rows > 0:
        output_file_path = f"{cos_output_parquet_folder}{str(uuid.uuid4())}.parquet"
        try:
            with s3_fs.open_output_stream(output_file_path) as sink:
                pq.write_table(buffered_results_table, sink)
            current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S') 
            print(f"{current_time} Task: {task_id} Wrote final {buffered_results_table.num_rows} rows to {output_file_path}")
        except Exception as e:
            print(f"Error writing output parquet: {e}")
    return centroid_document_counts

def find_similar_with_multi_query_global_threshold(
        s3_credentials: dict,
        uri: str, 
        table_name: str,
        cos_output_parquet_folder: str,
        cos_output_centroid_doc_counts_json_path: str,
        num_parallel_searches: int,
        chunk_size: int,
        vector_column: str, 
        threshold: float
):
    all_centroids =[]
    for cos_centroids_path in centroids_npy_paths:
        centroids = load_centroids_pyarrow_npy_s3(cos_centroids_path)
        all_centroids.append(centroids)
    centroids = np.concatenate(all_centroids, axis=0)
    # centroids = total_centroids[[222, 543]]
    # centroids = tmp_centroids[[0]]
    print(f"{type(centroids)}")
    print(f"{centroids}")
    tot_centroids = len(centroids)
    print(f"{tot_centroids=}")
    # compute the beginning offsets for each parallel search with table scan
    db = lancedb.connect(uri)
    lance_table = db.open_table(table_name)
    total_rows = lance_table.count_rows()
    print(f"{total_rows=}")
    scan_size = total_rows // (num_parallel_searches)
    remainder = total_rows % (num_parallel_searches)
    ranges = []
    start = 0
    for i in range(num_parallel_searches):
        end = start + scan_size + 1 if i < remainder else scan_size + start
        ranges.append((start, end))
        start = end
    print(f"{ranges=}")
    ray_futures = []
    for start, end in ranges:
        ray_future = process_table_chunk_with_multi_query_global_threshold.remote(
            s3_credentials=s3_credentials,
            lancedb_uri=uri,
            table_name=table_name, 
            chunk_size=chunk_size,
            cos_output_parquet_folder=cos_output_parquet_folder, 
            queries=centroids, 
            start=start,
            end=end,
            vector_column=vector_column, 
            threshold=threshold)
        ray_futures.append(ray_future)
        time.sleep(2)
    centroid_document_counts_sum = [0] * len(centroids)
    while len(ray_futures) > 0:
        ready, _ = ray.wait(ray_futures, num_returns=1, timeout=6)
        if ready:
            centroid_document_counts = ray.get(ready[0])
            centroid_document_counts_sum = [sum(x) for x in zip(centroid_document_counts_sum, centroid_document_counts)]
            ray_futures.remove(ready[0])
        else:
            time.sleep(2)
    # print(f"{centroid_document_counts_sum=}")
    centroid_doc_counts = [{"centroid": i, "count": count} for i, count in enumerate(centroid_document_counts_sum)]
    with s3.open_output_stream(cos_output_centroid_doc_counts_json_path) as s3_file:
        s3_file.write(json.dumps(centroid_doc_counts, indent=4).encode('utf-8'))
    print(f"All {num_parallel_searches} parallel searches completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for Top-K most similar docs from lancedb table")    
    parser.add_argument(
        f"--cos_output_parquet_folder",
        type=str,
        required=False,
        default="cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/chunking_experiments/parquets_java_6K_finepdf_no_chunk_r2_table_scan/",
        help="Output path for the found results of all table chunks as parquet files.",
    )
    parser.add_argument(
        f"--cos_output_centroid_doc_counts_json_path",
        type=str,
        required=False,
        default="cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/centroid_doc_counts/finepdf_no_chunk_java_6K_r2_table_scan.json",
        help="Output json path for the document counts for each centroid.",
    )
    parser.add_argument(
        f"--lancedb_uri",
        type=str,
        required=False,
        default="s3://cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/chunking_experiments/lance/finepdf_no_chunk_r2.db/",
        help="URI of the lancedb table to search.",
    )
    parser.add_argument(
        f"--table_name",
        type=str,
        required=False,
        default="finepdf_no_chunk_r2",
        help="Name of the lancedb table to search.",
    )
    parser.add_argument(
        f"--num_parallel_searches",
        type=int,
        required=False,
        default=100,
        help="Number of parallel searches to run.",
    )
    parser.add_argument(
        f"--similarity_threshold",
        type=float,
        required=False,
        default=0.70,
        help="Similarity threshold for finding similar documents.",    
    )
    parser.add_argument(
        f"--s3_access_key",
        type=str,
        required=False,
        default="9cf09c22030545f4adfe06e936387db4",
        help="S3 access key for the S3/COS bucket.",
    )
    parser.add_argument(
        f"--s3_secret_key",
        type=str,
        required=False,
        default="e0493ef603ec6a214f40065477180ac2cf5aee317aee72ce",
        help="S3 secret key for the S3/COS bucket.",
    )
    parser.add_argument(
        f"--s3_endpoint_override",
        type=str,
        required=False,
        default="s3.us-east.cloud-object-storage.appdomain.cloud",
        help="S3 endpoint override for the S3/COS bucket.",
    )
    args = parser.parse_args()
    s3_credentials ={}
    s3_credentials["access_key"] = args.s3_access_key
    s3_credentials["secret_key"] = args.s3_secret_key
    s3_credentials["endpoint_override"] = args.s3_endpoint_override
    
    # Example usage:
    db_uri = args.lancedb_uri
    table_name = args.table_name
    cos_output_parquet_folder = args.cos_output_parquet_folder
    cos_output_centroid_doc_counts_json_path = args.cos_output_centroid_doc_counts_json_path
    # ensure cos_output_parquet_folder ends with /
    cos_output_parquet_folder = cos_output_parquet_folder if cos_output_parquet_folder.endswith("/") else cos_output_parquet_folder + "/"
    num_parallel_searches = int(args.num_parallel_searches)
    vector_column = vector_column_name
    batch_size = lance_table_scan_chunk_size
    threshold = float(args.similarity_threshold)
    print(f"{threshold=}")
    ray.init()

    find_similar_with_multi_query_global_threshold(
        s3_credentials=s3_credentials,
        uri=db_uri, 
        table_name=table_name,
        cos_output_parquet_folder=cos_output_parquet_folder,
        cos_output_centroid_doc_counts_json_path=cos_output_centroid_doc_counts_json_path,
        num_parallel_searches=num_parallel_searches,
        chunk_size=batch_size,
        vector_column=vector_column, 
        threshold=threshold
    )
    ray.shutdown()
