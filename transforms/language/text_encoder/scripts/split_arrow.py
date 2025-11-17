import lancedb
import numpy as np
from pyarrow import fs
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import uuid
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import ray
import time
import os

# Assume you have a LanceDB connection object 'db' and the table name 'my_table'
# Assume 'final_centroids' is a NumPy array where each row is a centroid embedding
# Assume 'embedding_column_name' is the name of your embedding column


s3_east = fs.S3FileSystem(
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    request_timeout=30,
    connect_timeout=30,
    retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=30),
    endpoint_override="s3.us-east.cloud-object-storage.appdomain.cloud",
)

s3_south = fs.S3FileSystem(
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    request_timeout=10,
    connect_timeout=10,
    retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=10),
    endpoint_override="s3.us-south.cloud-object-storage.appdomain.cloud",
)


@ray.remote(memory=30*1024*1024*1024, num_cpus=5)
def split_large_arrow_batch(
    s3: fs.S3FileSystem,
    input_file: str,
    input_folder: str,
    output_folder: str,
    max_num_tokens_per_arrow_batch: int):

    try:
        with s3.open_input_stream(input_file) as f:
            arrow_bytes = f.readall()
            arrow_reader = pa.ipc.open_file(arrow_bytes)
            _ = arrow_reader.read_all()
            input_arrow_tokens_batches = [arrow_reader.get_batch(i)["tokens"].to_pylist() for i in range(arrow_reader.num_record_batches)]
        print(f"{input_file=} {len(input_arrow_tokens_batches)=}")
    except Exception as e:
        print(f"Error reading arrow file: {e}")
    # now start examine each batch and do the splitting
    output_arrow_tokens_batches = []
    output_arrow_file_path = input_file.replace(input_folder, output_folder)
    print(f"{output_arrow_file_path=}")
    for i, tokens_batch in enumerate(input_arrow_tokens_batches):
        for j in range(len(tokens_batch)//max_num_tokens_per_arrow_batch +1):
            output_arrow_tokens_batches.append(tokens_batch[j*max_num_tokens_per_arrow_batch:(j+1)*max_num_tokens_per_arrow_batch])
    print(f"{len(output_arrow_tokens_batches)=}")
    schema = pa.schema([("tokens", pa.uint32())])
    output_arrow_tokens_array =[pa.array(batch_tokens) for batch_tokens in output_arrow_tokens_batches]
    output_arrow_record_batches = [pa.RecordBatch.from_arrays([array], schema=schema) for array in output_arrow_tokens_array]
    try:
        with s3.open_output_stream(output_arrow_file_path) as f:
            with pa.ipc.RecordBatchFileWriter(f, schema) as writer:
                for record_batch in output_arrow_record_batches:
                    writer.write_batch(record_batch)
        print(f"writing to {output_arrow_file_path} {len(output_arrow_tokens_batches)=}")
    except Exception as e:
        print(f"Error writing arrow or meta files: {e}")
                
def main(args):
    s3 = s3_east
    max_num_tokens_per_arrow_batch = int(args.max_num_tokens_per_arrow_batch)
    num_parallel_splits = int(args.num_parallel_splits)
    input_folder = args.input_folder
    input_folder = input_folder if input_folder.endswith("/") else input_folder + "/"
    output_folder = args.output_folder
    output_folder = output_folder if output_folder.endswith("/") else output_folder + "/"
    if not bool(input_folder) or not bool(output_folder):
        print("Please provide input and output folder")
        exit(1)
    input_files = [file for file in s3.get_file_info(fs.FileSelector(input_folder, recursive=True))]
    output_files = [file for file in s3.get_file_info(fs.FileSelector(output_folder, recursive=True))]
    input_arrow_files = [file for file in input_files if file.type == fs.FileType.File and file.path.endswith(".arrow")]
    output_arrow_files_basenames = [os.path.basename(file.path) for file in output_files if file.type == fs.FileType.File and file.path.endswith(".arrow")]
    remaining_arrow_files_to_split = [file.path for file in input_arrow_files if os.path.basename(file.path) not in output_arrow_files_basenames]
    print(f"{len(input_arrow_files)=} {len(remaining_arrow_files_to_split)=}")
    object_refs = []
    for j, file in enumerate(remaining_arrow_files_to_split):
        object_refs.append(
            split_large_arrow_batch.remote(
                s3=s3, 
                input_file=file, 
                input_folder=input_folder, 
                output_folder=output_folder, 
                max_num_tokens_per_arrow_batch=max_num_tokens_per_arrow_batch,
            )
        )
        while len(object_refs) > num_parallel_splits:
            ready, _ = ray.wait(object_refs, num_returns=1)
            if ready:
                object_refs.remove(ready[0])
            else:
                time.sleep(0.5) # Avoid busy-waiting
    while len(object_refs) > 0:
        print(f"{len(object_refs)=}")
        ready, _ = ray.wait(object_refs, num_returns=1)
        if ready:
            object_refs.remove(ready[0])
        else:
            time.sleep(0.5)
    print(f"All parallel split tasks completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for Top-K most similar docs from lancedb table")
    parser.add_argument(
        f"--max_num_tokens_per_arrow_batch",
        type=int,
        required=False,
        default = 10000,
        help="max num of tokens per arrow batch"
    )
    parser.add_argument(
        f"--num_parallel_splits",
        type=int,
        required=False,
        default = 100,
        help="number of parallel splits"
    )
    parser.add_argument(
        f"--input_folder",
        type=str,   
        required=False,
        default = "",
        help="input folder"
    )
    parser.add_argument(
        f"--output_folder",
        type=str,
        required=False,
        default = "",
        help="output folder"
    )
    args = parser.parse_args()
    ray.init()
    main(args)
    ray.shutdown()