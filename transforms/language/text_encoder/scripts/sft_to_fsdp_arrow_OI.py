from datasets import load_from_disk
import pyarrow as pa
import os
import gc
import argparse
from pyarrow import fs
import pyarrow.ipc as ipc
import pyarrow.dataset as ds

storage_options = {
                "anon": False,
                "key": os.environ['AWS_ACCESS_KEY_ID'], 
                "secret": os.environ['AWS_SECRET_ACCESS_KEY'],
                "endpoint_url": os.environ['AWS_ENDPOINT']
            }

s3_east = fs.S3FileSystem (
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    request_timeout=30,
    connect_timeout=30,
    retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=30),
    endpoint_override=os.environ['AWS_ENDPOINT'],
)

from datasets import Dataset, load_from_disk
from typing import Generator, Dict, List


def main(args):
    input_folder = args.input_folder
    output_folder = args.output_folder
    max_buffer_tokens = args.max_buffer_tokens
    excluded_folders = args.excluded_folders
    # num_parallel_tasks = args.num_parallel_tasks
    s3 = s3_east
    folders = [file for file in s3.get_file_info(fs.FileSelector(input_folder, recursive=False))]
    for folder in folders:
        print(f"{folder.path=}")
    for i, folder in enumerate(folders):
        exclusion = False
        for ex in excluded_folders:
            if ex in folder.path:
                exclusion = True
        if exclusion:
            continue
        try:
            if folder.type == fs.FileType.Directory:
                file_infos = [file for file in s3.get_file_info(fs.FileSelector(folder.path, recursive=True))]
                data_files = [
                    info.path for info in file_infos
                    if info.type == fs.FileType.File and info.path.endswith('.arrow') and '/data-' in info.path
                ]
                if not data_files:
                    raise FileNotFoundError(f"No .arrow files found at {folder.path}")
        except Exception as e:
            print(f"Error creating PyArrow Dataset from {folder.path}: {e}")

        output_folder = output_folder if output_folder.endswith('/') else output_folder+'/'
        input_folder = input_folder if input_folder.endswith('/') else input_folder+'/'
        # output_file_path is now the path prefix for multiple arrow files 
        output_file_path = folder.path.replace(input_folder, output_folder) 

        token_buffer = []
        total_token_count = 0
        current_buffer_token_count = 0
        batch_file_counter = 0
        try:
            print(f"Start processing {len(data_files)=} arrow files in {folder.path}.")
            for source_file in data_files:
                print(f"Processing {source_file=}")
                with s3.open_input_stream(source_file) as input_stream:
                # Use PyArrow IPC reader to stream batches from the file
                    reader = ipc.open_stream(input_stream)
                    for incoming_batch in reader:
                        # 1. Extract the tokens for one row/example
                        input_ids_array = incoming_batch.column('input_ids')
                        new_rows = input_ids_array.to_pylist()
                        for batch_tokens in new_rows:
                            tokens_in_row = len(batch_tokens)
                            
                            # 2. Use .append() to maintain list-of-lists structure
                            token_buffer.append(batch_tokens)
                            
                            # 3. Update token counts
                            current_buffer_token_count += tokens_in_row
                            total_token_count += tokens_in_row
                            
                            # 4. Check if buffer size exceeds the limit (based on tokens)
                            if current_buffer_token_count >= max_buffer_tokens:

                                schema = pa.schema([("tokens", pa.int32())])
                                output_arrow_tokens_array =[pa.array(batch_tokens) for batch_tokens in token_buffer]
                                output_arrow_record_batches = [pa.RecordBatch.from_arrays([array], schema=schema) for array in output_arrow_tokens_array]
                                batch_file = f"{output_file_path}/{batch_file_counter:04d}.arrow"
                                try:
                                    with s3.open_output_stream(batch_file) as output_stream:
                                        with pa.ipc.RecordBatchFileWriter(output_stream, schema) as writer:
                                            for record_batch in output_arrow_record_batches:
                                                writer.write_batch(record_batch)
                                except Exception as e:
                                    print(f" Error writing out fsdp arrow file {batch_file}: {e}")

                                print(f"Successfully wrote {batch_file=}")
                                batch_file_counter += 1
                                # Reset the buffer and the token count for the next batch
                                token_buffer = []
                                current_buffer_token_count = 0

                                # Force garbage collection
                                gc.collect() 
                                
                                
            # 5. Write any remaining tokens (the final partial batch)
            if token_buffer:
                print("--> Writing final partial batch...")
                batch_file = f"{output_file_path}/{batch_file_counter:04d}.arrow"
                schema = pa.schema([("tokens", pa.int32())])
                output_arrow_tokens_array =[pa.array(batch_tokens) for batch_tokens in token_buffer]
                output_arrow_record_batches = [pa.RecordBatch.from_arrays([array], schema=schema) for array in output_arrow_tokens_array]
                try:
                    with s3.open_output_stream(batch_file) as output_stream:
                        with pa.ipc.RecordBatchFileWriter(output_stream, schema) as writer:
                            for record_batch in output_arrow_record_batches:
                                writer.write_batch(record_batch)
                except Exception as e:
                    print(f" Error writing out fsdp arrow file {batch_file}: {e}")
                
                print(f"Successfully wrote {batch_file=}")
                batch_file_counter += 1
                gc.collect() 
                            
            print(f"--> Streaming write complete for {folder.path}. Total tokens processed: {total_token_count}")
        except Exception as e:
            print(f"Error processing scanner: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for Top-K most similar docs from lancedb table")
    parser.add_argument(
        f"--input_folder",
        type=str,
        required=False,
        default = "cos-optimal-llm-pile/bluepile-processing/sgd_sft_tokenized/sft_dataset/",
        help="input folder to count all the tokens in arrow files",
    )
    parser.add_argument(
        f"--output_folder",
        type=str,
        required=False,
        default='cos-optimal-llm-pile/bluepile-processing/sgd_sft_tokenized/sft_dataset_fsdp/',
        help="output folder for fstd arrow files"
    )
    parser.add_argument(
        f"--max_buffer_tokens",
        type=int,
        required=False,
        default=100_000_000,
        help='max buffer tokens to write out to an .arrow file',
    )
    parser.add_argument(
        '--excluded_folders',
        nargs='*',
        default=['v8d2_no_code', 'v8d2_only_code', 'ramon_math', 'amon_science'],
        help=(
            "Accepts zero or more items, collected as a list.\n"
            "Example: --items apple banana cherry"
        )
    )
    args = parser.parse_args()
    main(args)
