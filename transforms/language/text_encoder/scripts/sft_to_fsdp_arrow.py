from datasets import load_dataset
import pyarrow as pa
import os
import ray
import argparse
from pyarrow import fs

storage_options = {
                "anon": False,
                "key": os.environ['AWS_ACCESS_KEY_ID'], 
                "secret": os.environ['AWS_SECRET_ACCESS_KEY'],
                "endpoint_url": os.environ['AWS_ENDPOINT']
            }

s3_east = fs.S3FileSystem (
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    request_timeout=10,
    connect_timeout=10,
    retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=10),
    endpoint_override=os.environ['AWS_ENDPOINT'],
)

@ray.remote(memory=85*1024*1024*1024, num_cpus=10)
def convert_sft_to_fsdp(s3: fs.S3FileSystem, sft_file_list: str, input_folder: str, output_folder: str):
    print(f" converting {len(sft_file_list)} sft arrow files")
    dataset_loaded = 0
    dataset_written = 0
    for sft_file in sft_file_list:
        s3_uri = f"s3://{sft_file}"
        print(f"loading file: {s3_uri}")
        try:
            sft_dataset = load_dataset(
                'arrow',
                data_files={'train': s3_uri},
                storage_options=storage_options
            )
            dataset_loaded += 1
        except Exception as e:
            print(f" Error loading sft arrow files {sft_file}: {e}")
        print(f"**** {s3_uri} loaded successfully.")
        print(f"**** {sft_dataset['train'].features}=")
        fsdp_tokens_batches = sft_dataset['train'].data.column('input_ids').to_pylist()
        print(f"loaded sft arrow {sft_file}")
        print(f"{sft_dataset}")
        schema = pa.schema([("tokens", pa.int32())])
        fsdp_file = sft_file.replace(input_folder, output_folder)
        output_arrow_tokens_array =[pa.array(batch_tokens) for batch_tokens in fsdp_tokens_batches]
        output_arrow_record_batches = [pa.RecordBatch.from_arrays([array], schema=schema) for array in output_arrow_tokens_array]
        try:
            with s3.open_output_stream(fsdp_file) as output_stream:
                with pa.ipc.RecordBatchFileWriter(output_stream, schema) as writer:
                    for record_batch in output_arrow_record_batches:
                        writer.write_batch(record_batch)
            dataset_written += 1
        except Exception as e:
            print(f" Error writing out fsdp arrow file {fsdp_file}: {e}")
    if dataset_loaded != dataset_written:
        print(f" Something is wrong: {dataset_loaded=} but {dataset_written=}")
    return dataset_written


def main(args):
    input_folder = args.input_folder
    output_folder = args.output_folder
    num_parallel_tasks = args.num_parallel_tasks
    s3 = s3_east
    input_files = [file for file in s3.get_file_info(fs.FileSelector(input_folder, recursive=True))]
    output_files = [file for file in s3.get_file_info(fs.FileSelector(output_folder, recursive=True))]
    output_file_basenames = [os.path.basename(file.path) for file in output_files]
    object_ref = []
    sft_file_lists = [[] for _ in range(num_parallel_tasks)]
    num_files_to_convert = 0
    for i, file in enumerate(input_files):
        if file.type == fs.FileType.File and file.path.endswith(".arrow") and '/data-' in file.path:
            if os.path.basename(file.path) in output_file_basenames:
                continue
            sft_file = file.path
            index = i % num_parallel_tasks
            sft_file_lists[index].append(sft_file)
            num_files_to_convert += 1
        else:
            continue
    print(f" {num_files_to_convert=}")
    for i in range(num_parallel_tasks):
        object_ref.append(convert_sft_to_fsdp.remote(s3, sft_file_lists[i], input_folder, output_folder))
    
    total_dataset_written = 0
    while len(object_ref) > 0:
        ready, object_ref = ray.wait(object_ref, num_returns=1)
        total_dataset_written += ray.get(ready[0])
    print(f" all {num_parallel_tasks} tasks completed." )
    print(f" total dataset written: {total_dataset_written}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for Top-K most similar docs from lancedb table")
    parser.add_argument(
        f"--input_folder",
        type=str,
        required=False,
        default = "cos-optimal-llm-pile/bluepile-processing/sgd_sft_tokenized/hajar_trl/",
        help="input folder to count all the tokens in arrow files",
    )
    parser.add_argument(
        f"--output_folder",
        type=str,
        required=False,
        default='cos-optimal-llm-pile/bluepile-processing/sgd_sft_tokenized/hajar_trl_fsdp_v2/',
        help="output folder for fstd arrow files"
    )
    parser.add_argument(
        f"--num_parallel_tasks",
        type=int,
        required=False,
        default=1,
        help="num of parallel ray tasks"
    )
    args = parser.parse_args()
    ray.init()
    main(args)
    ray.shutdown()
