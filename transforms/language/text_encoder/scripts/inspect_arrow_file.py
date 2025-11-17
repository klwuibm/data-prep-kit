import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.fs as fs
from typing import Dict, Any
import os

# ----------------------------------------------------------------------
# IMPORTANT: Replace these with your actual configuration
# ----------------------------------------------------------------------

# Replace with your actual fsspec-based S3FileSystem object
# Example: s3 = fs.S3FileSystem(...) 
class MockS3:
    """Mocks the S3 file system for demonstration. In real use, this must be your actual S3 client."""
    def open_input_stream(self, path):
        print(f"--> Mock opening input stream for S3 path: {path}")
        
        # NOTE: This mock section cannot actually connect to S3.
        # It MUST be replaced with your working s3 client configuration.
        # Example of creating a real S3FileSystem:
        # return fs.S3FileSystem().open_input_stream(path)
        
        # --- MOCK DATA GENERATION FOR DEMO ---
        # If your actual file has the schema {'indices': uint64}, 
        # the reader must be able to load it. 
        # Since we cannot mock the IPC stream content, we return None here
        # and rely on you replacing this with the real S3 object.
        return None 
        
    # Mock for file info listing in the previous script (not needed here, but kept for context)
    def open_output_stream(self, path): 
        print(f"--> Mock opening output stream for S3 path: {path}")
        class MockStream:
            def write(self, data): pass
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        return MockStream()

# The specific .arrow file path you are debugging
# Example: sft_file_path = "cos-optimal-llm-pile/bluepile-processing/sgd_sft_tokenized/sft_dataset/v8d2_no_code/data-00000-of-00219.arrow"
sft_file_path = "cos-optimal-llm-pile/bluepile-processing/sgd_sft_tokenized/hajar_trl/finemath4plus_gptoss20B/data-00000-of-00008.arrow"

# Initialize your actual S3 filesystem object here
# s3 = fs.S3FileSystem(key=..., secret=...)
storage_options = {
                "anon": False,
                "key": os.environ['AWS_ACCESS_KEY_ID'], 
                "secret": os.environ['AWS_SECRET_ACCESS_KEY'],
                "endpoint_url": os.environ['AWS_ENDPOINT']
            }

s3 = fs.S3FileSystem (
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    request_timeout=10,
    connect_timeout=10,
    retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=10),
    endpoint_override=os.environ['AWS_ENDPOINT'],
)


# ----------------------------------------------------------------------

def inspect_arrow_file(sft_file_path: str, s3: Any):
    """
    Opens an S3-hosted Arrow IPC file, prints its schema, and displays the first batch of data.
    """
    if s3.open_input_stream(sft_file_path) is None:
        print("\nERROR: Cannot proceed with inspection.")
        print("Please ensure you replace 's3 = MockS3()' with your actual, configured S3FileSystem instance.")
        print("The pyarrow.fs.S3FileSystem must be able to open an input stream to the file path.")
        return

    try:
        # 1. Open the file stream from S3
        with s3.open_input_stream(sft_file_path) as input_stream:
            
            # 2. Open the Arrow IPC STREAM Reader (Fix for 'seekable files' error)
            # The stream reader is designed for non-seekable streams like network pipes.
            reader = ipc.open_stream(input_stream)

            # 3. Print the Schema (Critical step to see all column names)
            print("\n=======================================================")
            print(f"Successfully loaded file: {sft_file_path}")
            print("--- SCHEMA (Column Names and Types) ---")
            print(reader.schema)
            print("=======================================================\n")

            # 4. Read and display the first Record Batch
            try:
                # Use read_next_batch() for stream readers
                first_batch = reader.read_next_batch() 
                print("--- FIRST RECORD BATCH (Sample Data) ---")
                
                # Convert the PyArrow RecordBatch to a Pandas DataFrame for easy viewing
                # NOTE: If pandas is not available, comment out the next line and use first_batch.to_pydict()
                try:
                    import pandas as pd
                    df = first_batch.to_pandas()
                    print(df.head())
                except ImportError:
                    print("Pandas not found. Displaying raw PyArrow dictionary:")
                    print(first_batch.to_pydict())

            except StopIteration:
                print("File stream was opened successfully, but it contained no record batches (is empty).")
                
    except Exception as e:
        print("\n=======================================================")
        print(f"FATAL ERROR: Could not inspect file {sft_file_path}")
        print(f"Error details: {e}")
        print("This usually means the file path is wrong, the S3 permissions are wrong, or the file is corrupted.")
        print("=======================================================")
        
        
# Execute the inspection
if __name__ == '__main__':
    inspect_arrow_file(sft_file_path, s3)

