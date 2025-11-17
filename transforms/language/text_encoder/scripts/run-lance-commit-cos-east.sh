#!/bin/bash
python3 -u -m dpk_text_encoder.lance_commit \
--lanceDB_storage_type 's3' \
--lanceDB_uri "s3://cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/lance/math_hard_test_r2.db/" \
--lanceDB_data_uri "s3://cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/lance/math_hard_test_r2.db/math_hard_test_r2.lance/" \
--lanceDB_table_name "math_hard_test_r2" \
--lanceDB_fragments_json_folder "cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/lance/math_hard_test_r2_fragments_json/" \
--lanceDB_table_schema_folder "cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/math_hard_test_tmp4embeddings/"
