#!/bin/bash
python3 -u -m dpk_text_encoder.ray.runtime \
--run_locally False \
--text_encoder_model_name "ibm-granite/granite-embedding-small-english-r2" \
--text_encoder_content_column_name "contents" \
--text_encoder_output_embeddings_column_name "embeddings" \
--text_encoder_lanceDB_batch_size 262144 \
--text_encoder_embedding_batch_size 8 \
--text_encoder_embeddings_in_lanceDB True \
--text_encoder_model_max_seq_length 2048 \
--text_encoder_lanceDB_table_name "math_hard_test_r2" \
--text_encoder_lanceDB_fragments_json_folder "cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/lance/math_hard_test_r2_fragments_json/" \
--text_encoder_lanceDB_data_uri "s3://cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/lance/math_hard_test_r2.db/math_hard_test_r2.lance/" \
--data_s3_config '{"input_folder": "cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/math_hard_test/", "output_folder": "cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/math_hard_test_tmp4embeddings/"}' \
--runtime_worker_options '{"num_cpus": 14, "memory": 95000000000, "max_restarts": 0, "max_task_retries": 0,}' \
--runtime_creation_delay 3 \
--runtime_num_workers 1 --data_checkpointing False --runtime_code_location '{"github": "github", "commit_hash": "12345", "path": "path"}'