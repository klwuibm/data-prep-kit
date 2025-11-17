#!/bin/bash
python3 -u ./ann_faiss_search.py \
--output_folder 'cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/parquets_java_finepdf_eng_r2_2K_0.85/' \
--lancedb_uri 's3://cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/lance/finepdf_eng_r2_2K.db/' \
--table_name 'finepdf_eng_r2_2K' \
--num_parallel_searches 100 \
--lance_partition_size 524288 \
--top_K 15 \
--sim_threshold 0.85 \
--query_embeddings_path 'cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/centroids/java_6K_sample_embeddings_r2.npy'