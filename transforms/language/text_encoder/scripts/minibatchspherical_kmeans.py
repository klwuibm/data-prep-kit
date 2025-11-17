import lance
import lancedb
import pyarrow as pa
from pyarrow import fs
import pyarrow.compute as pc
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import random
import time, datetime
from io import BytesIO
import os


DCLM_DB_PATHS = ["s3://cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/lance/dclm_parquets_filtered.db/"]
DCLM_TABLE_NAMES = ["dclm_parquets_filtered"]
dclm_cos_centroid_path = 'cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/centroids/dclm_parquets_filtered_11000.npy'

CC_DB_PATHS = ["s3://cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/lance/dclm_parquets_filtered.db/",
           "s3://cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/lance/Nemotron_CC.db/"]
CC_TABLE_NAMES = ["dclm_parquets_filtered", "Nemotron_CC"]  
cc_cos_centroid_path = 'cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/centroids-1500/dclm_Nemotron_CC.npy'

WK_DB_PATHS = ["s3://cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/lance/wikipedia.db/",
           "s3://cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/lance/stackexchange.db/"]
WK_TABLE_NAMES = ["wikipedia", "stackexchange"]
wk_cos_centroid_path = 'cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/centroids/wikipedia_stachexchange.npy'

MATH_DB_PATHS= ["s3://cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/lance/finemath4plus.db/"]
MATH_TABLE_NAMES= ["finemath4plus"]
math_cos_centroid_path = 'cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/centroids/finemath4plus.npy'

MMLU_ARC_DB_PATHS= ["s3://cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/lance/MMLU_validation.db/",
           "s3://cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/lance/ARC_validation.db/"]
MMLU_ARC_TABLE_NAMES= ["MMLU_validation", "ARC_validation"]
mmlu_arc_cos_centroid_path= 'cos-llm-pile-south/bluepile-processing/fineweb_ablation/R183_embedding-based_clustering/centroids/mmlu_arc_validation.npy'

MAINFRAME_DB_PATHS = ["s3://cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/lance/mainframe.db/"]   
MAINFRAME_TABLE_NAMES = ["mainframe"]
mainframe_cos_centroid_path = 'cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/centroids/mainframe.npy'

ZOS_DB_PATHS = ["s3://cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/lance/zos_data.db/",
                "s3://cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/lance/zos_data_delta.db/"]
ZOS_TABLE_NAMES = ["zos_data", "zos_data_delta"]
zos_cos_centroid_path = 'cos-optimal-llm-pile/bluepile-processing/rel_10/embeddings/centroids/zos_data_and_delta_100.npy'

EMBEDDING_COLUMN_NAME = 'embeddings'


s3_east = fs.S3FileSystem(
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    request_timeout=60,
    connect_timeout=60,
    retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=20),
    endpoint_override="s3.us-east.cloud-object-storage.appdomain.cloud",
)

s3_south = fs.S3FileSystem(
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    request_timeout=60,
    connect_timeout=60,
    retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=20),
    endpoint_override="s3.us-south.cloud-object-storage.appdomain.cloud",
)

s3 = s3_east



def store_centroids_pyarrow_npy_s3(centroids: np.ndarray, s3: fs.S3FileSystem, path: str):
    """Stores a NumPy array of centroids to S3/COS as a .npy file using PyArrow's S3FileSystem."""
    try:
        # Save the NumPy array to a BytesIO object
        buffer = BytesIO()
        np.save(buffer, centroids)
        buffer.seek(0)

        # Open a file for writing in the S3 bucket using PyArrow
        with s3.open_output_stream(path) as sink:
            # Write the content of the BytesIO buffer to the S3 file
            sink.write(buffer.getvalue())
            print(f"Centroids saved to {path} in .npy format using PyArrow")

    except Exception as e:
        print(f"Error saving centroids to S3/COS using PyArrow: {e}")
def print_documents_in_clusters(
        lance_uri: str, 
        table_name: str, 
        final_centroids: np.ndarray, 
        num_clusters: int,
        embedding_column_name: str):
    db = lancedb.connect(lance_uri)
    lance_table = db.open_table(table_name)
    num_rows = lance_table.count_rows()
    dataset_uri = f"{lance_uri}/{table_name}.lance"
    ds = lance.dataset(dataset_uri)
    scanner_table = ds.scanner(columns=[embedding_column_name, "document_id", "contents", "subject"], limit=num_rows, offset=0).to_table()
    doc_ids = scanner_table.column("document_id")
    contents = scanner_table.column("contents")
    subjects = scanner_table.column("subject")
    all_embeddings = np.array(scanner_table.column(embedding_column_name).to_pylist())
    embeddings = normalize(all_embeddings, axis=1)
    similarity = cosine_similarity(embeddings, final_centroids)
    all_labels = pa.array(np.argmax(similarity, axis=1), type=pa.int32())
    table = pa.Table.from_arrays(
        [doc_ids, contents, subjects, all_labels],
        names=['document_id', 'contents', 'subject', 'cluster_id']
    )
    cluster_ids = [i for i in range(num_clusters)]
    for cluster_id in cluster_ids:
        column_name = 'cluster_id'
        mask= pc.equal(table.column(column_name), pa.scalar(cluster_id))
        filtered_table = table.filter(mask)
        content_list = filtered_table.column("contents").to_pylist()
        doc_ids = filtered_table.column("document_id").to_pylist()
        subjects = filtered_table.column("subject").to_pylist()
        i = 0
        print(f"\n")
        for doc_id, content, subject in zip(doc_ids, content_list, subjects):
            if i < 10:
                print(f"cluster_id: {cluster_id} doc_id: {doc_id} subject: {subject}")
                print(f"{content}")
                print(f"\n")
                i += 1

def evaluate_silhouette(lancedb_uris: list[str], 
                        table_names: list[str],
                        embedding_column_name: str, 
                        final_centroids: np.ndarray, 
                        batch_sizes: list[int], 
                        sample_size_for_silhouette: int = 10000, 
                        seed: int = 42):
    """
    Evaluates the silhouette score for the data in Lance tables based on the given centroids.

    Args:
        lance_uris: List of URIs for your Lance tables.
        embedding_column: Name of the embedding column.
        final_centroids: The trained cluster centroids (NumPy array).
        batch_size: Batch size for reading data from Lance tables.
        sample_size_for_silhouette: Number of data points to sample for silhouette calculation.
        seed: Random seed for sampling.

    Returns:
        The silhouette score (float) on the sampled data, or None if not enough data.
    """
    all_embeddings = []
    all_labels = []
    random_state = np.random.RandomState(seed)

    for uri, table_name, batch_size in zip(lancedb_uris, table_names, batch_sizes):
        dataset_uri = f"{uri}{table_name}.lance"
        db = lancedb.connect(uri)
        ds = lance.dataset(dataset_uri)
        table = db.open_table(table_name)
        total_rows = table.count_rows()
        offset = 0
        # compute the portion of samples needed from this table
        total_batch_size = sum(batch_sizes)
        sample_size = int ((batch_size/total_batch_size) * sample_size_for_silhouette) + 1
        sample_indices = sorted(random_state.choice(total_rows, min(sample_size, total_rows), replace=False))
        while offset < total_rows:
            limit = min(batch_size, total_rows - offset)
            scanner_table = ds.scanner(columns=[embedding_column_name], limit=limit, offset=offset).to_table() 
            # print(f"fetched {scanner_table.num_rows} records from {table_name=}")
            # find local_indices using offset
            local_indices =[]
            index_pointer = 0
            while index_pointer < len(sample_indices) and sample_indices[index_pointer] < offset+batch_size:
                index = sample_indices[index_pointer] - offset
                if 0 <= index and index < batch_size:
                    local_indices.append(index)
                index_pointer += 1
            if len(local_indices) > 0:
                local_indices = sorted(local_indices)
                local_indices_array = pa.array(local_indices)
                selected_table = scanner_table.take(local_indices_array)
                embeddings_selected_batch = np.array(selected_table.column(embedding_column_name).to_pylist())
                embeddings_batch = normalize(embeddings_selected_batch, axis=1)
                similarity = cosine_similarity(embeddings_batch, final_centroids)
                labels_batch = np.argmax(similarity, axis=1)
                all_embeddings.extend(embeddings_batch)
                all_labels.extend(labels_batch)
            offset += batch_size

     # Assuming all_labels is a NumPy array of cluster assignments
    unique_labels = np.unique(all_labels)
    cluster_counts = np.bincount(all_labels)

    print("Number of embeddings in each cluster:")
    for label in unique_labels:
        if label < len(cluster_counts):
            print(f"{label}: {cluster_counts[label]}")



    if len(np.unique(all_labels)) > 1 and len(all_labels) > len(np.unique(all_labels)):
        silhouette = silhouette_score(np.array(all_embeddings), np.array(all_labels), metric='cosine')
        return silhouette
    else:
        print("Warning: Not enough clusters or data points to calculate silhouette score.")
        return None


def get_sample_from_batch(table_batch: pa.Table, num_samples: int, seed: int = None) -> np.ndarray:
    """
    Takes a random sample of embeddings from a PyArrow Table (batch).

    Args:
        table_batch: The input PyArrow Table batch containing an embedding column.
        num_samples: The number of embeddings to sample from this batch.
        seed: Optional random seed for reproducibility.

    Returns:
        A NumPy array of the sampled embeddings.
    """
    if not table_batch or table_batch.num_rows == 0:
        return np.array([])

    embedding_column_name = EMBEDDING_COLUMN_NAME

    n_rows = table_batch.num_rows
    if n_rows <= num_samples:
        embeddings = np.array(table_batch.column(embedding_column_name).to_pylist())
        return embeddings if embeddings.size > 0 else np.array([])
    else:
        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(n_rows, size=num_samples, replace=False)
        sampled_table = table_batch.take(pa.array(indices))
        embeddings = np.array(sampled_table.column(embedding_column_name).to_pylist())
        return embeddings if embeddings.size > 0 else np.array([])


def mini_batch_spherical_kmeans_multiple_tables(
    lancedb_uris: list[str],
    table_names: list[str],
    embedding_column_name: str,
    n_clusters: int,
    batch_sizes: list[int],
    max_iters: int = 100,
    sample_rate: int = 0.1,
    tol: float = 1e-4,
    not_mini_batch: bool = False,
    stability_window: int = 5,  # Number of past iterations to check for stability
    stability_threshold_mean_change: float = 1e-5, # Threshold for mean change in updates
    stability_threshold_variance: float = 1e-6,    # Threshold for variance of updates
    seed: int = 42  # For reproducibility
):
    random_state = np.random.RandomState(seed)
    lance_tables = [lancedb.connect(lancedb_uri).open_table(table_name) for lancedb_uri, table_name in zip(lancedb_uris, table_names)]
    table_num_rows = [lance_table.count_rows() for lance_table in lance_tables]
    dataset_uris = [f"{lance_uri}/{table}.lance" for lance_uri, table in zip(lancedb_uris, table_names)]

    # 1. Initialization: Sample initial centroids
    initial_sample = []
    total_batch_size = sum(batch_sizes)
    init_data_used = []
    for dataset_uri, batch_size in zip(dataset_uris, batch_sizes):
        limit = int((batch_size/total_batch_size) * n_clusters) + 1
        init_data_used.append(limit)
        ds = lance.dataset(dataset_uri)
        scanner_table = ds.scanner(columns=[embedding_column_name], limit=limit, offset=0).to_table()
        batch_list = scanner_table.column(embedding_column_name).to_pylist()
        if len(batch_list)> 0:
            initial_sample.extend(batch_list)
    if not initial_sample:
        raise ValueError("No data found in the Lance tables.")
    
    initial_indices = random.sample(range(len(initial_sample)), min(n_clusters, len(initial_sample)))
    initial_sample_np = np.array(initial_sample)
    centroids = normalize(initial_sample_np[initial_indices], axis=1)

    prev_centroids = np.zeros_like(centroids)

    update_history = []
    
    offsets = init_data_used
    print(f"init_data_used: {init_data_used}")

    for iteration in range(max_iters):
        total_centroid_updates = 0
        combined_batch_embeddings = []

        if not_mini_batch:
            for dataset_uri, table_num_row in zip(dataset_uris, table_num_rows):
                limit = table_num_row
                ds = lance.dataset(dataset_uri)
                scanner_table = ds.scanner(columns=[embedding_column_name], limit=limit, offset=0).to_table()
                sampled_embeddings = np.array(scanner_table.column(embedding_column_name).to_pylist())
                combined_batch_embeddings.extend(sampled_embeddings)
        else:
            # Sample one batch from each table skipping the inital n_clusters, which is used to create the initial centroids
            for dataset_uri, table_num_row, batch_size, offset in zip(dataset_uris, table_num_rows, batch_sizes, offsets):
                limit = min(batch_size, table_num_row - offset)
                ds = lance.dataset(dataset_uri)
                scanner_table = ds.scanner(columns=[embedding_column_name], limit=limit, offset=offset).to_table()
                samples_per_table = int(limit*sample_rate)
                if scanner_table.shape[0] > 0:
                    # increasing size
                    # sampled_embeddings = np.array(scanner_table.column(embedding_column_name).to_pylist())
                    sampled_embeddings = get_sample_from_batch(scanner_table, samples_per_table, seed=random_state.randint(0, batch_size))
                    if sampled_embeddings.size > 0:
                        combined_batch_embeddings.extend(sampled_embeddings)
            offsets = [offset+batch_size for offset, batch_size in zip(offsets, batch_sizes)]

        if not combined_batch_embeddings:
            print("No data sampled in this iteration.")
            continue
        print(f"{len(combined_batch_embeddings)=}")
        combined_batch_embeddings = normalize(combined_batch_embeddings, axis=1)

        # Assignment
        similarity = cosine_similarity(combined_batch_embeddings, centroids)
        labels_batch = np.argmax(similarity, axis=1)

        # Update Centroids
        new_centroids_sum = np.zeros_like(centroids, dtype=np.float64)
        cluster_counts = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_points = combined_batch_embeddings[labels_batch == i]
            if len(cluster_points) > 0:
                new_centroids_sum[i] += np.sum(cluster_points, axis=0)
                cluster_counts[i] += len(cluster_points)

        new_centroids = np.where(cluster_counts[:, np.newaxis] > 0,
                                 normalize(new_centroids_sum, axis=1, norm='l2'),
                                 centroids)

        # Convergence Check
        centroid_updates = np.sum((new_centroids - centroids)**2, axis=1)
        total_centroid_updates = np.sum(centroid_updates)
        update_history.append(total_centroid_updates)
        centroids = new_centroids

        print(f"{datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')}" )
        print(f"Iteration {iteration + 1}, Total Centroid Update: {total_centroid_updates:.6f}")
        if total_centroid_updates < tol:
            print("Centroids converged.")
            break

        if np.allclose(centroids, prev_centroids, atol=tol):
            print("Centroids converged (allclose check).")
            break
        prev_centroids = centroids.copy()

         # Stability Check
        if len(update_history) >= stability_window:
            window = update_history[-stability_window:]
            mean_update = np.mean(window)
            variance_update = np.var(window)
            mean_change = np.abs(mean_update - (np.mean(update_history[-stability_window-1:-1]) if len(update_history) > stability_window else mean_update))

            print(f"  Mean update over last {stability_window}: {mean_update:.6f}, Variance: {variance_update:.8f}, Mean Change: {mean_change:.8f}")

            if mean_change < stability_threshold_mean_change and variance_update < stability_threshold_variance:
                print("Centroid updates stabilized.")
                break

    return centroids

if __name__ == "__main__":
    lance_files = ZOS_DB_PATHS
    table_names = ZOS_TABLE_NAMES
    cos_centroid_path = zos_cos_centroid_path
    embedding_column_name = EMBEDDING_COLUMN_NAME
    num_clusters_list = [100]
    batch_sizes = [2048, 4096]
    max_iterations = 170
    convergence_tolerance = 1e-4
    sample_rate = 1.0
    seeds =[42]
    not_mini_batch = False

    for seed in seeds:
        for num_clusters in num_clusters_list:
            final_centroids = mini_batch_spherical_kmeans_multiple_tables(
                lance_files, 
                table_names, 
                embedding_column_name, 
                num_clusters, 
                batch_sizes=batch_sizes, 
                max_iters=max_iterations, 
                sample_rate=sample_rate, 
                tol=convergence_tolerance,
                not_mini_batch=not_mini_batch,
                seed=seed
            )
            print("\nFinal Centroids:\n", final_centroids)

            store_centroids_pyarrow_npy_s3(final_centroids, s3, cos_centroid_path)

            print(f"seed: {seed} num_clusters: {num_clusters} max_iters: {max_iterations} sample_rate: {sample_rate}")
            if final_centroids is not None:
                silhouette_score_value = evaluate_silhouette(
                    lance_files,
                    table_names,
                    embedding_column_name,
                    final_centroids,
                    batch_sizes,
                    sample_size_for_silhouette=10000,  # Adjust sample size as needed
                    seed=seed
                )
            if silhouette_score_value is not None:
                print(f"\nSilhouette Score based on final centroids: {silhouette_score_value:.4f}")
            

