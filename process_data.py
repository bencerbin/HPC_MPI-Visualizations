from mpi4py import MPI
import pandas as pd
import os
from collections import Counter
import networkx as nx
import pickle

# -----------------------------
# Main MPI Processing (OPTIMIZED)
# -----------------------------

def process_imdb_data(comm, config):
    rank = comm.Get_rank()
    size = comm.Get_size()

    data_path = config["data_path"]

    def log(msg):
        print(f"[Rank {rank}] {msg}", flush=True)

    # -----------------------------
    # Load ratings (small → replicate)
    # -----------------------------
    if rank == 0:
        print("\nLoading ratings...", flush=True)

    ratings = pd.read_csv(
        os.path.join(data_path, "title.ratings.tsv"),
        sep="\t",
        usecols=["tconst", "averageRating"]
    )

    log(f"Loaded ratings: {len(ratings)} rows")

    # -----------------------------
    # Process BASICS using chunking
    # -----------------------------
    basics_file = os.path.join(data_path, "title.basics.tsv")

    chunk_size = 200_000  # tune this if needed

    local_top_list = []
    local_counter = Counter()

    log("Starting basics processing...")

    for i, chunk in enumerate(pd.read_csv(
        basics_file,
        sep="\t",
        chunksize=chunk_size,
        usecols=["tconst", "primaryTitle", "startYear", "genres"]
    )):
        # Distribute chunks across ranks
        if i % size != rank:
            continue

        if i % 10 == 0:
            log(f"Processing basics chunk {i}")

        # Filter fantasy
        fantasy = chunk[chunk["genres"].str.contains("Fantasy", na=False)].copy()

        # Clean year
        fantasy["startYear"] = pd.to_numeric(fantasy["startYear"], errors="coerce")
        fantasy = fantasy.dropna(subset=["startYear"])
        fantasy["startYear"] = fantasy["startYear"].astype(int)

        # Join ratings
        fantasy = fantasy.merge(ratings, on="tconst", how="inner")

        if len(fantasy) == 0:
            continue

        # ---- Task 1 (local) ----
        local_top_list.append(fantasy)

        # ---- Task 2 (local) ----
        for genres in fantasy["genres"].dropna():
            split_genres = genres.split(",")
            if "Fantasy" in split_genres:
                for g in split_genres:
                    if g != "Fantasy":
                        local_counter[g] += 1

    log("Finished basics processing")

    # -----------------------------
    # Process PRINCIPALS using chunking
    # -----------------------------
    principals_file = os.path.join(data_path, "title.principals.tsv")

    local_edges = []

    log("Starting principals processing...")

    for i, chunk in enumerate(pd.read_csv(
        principals_file,
        sep="\t",
        chunksize=chunk_size,
        usecols=["tconst", "nconst", "category"]
    )):
        if i % size != rank:
            continue

        if i % 10 == 0:
            log(f"Processing principals chunk {i}")

        chunk = chunk[
            chunk["category"].isin(["director", "actor", "actress"])
        ]

        for _, group in chunk.groupby("tconst"):
            directors = group[group["category"] == "director"]["nconst"].tolist()
            actors = group[group["category"].isin(["actor", "actress"])]["nconst"].tolist()

            for d in directors:
                for a in actors:
                    local_edges.append((d, a))

    log("Finished principals processing")

    # -----------------------------
    # GATHER RESULTS
    # -----------------------------
    log("Gathering results...")

    all_top = comm.gather(local_top_list, root=0)
    all_counts = comm.gather(local_counter, root=0)
    all_edges = comm.gather(local_edges, root=0)

    if rank == 0:
        print("\nCombining results...", flush=True)

        results = {}

        # ---- Task 1: All fantasy movies ----
        combined = pd.concat([df for sublist in all_top for df in sublist])
        results["fantasy_movies"] = combined

        # ---- Task 2: Genre counts ----
        global_counter = Counter()
        for c in all_counts:
            global_counter.update(c)

        genre_df = pd.DataFrame(
            global_counter.items(),
            columns=["Genre", "Count"]
        ).sort_values(by="Count", ascending=False)

        results["genre_counts"] = genre_df

        # ---- Task 3: Graph ----
        edges = [e for sublist in all_edges for e in sublist]
        edges = edges[:1000]

        G = nx.Graph()
        G.add_edges_from(edges)

        results["graph"] = G

        print("Done.\n", flush=True)

        return results

    return None


# -----------------------------
# Run Script
# -----------------------------

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    config = {
        "data_path": "./data"
    }

    results = process_imdb_data(comm, config)

    if comm.Get_rank() == 0:

        with open("results.pkl", "wb") as f:
            pickle.dump(results, f)

            print("Saved results to results.pkl", flush=True)
        
        print("\nTop Fantasy Movies Per Year:")
        print(results["fantasy_movies"].head())

        print("\nTop Genre Pairings with Fantasy:")
        print(results["genre_counts"].head())

        print("\nGraph info:")
        print(results["graph"])