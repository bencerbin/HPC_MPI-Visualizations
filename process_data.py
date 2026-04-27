from mpi4py import MPI
import pandas as pd
import os
from collections import Counter
import networkx as nx
import pickle
import random

# -----------------------------
# Main MPI Processing 
# -----------------------------

def process_imdb_data(comm, config):
    rank = comm.Get_rank()
    size = comm.Get_size()

    data_path = config["data_path"]

    def log(msg):
        print(f"[Rank {rank}] {msg}", flush=True)

    # -----------------------------
    # Load ratings
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

    local_valid_ids = set()
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

        high_rating = fantasy[fantasy["averageRating"] >= 7]

        local_valid_ids.update(high_rating["tconst"].tolist())

        # ---- Task 2 (local) ----
        for genres in fantasy["genres"].dropna():
            split_genres = genres.split(",")
            if "Fantasy" in split_genres:
                for g in split_genres:
                    if g != "Fantasy":
                        local_counter[g] += 1

    log("Finished basics processing")

    all_valid_ids = comm.gather(local_valid_ids, root=0)
    
    if rank == 0:
        valid_ids = set().union(*all_valid_ids)
    
    valid_ids = comm.bcast(valid_ids if rank == 0 else None, root=0)

    # -----------------------------
    # Process PRINCIPALS using chunking
    # -----------------------------
    principals_file = os.path.join(data_path, "title.principals.tsv")

    from collections import defaultdict
    local_edges = defaultdict(list)

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
            chunk["category"].isin(["director", "actor", "actress"]) &
            chunk["tconst"].isin(valid_ids)
]

        for _, group in chunk.groupby("tconst"):
            directors = group[group["category"] == "director"]["nconst"].tolist()
            actors = group[group["category"].isin(["actor", "actress"])]["nconst"].tolist()

            for d in directors:
                for a in actors:
                    local_edges[d].append((d, a))

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
        
        # 1. Merge director → edges mappings from all ranks
        director_edges = defaultdict(list)
        for d in all_edges:
            for director, edges_list in d.items():
                director_edges[director].extend(edges_list)
        
        # 2. Randomly sample directors
        all_directors = list(director_edges.keys())
        sample_size = min(100, len(all_directors))  # adjust this number as needed
        sampled_directors = random.sample(all_directors, sample_size)
        
        # 3. Collect ALL edges from sampled directors
        edges = []
        for d in sampled_directors:
            edges.extend(director_edges[d])
        
        # 4. Collect unique IDs for naming
        needed_ids = set()
        for d, a in edges:
            needed_ids.add(d)
            needed_ids.add(a)
        
        # 5. Load names
        name_file = os.path.join(data_path, "name.basics.tsv")
        name_dict = {}
        
        for chunk in pd.read_csv(
            name_file,
            sep="\t",
            chunksize=200_000,
            usecols=["nconst", "primaryName"]
        ):
            filtered = chunk[chunk["nconst"].isin(needed_ids)]
            name_dict.update(dict(zip(filtered["nconst"], filtered["primaryName"])))
        
            if len(name_dict) == len(needed_ids):
                break
        
        # 6. Replace IDs with names
        named_edges = [
            (name_dict.get(d, d), name_dict.get(a, a))
            for d, a in edges
        ]
        
        # 7. Build graph
        G = nx.Graph()
        G.add_edges_from(named_edges)
        
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