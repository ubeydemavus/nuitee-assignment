import pandas as pd 
import numpy as np
from tqdm import tqdm
import torch
import math
import pickle
import copy
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from gliner import GLiNER
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
import umap
import random

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the lowest level to capture
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Ensures logs go to the console
)

# Config
batch_size = 256  # Adjust batch size as needed
labels = ["roomType", "classType", "bedCount", "view", "bedType", "features"] # if changed, self.labels and self.label_importance_coefficients attributes of RoomMatcher class need to be changed as well. In addition, this pre-training step needs to run to completion!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cpu":
    logging.warning("Running on CPU will take a lot longer. Run this script with a CUDA enabled GPU.")
else:
    logging.info("Running with CUDA. Make sure batch size won't overflow memory!")

logging.info("Initializing models.")
model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5", load_tokenizer=True)
model = model.to(device)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_model = embedding_model.to(device)
logging.info("Models are initialized succesfully!")

logging.info("Reading datasets.")
ref_df = pd.read_csv("./datasets/referance_rooms.csv")
core_df = pd.read_csv("./datasets/updated_core_rooms.csv")
logging.info("Datasets are loaded to the memory")


logging.info("Running Named Entity Recognition model on referance_rooms.csv dataset.")
# Extract room names to process in batches
room_names = ref_df.room_name.tolist()

refrooms = []

for i in tqdm(range(0, len(room_names), batch_size)):
    batch_room_names = room_names[i:i + batch_size]
    # Replace NaN values with an empty string
    batch_room_names = ["" if isinstance(name, float) and math.isnan(name) else name for name in batch_room_names]
    batch_room_names = [name.lower() for name in batch_room_names]

    batch_entities = model.batch_predict_entities(batch_room_names, labels)
    
    for row, entities in zip(ref_df.iloc[i:i + batch_size].itertuples(index=False), batch_entities):
        ners = {label: [] for label in labels}
        
        for entity in entities:
            ners[entity["label"]].append((
                entity["text"], 
                entity["score"], 
                ))
        
        room_item = {
            "hotel_id": row.hotel_id,
            "lp_id": row.lp_id,
            "room_id": row.room_id,
            "room_name": row.room_name,
            "ners": ners
        }
        
        refrooms.append(room_item)

logging.info("Storing refrooms_processed.p to the disk.")
pickle.dump(refrooms,open("./datasets/refrooms_processed.p","wb"))
logging.info("refrooms_processed.p is stored to the disk successfully!")


logging.info("Running Named Entity Recognition model on updated_core_rooms.csv dataset.")
# Extract room names to process in batches
room_names = core_df.supplier_room_name.tolist()

corerooms = []

for i in tqdm(range(0, len(room_names), batch_size)):
    batch_room_names = room_names[i:i + batch_size]
    # Replace NaN values with an empty string
    batch_room_names = ["" if isinstance(name, float) and math.isnan(name) else name for name in batch_room_names]
    batch_room_names = [name.lower() for name in batch_room_names]

    batch_entities = model.batch_predict_entities(batch_room_names, labels)
    
    for row, entities in zip(core_df.iloc[i:i + batch_size].itertuples(index=False), batch_entities):
        ners = {label: [] for label in labels}
        
        for entity in entities:
            ners[entity["label"]].append((
                entity["text"], 
                entity["score"], 
                ))
            
        room_item = {
            "core_room_id":row.core_room_id,
            "core_hotel_id":row.core_hotel_id,
            "lp_id":row.lp_id,
            "supplier_room_id":row.supplier_room_id,
            "supplier_name":row.supplier_name,
            "supplier_room_name":row.supplier_room_name,
            "ners":ners
        }
        
        corerooms.append(room_item)

logging.info("Storing corerooms_processed.p to the disk.")
pickle.dump(corerooms,open("./datasets/corerooms_processed.p","wb"))
logging.info("corerooms_processed.p is stored to the disk successfully!")

core_rooms = corerooms
ref_rooms = refrooms

s = set()
for idx,item in enumerate(core_rooms):
    ner = item["ners"]
    for rtype,score in ner["roomType"]:
        s.add(rtype)
    for ctype,score in ner["classType"]:
        s.add(ctype)
    for bcount,score in ner["bedCount"]:
        s.add(bcount)
    for vtype,score in ner["view"]:
        s.add(vtype)
    for btype,score in ner["bedType"]:
        s.add(btype)
    for ftype,score in ner["features"]:
        s.add(ftype)

for idx,item in enumerate(ref_rooms):
    ner = item["ners"]
    for rtype,score in ner["roomType"]:
        s.add(rtype)
    for ctype,score in ner["classType"]:
        s.add(ctype)
    for bcount,score in ner["bedCount"]:
        s.add(bcount)
    for vtype,score in ner["view"]:
        s.add(vtype)
    for btype,score in ner["bedType"]:
        s.add(btype)
    for ftype,score in ner["features"]:
        s.add(ftype)

logging.info("Generating embeddings for NER labels.")
# Sample words/phrases
words = list(s)

# Step 1: Load pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Encode the words into embeddings
embeddings = model.encode(words,normalize_embeddings=True)


logging.info("Running Clustering algorithm on NERs phrases. This will take a long time.")
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ["LOKY_MAX_CPU_COUNT"] = "16"

# Step 3: Elbow Method for optimal number of clusters
wcss = []  # Within-cluster sum of squares
for i in range(1, 300):  
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(embeddings)
    wcss.append(kmeans.inertia_)  # inertia_ is the WCSS
    print(f"WCSS for {i} clusters: {kmeans.inertia_}")  # Debugging line to check WCSS

print(f"Total WCSS values: {len(wcss)}")

# Step 4: Plot the elbow curve
plt.plot(range(1, len(wcss) + 1), wcss)  # Ensure the x-axis matches the number of WCSS values
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


logging.info("Running Clustering using optimal_k=200. Change this value based on elbow method.")
optimal_k = 200
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(embeddings)

# Step 6: Output the clusters
clusters = {}
for word, cluster in zip(words, kmeans.labels_):
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append(word)

# Print the words in each cluster
logging.info("Example clusters with example similar words:")
count = 0 
for cluster_idx, words_in_cluster in clusters.items():
    print(f"Cluster {cluster_idx}:")
    print(", ".join(words_in_cluster[:5]))
    print("-" * 50)
    count = count + 1 
    if count >= 10:
        break

logging.info("Refining Clusters (eliminating non-similar words with similarity threshold being 0.7, adjust experimentally)")
# Step 1: Normalize the Embeddings
normalized_embeddings = embeddings #normalize(embeddings, norm='l2')

# Step 2: Compute Cosine Similarity and Filter Outliers
cluster_centroids = normalize(kmeans.cluster_centers_, norm='l2')
threshold = 0.7  # Adjust based on experiments

refined_clusters = {}
filtered_words = []
filtered_embeddings = []

for word, cluster, embedding in zip(words, kmeans.labels_, normalized_embeddings):
    similarity = cosine_similarity([embedding], [cluster_centroids[cluster]])[0][0]

    if similarity >= threshold:  # Keep words above the threshold
        if cluster not in refined_clusters:
            refined_clusters[cluster] = []
        refined_clusters[cluster].append(word)
        
        filtered_words.append(word)
        filtered_embeddings.append(embedding)

# Convert to NumPy array
filtered_embeddings = np.array(filtered_embeddings)

# Step 3: Apply UMAP for Dimensionality Reduction
umap_model = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, n_jobs=-1)
umap_embeddings = umap_model.fit_transform(filtered_embeddings)

# Number of words to annotate (adjust as needed)
num_annotations = 50  # Adjust to how many random words you want to annotate

# Randomly select `num_annotations` words to annotate
selected_indices = random.sample(range(len(filtered_words)), num_annotations)

# Visualize the Retained Words (UMAP Plot)
plt.figure(figsize=(20, 20))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1],s=0.5, marker='o', color='green', alpha=0.3, label="Retained Words")

# Annotate only the selected random words
for i in selected_indices:
    plt.annotate(filtered_words[i], (umap_embeddings[i, 0], umap_embeddings[i, 1]), fontsize=9, alpha=0.75)

# Step 4: Annotate Cluster Centers with One Word from Each Cluster
for cluster_idx, cluster_center in enumerate(cluster_centroids):
    # Find the word closest to the cluster center
    cluster_embeddings = [embedding for word, cluster, embedding in zip(filtered_words, kmeans.labels_, filtered_embeddings) if cluster == cluster_idx]
    cluster_embeddings = np.array(cluster_embeddings)

    # Compute cosine similarities between the cluster center and words in the cluster
    similarities = cosine_similarity([cluster_center], cluster_embeddings)[0]
    
    # Find the word with the highest similarity to the cluster center
    closest_word_idx = np.argmax(similarities)
    closest_word = filtered_words[[idx for idx, cluster in enumerate(kmeans.labels_) if cluster == cluster_idx][closest_word_idx]]

    # Annotate the cluster center on the UMAP plot with the closest word
    plt.annotate(closest_word, (umap_embeddings[[idx for idx, cluster in enumerate(kmeans.labels_) if cluster == cluster_idx][closest_word_idx], 0],
                               umap_embeddings[[idx for idx, cluster in enumerate(kmeans.labels_) if cluster == cluster_idx][closest_word_idx], 1]),
                 fontsize=8, color='blue', weight='bold', alpha=0.8)

    # Scatter the cluster centers
    plt.scatter(umap_embeddings[[idx for idx, cluster in enumerate(kmeans.labels_) if cluster == cluster_idx][closest_word_idx], 0],
                umap_embeddings[[idx for idx, cluster in enumerate(kmeans.labels_) if cluster == cluster_idx][closest_word_idx], 1],
                color='blue', s=2, marker='*', label=f'Cluster {cluster_idx} Center')

plt.title("UMAP Visualization of Retained Word Clusters with Cluster Centers")
plt.show()

# Print the words in each cluster
logging.info("Refined clusters with example similar words:")
count = 0 
for cluster_idx, words_in_cluster in refined_clusters.items():
    print(f"Cluster {cluster_idx}:")
    print(", ".join(words_in_cluster[:10]))
    print("-" * 50)
    count = count + 1 
    if count >= 10:
        break

logging.info("Generating and storing synonms dictionary.")
synonym_dict = {}
for cluster_idx, words_in_cluster in refined_clusters.items():
    words_in_cluster = [item.lower() for item in words_in_cluster]
    synonym_dict[words_in_cluster[0]] = set(words_in_cluster[1:])
pickle.dump(synonym_dict,open("./datasets/similar_dict.p","wb"))

logging.info("Storing embeddings for unique NERs for retrieval.")
embeddings_dict = {}
for word, embedding in zip(words,embeddings):
    embeddings_dict[word] = embedding.reshape(1,-1)
pickle.dump(embeddings_dict,open("./datasets/embeddings.p","wb"))

logging.info("Dataset transformation is successfully completed! You can now launch the room matching app!")
