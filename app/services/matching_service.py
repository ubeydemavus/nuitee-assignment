"""
Service for room matching algorithm
"""
import copy
import logging
import numpy as np
import os
import pickle
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
from sentence_transformers import SentenceTransformer
from gliner import GLiNER
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure there's at least one handler (console output)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

logger.info("Logger initialized: INFO level")  # Test log

class UnionFind:
    """UnionFind data structure to handle synonym groups"""
    def __init__(self):
        self.parent = {}

    def find(self, word):
        if word not in self.parent:
            self.parent[word] = word
        if self.parent[word] != word:
            self.parent[word] = self.find(self.parent[word])  # Path compression
        return self.parent[word]

    def union(self, word1, word2):
        root1, root2 = self.find(word1), self.find(word2)
        if root1 != root2:
            self.parent[root2] = root1  # Merge sets

class RoomMatcher:
    """Service for matching hotel rooms"""
    
    def __init__(self):
        # Default parameters
        self.match_threshold = 0.1
        self.labels = ["roomType", "classType", "bedCount", "view", "bedType", "features"]
        self.label_importance_coefficients = {
            "roomType": 2./6.,
            "classType": 1./6.,
            "bedCount": 1./6.,
            "view": 1./6.,
            "bedType": 1./6.,
            "features": 1./6.,
        }
        self._initialize_models()
        self._load_data()
        
    def _initialize_models(self):
        """Initialize NLP models"""
        logger.info("Initializing NLP models...")
        logger.warning("First run will download models to disk. This will take some time...")
        
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load NER model
        self.ners_model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5", load_tokenizer=True)
        self.ners_model = self.ners_model.to(self.device)
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_model = self.embedding_model.to(self.device)
        
        logger.info("Models initialized successfully")
        
    def _load_data(self):
        """Load pre-trained data from pickle files"""
        logger.info("Loading pre-trained data...")
        
        try:
            # Set up paths to dataset files
            dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets')
            
            # Load reference and core room dataframes
            self.ref_df = pd.read_csv(os.path.join(dataset_dir, 'referance_rooms.csv'))
            self.core_df = pd.read_csv(os.path.join(dataset_dir, 'updated_core_rooms.csv'))
            
            # Load processed room data
            self.ref_d = pickle.load(open(os.path.join(dataset_dir, 'refrooms_processed.p'), 'rb'))
            self.core_d = pickle.load(open(os.path.join(dataset_dir, 'corerooms_processed.p'), 'rb'))
            
            # Load synonym dictionary and embeddings
            self.synonym_dict = pickle.load(open(os.path.join(dataset_dir, 'similar_dict.p'), 'rb'))
            self.embeds = pickle.load(open(os.path.join(dataset_dir, 'embeddings.p'), 'rb'))
            
            # Initialize UnionFind with synonym data
            self.union_find = UnionFind()
            self.build_synonym_groups(self.synonym_dict)
            
            logger.info("Pre-trained data loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load pre-trained data: {str(e)}")
            logger.warning("Falling back to empty dictionaries for demonstration purposes")
            
            # Fallback to empty dictionaries for demonstration
            self.ref_df = pd.DataFrame()
            self.core_df = pd.DataFrame()
            self.ref_d = []
            self.core_d = []
            self.synonym_dict = {}
            self.embeds = {}
            self.union_find = UnionFind()
        
    def build_synonym_groups(self, synonym_dict):
        """Build synonym groups using UnionFind"""
        for key, synonyms in synonym_dict.items():
            for synonym in synonyms:
                self.union_find.union(key, synonym)
        
    def are_similar(self, word1, word2):
        """Check if two words are similar based on synonym groups"""
        return (self.union_find.find(word1) == self.union_find.find(word2) 
                if word1 in self.union_find.parent and word2 in self.union_find.parent 
                else False)
    
    def extract_entities(self, room_name: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract named entities from room name"""
        preds = self.ners_model.predict_entities(room_name, self.labels)
        ners = {label: [] for label in self.labels}
        
        for entity in preds:
            ners[entity["label"]].append((
                entity["text"], 
                entity["score"], 
            ))
            
        # Also update embeddings dictionary
        words = [entity["text"] for entity in preds]
        embeddings = self.embedding_model.encode(words, normalize_embeddings=True)
        
        for word, embed in zip(words, embeddings):
            self.embeds[word] = embed.reshape(1, -1)
            
        return ners
    
    def get_final_score(self, scores_dict: Dict[str, float]) -> float:
        """Calculate final match score using weighted average"""
        total = 0
        for key, val in scores_dict.items():
            total = total + (val * self.label_importance_coefficients[key])
        return total
    
    def process_match_request(self, reference_catalog: List[Dict], input_catalog: List[Dict]) -> Dict[str, Any]:
        """Process room match request"""
        # Process reference rooms
        refs_retrieved = []
        for ref in reference_catalog:
            rhid = ref["hotel_id"]
            rlpid = ref["lp_id"]
            rrid = ref["room_id"]
            
            # Check if it exists in the reference database
            idx = self.ref_df.query(f"hotel_id == @rhid and lp_id == @rlpid and room_id == @rrid").index.to_list()
            dict_idx = idx[0] if idx else None
            if dict_idx is not None:
                ref_w_ners = self.ref_d[dict_idx]
            else:
                # If database not available, extract entities on the fly
                ref_w_ners = copy.deepcopy(ref)
                ref_w_ners["ners"] = self.extract_entities(ref["room_name"])
                
            refs_retrieved.append(ref_w_ners)
            
        # Process input rooms
        cats_retrieved = []
        for query in input_catalog:
            qhid = query["hotel_id"]
            qlpid = query["lp_id"]
            qsrid = query["supplier_room_id"]
            
            # Check if it exists in the core database
            idx = self.core_df.query(f"core_hotel_id == @qhid and lp_id == @qlpid and supplier_room_id == @qsrid").index.to_list()
            dict_idx = idx[0] if idx else None
            if dict_idx is not None:
                core_w_ners = self.core_d[dict_idx]
            else:
                # If database not available, extract entities on the fly
                core_w_ners = copy.deepcopy(query)
                core_w_ners["ners"] = self.extract_entities(query["supplier_room_name"])
                # Rename hotel_id to core_hotel_id for consistency
                if "hotel_id" in core_w_ners:
                    core_w_ners["core_hotel_id"] = core_w_ners.pop("hotel_id")
                
            cats_retrieved.append(core_w_ners)
            
        # Match rooms
        response_dict = {}
        matched_query_idx = []
        
        for idx, r in enumerate(refs_retrieved):
            rhid = r["hotel_id"]
            rlpid = r["lp_id"]
            rrid = r["room_id"]
            rname = r["room_name"]
            rners = r["ners"]
            
            for qidx, q in enumerate(cats_retrieved):
                qhid = q["core_hotel_id"]
                qlpid = q["lp_id"]
                qsrid = q["supplier_room_id"]
                qsname = q["supplier_name"]
                qsrname = q["supplier_room_name"]
                qners = q["ners"]
                
                # First check if lp id match or name match, if not go into score calculation
                if qlpid.lower() == rlpid.lower() or qsrname.lower() == rname.lower():
                    if idx not in response_dict:
                        response_dict[idx] = []
                    q_dp = copy.deepcopy(q)
                    q_dp["reference_match_score"] = 1.0
                    response_dict[idx].append(q_dp)
                    matched_query_idx.append(qidx)
                    logger.info(f"'{rname}' is similar to '{qsrname}' with score of {1.0}")
                else:
                    # Calculate score
                    scores = {}
                    for rnerkey, rnerlist in rners.items():
                        qnerlist = qners[rnerkey]
                        scores_across_entity_category = []
                        
                        for rentity in rnerlist:
                            for qentity in qnerlist:
                                if self.are_similar(rentity[0], qentity[0]):
                                    score = max(rentity[1], qentity[1])
                                    scores_across_entity_category.append(score)
                                else:
                                    score = min(rentity[1], qentity[1])
                                    similarity = cosine_similarity(self.embeds[rentity[0]], self.embeds[qentity[0]])[0][0]
                                    score = score * similarity
                                    scores_across_entity_category.append(score.item())
                                        
                        if scores_across_entity_category==[]:
                            scores_across_entity_category = 0.0
                        else:
                            scores_across_entity_category = np.mean(scores_across_entity_category)
                            
                        scores[rnerkey] = float(scores_across_entity_category)

                        
                    final_score = self.get_final_score(scores)
                    if final_score >= self.match_threshold:
                        if idx not in response_dict:
                            response_dict[idx] = []
                        q_dp = copy.deepcopy(q)
                        q_dp["reference_match_score"] = final_score
                        response_dict[idx].append(q_dp)
                        matched_query_idx.append(qidx)

                    logger.info(f"'{rname}' is similar to '{qsrname}' with score of {final_score}")
        # Prepare response
        unmatched_query_idx = [idx for idx in range(len(cats_retrieved)) if idx not in matched_query_idx]
        
        result = {
            "results": [],  # --> will have the mapped rooms, ref room then matched rooms
            "unmapped_rooms": []
        }
        
        for idx, r in enumerate(refs_retrieved):
            matches = response_dict.get(idx)
            if matches:
                result["results"].append({
                    "room_name": r["room_name"],
                    "room_id": r["room_id"],
                    "hotel_id": r["hotel_id"],
                    "lp_id": r["lp_id"],
                    "mapped_rooms": matches
                })
                
        result["unmapped_rooms"] = [cats_retrieved[idx] for idx in unmatched_query_idx]
        
        return result 