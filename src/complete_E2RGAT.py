#!/usr/bin/env python3
"""
Complete E2RGAT (Edge-Enhanced Relational Graph Attention) Implementation
This code provides a complete framework for disease prediction using
relational graph attention networks via edge embeddings with temporal patient data.
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import warnings
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm
import re
from datetime import datetime
from collections import defaultdict



# Read data

class DataProcessor:
    """Data processing utilities for E2RGAT with static and longitudinal data"""
    
    @staticmethod
    def parse_vector(s):
        """Parse vector string to numpy array"""
        if isinstance(s, str):
            nums = re.findall(r"[-+]?\d*\.\d+e[-+]\d+|[-+]?\d+\.\d+|[-+]?\d+", s)
            return np.array([float(num) for num in nums])
        return s
    
    @staticmethod
    def load_static_data(static_file_path: str):
        """
        Load static patient information (preprocessed data, no NA values)
        
        Args:
            static_file_path: Path to static data file containing patient ID, age, sex
            
        Returns:
            DataFrame with static patient information
        """
        print(f"Loading static data from {static_file_path}...")
        
        try:
            static_df = pd.read_csv(static_file_path)
            
            # Ensure required columns exist
            required_columns = ['patient_id', 'age', 'sex']
            missing_columns = [col for col in required_columns if col not in static_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in static file: {missing_columns}")
            
            print(f"Static data loaded: {len(static_df)} patients")
            
            return static_df
            
        except Exception as e:
            print(f"Error loading static data: {e}")
            return None
    
    @staticmethod
    def load_longitudinal_data(longitudinal_file_path: str):
        """
        Load longitudinal visit data with ICD codes and disease labels
        
        Args:
            longitudinal_file_path: Path to longitudinal data file containing:
                patient_id, visit_date, icd_codes, disease_labels
                
        Returns:
            DataFrame with longitudinal visit information
        """
        print(f"Loading longitudinal data from {longitudinal_file_path}...")
        
        try:
            long_df = pd.read_csv(longitudinal_file_path)
            
            # Ensure required columns exist
            required_columns = ['patient_id', 'visit_date', 'is_last_visit', 'ICD10', 'disease_label']
            missing_columns = [col for col in required_columns if col not in long_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in longitudinal file: {missing_columns}")
            
            # Convert visit_date to datetime (assuming data is already clean)
            long_df['visit_date'] = pd.to_datetime(long_df['visit_date'])
            
            # Sort by patient ID and visit date
            long_df = long_df.sort_values(['patient_id', 'visit_date']).reset_index(drop=True)
            
            # Process disease label
            if 'disease_label' in long_df.columns:
                # Convert to numeric (assuming data is already clean)
                long_df['disease_label'] = pd.to_numeric(long_df['disease_label'])
                disease_columns = ['disease_label']
                print(f"Found disease label column: disease_label")
            else:
                raise ValueError("Required column 'disease_label' not found in longitudinal data")
            
            return long_df, disease_columns
            
        except Exception as e:
            print(f"Error loading longitudinal data: {e}")
            return None, []
    
    @staticmethod
    def create_visit_features(long_df: pd.DataFrame, static_df: pd.DataFrame, disease_columns: List[str]):
        """
        Create visit features from ICD codes and static information
        
        Args:
            long_df: Longitudinal data DataFrame
            static_df: Static data DataFrame
            disease_columns: List of disease label column names
            
        Returns:
            DataFrame with visit features
        """
        
        # Get all unique ICD codes
        all_icd_codes = set()
        for icd_list in long_df['icd_list']:
            all_icd_codes.update(icd_list)
        
        icd_codes_list = sorted(list(all_icd_codes))
        print(f"Found {len(icd_codes_list)} unique ICD codes")
        
        # Create ICD code features
        visit_features = []
        
        for _, visit in long_df.iterrows():
            patient_id = visit['patient_id']
            visit_date = visit['visit_date']
            icd_codes = visit['icd_list']
            
            # Create ICD code vector (one-hot encoding)
            icd_vector = [1 if icd in icd_codes else 0 for icd in icd_codes_list]
            
            # Get static features (all patients have age and sex information)
            static_info = static_df[static_df['patient_id'] == patient_id]
            age = static_info.iloc[0]['age']
            sex = static_info.iloc[0]['sex']
            
            # Combine features (age is already in desired format)
            visit_vector = icd_vector + [age, sex]
            
            # Get disease label (single column)
            disease_label = visit['disease_label']
            
            visit_features.append({
                'patient_id': patient_id,
                'visit_date': visit_date,
                'visit_vector': visit_vector,
                'icd_codes': icd_codes,
                'disease_label': disease_label
            })
        
        features_df = pd.DataFrame(visit_features)
        print(f"Created {len(features_df)} visit features")
        print(f"Feature dimension: {len(features_df['visit_vector'].iloc[0])}")
        
        return features_df, icd_codes_list
    
    @staticmethod
    def load_and_process_data(static_file_path: str, longitudinal_file_path: str, 
                            kinship_file_path: str = None):
        """
        Load and process both static and longitudinal data
        
        Args:
            static_file_path: Path to static patient data file
            longitudinal_file_path: Path to longitudinal visit data file
            kinship_file_path: Path to kinship data file (optional)
            
        Returns:
            Tuple of (processed_data, graph, disease_columns, icd_codes_list)
        """
        print("Loading and processing data...")
        
        # Load static data
        static_df = DataProcessor.load_static_data(static_file_path)
        if static_df is None:
            raise ValueError("Failed to load static data")
        
        # Load longitudinal data
        long_df, disease_columns = DataProcessor.load_longitudinal_data(longitudinal_file_path)
        if long_df is None:
            raise ValueError("Failed to load longitudinal data")
        
        # Create visit features
        features_df, icd_codes_list = DataProcessor.create_visit_features(long_df, static_df, disease_columns)
        
        # Prepare data for graph building
        processed_data = []
        for _, row in features_df.iterrows():
            processed_data.append({
                'eid': row['patient_id'],
                'event_dt': row['visit_date'],
                'full_visit_vector': np.array(row['visit_vector']),
                'has_disease': row['disease_label'],
                'familyid': row['patient_id'] // 1000  # Simple family ID assignment
            })
        
        processed_df = pd.DataFrame(processed_data)
        print(f"Processed data: {len(processed_df)} records, {processed_df['eid'].nunique()} unique patients")
        
        # Build graph
        G, grouped = DataProcessor.build_graph(processed_df, kinship_file_path)
        
        return processed_df, G, disease_columns, icd_codes_list
    
    @staticmethod
    def build_graph(df: pd.DataFrame, kinship_file: str = None):
        """Build NetworkX graph from processed data"""
        print("Building graph...")
        
        # Group by patient
        grouped = df.groupby("eid").agg({
            "full_visit_vector": list,
            "event_dt": list,
            "has_disease": "max",
            "familyid": "first"
        })
        
        # Calculate padding length
        max_visits = grouped["full_visit_vector"].apply(len).max()
        vector_dim = len(df["full_visit_vector"].iloc[0])
        
        print(f"Max visits: {max_visits}, Vector dimension: {vector_dim}")
        
        # Build graph
        G = nx.Graph()
        
        for eid, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Adding nodes"):
            visits = row["full_visit_vector"]
            times = row["event_dt"]
            
            # Padding vectors
            padded_vectors = visits + [np.zeros(vector_dim)] * (max_visits - len(visits))
            padded_vectors = np.stack(padded_vectors)
            
            # Padding times
            padded_times = times + [pd.NaT] * (max_visits - len(times))
            
            G.add_node(
                eid,
                visit_matrix=padded_vectors,
                visit_dates=padded_times,
                has_disease=row["has_disease"],
                familyid=row["familyid"]
            )
        
        # Add edges based on kinship (if kinship file provided)
        if kinship_file:
            DataProcessor.add_kinship_edges(G, kinship_file)
        else:
            # Create synthetic family edges
            DataProcessor.add_synthetic_family_edges(G)
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(G))
        if isolated_nodes:
            print(f"Removing {len(isolated_nodes)} isolated nodes")
            G.remove_nodes_from(isolated_nodes)
    
        return G, grouped
    
    @staticmethod
    def add_kinship_edges(G: nx.Graph, kinship_file: str):
        """Add edges based on kinship relationships"""
        print("Adding kinship edges...")
        
        try:
            kin = pd.read_csv(kinship_file, sep=" ", header=0)[["ID1", "ID2", "Kinship"]]
        except:
            print("Kinship file not found, creating synthetic edges based on family ID")
            # Create synthetic edges based on family ID
            family_groups = {}
            for node, data in G.nodes(data=True):
                familyid = data['familyid']
                if familyid not in family_groups:
                    family_groups[familyid] = []
                family_groups[familyid].append(node)
            
            edges_added = 0
            for familyid, nodes in family_groups.items():
                if len(nodes) > 1:
                    for i, node1 in enumerate(nodes):
                        for node2 in nodes[i+1:]:
                            # Create edge vector based on family relationship
                            edge_vector = [1, 0, 0, 0.5]  # Synthetic kinship
                            G.add_edge(node1, node2, edge_vector=edge_vector, kin_degree=0)
                            edges_added += 1
            
            print(f"Added {edges_added} synthetic family edges")
            return
        
        def get_edge_vector(kinship):
            # Patients are connected if kinship coefficient >= 0.0442
            if kinship >= 0.0442:
                # 1st degree relation: kinship >= 0.117
                if kinship >= 0.117:
                    return [1, 0, 0, kinship], 0
                # 2nd degree relation: kinship >= 0.0884
                elif kinship >= 0.0884:
                    return [0, 1, 0, kinship], 1
                # 3rd degree relation: kinship >= 0.0442
                else:
                    return [0, 0, 1, kinship], 2
            else:
                return None, None
        
        existing_eids = set(G.nodes())
        edges_added = 0
        
        for _, row in tqdm(kin.iterrows(), total=len(kin), desc="Adding kinship edges"):
            id1, id2 = int(row["ID1"]), int(row["ID2"])
            edge_vec, deg = get_edge_vector(row["Kinship"])
            
            if edge_vec is not None and id1 in existing_eids and id2 in existing_eids:
                G.add_edge(id1, id2, edge_vector=edge_vec, kin_degree=deg)
                edges_added += 1
                

# Custom Loss Functions

class CustomLossFunctions:
    
    @staticmethod
    def info_nce_loss(h: torch.Tensor, pos_pairs: torch.Tensor, neg_pairs: torch.Tensor, 
                     relation_types: torch.Tensor = None, learnable_temperature: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
        """
        Extended contrastive loss implementation with shared learnable temperature
        
        Formula: -∑_r ∑_i log(∑_j exp(sim(hi,hj+)/T) / ∑_k exp(sim(hi,hk)/T))
        
        Where:
        - r indexes over relation types (1st, 2nd, 3rd degree)
        - i indexes over anchor nodes
        - j indexes over positive samples of relation type r
        - k indexes over all samples (positive + negative)
        - T is shared learnable temperature for all relation types
        
        Args:
            h: (N, F) Node representations
            pos_pairs: (2, P) Positive sample pairs (closer relatives)
            neg_pairs: (2, K) Negative sample pairs (distant relatives)
            relation_types: (P,) Relation types for positive pairs (0=1st, 1=2nd, 2=3rd degree)
            learnable_temperature: (1,) Shared learnable temperature parameter for all relations
            eps: Numerical stability parameter
            
        Returns:
            Extended InfoNCE loss tensor
        """
        if pos_pairs.size(1) == 0:
            return torch.tensor(0.0, device=h.device, requires_grad=True)
        
        # Use shared learnable temperature if provided, otherwise use default
        if learnable_temperature is not None:
            temperature = learnable_temperature
        else:
            # Default temperature if learnable one not provided
            temperature = torch.tensor(0.2, device=h.device)
        
        # Calculate similarity with shared temperature
        def sim(a, b):
            return F.cosine_similarity(a, b, dim=1) / temperature
        
        # Group positive pairs by relation type
        relation_groups = defaultdict(lambda: {'anchors': [], 'positives': []})
        
        for i in range(pos_pairs.size(1)):
            anchor = pos_pairs[0, i].item()
            pos = pos_pairs[1, i].item()
            rel_type = relation_types[i].item() if relation_types is not None else 0
            
            relation_groups[rel_type]['anchors'].append(anchor)
            relation_groups[rel_type]['positives'].append(pos)
        
        # Collect all negative samples
        all_neg_indices = []
        for i in range(neg_pairs.size(1)):
            neg = neg_pairs[1, i].item()
            all_neg_indices.append(neg)
        
        total_loss = 0.0
        num_valid_anchors = 0
        
        # Process each relation type separately using shared temperature
        for rel_type, group_data in relation_groups.items():
            if not group_data['anchors']:
                continue
                
            anchors = group_data['anchors']
            positives = group_data['positives']
            
            # Process each anchor for this relation type
            for anchor, pos in zip(anchors, positives):
                # Calculate similarity with positive sample
                anchor_emb = h[anchor:anchor+1]  # (1, F)
                pos_sim = sim(anchor_emb, h[pos:pos+1])  # (1,)
                
                # Calculate similarity with all negative samples
                if all_neg_indices:
                    neg_indices = torch.tensor(all_neg_indices, device=h.device)
                    neg_sims = sim(anchor_emb.expand(len(neg_indices), -1), h[neg_indices])  # (num_neg,)
                    
                    # Combine positive and negative similarities
                    all_sims = torch.cat([pos_sim, neg_sims])  # (1 + num_neg,)
                else:
                    all_sims = pos_sim  # Only positive sample
                
                # Numerical stability: subtract maximum value
                max_sim = torch.max(all_sims)
                all_sims_stable = all_sims - max_sim
                
                # Calculate denominator log-sum-exp
                denominator_log = torch.logsumexp(all_sims_stable, dim=0)
                
                # Calculate numerator log-sum-exp (only positive samples)
                numerator_log = (pos_sim - max_sim).squeeze()
                
                # Calculate InfoNCE loss for this anchor-positive pair
                loss_item = -(numerator_log - denominator_log)
                total_loss += loss_item
                num_valid_anchors += 1
        
        return total_loss / max(num_valid_anchors, 1)
    
    @staticmethod
    def family_loss(h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, 
                   margin: float = 0.5) -> torch.Tensor:
        """
        Improved Family Loss: Make node representations more similar within families
        
        Args:
            h: (N, F) Node representations
            edge_index: (2, E) Edge indices
            edge_type: (E,) Edge types, 0 for direct relatives, 1 for other relatives
            margin: Margin parameter for contrastive learning
        """
        if edge_type is None or len(edge_type) == 0:
            return torch.tensor(0.0, device=h.device, requires_grad=True)
        
        total_loss = 0.0
        loss_count = 0
        
        # Process different types of family relationships
        relation_types = torch.unique(edge_type)
        
        for rel_type in relation_types:
            # Get edges of specific relationship type
            type_mask = edge_type == rel_type
            if type_mask.sum() == 0:
                continue
                
            type_edges = edge_index[:, type_mask]
            
            # Calculate similarity for this relationship type
            similarities = F.cosine_similarity(h[type_edges[0]], h[type_edges[1]], dim=1)
            
            # Set different weights based on relationship type
            if rel_type == 0:  # Direct relatives, higher weight
                weight = 2.0
                target_sim = 0.8  # Expect higher similarity
            else:  # Other relatives
                weight = 1.0
                target_sim = 0.6  # Expect moderate similarity
            
            # Use smooth L1 loss (Huber loss)
            loss = F.smooth_l1_loss(similarities, 
                                   torch.full_like(similarities, target_sim))
            total_loss += weight * loss
            loss_count += 1
        
        return total_loss / max(loss_count, 1)

    @staticmethod
    def model_regularization(model: nn.Module, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Cross entropy loss of the entire model, including parameters from prediction layer
        
        Args:
            model: The complete model (E2RGAT)
            logits: Model predictions (logits)
            labels: Ground truth labels
            
        Returns:
            Cross entropy loss tensor
        """
        # Use BCEWithLogitsLoss for binary classification
        criterion = nn.BCEWithLogitsLoss()
        return criterion(logits.squeeze(), labels)
    
    @staticmethod
    def combined_loss(h: torch.Tensor, pos_pairs: torch.Tensor, neg_pairs: torch.Tensor,
                     edge_index: torch.Tensor, edge_type: torch.Tensor,
                     model: Optional[nn.Module] = None, logits: Optional[torch.Tensor] = None, 
                     labels: Optional[torch.Tensor] = None,
                     alpha_family: float = 0.5, alpha_reg: float = 1e-4, 
                     alpha_custom: float = 1.0, temperature: float = 0.2) -> Tuple[torch.Tensor, Dict]:
        """
        Combined loss function: alpha_family * family_loss + alpha_reg * model_loss + alpha_custom * custom_loss
        Excludes temporal consistency loss as requested
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 1. Family Loss
        if alpha_family > 0:
            fam_loss = CustomLossFunctions.family_loss(h, edge_index, edge_type)
            total_loss += alpha_family * fam_loss
            loss_dict['family'] = fam_loss.item()
        
        # 2. Model Loss (Cross entropy of entire model)
        if alpha_reg > 0 and model is not None and logits is not None and labels is not None:
            model_loss = CustomLossFunctions.model_regularization(model, logits, labels)
            total_loss += alpha_reg * model_loss
            loss_dict['model_loss'] = model_loss.item()
        
        # 3. Custom Loss (InfoNCE)
        if alpha_custom > 0:
            custom_loss = CustomLossFunctions.info_nce_loss(h, pos_pairs, neg_pairs, temperature=temperature)
            total_loss += alpha_custom * custom_loss
            loss_dict['custom'] = custom_loss.item()
        
        return total_loss, loss_dict

class E2RGATLoss(nn.Module):
    """Three-part loss module for E2RGAT training:
    1. Model Loss: Cross entropy loss of the entire model
    2. Graph Loss: Entropy loss of the graph attention networks
    3. Custom Loss: Contrastive loss accounting for all relation types
    """
    
    def __init__(self, alpha_model: float = 0.5, alpha_graph: float = 0.5, 
                 alpha_custom: float = 1, temperature: float = 0.2):
        super().__init__()
        
        self.alpha_model = alpha_model    # Weight for model loss (cross entropy)
        self.alpha_graph = alpha_graph    # Weight for graph loss (entropy)
        self.alpha_custom = alpha_custom  # Weight for custom loss (contrastive)
        
        # Shared learnable temperature parameter for all relations
        self.learnable_temperature = nn.Parameter(torch.tensor(temperature))
        
        # Model loss: Cross entropy for classification
        self.model_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, 
                node_embeddings: torch.Tensor, G: nx.Graph, 
                model: nn.Module, node_ids: List[int] = None) -> Tuple[torch.Tensor, Dict]:

        if node_ids is None:
            node_ids = list(G.nodes())
        
        # 1. MODEL LOSS: Cross entropy loss of the entire model
        model_loss = self.model_loss(logits.squeeze(), labels)
        total_loss = self.alpha_model * model_loss
        loss_dict = {'model_loss': model_loss.item()}
        
        # 2. GRAPH LOSS: Entropy loss of the graph attention networks
        graph_loss = self._compute_graph_entropy_loss(model, G, node_ids)
        total_loss += self.alpha_graph * graph_loss
        loss_dict['graph_loss'] = graph_loss.item()
        
        # 3. CUSTOM LOSS: Contrastive loss accounting for all relation types
        custom_loss = self._compute_contrastive_loss(node_embeddings, G, node_ids)
        total_loss += self.alpha_custom * custom_loss
        loss_dict['custom_loss'] = custom_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_graph_entropy_loss(self, model: nn.Module, G: nx.Graph, node_ids: List[int]) -> torch.Tensor:
        """
        Compute entropy loss of the graph attention networks
        This encourages the attention weights to be more focused and less uniform
        """
        device = next(model.parameters()).device
        total_entropy_loss = 0.0
        num_attention_layers = 0
        
        # Iterate through attention layers in the model
        for layer in model.modules():
            if hasattr(layer, 'attention_weights') and hasattr(layer, '_compute_relation_attention'):
                num_attention_layers += 1
                
                # Get attention weights for each node
                for node_id in node_ids:
                    if node_id not in G:
                        continue
                    
                    neighbors = list(G.neighbors(node_id))
                    if not neighbors:
                        continue
                    
                    # Group neighbors by relation type
                    relation_groups = {}
                    for neighbor_id in neighbors:
                        edge_data = G.get_edge_data(node_id, neighbor_id) or G.get_edge_data(neighbor_id, node_id)
                        if edge_data is None:
                            continue
                        
                        kin_degree = edge_data.get('kin_degree', 0)
                        if kin_degree not in relation_groups:
                            relation_groups[kin_degree] = []
                        relation_groups[kin_degree].append(neighbor_id)
                    
                    # Compute entropy for each relation type
                    for relation_type, neighbor_list in relation_groups.items():
                        if len(neighbor_list) <= 1:
                            continue
                        
                        # Create dummy features for entropy computation
                        neighbor_features = torch.randn(len(neighbor_list), layer.out_dim, device=device)
                        edge_features = torch.randn(len(neighbor_list), layer.edge_dim, device=device)
                        
                        # Get attention weights (this is a simplified version)
                        h_i = torch.randn(layer.out_dim, device=device)
                        attention_scores = torch.randn(len(neighbor_list), device=device)
                        attention_weights = F.softmax(attention_scores, dim=0)
                        
                        # Compute entropy loss (encourage focused attention)
                        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8))
                        max_entropy = torch.log(torch.tensor(len(neighbor_list), dtype=torch.float32, device=device))
                        normalized_entropy = entropy / max_entropy
                        
                        # We want to minimize entropy (maximize focus)
                        entropy_loss = normalized_entropy
                        total_entropy_loss += entropy_loss
        
        if num_attention_layers == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_entropy_loss / num_attention_layers
    
    def _compute_contrastive_loss(self, node_embeddings: torch.Tensor, G: nx.Graph, node_ids: List[int]) -> torch.Tensor:
        """
        Compute extended contrastive loss accounting for different relation types
        This encourages similar representations for nodes with the same relation types
        """
        device = node_embeddings.device
        
        # Collect all positive pairs with their relation types
        pos_pairs = []
        relation_types = []
        
        for edge in G.edges(data=True):
            node1, node2 = edge[0], edge[1]
            kin_degree = edge[2].get('kin_degree', 0)
            
            if node1 in node_ids and node2 in node_ids:
                idx1 = node_ids.index(node1)
                idx2 = node_ids.index(node2)
                pos_pairs.append([idx1, idx2])
                relation_types.append(kin_degree)
        
        if len(pos_pairs) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Create negative pairs (nodes with different relation types or no relation)
        neg_pairs = []
        num_nodes = len(node_ids)
        num_neg = min(len(pos_pairs) * 2, num_nodes * (num_nodes - 1) // 4)
        
        for _ in range(num_neg):
            i, j = np.random.choice(num_nodes, 2, replace=False)
            # Ensure this is not a positive pair
            if [i, j] not in pos_pairs and [j, i] not in pos_pairs:
                neg_pairs.append([i, j])
        
        # Convert to tensors
        pos_pairs_tensor = torch.tensor(pos_pairs, device=device).t()
        neg_pairs_tensor = torch.tensor(neg_pairs, device=device).t() if neg_pairs else torch.empty((2, 0), dtype=torch.long, device=device)
        relation_types_tensor = torch.tensor(relation_types, device=device)
        
        # Compute extended contrastive loss with shared learnable temperature
        contrastive_loss = CustomLossFunctions.info_nce_loss(
            node_embeddings, 
            pos_pairs_tensor, 
            neg_pairs_tensor, 
            relation_types_tensor,
            self.learnable_temperature
        )
        
        return contrastive_loss

class RelationalAttentionLayer(nn.Module):
    """Relational Attention Layer for E2RGAT with Graph Attention Mechanism"""
    
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, 
                 num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.num_heads = 1  # Force single head
        self.head_dim = out_dim  # Use full dimension for single head
        
        # Linear projections for node features
        self.node_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False)
        
        # Graph attention mechanism - learnable attention weights (single head)
        # Attention weight vector for computing attention scores
        self.attention_weights = nn.Parameter(torch.Tensor(2 * out_dim + edge_dim))
        
        # Output projection
        self.output_proj = nn.Linear(out_dim, out_dim)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize attention weights
        nn.init.xavier_uniform_(self.attention_weights)
    
    def temporal_aggregate_neighbor(self, neighbor_features: torch.Tensor,
                                  neighbor_times: List[pd.Timestamp],
                                  current_time: pd.Timestamp) -> torch.Tensor:
        """Aggregate neighbor features based on temporal relationships"""
        valid_mask = []
        for t in neighbor_times:
            if pd.notna(t) and t < current_time:
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        
        if not any(valid_mask):
            return torch.zeros(neighbor_features.size(1), device=neighbor_features.device)
        
        valid_features = neighbor_features[valid_mask]
        valid_times = [t for i, t in enumerate(neighbor_times) if valid_mask[i]]
        time_diffs = [(current_time - t).days for t in valid_times]
        
        weights = torch.tensor([1.0 / (1.0 + diff) for diff in time_diffs],
                              device=neighbor_features.device)
        weights = F.softmax(weights, dim=0)
        
        aggregated = (valid_features * weights.unsqueeze(1)).sum(dim=0)
        return aggregated
    
    def forward(self, G, node_ids=None):
        """Forward pass with Graph Attention Mechanism"""
        if node_ids is None:
            node_ids = list(G.nodes())
        
        device = next(self.parameters()).device
        T_max = max(G.nodes[eid]["visit_matrix"].shape[0] for eid in node_ids)
        updated_features = {}
        
        for eid in node_ids:
            node_data = G.nodes[eid]
            visit_matrix = torch.tensor(node_data["visit_matrix"], 
                                      dtype=torch.float32, device=device)
            visit_dates = node_data["visit_dates"]
            
            h_i = self.node_proj(visit_matrix)
            h_i_updated = []
            
            for t in range(len(visit_dates)):
                if pd.isna(visit_dates[t]):
                    h_i_updated.append(torch.zeros(self.out_dim, device=device))
                    continue
                
                current_time = visit_dates[t]
                h_i_t = h_i[t]
                
                # Get neighbors for current node
                neighbors = list(G.neighbors(eid))
                if not neighbors:
                    h_i_updated.append(h_i_t)
                    continue
                
                # Group neighbors by relation type for localized normalization
                relation_groups = {}
                
                for neighbor_id in neighbors:
                    if neighbor_id not in G:
                        continue
                    
                    neighbor_data = G.nodes[neighbor_id]
                    neighbor_visit_matrix = torch.tensor(neighbor_data["visit_matrix"],
                                                       dtype=torch.float32, device=device)
                    neighbor_visit_dates = neighbor_data["visit_dates"]
                    
                    h_j = self.node_proj(neighbor_visit_matrix)
                    h_j_agg = self.temporal_aggregate_neighbor(h_j, neighbor_visit_dates, current_time)
                    
                    edge_data = G.get_edge_data(eid, neighbor_id) or G.get_edge_data(neighbor_id, eid)
                    if edge_data is None or "edge_vector" not in edge_data:
                        continue
                    
                    edge_vec = torch.tensor(edge_data["edge_vector"], 
                                          dtype=torch.float32, device=device)
                    
                    # Determine relation type from edge data
                    kin_degree = edge_data.get('kin_degree', 0)
                    relation_type = kin_degree  # Use kin_degree as relation type
                    
                    if relation_type not in relation_groups:
                        relation_groups[relation_type] = {
                            'neighbor_features': [],
                            'edge_features': [],
                            'neighbor_ids': []
                        }
                    
                    relation_groups[relation_type]['neighbor_features'].append(h_j_agg)
                    relation_groups[relation_type]['edge_features'].append(edge_vec)
                    relation_groups[relation_type]['neighbor_ids'].append(neighbor_id)
                
                if not relation_groups:
                    h_i_updated.append(h_i_t)
                    continue
                
                # Apply localized attention for each relation type
                h_i_t_updated = self._localized_graph_attention(
                    h_i_t, relation_groups
                )
                
                h_i_updated.append(h_i_t_updated)
            
            while len(h_i_updated) < T_max:
                h_i_updated.append(torch.zeros(self.out_dim, device=device))
            
            updated_features[eid] = torch.stack(h_i_updated)
        
        return updated_features
    
    def _graph_attention(self, h_i: torch.Tensor, neighbor_features: torch.Tensor, 
                        edge_features: torch.Tensor) -> torch.Tensor:
        """
        Graph attention mechanism with learnable weights (single head)
        
        Args:
            h_i: (out_dim,) Current node features
            neighbor_features: (num_neighbors, out_dim) Neighbor features
            edge_features: (num_neighbors, edge_dim) Edge features
            
        Returns:
            Updated node features
        """
        num_neighbors = neighbor_features.size(0)
        
        # Concatenate features for attention computation
        # [h_i || h_j || edge_features] for each neighbor
        h_i_expanded = h_i.unsqueeze(0).expand(num_neighbors, -1)  # (num_neighbors, out_dim)
        
        # Concatenate along feature dimension
        attention_input = torch.cat([
            h_i_expanded,  # (num_neighbors, out_dim)
            neighbor_features,  # (num_neighbors, out_dim)
            edge_features,  # (num_neighbors, edge_dim)
        ], dim=-1)  # (num_neighbors, 2*out_dim + edge_dim)
        
        # Compute attention scores using learnable weights
        attention_scores = torch.sum(
            self.attention_weights * attention_input, dim=-1
        )  # (num_neighbors,)
        
        # Apply LeakyReLU and softmax
        attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)
        attention_weights = F.softmax(attention_scores, dim=0)  # (num_neighbors,)
        
        # Apply attention weights to neighbor features
        attended_features = torch.sum(
            attention_weights.unsqueeze(-1) * neighbor_features, dim=0
        )  # (out_dim,)
        
        # Residual connection and layer norm
        h_i_updated = self.layer_norm1(h_i + self.dropout(attended_features))
        
        # Feed-forward network
        ff_output = self.ffn(h_i_updated)
        h_i_updated = self.layer_norm2(h_i_updated + self.dropout(ff_output))
        
        return h_i_updated
    
    def _localized_graph_attention(self, h_i: torch.Tensor, relation_groups: Dict) -> torch.Tensor:
        """
        Graph attention mechanism with localized normalization for each relation type
        
        Args:
            h_i: (out_dim,) Current node features
            relation_groups: Dictionary with relation types as keys, containing:
                - 'neighbor_features': List of neighbor features for this relation type
                - 'edge_features': List of edge features for this relation type
                - 'neighbor_ids': List of neighbor IDs for this relation type
                
        Returns:
            Updated node features
        """
        all_attended_features = []
        
        # Process each relation type separately
        for relation_type, group_data in relation_groups.items():
            if not group_data['neighbor_features']:
                continue
                
            # Stack features for this relation type
            neighbor_features = torch.stack(group_data['neighbor_features'])  # (num_neighbors, out_dim)
            edge_features = torch.stack(group_data['edge_features'])  # (num_neighbors, edge_dim)
            
            # Apply attention for this relation type
            attended_features = self._compute_relation_attention(
                h_i, neighbor_features, edge_features, relation_type
            )
            all_attended_features.append(attended_features)
        
        # Combine attended features from all relation types
        if all_attended_features:
            combined_features = torch.stack(all_attended_features).mean(dim=0)  # (out_dim,)
        else:
            combined_features = torch.zeros_like(h_i)
        
        # Residual connection and layer norm
        h_i_updated = self.layer_norm1(h_i + self.dropout(combined_features))
        
        # Feed-forward network
        ff_output = self.ffn(h_i_updated)
        h_i_updated = self.layer_norm2(h_i_updated + self.dropout(ff_output))
        
        return h_i_updated
    
    def _compute_relation_attention(self, h_i: torch.Tensor, neighbor_features: torch.Tensor, 
                                  edge_features: torch.Tensor, relation_type: int) -> torch.Tensor:
        """
        Compute attention for a specific relation type with localized normalization
        
        Args:
            h_i: (out_dim,) Current node features
            neighbor_features: (num_neighbors, out_dim) Neighbor features for this relation type
            edge_features: (num_neighbors, edge_dim) Edge features for this relation type
            relation_type: Integer indicating the relation type
            
        Returns:
            Attended features for this relation type
        """
        num_neighbors = neighbor_features.size(0)
        
        # Concatenate features for attention computation
        # [h_i  || edge_features || h_j] for each neighbor
        h_i_expanded = h_i.unsqueeze(0).expand(num_neighbors, -1)  # (num_neighbors, out_dim)
        
        # Concatenate along feature dimension
        attention_input = torch.cat([
            h_i_expanded,  # (num_neighbors, out_dim)
            edge_features,  # (num_neighbors, edge_dim)
            neighbor_features,  # (num_neighbors, out_dim)
        ], dim=-1)  # (num_neighbors, 2*out_dim + edge_dim)
        
        # Compute attention scores using learnable weights
        attention_scores = torch.sum(
            self.attention_weights * attention_input, dim=-1
        )  # (num_neighbors,)
        
        # Apply LeakyReLU and LOCALIZED softmax (only within this relation type)
        attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)
        attention_weights = F.softmax(attention_scores, dim=0)  # (num_neighbors,) - localized to this relation type
        
        # Apply attention weights to neighbor features
        attended_features = torch.sum(
            attention_weights.unsqueeze(-1) * neighbor_features, dim=0
        )  # (out_dim,)
        
        return attended_features

class E2RGAT(nn.Module):
    """E2RGAT model"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3,
                 num_heads: int = 1, dropout: float = 0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            RelationalAttentionLayer(
                in_dim=hidden_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                edge_dim=4,
                num_heads=num_heads,
                dropout=dropout
            ) for i in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Temporal aggregation (simplified without query/key/value)
        self.temporal_aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, G, node_ids=None, return_embeddings=False):
        """Forward pass with optional embedding return"""
        if node_ids is None:
            node_ids = list(G.nodes())
        
        # Prepare node features
        node_features = {}
        for eid in node_ids:
            visit_matrix = torch.tensor(G.nodes[eid]["visit_matrix"], dtype=torch.float32)
            node_features[eid] = self.input_proj(visit_matrix)
        
        # Apply attention layers
        current_features = node_features
        for i, (attention_layer, layer_norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            attended_features = attention_layer(G)
            
            for eid in node_ids:
                if eid in attended_features:
                    current_features[eid] = layer_norm(
                        current_features[eid] + attended_features[eid]
                    )
        
        # Temporal aggregation for final prediction
        batch_features = []
        for eid in node_ids:
            features = current_features[eid]
            # Simple temporal aggregation: mean pooling + MLP
            temporal_agg = features.mean(dim=0)  # Average across time steps
            aggregated = self.temporal_aggregation(temporal_agg)
            batch_features.append(aggregated)
        
        batch_features = torch.stack(batch_features)
        logits = self.classifier(batch_features)
        
        if return_embeddings:
            return logits, batch_features
        else:
            return logits