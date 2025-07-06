"""
Memory management system for Archie
Handles persistent memory using ChromaDB for vector storage
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import math
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class Memory:
    """Memory entry structure"""
    id: str
    content: str
    memory_type: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    importance: float = 0.5
    tags: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.accessed_at is None:
            self.accessed_at = datetime.now()
        if self.tags is None:
            self.tags = []

@dataclass
class MemoryQuery:
    """Memory query structure"""
    query_text: str
    memory_types: Optional[List[str]] = None
    limit: int = 10
    similarity_threshold: float = 0.7
    time_filter: Optional[Dict[str, datetime]] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    importance_threshold: float = 0.3
    include_context: bool = True
    context_window: int = 3

@dataclass
class MemoryCluster:
    """Memory cluster for semantic grouping"""
    id: str
    centroid: List[float]
    memory_ids: List[str]
    theme: str
    importance: float
    created_at: datetime
    last_updated: datetime

class MemoryManager:
    """Manages persistent memory using ChromaDB"""
    
    def __init__(self, 
                 chroma_host: str = "localhost",
                 chroma_port: int = 8000,
                 collection_name: str = "archie_memory",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Initialize components
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.memory_cache = {}
        self.cache_size = 1000
        
        # Advanced memory features
        self.memory_clusters = {}
        self.memory_connections = defaultdict(set)
        self.context_graph = defaultdict(list)
        self.importance_decay_rate = 0.95
        self.clustering_threshold = 0.8
        
        # Memory statistics
        self.stats = {
            "total_memories": 0,
            "queries_count": 0,
            "cache_hits": 0,
            "clusters_count": 0,
            "connections_count": 0,
            "last_updated": datetime.now()
        }
    
    async def initialize(self):
        """Initialize the memory manager"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                settings=Settings(allow_reset=True)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Connected to existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Archie's persistent memory"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            
            # Load memory statistics
            await self._load_stats()
            
            logger.info("Memory manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            raise
    
    async def _load_stats(self):
        """Load memory statistics"""
        try:
            count = self.collection.count()
            self.stats["total_memories"] = count
            self.stats["last_updated"] = datetime.now()
            logger.info(f"Loaded {count} memories from database")
        except Exception as e:
            logger.error(f"Error loading memory stats: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = self.embedding_model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def _create_memory_id(self, content: str, memory_type: str) -> str:
        """Create unique memory ID"""
        import hashlib
        content_hash = hashlib.md5(
            f"{content}:{memory_type}:{datetime.now().date()}".encode()
        ).hexdigest()
        return f"mem_{content_hash}"
    
    async def store_memory(self, 
                          content: str, 
                          memory_type: str,
                          metadata: Dict[str, Any] = None,
                          importance: float = 0.5,
                          tags: List[str] = None) -> str:
        """Store a new memory"""
        try:
            if metadata is None:
                metadata = {}
            if tags is None:
                tags = []
            
            # Create memory object
            memory_id = self._create_memory_id(content, memory_type)
            
            # Check if memory already exists
            existing = await self.get_memory(memory_id)
            if existing:
                logger.info(f"Memory {memory_id} already exists, updating access time")
                existing.accessed_at = datetime.now()
                await self._update_memory_access(memory_id)
                return memory_id
            
            # Generate embedding
            embedding = self._generate_embedding(content)
            
            # Create memory
            memory = Memory(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                metadata=metadata,
                embedding=embedding,
                importance=importance,
                tags=tags
            )
            
            # Prepare ChromaDB data
            chroma_metadata = {
                "memory_type": memory_type,
                "importance": importance,
                "created_at": memory.created_at.isoformat(),
                "accessed_at": memory.accessed_at.isoformat(),
                "tags": json.dumps(tags),
                **metadata
            }
            
            # Store in ChromaDB
            self.collection.add(
                ids=[memory_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[chroma_metadata]
            )
            
            # Update cache
            self.memory_cache[memory_id] = memory
            
            # Update stats
            self.stats["total_memories"] += 1
            self.stats["last_updated"] = datetime.now()
            
            # Create semantic connections
            await self.create_memory_connections(memory_id, threshold=0.75)
            
            logger.info(f"Stored memory: {memory_id} (type: {memory_type})")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory"""
        try:
            # Check cache first
            if memory_id in self.memory_cache:
                self.stats["cache_hits"] += 1
                return self.memory_cache[memory_id]
            
            # Query ChromaDB
            results = self.collection.get(
                ids=[memory_id],
                include=["documents", "embeddings", "metadatas"]
            )
            
            if not results['ids']:
                return None
            
            # Parse results
            document = results['documents'][0]
            embedding = results['embeddings'][0]
            metadata = results['metadatas'][0]
            
            # Create memory object
            memory = Memory(
                id=memory_id,
                content=document,
                memory_type=metadata.get('memory_type', 'unknown'),
                metadata={k: v for k, v in metadata.items() 
                         if k not in ['memory_type', 'importance', 'created_at', 'accessed_at', 'tags']},
                embedding=embedding,
                created_at=datetime.fromisoformat(metadata.get('created_at', datetime.now().isoformat())),
                accessed_at=datetime.fromisoformat(metadata.get('accessed_at', datetime.now().isoformat())),
                importance=metadata.get('importance', 0.5),
                tags=json.loads(metadata.get('tags', '[]'))
            )
            
            # Update cache
            self.memory_cache[memory_id] = memory
            
            # Update access time
            await self._update_memory_access(memory_id)
            
            return memory
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    async def query_memories(self, query: MemoryQuery) -> List[Tuple[Memory, float]]:
        """Query memories by similarity"""
        try:
            self.stats["queries_count"] += 1
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query.query_text)
            
            # Prepare metadata filter
            where_clause = {}
            if query.memory_types:
                where_clause["memory_type"] = {"$in": query.memory_types}
            
            if query.metadata_filter:
                where_clause.update(query.metadata_filter)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=query.limit,
                where=where_clause if where_clause else None,
                include=["documents", "embeddings", "metadatas", "distances"]
            )
            
            # Process results
            memories_with_scores = []
            
            for i, (doc, embedding, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['embeddings'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score
                similarity = 1 - distance
                
                # Apply similarity threshold
                if similarity < query.similarity_threshold:
                    continue
                
                # Create memory object
                memory_id = results['ids'][0][i]
                memory = Memory(
                    id=memory_id,
                    content=doc,
                    memory_type=metadata.get('memory_type', 'unknown'),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ['memory_type', 'importance', 'created_at', 'accessed_at', 'tags']},
                    embedding=embedding,
                    created_at=datetime.fromisoformat(metadata.get('created_at', datetime.now().isoformat())),
                    accessed_at=datetime.fromisoformat(metadata.get('accessed_at', datetime.now().isoformat())),
                    importance=metadata.get('importance', 0.5),
                    tags=json.loads(metadata.get('tags', '[]'))
                )
                
                memories_with_scores.append((memory, similarity))
                
                # Update cache
                self.memory_cache[memory_id] = memory
            
            # Sort by similarity (highest first)
            memories_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Found {len(memories_with_scores)} memories for query")
            return memories_with_scores
            
        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            return []
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory"""
        try:
            # Get current memory
            memory = await self.get_memory(memory_id)
            if not memory:
                logger.warning(f"Memory {memory_id} not found for update")
                return False
            
            # Update fields
            updated_metadata = memory.metadata.copy()
            
            if 'content' in updates:
                memory.content = updates['content']
                # Regenerate embedding for new content
                memory.embedding = self._generate_embedding(memory.content)
            
            if 'memory_type' in updates:
                memory.memory_type = updates['memory_type']
            
            if 'metadata' in updates:
                updated_metadata.update(updates['metadata'])
                memory.metadata = updated_metadata
            
            if 'importance' in updates:
                memory.importance = updates['importance']
            
            if 'tags' in updates:
                memory.tags = updates['tags']
            
            # Update access time
            memory.accessed_at = datetime.now()
            
            # Prepare ChromaDB update
            chroma_metadata = {
                "memory_type": memory.memory_type,
                "importance": memory.importance,
                "created_at": memory.created_at.isoformat(),
                "accessed_at": memory.accessed_at.isoformat(),
                "tags": json.dumps(memory.tags),
                **memory.metadata
            }
            
            # Update in ChromaDB
            self.collection.update(
                ids=[memory_id],
                documents=[memory.content],
                embeddings=[memory.embedding],
                metadatas=[chroma_metadata]
            )
            
            # Update cache
            self.memory_cache[memory_id] = memory
            
            logger.info(f"Updated memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        try:
            # Delete from ChromaDB
            self.collection.delete(ids=[memory_id])
            
            # Remove from cache
            if memory_id in self.memory_cache:
                del self.memory_cache[memory_id]
            
            # Update stats
            self.stats["total_memories"] -= 1
            self.stats["last_updated"] = datetime.now()
            
            logger.info(f"Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    async def _update_memory_access(self, memory_id: str):
        """Update memory access time"""
        try:
            # Update in ChromaDB
            current_time = datetime.now().isoformat()
            self.collection.update(
                ids=[memory_id],
                metadatas=[{"accessed_at": current_time}]
            )
            
            # Update cache
            if memory_id in self.memory_cache:
                self.memory_cache[memory_id].accessed_at = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating access time for {memory_id}: {e}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            # Update total count
            self.stats["total_memories"] = self.collection.count()
            self.stats["last_updated"] = datetime.now()
            
            # Add cache stats
            cache_stats = {
                "cache_size": len(self.memory_cache),
                "cache_limit": self.cache_size,
                "cache_usage": len(self.memory_cache) / self.cache_size * 100
            }
            
            return {
                **self.stats,
                **cache_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return self.stats
    
    async def cleanup_old_memories(self, days_old: int = 30, max_memories: int = 10000):
        """Clean up old or excess memories"""
        try:
            # Get all memories with metadata
            all_memories = self.collection.get(include=["metadatas"])
            
            if not all_memories['ids']:
                return
            
            # Find old memories
            cutoff_date = datetime.now() - timedelta(days=days_old)
            old_memory_ids = []
            
            for memory_id, metadata in zip(all_memories['ids'], all_memories['metadatas']):
                accessed_at = datetime.fromisoformat(metadata.get('accessed_at', datetime.now().isoformat()))
                if accessed_at < cutoff_date:
                    old_memory_ids.append(memory_id)
            
            # Delete old memories
            if old_memory_ids:
                self.collection.delete(ids=old_memory_ids)
                
                # Remove from cache
                for memory_id in old_memory_ids:
                    if memory_id in self.memory_cache:
                        del self.memory_cache[memory_id]
                
                logger.info(f"Cleaned up {len(old_memory_ids)} old memories")
            
            # Check if we need to remove excess memories
            current_count = self.collection.count()
            if current_count > max_memories:
                # This would require more complex logic to remove least important memories
                logger.warning(f"Memory count ({current_count}) exceeds limit ({max_memories})")
            
        except Exception as e:
            logger.error(f"Error cleaning up memories: {e}")
    
    async def export_memories(self, memory_type: str = None) -> List[Dict]:
        """Export memories to JSON format"""
        try:
            where_clause = {"memory_type": memory_type} if memory_type else None
            
            results = self.collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )
            
            exported_memories = []
            for memory_id, document, metadata in zip(
                results['ids'],
                results['documents'],
                results['metadatas']
            ):
                exported_memories.append({
                    "id": memory_id,
                    "content": document,
                    "metadata": metadata
                })
            
            logger.info(f"Exported {len(exported_memories)} memories")
            return exported_memories
            
        except Exception as e:
            logger.error(f"Error exporting memories: {e}")
            return []
    
    async def calculate_memory_importance(self, memory: Memory) -> float:
        """Calculate dynamic importance score for a memory"""
        try:
            base_importance = memory.importance
            
            # Time decay factor
            days_old = (datetime.now() - memory.created_at).days
            time_decay = self.importance_decay_rate ** days_old
            
            # Access frequency boost
            access_boost = 1.0
            if memory.accessed_at:
                hours_since_access = (datetime.now() - memory.accessed_at).total_seconds() / 3600
                access_boost = 1.0 + (1.0 / (1.0 + hours_since_access / 24))
            
            # Connection strength boost
            connection_boost = 1.0 + (len(self.memory_connections[memory.id]) * 0.1)
            
            # Calculate final importance
            final_importance = base_importance * time_decay * access_boost * connection_boost
            
            return min(1.0, final_importance)
            
        except Exception as e:
            logger.error(f"Error calculating importance for {memory.id}: {e}")
            return memory.importance
    
    async def create_memory_connections(self, memory_id: str, threshold: float = 0.8):
        """Create semantic connections between memories"""
        try:
            memory = await self.get_memory(memory_id)
            if not memory or not memory.embedding:
                return
            
            # Get all memories for comparison
            all_memories = self.collection.get(include=["embeddings", "metadatas"])
            
            if not all_memories['ids']:
                return
            
            # Calculate similarities
            memory_embedding = np.array(memory.embedding).reshape(1, -1)
            all_embeddings = np.array(all_memories['embeddings'])
            
            similarities = cosine_similarity(memory_embedding, all_embeddings)[0]
            
            # Create connections above threshold
            connections = []
            for i, (other_id, similarity) in enumerate(zip(all_memories['ids'], similarities)):
                if other_id != memory_id and similarity >= threshold:
                    connections.append((other_id, similarity))
                    self.memory_connections[memory_id].add(other_id)
                    self.memory_connections[other_id].add(memory_id)
            
            # Update context graph
            self.context_graph[memory_id] = sorted(connections, key=lambda x: x[1], reverse=True)
            
            # Update stats
            self.stats["connections_count"] = sum(len(connections) for connections in self.memory_connections.values())
            
            logger.info(f"Created {len(connections)} connections for memory {memory_id}")
            
        except Exception as e:
            logger.error(f"Error creating connections for {memory_id}: {e}")
    
    async def cluster_memories(self, n_clusters: int = None, min_cluster_size: int = 3):
        """Cluster memories by semantic similarity"""
        try:
            # Get all memories with embeddings
            all_memories = self.collection.get(include=["embeddings", "documents", "metadatas"])
            
            if not all_memories['ids'] or len(all_memories['ids']) < min_cluster_size:
                return
            
            # Prepare embeddings
            embeddings = np.array(all_memories['embeddings'])
            
            # Determine optimal cluster count
            if n_clusters is None:
                n_clusters = min(10, max(2, len(all_memories['ids']) // 10))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Create clusters
            clusters = {}
            for i, (memory_id, document, metadata) in enumerate(zip(
                all_memories['ids'], 
                all_memories['documents'], 
                all_memories['metadatas']
            )):
                cluster_id = f"cluster_{cluster_labels[i]}"
                
                if cluster_id not in clusters:
                    clusters[cluster_id] = {
                        'memory_ids': [],
                        'documents': [],
                        'centroid': kmeans.cluster_centers_[cluster_labels[i]],
                        'importance': 0.0
                    }
                
                clusters[cluster_id]['memory_ids'].append(memory_id)
                clusters[cluster_id]['documents'].append(document)
                clusters[cluster_id]['importance'] += metadata.get('importance', 0.5)
            
            # Generate cluster themes
            for cluster_id, cluster_data in clusters.items():
                theme = await self._generate_cluster_theme(cluster_data['documents'])
                avg_importance = cluster_data['importance'] / len(cluster_data['memory_ids'])
                
                self.memory_clusters[cluster_id] = MemoryCluster(
                    id=cluster_id,
                    centroid=cluster_data['centroid'].tolist(),
                    memory_ids=cluster_data['memory_ids'],
                    theme=theme,
                    importance=avg_importance,
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
            
            # Update stats
            self.stats["clusters_count"] = len(self.memory_clusters)
            
            logger.info(f"Created {len(self.memory_clusters)} memory clusters")
            
        except Exception as e:
            logger.error(f"Error clustering memories: {e}")
    
    async def _generate_cluster_theme(self, documents: List[str]) -> str:
        """Generate theme for a cluster of documents"""
        try:
            # Simple keyword extraction for theme
            from collections import Counter
            import re
            
            # Extract keywords from all documents
            all_words = []
            for doc in documents:
                words = re.findall(r'\b\w+\b', doc.lower())
                all_words.extend([word for word in words if len(word) > 3])
            
            # Get most common words
            word_counts = Counter(all_words)
            top_words = [word for word, count in word_counts.most_common(3)]
            
            return " ".join(top_words) if top_words else "general"
            
        except Exception as e:
            logger.error(f"Error generating cluster theme: {e}")
            return "unknown"
    
    async def get_contextual_memories(self, memory_id: str, context_window: int = 3) -> List[Memory]:
        """Get contextually related memories"""
        try:
            contextual_memories = []
            
            # Get direct connections
            connected_ids = list(self.memory_connections[memory_id])[:context_window]
            
            for connected_id in connected_ids:
                memory = await self.get_memory(connected_id)
                if memory:
                    contextual_memories.append(memory)
            
            # Get cluster-based context if needed
            if len(contextual_memories) < context_window:
                for cluster in self.memory_clusters.values():
                    if memory_id in cluster.memory_ids:
                        for cluster_memory_id in cluster.memory_ids:
                            if cluster_memory_id != memory_id and len(contextual_memories) < context_window:
                                memory = await self.get_memory(cluster_memory_id)
                                if memory and memory not in contextual_memories:
                                    contextual_memories.append(memory)
            
            return contextual_memories
            
        except Exception as e:
            logger.error(f"Error getting contextual memories for {memory_id}: {e}")
            return []
    
    async def smart_memory_retrieval(self, query: MemoryQuery) -> List[Tuple[Memory, float, List[Memory]]]:
        """Enhanced memory retrieval with context and importance"""
        try:
            # Get base memories
            base_results = await self.query_memories(query)
            
            # Enhance with context and importance
            enhanced_results = []
            
            for memory, similarity in base_results:
                # Recalculate importance
                current_importance = await self.calculate_memory_importance(memory)
                
                # Apply importance threshold
                if current_importance < query.importance_threshold:
                    continue
                
                # Get contextual memories if requested
                context_memories = []
                if query.include_context:
                    context_memories = await self.get_contextual_memories(
                        memory.id, query.context_window
                    )
                
                # Calculate enhanced score
                enhanced_score = similarity * current_importance
                
                enhanced_results.append((memory, enhanced_score, context_memories))
            
            # Sort by enhanced score
            enhanced_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Retrieved {len(enhanced_results)} enhanced memories")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in smart memory retrieval: {e}")
            return []
    
    async def update_memory_importance_scores(self):
        """Update importance scores for all memories"""
        try:
            updated_count = 0
            
            # Get all memories
            all_memories = self.collection.get(include=["metadatas"])
            
            for memory_id, metadata in zip(all_memories['ids'], all_memories['metadatas']):
                # Get memory object
                memory = await self.get_memory(memory_id)
                if not memory:
                    continue
                
                # Calculate new importance
                new_importance = await self.calculate_memory_importance(memory)
                
                # Update if significantly different
                if abs(new_importance - memory.importance) > 0.1:
                    await self.update_memory(memory_id, {'importance': new_importance})
                    updated_count += 1
            
            logger.info(f"Updated importance scores for {updated_count} memories")
            
        except Exception as e:
            logger.error(f"Error updating importance scores: {e}")
    
    async def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights about memory patterns"""
        try:
            insights = {
                "memory_distribution": {},
                "top_themes": [],
                "connection_strength": 0.0,
                "importance_trends": {},
                "cluster_health": {}
            }
            
            # Get all memories
            all_memories = self.collection.get(include=["metadatas"])
            
            if not all_memories['ids']:
                return insights
            
            # Memory type distribution
            type_counts = defaultdict(int)
            importance_sums = defaultdict(float)
            
            for metadata in all_memories['metadatas']:
                memory_type = metadata.get('memory_type', 'unknown')
                importance = metadata.get('importance', 0.5)
                
                type_counts[memory_type] += 1
                importance_sums[memory_type] += importance
            
            insights["memory_distribution"] = dict(type_counts)
            insights["importance_trends"] = {
                t: importance_sums[t] / type_counts[t] for t in type_counts
            }
            
            # Top themes from clusters
            cluster_themes = [(cluster.theme, cluster.importance) 
                            for cluster in self.memory_clusters.values()]
            cluster_themes.sort(key=lambda x: x[1], reverse=True)
            insights["top_themes"] = cluster_themes[:5]
            
            # Connection strength
            if self.memory_connections:
                total_connections = sum(len(conns) for conns in self.memory_connections.values())
                insights["connection_strength"] = total_connections / len(self.memory_connections)
            
            # Cluster health
            insights["cluster_health"] = {
                "total_clusters": len(self.memory_clusters),
                "avg_cluster_size": sum(len(c.memory_ids) for c in self.memory_clusters.values()) / len(self.memory_clusters) if self.memory_clusters else 0,
                "cluster_coverage": len([mid for cluster in self.memory_clusters.values() for mid in cluster.memory_ids]) / len(all_memories['ids']) if all_memories['ids'] else 0
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting memory insights: {e}")
            return {}

# Factory function for creating memory manager
async def create_memory_manager(config: Dict[str, Any]) -> MemoryManager:
    """Create and initialize memory manager"""
    memory_config = config.get('memory', {})
    
    # Parse ChromaDB URL
    chroma_url = memory_config.get('url', 'http://localhost:8000')
    if '://' in chroma_url:
        chroma_url = chroma_url.split('://', 1)[1]
    
    host, port = chroma_url.split(':') if ':' in chroma_url else (chroma_url, 8000)
    
    manager = MemoryManager(
        chroma_host=host,
        chroma_port=int(port),
        collection_name=memory_config.get('collection_name', 'archie_memory'),
        embedding_model=memory_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    )
    
    await manager.initialize()
    return manager