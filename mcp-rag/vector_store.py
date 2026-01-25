"""
向量存储服务 - 封装 ChromaDB 操作
"""
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import uuid

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL
)


class VectorStore:
    """ChromaDB 向量存储封装类"""
    
    def __init__(self):
        """初始化 ChromaDB 客户端和集合"""
        # 创建持久化客户端
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        
        # 使用 sentence-transformers 嵌入函数
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"description": "MCP RAG Demo 知识库"}
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        添加文档到向量库
        
        Args:
            texts: 文档文本列表
            metadatas: 元数据列表
            ids: 文档 ID 列表（可选，自动生成）
            
        Returns:
            添加的文档 ID 列表
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def query(
        self,
        query_text: str,
        n_results: int = 3
    ) -> Dict[str, Any]:
        """
        向量相似度检索
        
        Args:
            query_text: 查询文本
            n_results: 返回结果数量
            
        Returns:
            检索结果字典
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "collection_name": CHROMA_COLLECTION_NAME,
            "document_count": self.collection.count(),
            "persist_directory": str(CHROMA_PERSIST_DIR)
        }
    
    def list_documents(self, limit: int = 100) -> Dict[str, Any]:
        """
        列出知识库中的文档
        
        Args:
            limit: 返回数量限制
            
        Returns:
            文档列表
        """
        results = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        return {
            "ids": results["ids"],
            "documents": results["documents"],
            "metadatas": results["metadatas"],
            "count": len(results["ids"])
        }
    
    def delete_collection(self) -> None:
        """清空并删除集合"""
        self.client.delete_collection(CHROMA_COLLECTION_NAME)
        # 重新创建空集合
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )


# 全局单例
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """获取向量存储单例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
