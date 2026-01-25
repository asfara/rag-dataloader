"""
RAG 核心服务 - 文档加载、分块和检索
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from vector_store import get_vector_store
from config import DATA_DIR, DEFAULT_TOP_K


class RAGService:
    """RAG 服务类"""
    
    def __init__(self):
        """初始化 RAG 服务"""
        self.vector_store = get_vector_store()
    
    def load_text_file(self, file_path: Path) -> str:
        """
        加载文本文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        将文本分块
        
        Args:
            text: 原始文本
            chunk_size: 块大小（字符数）
            overlap: 重叠字符数
            
        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            print(f"start: {start}, len: {text_len}")
            print(f"处理文本: {text[start:start + chunk_size]}...")
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            
            # 如果不是最后一块，尝试在句子边界处截断
            if end < text_len:
                # 查找最后一个句子结束符
                last_period = max(
                    chunk.rfind("。"),
                    chunk.rfind("."),
                    chunk.rfind("\n\n")
                )
                if last_period > chunk_size // 2:
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            # 确保 start 始终向前推进，避免无限循环
            start = max(end - overlap, start + 1)
        
        return [c for c in chunks if c]  # 过滤空块
    
    def add_document(
        self,
        content: str,
        title: str = "Untitled",
        chunk_size: int = 500
    ) -> Dict[str, Any]:
        """
        添加文档到知识库
        
        Args:
            content: 文档内容
            title: 文档标题
            chunk_size: 分块大小
            
        Returns:
            添加结果
        """
        chunks = self.chunk_text(content, chunk_size=chunk_size)
        
        metadatas = [
            {"title": title, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]
        
        ids = self.vector_store.add_documents(chunks, metadatas)
        
        return {
            "success": True,
            "title": title,
            "chunk_count": len(chunks),
            "ids": ids
        }
    
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            相关文档列表
        """
        results = self.vector_store.query(query, n_results=top_k)
        
        documents = []
        for i, doc in enumerate(results["documents"]):
            documents.append({
                "content": doc,
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
                "score": 1 - results["distances"][i] if results["distances"] else 0,
                "id": results["ids"][i] if results["ids"] else None
            })
        
        return documents
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        格式化检索结果为上下文字符串
        
        Args:
            documents: 检索到的文档列表
            
        Returns:
            格式化的上下文字符串
        """
        if not documents:
            return "未找到相关文档。"
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get("metadata", {}).get("title", "Unknown")
            score = doc.get("score", 0)
            content = doc.get("content", "")
            context_parts.append(
                f"[文档 {i}] 来源: {title}, 相关度: {score:.2f}\n{content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def load_data_directory(self) -> Dict[str, Any]:
        """
        加载 data 目录下的所有文本文件
        
        Returns:
            加载结果
        """
        loaded_files = []
        
        for file_path in DATA_DIR.glob("*.txt"):
            print(f"处理文件: {file_path}")
            content = self.load_text_file(file_path)
            result = self.add_document(content, title=file_path.name)
            loaded_files.append({
                "file": file_path.name,
                "chunks": result["chunk_count"]
            })
        
        return {
            "success": True,
            "loaded_files": loaded_files,
            "total_files": len(loaded_files)
        }


# 全局单例
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """获取 RAG 服务单例"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
