"""
MCP RAG Demo 服务器 - 使用 FastMCP 暴露 RAG 功能
"""
from fastmcp import FastMCP
from typing import Optional

from rag_service import get_rag_service
from vector_store import get_vector_store

# 创建 FastMCP 服务器实例
mcp = FastMCP(
    name="RAG Demo",
    instructions="基于 ChromaDB 的 RAG 知识库服务，提供文档管理和语义检索功能"
)


# ===== MCP Tools =====

@mcp.tool()
def add_document(content: str, title: str = "Untitled") -> dict:
    """
    添加文档到知识库
    
    将文档内容分块后存储到向量数据库中，用于后续的语义检索。
    
    Args:
        content: 要添加的文档内容
        title: 文档标题，用于标识文档来源
        
    Returns:
        包含添加结果的字典，包括分块数量和文档ID
    """
    rag_service = get_rag_service()
    result = rag_service.add_document(content, title)
    return {
        "message": f"成功添加文档 '{title}'",
        "chunk_count": result["chunk_count"],
        "ids": result["ids"][:5] if len(result["ids"]) > 5 else result["ids"]  # 只返回前5个ID
    }


@mcp.tool()
def search(query: str, top_k: int = 3) -> dict:
    """
    在知识库中搜索相关文档
    
    使用语义相似度搜索，返回与查询最相关的文档片段。
    
    Args:
        query: 搜索查询文本
        top_k: 返回的最大结果数量，默认为3
        
    Returns:
        包含搜索结果的字典，包括相关文档内容和相关度分数
    """
    rag_service = get_rag_service()
    documents = rag_service.retrieve(query, top_k=top_k)
    
    if not documents:
        return {
            "message": "未找到相关文档",
            "results": [],
            "context": ""
        }
    
    results = []
    for doc in documents:
        results.append({
            "title": doc.get("metadata", {}).get("title", "Unknown"),
            "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
            "score": round(doc["score"], 3)
        })
    
    context = rag_service.format_context(documents)
    
    return {
        "message": f"找到 {len(results)} 个相关结果",
        "results": results,
        "context": context
    }


@mcp.tool()
def list_documents(limit: int = 20) -> dict:
    """
    列出知识库中的所有文档
    
    Args:
        limit: 返回的最大文档数量，默认为20
        
    Returns:
        包含文档列表的字典
    """
    vector_store = get_vector_store()
    result = vector_store.list_documents(limit=limit)
    
    # 整理文档信息
    documents = []
    for i, doc_id in enumerate(result["ids"]):
        metadata = result["metadatas"][i] if result["metadatas"] else {}
        content = result["documents"][i] if result["documents"] else ""
        documents.append({
            "id": doc_id,
            "title": metadata.get("title", "Unknown"),
            "preview": content[:100] + "..." if len(content) > 100 else content
        })
    
    return {
        "message": f"共有 {result['count']} 个文档片段",
        "documents": documents
    }


@mcp.tool()
def load_sample_data() -> dict:
    """
    加载 data 目录下的示例文档
    
    自动扫描并加载 data/ 目录下的所有 .txt 文件到知识库中。
    
    Returns:
        加载结果，包括加载的文件列表和数量
    """
    rag_service = get_rag_service()
    result = rag_service.load_data_directory()
    
    return {
        "message": f"成功加载 {result['total_files']} 个文件",
        "loaded_files": result["loaded_files"]
    }


@mcp.tool()
def clear_knowledge_base() -> dict:
    """
    清空知识库
    
    删除知识库中的所有文档。此操作不可恢复。
    
    Returns:
        操作结果
    """
    vector_store = get_vector_store()
    vector_store.delete_collection()
    
    return {
        "message": "知识库已清空",
        "success": True
    }


# ===== MCP Resources =====

@mcp.resource("rag://stats")
def get_stats() -> str:
    """
    获取知识库统计信息
    
    Returns:
        知识库状态信息（JSON 格式字符串）
    """
    import json
    vector_store = get_vector_store()
    stats = vector_store.get_stats()
    
    return json.dumps({
        "collection_name": stats["collection_name"],
        "document_count": stats["document_count"],
        "storage_path": stats["persist_directory"]
    }, ensure_ascii=False, indent=2)


# 入口点
if __name__ == "__main__":
    # 使用 stdio 传输运行 MCP 服务器
    mcp.run()
