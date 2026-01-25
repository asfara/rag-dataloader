"""
MCP RAG Demo 功能测试脚本
"""
import sys
sys.path.insert(0, '/Users/lipucheng/workspace/rag-dataloader/mcp-rag')

from vector_store import get_vector_store
from rag_service import get_rag_service

def test_vector_store():
    """测试向量存储"""
    print("=" * 50)
    print("测试向量存储...")
    
    vs = get_vector_store()
    
    # 添加测试文档
    ids = vs.add_documents(
        texts=["这是测试文档一", "这是测试文档二", "Python 是一种流行的编程语言"],
        metadatas=[{"title": "doc1"}, {"title": "doc2"}, {"title": "doc3"}]
    )
    print(f"✓ 添加文档成功，ID: {ids}")
    
    # 检索测试
    results = vs.query("编程语言", n_results=2)
    print(f"✓ 检索结果: {len(results['documents'])} 个文档")
    for i, doc in enumerate(results['documents']):
        print(f"  - {doc[:50]}...")
    
    # 统计信息
    stats = vs.get_stats()
    print(f"✓ 知识库统计: {stats['document_count']} 个文档")
    
    print("向量存储测试通过！\n")

def test_rag_service():
    """测试 RAG 服务"""
    print("=" * 50)
    print("测试 RAG 服务...")
    
    rag = get_rag_service()
    
    # 加载示例数据
    result = rag.load_data_directory()
    print(f"✓ 加载示例数据: {result['total_files']} 个文件")
    
    # 检索测试
    docs = rag.retrieve("什么是 RAG", top_k=2)
    print(f"✓ 检索 'RAG' 相关文档: {len(docs)} 个结果")
    
    # 格式化上下文
    context = rag.format_context(docs)
    print(f"✓ 上下文格式化成功，长度: {len(context)} 字符")
    
    print("RAG 服务测试通过！\n")

def test_mcp_tools():
    """测试 MCP 工具导入"""
    print("=" * 50)
    print("测试 MCP 服务器模块...")
    
    from server import mcp, add_document, search, list_documents, get_stats
    
    print(f"✓ MCP 服务器名称: {mcp.name}")
    print("✓ 工具函数导入成功")
    
    # 测试搜索工具 - FastMCP 的 @mcp.tool() 装饰器返回 FunctionTool 对象
    # 需要通过 .fn 属性访问底层函数
    result = search.fn("MCP 协议是什么")
    print(f"✓ 搜索工具调用成功: {result['message']}")
    
    print("MCP 模块测试通过！\n")

if __name__ == "__main__":
    print("\n🚀 开始 MCP RAG Demo 功能测试\n")
    
    try:
        test_vector_store()
        test_rag_service()
        test_mcp_tools()
        
        print("=" * 50)
        print("🎉 所有测试通过！")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
