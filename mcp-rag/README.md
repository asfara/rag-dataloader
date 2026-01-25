# MCP RAG Demo 服务

基于 FastMCP 和 ChromaDB 的 RAG（检索增强生成）演示服务。

## 功能特性

- 📚 **文档管理**：添加、列出、清空知识库文档
- 🔍 **语义搜索**：基于向量相似度的智能检索
- 🔌 **MCP 协议**：标准化的 AI 工具接口
- 💾 **持久化存储**：使用 ChromaDB 本地存储向量数据

## 安装

### 1. 激活 Conda 环境

```bash
conda activate rag-project01
```

### 2. 安装依赖

```bash
cd /Users/lipucheng/workspace/rag-dataloader/mcp-rag
pip install -r requirements.txt
```

## 使用方法

### 开发模式

使用 FastMCP 开发工具进行测试：

```bash
fastmcp dev server.py
```

这将启动一个交互式界面，可以直接测试各个 MCP 工具。

### 生产模式

直接运行服务器：

```bash
python server.py
```

### 在 AI 客户端中使用

将此服务配置为 MCP 服务器，在支持 MCP 的 AI 客户端（如 Claude Desktop）中使用：

```json
{
  "mcpServers": {
    "rag-demo": {
      "command": "python",
      "args": ["/Users/lipucheng/workspace/rag-dataloader/mcp-rag/server.py"],
      "env": {}
    }
  }
}
```

## MCP 工具列表

| 工具名 | 描述 | 参数 |
|--------|------|------|
| `add_document` | 添加文档到知识库 | `content`, `title` |
| `search` | 语义搜索相关文档 | `query`, `top_k` |
| `list_documents` | 列出所有文档 | `limit` |
| `load_sample_data` | 加载示例数据 | - |
| `clear_knowledge_base` | 清空知识库 | - |

## MCP 资源

| 资源 URI | 描述 |
|----------|------|
| `rag://stats` | 获取知识库统计信息 |

## 项目结构

```
mcp-rag/
├── server.py           # FastMCP 服务器入口
├── rag_service.py      # RAG 核心服务
├── vector_store.py     # ChromaDB 向量存储封装
├── config.py           # 配置管理
├── requirements.txt    # Python 依赖
├── data/               # 示例文档目录
│   └── sample.txt      # 示例文档
└── README.md           # 本文件
```

## 技术栈

- **FastMCP**: MCP 服务器框架
- **ChromaDB**: 向量数据库
- **sentence-transformers**: 文本嵌入模型

## License

MIT
