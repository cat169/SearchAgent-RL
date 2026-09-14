# SearchAgent-RL：Local BM25 Retriever 阶段总结

> 本文记录 SearchAgent-RL 在 Local BM25 Retriever 环境搭建、wiki-18 索引与语料准备、检索链路验证过程中遇到的问题、原因与处理方式，作为后续开发与复习资料。

## 1. 本阶段目标

SearchAgent-RL 当前固定路线：

```text
NQ + HotpotQA + AetherSearch
        ↓
Local BM25 Retriever
        ↓
DeepSeek Teacher Trajectory
        ↓
Agentic SFT
        ↓
veRL GRPO
```

本阶段不直接训练 Agent，而是先验证最底层检索链路：

```text
query
→ BM25 index
→ Top-k docid
→ Wikipedia corpus
→ Top-k passages
```

最终计划封装统一接口：

```python
results = retriever.search(query, top_k=3)
```

返回结构化结果，例如：

```python
[
    {
        "rank": 1,
        "doc_id": "...",
        "title": "...",
        "text": "...",
        "score": 8.12,
    }
]
```

---

## 2. BM25、Lucene、Pyserini、LuceneSearcher 的关系

### BM25

BM25 是 sparse lexical retrieval 算法，根据 query term、term frequency、document frequency 和文档长度等因素对文档进行排序。

```text
Sparse Retrieval
└── BM25
```

它主要依赖词法匹配，不像 E5、DPR 等 dense retriever 那样依赖向量语义表示。

### Lucene

Apache Lucene 是成熟的全文搜索引擎库，负责 inverted index、词项统计、BM25 scoring 等核心检索能力。

### Pyserini

Pyserini 是 Python 信息检索工具包，为 Lucene / Anserini 提供 Python 侧接口。

### LuceneSearcher

项目实际使用：

```python
from pyserini.search.lucene import LuceneSearcher
```

关系可概括为：

```text
BM25：检索算法
Lucene：索引与搜索实现
Pyserini：Python 侧 IR 工具包
LuceneSearcher：Python 调用 Lucene 搜索的接口
```

`LuceneSearcher` 不是一种与 BM25 同级的新检索算法。

---

## 3. 为什么当前不先做 HTTP Retriever Server

Local Retriever 不等于只能单线程、单请求搜索。

“Local”描述的是 index / corpus 部署位置，而不是并发能力。

当前开发阶段只需要：

```text
Python code
    ↓
BM25Retriever.search(query)
    ↓
Local Lucene index
```

未来 GRPO 中会出现多个 rollout worker：

```text
rollout worker 1 ─┐
rollout worker 2 ─┤
rollout worker 3 ─┼→ shared Retriever
rollout worker 4 ─┘
```

届时可以把同一个 `BM25Retriever` 封装为 HTTP、Ray Actor 或其他共享服务，而无需重写核心检索逻辑。

因此当前原则是：

```text
现在：Direct Python Call
以后 GRPO：Shared Retriever Service
```

不提前引入 FastAPI 等额外工程复杂度。

---

## 4. 问题一：原 Python 环境不适合作为 Retriever 环境

原 base 环境 Python 版本较新，因此为降低 Pyserini / Java / Lucene 兼容风险，创建独立 Retriever 环境：

```text
Conda env : searchagent-retriever
Python    : 3.10.21
Java      : 21.0.8
Pyserini  : 0.44.0
datasets  : 5.0.1
```

创建方式：

```bat
conda create -n searchagent-retriever python=3.10 -y
conda activate searchagent-retriever
```

### 经验

Retriever、训练环境、veRL 环境不一定必须共用同一个 Conda 环境。独立环境往往更容易控制依赖和复现。

---

## 5. 问题二：普通 CMD 找不到 conda

最初执行：

```bat
conda create ...
```

报：

```text
'conda' 不是内部或外部命令
```

原因不是当前工作目录错误，而是普通 CMD 没有初始化 Conda。

处理方式：

```bat
D:\environment\Anaconda\Scripts\activate.bat
```

随后再执行：

```bat
conda activate searchagent-retriever
```

### 经验

当前目录和 Conda 是否已加载是两个独立问题。`conda create` 不要求在 Anaconda 安装目录下运行。

---

## 6. 问题三：pip 清华源 SSL 失败

安装：

```bat
python -m pip install pyserini==0.44.0
```

出现：

```text
SSL: UNEXPECTED_EOF_WHILE_READING
No matching distribution found
```

实际并不是 Pyserini 0.44.0 不存在，而是 pip 当前默认镜像连接失败，导致没有成功获取版本列表。

处理方式：

```bat
python -m pip install pyserini==0.44.0 -i https://pypi.org/simple
```

### 经验

看到 `No matching distribution found` 时，不应直接判断“版本不存在”，还要检查：

- 当前 pip index；
- 网络连接；
- SSL；
- Python 版本兼容性。

---

## 7. 问题四：导入 LuceneSearcher 时要求 OpenAI API Key

执行：

```python
from pyserini.search.lucene import LuceneSearcher
```

出现：

```text
openai.OpenAI(...)
Missing credentials
```

原因是 Pyserini 0.44.0 的 import chain 会顺带加载 `pyserini.encode._openai`，并在 import 阶段初始化 OpenAI client。

但 Local BM25 本身不使用 OpenAI API。

临时处理：

```bat
set OPENAI_API_KEY=local-placeholder-pyserini-import-only
```

之后：

```bat
python -c "from pyserini.search.lucene import LuceneSearcher; print('LuceneSearcher import OK')"
```

成功。

### 经验

需要区分：

```text
import-time dependency
```

和：

```text
runtime functionality
```

某依赖在 import 时被加载，不代表真正的 BM25 搜索会调用它。

---

## 8. 问题五：hf CLI 实际来自错误 Python 环境

下载 wiki-18 index 时，报错堆栈出现：

```text
C:\Users\lenovo\AppData\Local\Programs\Python\Python39\...
```

说明当时执行的 `hf.exe` 来自系统 Python 3.9，而不是 `searchagent-retriever`。

应检查：

```bat
where python
where hf
```

确保优先路径属于：

```text
D:\environment\Anaconda\envs\searchagent-retriever\
```

### 经验

涉及 CLI 工具时，不仅要检查 `python --version`，还应检查 CLI executable 本身来自哪个环境。

---

## 9. 问题六：Hugging Face 大文件下载中断

第一次下载 BM25 index 时出现：

```text
httpx.RemoteProtocolError:
Server disconnected without sending a response
```

检查：

```bat
echo %ERRORLEVEL%
```

结果为 `1`，本地仅有约 1.36 GB 文件，说明下载未完成。

处理方式不是删除目录重新下载，而是再次执行相同 `hf download` 命令。

Hugging Face 会利用已有缓存和已完成文件继续补齐，最终成功：

```text
Fetching 29 files: 100%
Download complete
Reconstruction complete
```

### 经验

大文件下载中断后优先使用已有 cache / resume，不要立即删除已经下载的数据。

---

## 10. BM25 index 第一次真实搜索成功

Index 路径：

```text
data/retrieval/wiki-18-bm25-index/bm25
```

测试：

```python
hits = searcher.search("Inception director", 3)
```

返回：

```text
RANK 1 DOCID 2495161 SCORE 8.0382
RANK 2 DOCID 18981818 SCORE 7.6592
RANK 3 DOCID 12550598 SCORE 7.4624
```

这证明以下链路已经正常：

```text
query
→ Pyserini
→ Lucene BM25 index
→ Top-k docid + score
```

---

## 11. 问题七：BM25 index 不包含 Wikipedia 原文

测试：

```python
doc = searcher.doc(0)
print(doc.raw())
```

返回：

```text
RAW_DOCUMENT_NONE
```

说明该 Lucene index 只用于检索，没有保存原始 passage。

因此完整架构必须是：

```text
query
   ↓
BM25 index
   ↓
docid + score
   ↓
wiki-18 corpus
   ↓
passage
```

随后下载官方 wiki-18 corpus：

```text
wiki-18.jsonl.gz
5,123,307,260 bytes
```

---

## 12. 问题八：`wiki-18.jsonl.gz` 实际不是普通 JSONL gzip

最初尝试：

```python
gzip.open(path, "rt", encoding="utf-8")
```

出现：

```text
UnicodeDecodeError
```

先校验 SHA256：

```text
7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db
```

确认文件没有损坏。

随后读取 gzip 解压后的 binary header，出现：

```text
data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
...
ustar
```

`ustar` 表明内部实际是 TAR archive。

因此真实结构为：

```text
wiki-18.jsonl.gz
    ↓ gzip
TAR archive
    ↓
data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
```

虽然扩展名是 `.jsonl.gz`，实际上更接近 `.tar.gz`。

---

## 13. 正确提取 wiki-18 corpus

使用：

```python
tarfile.open(path, "r:gz")
```

查看成员后确认只有一个主要 corpus：

```text
data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
```

解压后大小：

```text
14,393,573,105 bytes
≈ 14.39 GB
```

没有保留原始深层目录，而是将该成员流式提取并重命名：

```text
data/retrieval/wiki-18-corpus/wiki-18.jsonl
```

最终目录：

```text
wiki-18-corpus/
├── wiki-18.jsonl.gz
└── wiki-18.jsonl
```

---

## 14. Corpus 格式验证

读取第一条：

```python
{
    "id": "0",
    "contents": "\"Evan Morris\"\nEvan Morris Evan L. Morris ..."
}
```

确认 schema：

```text
id
contents
```

其中 `contents` 基本形式为：

```text
title
passage text
```

后续 Retriever 可以按第一个换行符拆分 title 和 text。

---

## 15. Lucene docid 与 corpus id 映射验证

BM25 搜索返回：

```text
DOCID = 2495161
```

读取 corpus 的 0-based row index `2495161`：

```text
ID: 2495161
```

因此当前官方 index / corpus 配对下验证得到：

```text
Lucene docid
=
corpus row index
=
corpus id
```

这也是后续能够使用：

```python
corpus[int(hit.docid)]
```

取回 passage 的基础。

---

## 16. BM25 的典型词法歧义案例

Query：

```text
Inception director
```

Top-1 `docid=2495161` 对应 passage：

```text
UNESCO
... UNESCO Director-General ...
... since inception ...
... Director-Generals ...
```

这不是 index/corpus 映射错误，而是 BM25 的 lexical retrieval 特性。

对 BM25 来说：

```text
Inception
```

在电影标题和 `since inception` 中都只是相同 term；

```text
director
```

在电影导演和 `Director-General` 中也可能产生词法匹配。

因此 Retriever implementation 正确，不代表每次 retrieval result 都一定语义正确。

---

## 17. Query Reformulation 的意义

将 query 改成：

```text
Inception 2010 film director
```

得到新的 Top-3：

```text
RANK 1 DOCID 1549096  SCORE 11.0463
RANK 2 DOCID 10559120 SCORE 10.2635
RANK 3 DOCID 6451889  SCORE 10.0136
```

加入 `2010`、`film` 后，提供了更强的词法消歧条件。

这说明 Search Agent 的重要能力之一并不是“第一次搜索必须命中”，而是：

```text
Question
↓
generate search query
↓
observe retrieval result
↓
判断结果是否相关
↓
reformulate query
↓
search again
```

因此未来 Teacher trajectory 中，`<search>` 的质量非常重要。

---

## 18. 需要区分 Implementation Error 和 Retrieval Error

### Implementation Error

例如：

- index 损坏；
- docid 映射错误；
- corpus 不匹配；
- score 排序逻辑错误；
- passage lookup 错误。

### Retrieval Error

例如：

- query 本身有歧义；
- lexical match 命中错误语义；
- BM25 缺乏语义理解；
- query formulation 不够好。

UNESCO 例子属于 Retrieval Error，不属于 Retriever 实现 bug。

---

## 19. 当前已经验证完成

```text
Conda Retriever environment             ✓
Python 3.10                              ✓
Java 21                                  ✓
Pyserini LuceneSearcher                  ✓
wiki-18 BM25 index                       ✓
真实 BM25 Top-k search                   ✓
wiki-18 corpus 下载                      ✓
corpus SHA256                            ✓
特殊 tar.gz 格式识别                     ✓
14.39 GB JSONL corpus 提取               ✓
corpus schema                            ✓
Lucene docid ↔ corpus id                 ✓
BM25 lexical retrieval behavior         ✓
query reformulation 现象                 ✓
```

因此 Local BM25 的底层基础设施验证已经基本完成。

---

## 20. 下一阶段：正式实现 `BM25Retriever`

下一步计划实现：

```text
retrieval/
└── bm25_retriever.py
```

接口目标：

```python
retriever = BM25Retriever(
    index_path="...",
    corpus_path="...",
)

results = retriever.search(
    "Inception 2010 film director",
    top_k=3,
)
```

返回：

```python
[
    {
        "rank": 1,
        "doc_id": "...",
        "title": "...",
        "text": "...",
        "score": 11.0463,
    }
]
```

Retriever 本身只负责：

```text
query → structured documents
```

暂时不负责：

- `<information>...</information>` 序列化；
- DeepSeek Teacher；
- Qwen3 Agent loop；
- FastAPI / HTTP；
- GRPO rollout。

`<information>` 的构造属于后续 Agent environment / renderer 层。

---

## 21. 后续整体路线

```text
[✓] Data Cleaning
    ├── NQ
    ├── HotpotQA
    └── AetherSearch

[进行中] Local Retrieval
    ├── wiki-18 corpus           ✓
    ├── wiki-18 BM25 index      ✓
    ├── Pyserini / Lucene       ✓
    └── BM25Retriever code      ← 下一步

[后续]
BM25Retriever
    ↓
Agent Environment
    ↓
DeepSeek Teacher Trajectory
    ↓
Aligned NQ / HotpotQA Trajectories
    ↓
AetherSearch + Teacher Trajectories
    ↓
Agentic SFT
    ↓
veRL GRPO
```

---

## 22. 面试 / 复习时至少要能回答的问题

### 1. BM25 index 和 corpus 有什么区别？

```text
Index = 为快速检索构建的数据结构
Corpus = 真正的知识文本
```

链路：

```text
query → index → docid → corpus → passage
```

### 2. Pyserini、Lucene、BM25 是什么关系？

```text
BM25           = retrieval algorithm
Lucene         = search/index implementation
Pyserini       = Python IR toolkit
LuceneSearcher = Python-side search interface
```

### 3. 为什么 Local BM25 可以支持后面的 GRPO？

Local 描述数据和 Retriever 的部署位置，不代表只能串行。后续多个 rollout worker 可以共享同一个 Retriever service。

### 4. 为什么 Agent 需要学习 query generation / reformulation？

因为 BM25 是词法检索，无法可靠理解词义和实体歧义。Agent 需要根据 observation 判断结果质量，并在必要时改写 query 再搜索。

### 5. 为什么 Retriever 和 Agent Environment 要分层？

Retriever 应稳定提供：

```text
query → structured documents
```

Agent Environment 再处理：

```text
documents → <information> → model continuation
```

这样 Teacher、SFT、GRPO、HTTP 服务等都可以复用同一 Retriever 核心。

---

## 23. 一句话总结

本阶段已经完成从“没有 Retriever”到“真实 wiki-18 + 官方 Lucene BM25 index + Pyserini search + docid→passage 映射”的完整底层验证；下一步将这些经过白盒验证的组件封装成正式的 `BM25Retriever.search(query, top_k)` 接口。
