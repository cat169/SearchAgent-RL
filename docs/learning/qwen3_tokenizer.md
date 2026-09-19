# Qwen3-4B-Base：模型仓库与 Tokenizer 学习笔记

> 用途：SearchAgent-RL 项目学习、后续复习与面试准备。
>
> 本文只讲 **Hugging Face 模型仓库、Tokenizer、BPE、Chat Template、Token/Token ID** 等基础知识。SFT 的 `labels`、loss mask、Trainer、LoRA、GRPO 等内容留到后续单独整理。
>
> 主要参考模型：`Qwen/Qwen3-4B-Base`。

---

## 1. 先建立整体心智模型

第一次使用 Hugging Face 模型时，最容易把下面几件事混在一起：

- Hugging Face 模型仓库
- `transformers` Python 库
- Tokenizer
- 模型权重
- Chat Template
- `messages`
- 文本字符串
- token
- token ID
- Transformer

先记住整个链路：

```text
Hugging Face Model Repository
        │
        ├── 模型结构配置
        ├── 模型权重
        ├── Tokenizer 文件
        └── 生成配置

应用层数据
messages = list[dict]
        │
        ▼
Chat Template
        │
        ▼
完整文本字符串 text
        │
        ▼
Tokenizer
        │
        ├── pre-tokenization
        └── BPE
        │
        ▼
tokens
        │
        ▼
token IDs / input_ids
        │
        ▼
Embedding
        │
        ▼
Transformer
```

最重要的两句话：

> **Chat Template 解决“结构化对话如何写成一个字符串”。**
>
> **Tokenizer 解决“字符串如何变成模型能处理的 token IDs”。**

这两个步骤不是一回事。

---

## 2. `Qwen/Qwen3-4B-Base` 仓库里到底有什么

当前官方 Hugging Face 仓库中，可以看到类似下面的文件：

```text
Qwen3-4B-Base/
│
├── README.md
├── config.json
├── generation_config.json
│
├── tokenizer_config.json
├── tokenizer.json
├── vocab.json
├── merges.txt
│
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
└── model.safetensors.index.json
```

这些文件可以分成四类。

### 2.1 模型结构：`config.json`

`config.json` 回答的是：

> “应该创建一个什么样的神经网络？”

它不保存真正的模型参数。

当前 `Qwen3-4B-Base` 的配置中可以看到例如：

```text
architecture              Qwen3ForCausalLM
hidden_size               2560
intermediate_size          9728
num_hidden_layers          36
num_attention_heads        32
num_key_value_heads         8
max_position_embeddings 32768
vocab_size             151936
torch_dtype             bfloat16
```

其中：

- `Qwen3ForCausalLM`：Causal Language Model，核心任务是根据前面的 token 预测下一个 token。
- `hidden_size=2560`：Transformer 内部每个位置的主隐藏表示维度。
- `num_hidden_layers=36`：36 层 Transformer block。
- `num_attention_heads=32`、`num_key_value_heads=8`：使用 Grouped Query Attention（GQA）。
- `max_position_embeddings=32768`：模型架构配置中的最大位置长度。
- `vocab_size=151936`：模型 embedding / LM head 所对应的词表规模配置。
- `bfloat16`：官方权重主要使用 BF16 存储。

### 2.2 模型权重：`*.safetensors`

```text
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
```

这些才是真正的神经网络参数，例如：

```text
embedding weights
attention q/k/v projection weights
MLP weights
RMSNorm weights
LM head weights
...
```

4B 参数如果主要使用 BF16，每个参数约占 2 bytes，因此仅权重文件大约就是数 GB 量级。

注意：

> **模型权重大小 ≠ 训练所需显存。**

训练时还需要保存或计算：

```text
weights
gradients
optimizer states
activations
temporary buffers
```

因此全参数训练的显存需求会明显高于“把模型权重加载进来”的大小。

### 2.3 权重索引：`model.safetensors.index.json`

当模型权重被拆成多个 shard 时，这个文件记录：

```text
parameter name → which safetensors shard
```

正常使用 `AutoModelForCausalLM.from_pretrained()` 时，不需要自己处理这些 shard。

### 2.4 Tokenizer 文件

```text
tokenizer_config.json
tokenizer.json
vocab.json
merges.txt
```

这是本文的重点，后面详细讲。

### 2.5 生成配置：`generation_config.json`

它负责保存 `model.generate()` 的一些默认生成设置，例如：

```text
do_sample
temperature
top_p
top_k
max_new_tokens
EOS token
```

它和网络结构不是一回事，也和 tokenizer 的分词规则不是一回事。

---

## 3. `transformers` 库和 Qwen 模型不是一回事

安装：

```bash
pip install transformers
```

安装的是 Hugging Face 的 Python 工具库。

它提供：

```text
AutoTokenizer
AutoModelForCausalLM
AutoConfig
Trainer
Generation utilities
各种模型架构适配代码
```

它不会因为安装 `transformers` 就自动下载 Qwen3-4B 的数 GB 权重。

真正下载某个模型的文件，通常发生在：

```python
AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
```

或：

```python
AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Base")
```

其中：

- `AutoTokenizer` 主要下载 / 读取 tokenizer 相关文件。
- `AutoModelForCausalLM` 才会进一步加载模型配置和大体积权重。

Tokenizer 本身可以在没有 PyTorch 的情况下进行很多工作；真正运行 Qwen 模型的 forward、generation、SFT，则需要 PyTorch 等深度学习运行时。

---

## 4. `AutoTokenizer.from_pretrained()` 到底做了什么

最常见的代码：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B-Base"
)
```

不要把它理解成：

> “只下载一个 tokenizer 文件。”

更准确的心智模型是：

```text
① 定位模型仓库
   Qwen/Qwen3-4B-Base

        ↓

② 读取 tokenizer 配置
   tokenizer_config.json

        ↓

③ 确定应该使用的 tokenizer 实现

        ↓

④ 加载 tokenizer 本体数据
   tokenizer.json
   以及/或者 vocab.json + merges.txt 等

        ↓

⑤ 恢复 added tokens / special tokens / EOS / PAD 等信息

        ↓

⑥ 加载 chat_template

        ↓

⑦ 构造可直接调用的 Python tokenizer 对象
```

`AutoTokenizer` 中的 “Auto” 表示：

> 它根据仓库配置自动选择合适的 tokenizer 实现，而不是要求用户手工指定某个具体 tokenizer class。

---

## 5. 四个最重要的 Tokenizer 文件

### 5.1 `vocab.json`：token 到 ID 的词表

可以把它理解成一个非常大的映射：

```text
token string → integer ID
```

教学示例：

```json
{
  "Who": 1234,
  " wrote": 5678,
  " Hamlet": 9012,
  "?": 30
}
```

模型本身不会处理字符串 `"Hamlet"`，它最终处理的是整数 token ID，再通过 embedding table 查成向量。

所以：

> **`vocab.json` 的核心作用是规定 token 与整数 ID 的对应关系。**

但是，只知道 vocabulary 还不够。因为一个字符串可能有多种切法，最终如何形成 token 还需要 BPE 规则。

### 5.2 `merges.txt`：BPE 合并规则

BPE：Byte Pair Encoding。

最简单的理解：

> 从较小单位出发，根据训练 tokenizer 时学到的合并规则，把经常一起出现的相邻片段逐步合并成更大的 token。

纯教学例子：

```text
h e l l o
```

假设 `merges.txt` 中按优先级存在：

```text
h e
he l
hel l
hell o
```

那么可能得到：

```text
h e l l o
↓
he l l o
↓
hel l o
↓
hell o
↓
hello
```

这些规则是在 **训练 tokenizer 时学习完成的**。使用 tokenizer 时不会重新学习 BPE。

粗略记成：

```text
merges.txt
→ 规定 BPE 如何逐步合并

vocab.json
→ 规定最终 token 对应哪个 ID
```

### 5.3 `tokenizer.json`：Fast Tokenizer 的完整序列化

这是第一次看 Hugging Face 仓库时最容易困惑的文件：

> “已经有 `vocab.json + merges.txt`，为什么还需要 `tokenizer.json`？”

因为 `tokenizer.json` 可以保存更完整的 tokenizer pipeline，例如：

```text
added tokens
normalizer
pre-tokenizer
model (BPE)
post-processor
decoder
其他 tokenizer 行为
```

可以把它理解为：

```text
vocab.json + merges.txt
→ 比较接近 BPE 核心模型的独立文件

tokenizer.json
→ 整个 Fast Tokenizer 流水线的一体化快照
```

因此不要机械理解成：

```text
AutoTokenizer 每次都先读 vocab，再读 merges，再重新构造 tokenizer.json。
```

实际加载路径由 Transformers、tokenizer class 和 Fast tokenizer 支持情况决定。

### 5.4 `tokenizer_config.json`：Transformers 怎么使用这个 tokenizer

如果前三个文件主要回答：

> “文本如何变成 token？”

那么 `tokenizer_config.json` 更多回答：

> “Transformers 应该怎样使用这个 tokenizer？”

其中可以包含：

```text
tokenizer_class
chat_template
eos_token
pad_token
bos_token
added_tokens_decoder
additional_special_tokens
model_max_length
split_special_tokens
...
```

对 SearchAgent-RL 来说，这个文件非常重要，因为后续我们要理解 ChatML、`<think>`、`<search>`、`<information>`、`<answer>` 如何进入模型序列。

---

## 6. Tokenizer 内部到底经过哪些阶段

完整流程不能简单理解成：

```text
一句话
→ 按空格切成英文单词
```

更接近：

```text
完整字符串
     ↓
Normalizer
     ↓
Pre-tokenizer
     ↓
BPE model
(vocab + merges)
     ↓
tokens
     ↓
token IDs
```

### 6.1 Normalizer

负责可能存在的文本规范化，例如 Unicode、空格或其他字符处理。不同模型策略不同，不要默认所有 tokenizer 都会 lowercase 或做同样的 normalization。

### 6.2 Pre-tokenizer

在 BPE 之前做初步边界处理。它处理的是整段字符串，不是“一个单词一个单词地送进模型”。

### 6.3 BPE

BPE 使用已经训练好的 `vocab + merges` 决定最终 subword token。

### 6.4 Decoder

执行反方向：

```text
token IDs
↓
tokens
↓
人类可读字符串
```

因此：

```python
tokenizer.decode(ids)
```

和：

```python
tokenizer.convert_ids_to_tokens(ids)
```

不是一回事。前者目标是恢复自然文本，后者只是查看 tokenizer 内部 token 表示。

---

## 7. Token 不是单词

下面三个判断都不成立：

```text
1 word      = 1 token       ×
1 message   = 1 token       ×
1 character = 1 token       ×
```

一个英文单词可能拆成多个 subword，也可能因为词表中已经存在而成为一个 token。

中文也不能简单认为：

```text
一个汉字 = 一个 token
```

因此对 LLM 来说，真正的计算长度应该看：

```text
token count
```

而不是 word count、character count 或 message count。

这对后续 SFT 的 `max_seq_length` 非常重要。

---

## 8. Message、Text、Token ID 是三个不同层级

### 8.1 应用层：messages

```python
messages = [
    {
        "role": "user",
        "content": "Who wrote Hamlet?"
    }
]
```

这是 Python 中方便程序组织对话的数据结构：`list[dict]`。

Transformer 不认识 Python 的 `list`、`dict`、`role`。

### 8.2 文本层：formatted text

经过 Chat Template 后，可能变成类似：

```text
<|im_start|>user
Who wrote Hamlet?<|im_end|>
<|im_start|>assistant
```

这是一个普通字符串。

### 8.3 模型输入层：token IDs

上面的整段字符串再经过 tokenizer：

```text
text
↓
tokens
↓
token IDs
```

最终模型接收到整数序列。

完整链路：

```text
messages
   ↓
chat template
   ↓
formatted text
   ↓
tokenizer
   ↓
tokens
   ↓
input_ids
   ↓
Transformer
```

核心结论：

> **Message 是应用层结构。**
>
> **Text 是序列化后的字符串。**
>
> **Token ID 才是 Transformer 真正接收的离散整数输入。**

---

## 9. Chat Template 到底是什么

假设：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who wrote Hamlet?"}
]
```

模型不能直接读取这个 Python list。

因此需要：

```python
tokenizer.apply_chat_template(...)
```

把结构化 messages 转换成模型约定的序列格式。

Qwen 使用 ChatML 风格，核心标记类似：

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
...
<|im_end|>
```

当前 `Qwen3-4B-Base` 官方 `tokenizer_config.json` 中已经包含 `chat_template`。

但是必须区分：

> **仓库里存在 Chat Template，不等于 Base 权重已经完成 instruction tuning。**

Chat Template 是输入格式规则；模型权重学会了什么，是训练阶段决定的。

---

## 10. `apply_chat_template()` 中几个关键参数

### 10.1 `tokenize=False`

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

表示只做 Chat Template 序列化，先返回字符串，不立刻变成 token IDs。非常适合学习和调试。

### 10.2 `tokenize=True`

表示：

```text
messages
↓
chat template
↓
text
↓
tokenizer
↓
token IDs
```

一步完成。

### 10.3 `add_generation_prompt=True`

推理时，如果最后一条是用户问题，我们希望提示“下一轮轮到 assistant 开始生成”，因此 template 通常会在末尾添加 assistant 开始标记，但不填 assistant 的答案。

### 10.4 `add_generation_prompt=False`

当完整训练样本中已经包含 assistant 输出时，通常不需要再额外添加新的 assistant generation prompt。具体 SFT renderer 如何使用，后续结合 loss mask 再确定。

---

## 11. Added Token 和 Special Token 不要混为一谈

Tokenizer 不只有基础 BPE vocabulary。

当前 Qwen3 tokenizer 中可以看到类似：

```text
<|im_start|>
<|im_end|>
<tool_call>
</tool_call>
<tool_response>
</tool_response>
<think>
</think>
```

一个重要细节：

> **added token 不一定等价于 `special=true` 的 special token。**

检查一个标签时，应该分别看：

```text
它是否已经存在于 tokenizer？
它对应几个 token ID？
它是否是 added token？
它是否被标记为 special？
```

这和 SearchAgent 标签直接相关：

```text
<search>
<information>
<answer>
```

这些标签即使不是单独的 special token，也不代表不能训练。

也不能看到一个自定义标签就立即：

```python
tokenizer.add_special_tokens(...)
```

因为新增 vocabulary 后，还要考虑模型 embedding / LM head 的尺寸和新参数初始化问题。

---

## 12. 常用 Tokenizer API

### `tokenizer.tokenize(text)`

```python
tokens = tokenizer.tokenize("Who wrote Hamlet?")
```

字符串 → token 字符串列表。

### `tokenizer.encode(text)`

```python
ids = tokenizer.encode(
    "Who wrote Hamlet?",
    add_special_tokens=False,
)
```

字符串 → tokens → token IDs。

### `tokenizer.convert_ids_to_tokens(ids)`

token IDs → tokenizer 内部 token 字符串。

### `tokenizer.decode(ids)`

token IDs → tokens → Decoder → 人类可读字符串。

### `tokenizer(text)`

```python
encoded = tokenizer(text)
```

通常会返回更接近模型实际输入的结构，例如：

```text
input_ids
attention_mask
```

---

## 13. `input_ids` 是什么

假设文本最终形成：

```text
["Who", " wrote", " Hamlet", "?"]
```

再通过 vocabulary 映射成：

```text
[1234, 5678, 9012, 30]
```

这些整数就是 token IDs。

模型 batch 中通常会变成 tensor：

```text
input_ids = tensor([
    [1234, 5678, 9012, 30]
])
```

Transformer 不直接看到字符串。

第一步通常是：

```text
token ID
↓
embedding lookup
↓
向量
↓
Transformer blocks
```

---

## 14. `attention_mask` 先记到什么程度

一个 batch 中不同样本长度不同时，常需要 padding：

```text
Sample A: [token token token token]
Sample B: [token PAD   PAD   PAD]
```

对应常见：

```text
A: [1, 1, 1, 1]
B: [1, 0, 0, 0]
```

基础理解：

> `attention_mask` 用于告诉模型哪些位置是真实输入，哪些位置是 padding。

但不要把它和后面的 **loss mask** 混为一谈。

SearchAgent-RL 中未来的 question / information 都是真实上下文，模型必须看到，但它们是否参与监督 loss 是另一件事，后续通过 `labels` 处理。

---

## 15. Hugging Face Cache

第一次执行：

```python
AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B-Base"
)
```

会把需要的文件放到 Hugging Face cache。

查看默认 cache：

```python
from huggingface_hub import constants

print(constants.HF_HUB_CACHE)
```

也可以指定：

```python
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B-Base",
    cache_dir=".hf_cache",
)
```

工程上通常保留全局 Hugging Face cache 更方便，因为多个项目可以复用同一份模型文件。

如果使用项目内 `.hf_cache`，一定加入 `.gitignore`，不能把模型缓存提交到 GitHub。

---

## 16. `tokenizer.vocab_size` 和 `len(tokenizer)` 为什么可能不同

```python
print(tokenizer.vocab_size)
print(len(tokenizer))
```

二者可能不同。

先这样理解：

```text
vocab_size
≈ 基础 vocabulary 的规模

len(tokenizer)
≈ 当前 tokenizer 实际可索引 token 数，包括额外加入的 tokens
```

因此后续如果人为增加 SearchAgent 标签，需要特别关注模型 embedding size 是否同步。

---

## 17. `model_max_length` 和模型真实 Context Length 不要机械等同

仓库中有时会看到：

```text
config.json 中一个长度
tokenizer_config.json 中另一个 model_max_length
```

不要看到 tokenizer metadata 的一个大数字，就直接认为模型一定能稳定支持该长度。

应该综合检查：

```text
Model Card
config.json
官方文档
RoPE/context extension 说明
实际推理与训练配置
```

本项目 SFT 的 `max_seq_length` 最终也应该根据 Narrow / Standard trajectory 的真实 token 长度分布来确定。

---

## 18. 与 SearchAgent-RL 的连接

我们现在已经有结构化 trajectory：

```json
{
  "id": "...",
  "source": "...",
  "question": "...",
  "answers": ["..."],
  "search_count": 2,
  "events": [
    {"type": "think", "content": "..."},
    {"type": "search", "content": "..."},
    {"type": "information", "content": "..."},
    {"type": "think", "content": "..."},
    {"type": "answer", "content": "..."}
  ]
}
```

它还不是模型能直接训练的 tensor。

后面会经历：

```text
trajectory JSON
       ↓
events → messages
       ↓
Qwen Chat Template
       ↓
formatted text
       ↓
Tokenizer
       ↓
input_ids
       ↓
labels / loss mask
       ↓
Qwen3-4B-Base
       ↓
SFT loss
```

所以：

> **Tokenizer 是“我们清洗好的 Agent 数据”和“模型真正训练 tensor”之间的关键桥梁。**

目前只学到 `input_ids` 这一层。下一阶段再学习：

```text
labels
-100 mask
assistant-only loss
information 为什么可见但不算 loss
causal mask 与 loss mask 的区别
```

---

## 19. 建议长期保留的 Notebook 实验

```python
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Who wrote Hamlet?"

print("Tokenizer type:")
print(type(tokenizer))

print("\nis_fast:")
print(tokenizer.is_fast)

print("\nvocab_size:")
print(tokenizer.vocab_size)

print("\nlen(tokenizer):")
print(len(tokenizer))

print("\ntokens:")
print(tokenizer.tokenize(text))

ids = tokenizer.encode(
    text,
    add_special_tokens=False,
)

print("\nids:")
print(ids)

print("\nids -> tokens:")
print(tokenizer.convert_ids_to_tokens(ids))

print("\ndecode:")
print(tokenizer.decode(ids))
```

重点观察：

1. 一个英文词是否一定对应一个 token？
2. 空格如何反映在内部 token 表示中？
3. `tokenize()` 和 `encode()` 返回值有什么区别？
4. `convert_ids_to_tokens()` 与 `decode()` 有什么区别？
5. `vocab_size` 与 `len(tokenizer)` 是否一致？

---

## 20. 容易混淆的概念速查

### `transformers` ≠ Qwen 模型权重

```text
transformers = Python 工具库
Qwen3-4B-Base = 具体模型配置 + tokenizer + 数十亿参数权重
```

### Chat Template ≠ Tokenizer

```text
Chat Template: messages → formatted text
Tokenizer:     formatted text → token IDs
```

### message ≠ text

```text
message = 结构化 Python 对话对象
text    = chat template 序列化后的字符串
```

### token ≠ word

一个词可以对应多个 token，一个 token 也可能覆盖完整词或常见子词片段。

### `vocab.json` ≠ `merges.txt`

```text
vocab  = token ↔ ID
merges = BPE 合并规则
```

### `tokenizer.json` ≠ `tokenizer_config.json`

```text
tokenizer.json        = Fast Tokenizer 完整流水线的序列化
tokenizer_config.json = Transformers 使用层面的配置
```

### added token ≠ special token

二者可以重叠，但不是同义词。

### attention mask ≠ loss mask

```text
attention mask = 输入位置是否有效/是否是 padding 等计算控制
loss mask      = 哪些 token 不参与监督 loss
```

---

## 21. 面试复习：常见问题怎么回答

### Q1：`AutoTokenizer.from_pretrained()` 做了什么？

> 它根据 Hugging Face 模型仓库中的 tokenizer 配置自动确定合适的 tokenizer 实现，加载 Fast Tokenizer 的序列化信息、词表、BPE 合并规则、added/special tokens 和 chat template 等配置，最终返回一个负责文本与 token ID 相互转换的 tokenizer 对象。

### Q2：`vocab.json` 和 `merges.txt` 有什么区别？

> `vocab.json` 主要保存 token 到整数 ID 的映射；`merges.txt` 保存 BPE 学到的合并规则和优先级，决定较小片段如何逐步组合成更大的 subword token。

### Q3：已经有 `vocab.json + merges.txt`，为什么还有 `tokenizer.json`？

> `vocab.json + merges.txt` 主要描述 BPE 模型本身，而 `tokenizer.json` 是 Fast Tokenizer 的完整序列化，可以包含 vocabulary、merges、normalizer、pre-tokenizer、decoder、added tokens 等整套 pipeline 信息。

### Q4：Chat Template 和 Tokenizer 是一回事吗？

> 不是。Chat Template 先把 `system/user/assistant` 这样的结构化 messages 序列化成模型规定的文本格式；Tokenizer 再把这段字符串转换成 tokens 和 token IDs。

### Q5：Transformer 能直接看到文本吗？

> 不能。字符串先经过 tokenizer 变成 token IDs，token IDs 再经过 embedding lookup 变成向量，Transformer 实际处理的是这些数值表示。

### Q6：一个英文单词是不是一个 token？

> 不一定。LLM tokenizer 通常使用 subword/byte-level BPE 思路，常见词可能成为一个 token，稀有词可能拆成多个 subword token。

### Q7：Base 模型为什么也可能有 chat template？

> Chat Template 是 tokenizer / 输入序列化层面的规则，不代表模型权重已经完成 instruction tuning。Base 与 Instruct 的核心区别仍然是训练阶段和权重学到的行为。

---

## 22. 最后一页速查表

| 对象 | 核心作用 |
|---|---|
| `README.md / Model Card` | 模型说明、用法、限制 |
| `config.json` | 模型网络结构配置 |
| `*.safetensors` | 真正模型参数 |
| `model.safetensors.index.json` | 参数到权重 shard 的索引 |
| `vocab.json` | token → ID 映射 |
| `merges.txt` | BPE merge rules |
| `tokenizer.json` | Fast Tokenizer 完整序列化 |
| `tokenizer_config.json` | tokenizer class、chat template、added/special tokens、EOS/PAD 等 |
| `generation_config.json` | 默认生成参数 |
| `messages` | 程序中的结构化对话 |
| Chat Template | messages → formatted text |
| Tokenizer | text → tokens → token IDs |
| `input_ids` | Transformer 真正接收的离散 token ID 序列 |
| `decode()` | token IDs → 人类可读文本 |

最终主线：

```text
SearchAgent trajectory
        ↓
messages
        ↓
Chat Template
        ↓
formatted text
        ↓
Tokenizer
        ↓
tokens
        ↓
input_ids
        ↓
Embedding
        ↓
Transformer
```

---

## 23. 官方资料

建议以后遇到版本差异时，以官方当前文件为准，不要只依赖博客或二手教程。

- Qwen3-4B-Base Model Card：<https://huggingface.co/Qwen/Qwen3-4B-Base>
- Qwen3-4B-Base Files：<https://huggingface.co/Qwen/Qwen3-4B-Base/tree/main>
- Qwen3-4B-Base `config.json`：<https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/config.json>
- Qwen3-4B-Base `tokenizer_config.json`：<https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/tokenizer_config.json>
- Hugging Face Tokenizer 文档：<https://huggingface.co/docs/transformers/main_classes/tokenizer>
- Hugging Face Tokenizers / BPE model API：<https://huggingface.co/docs/tokenizers/api/models>

> 注意：模型仓库文件会更新。本文中的具体字段和数字基于当前学习阶段看到的官方 Qwen3-4B-Base 配置；真正实现 SearchAgent-RL renderer / SFT 前，应再次读取本地实际加载的 tokenizer 配置进行核对。
