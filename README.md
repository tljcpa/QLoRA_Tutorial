# Qwen2.5-3B QLoRA 微调全流程教程

适用环境：Google Colab 免费层 T4 GPU（15GB 显存 + 12GB 内存）  
涵盖：QLoRA 训练 → 权重合并 → GGUF 量化 三阶段完整闭环

---

> **阅读说明**
>
> `[必须改]` 标注的内容：每次使用必须根据你的实际情况修改。  
> `[不用改]` 标注的内容：固定逻辑，直接复制使用，无需改动。  
> `[可选改]` 标注的内容：有默认经验值，理解含义后可以按需调整。
>
> 建议第一次使用时通读全文，之后可以直接跳到各阶段代码区复制执行。

---

## 目录

- [第零部分：核心概念](#第零部分核心概念先搞清楚为什么)
- [第一阶段：QLoRA 训练](#第一阶段qlora-训练)
- [第二阶段：权重合并](#第二阶段权重合并)
- [第三阶段：GGUF 量化](#第三阶段gguf-量化)
- [踩坑总结：5 个高发错误](#踩坑总结5-个高发错误)
- [附录 A：换数据集时怎么改](#附录-a换数据集时怎么改)
- [附录 B：换模型时怎么改](#附录-b换模型时怎么改)
- [附录 C：参数调优速查](#附录-c参数调优速查)
- [附录 D：在 Colab 中快速测试 GGUF 模型](#附录-d在-colab-中快速测试-gguf-模型)

---

## 第零部分：核心概念——先搞清楚为什么

在动手之前，你需要理解三个核心问题。不理解这些，出了问题你不知道从哪里排查。

### 0.1 微调是什么，和预训练有什么区别

**预训练**：从零开始，用海量数据（几千亿个词）训练模型，让它学会语言和通用知识。成本极高，普通人无法做到。

**微调**：在已有的预训练模型基础上，用少量专业数据继续训练，让模型适应特定任务或风格。成本可控，这就是我们要做的事。

类比：预训练是让一个人从零开始上学读书十几年。微调是让一个大学毕业生参加一周的岗前培训，学习你公司的特定业务流程。

### 0.2 为什么用 LoRA，不直接全量微调

全量微调需要修改模型的全部参数。Qwen2.5-3B 有 30 亿个参数，每个参数用 float16 存储占 2 字节，仅模型本身就需要 6GB 显存。加上训练时的梯度、优化器状态，总需求超过 40GB，T4 的 15GB 根本装不下。

LoRA（Low-Rank Adaptation）的核心思想：不修改原始参数，而是在旁边插入两个极小的矩阵 A 和 B，只训练这两个小矩阵。训练完成后，把 A×B 的结果加回原始权重。

```
W_new = W_original + (alpha / r) × A × B
```

实际效果：只训练约 0.6% 的参数，显存需求从 40GB 降到 6GB 以内，效果接近全量微调。

### 0.3 三个最重要的参数：r、alpha、dropout

| 参数 | 含义 | 经验值 |
|------|------|--------|
| r（rank） | LoRA 矩阵的秩，控制表达能力。r 越大能学习越复杂的变化，参数也越多 | 数据少用 8 或 16，数据多用 32 或 64 |
| alpha | 缩放系数，真正的缩放比例是 alpha/r，控制 LoRA 对原始模型的影响力 | 始终保持 alpha = 2 × r |
| dropout | 随机关闭部分神经元，防止过拟合 | 数据量少时 0.05~0.1，数据量大时可以设 0 |

**为什么需要 alpha？** 因为 r 的大小直接影响 A×B 的数值量级，r 越大数值越大。alpha/r 这个比值是补偿机制，让你调 r 的时候不用同时重新调学习率。固定 alpha=2r，缩放比例永远是 2，含义明确。

### 0.4 QLoRA 比 LoRA 多做了什么

QLoRA = Quantization + LoRA。在 LoRA 的基础上，把基座模型用 4bit 量化加载，显存从 6GB 进一步降到约 2GB。量化后的模型参数不参与训练，只有 LoRA 的 A、B 矩阵用 float16 训练。

代价：量化会损失一点点精度，但实验证明对最终效果影响很小，用显存换精度完全值得。

---

## 第一阶段：QLoRA 训练

> **注意事项（训练前必读）**
>
> 1. 运行前先挂载 Google Drive，所有产物保存到 Drive，防止断线丢失。
> 2. 确保你的数据文件已上传到 Drive 的 data/ 目录。
> 3. 第一次运行建议用少量数据（100 条以内）验证流程跑通，再换完整数据集。
> 4. 训练过程中不要关闭浏览器标签页，Colab 会话会断开。
> 5. 训练结束后，不要在同一个会话里直接做第二阶段，必须重启会话释放显存。

### Cell 1：环境初始化

作用：挂载 Drive，安装依赖，创建目录结构。每次新会话开始时运行一次。

```python
# 挂载 Google Drive，防止断线后产物丢失
from google.colab import drive
drive.mount('/content/drive')

# 安装所有核心依赖（-q 表示静默模式，-U 表示安装最新版）
!pip install -q -U transformers peft bitsandbytes accelerate datasets trl tensorboard

# 创建项目目录结构
import os
PROJECT_DIR = '/content/drive/MyDrive/Qwen-QLoRA-Project'  # [必须改] 改成你想要的项目名
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')             # 数据文件放在这里
OUTPUT_DIR  = os.path.join(PROJECT_DIR, 'output')           # 所有输出产物保存在这里
for d in [DATA_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)
print(f'环境就绪，工程主目录: {PROJECT_DIR}')
```

### Cell 2：训练主程序

作用：加载模型、注入 LoRA、加载数据、执行训练、保存 Adapter。

#### 第一块：清理环境 + 全局配置

这是每次换项目时唯一需要修改的区域，其他代码不用动。

```python
import os, torch, torch.nn as nn, bitsandbytes as bnb
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# [不用改] 清理可能导致 bf16 报错的环境变量（T4 不支持 bf16）
if 'ACCELERATE_MIXED_PRECISION' in os.environ:
    del os.environ['ACCELERATE_MIXED_PRECISION']

# ==================== 全局配置区（每次换项目只改这里）====================
CFG = {
    'model_id'     : 'Qwen/Qwen2.5-3B-Instruct',                             # [必须改] 换模型时改这里
    'dataset_path' : '/content/drive/MyDrive/Qwen-QLoRA-Project/data/train.jsonl', # [必须改] 你的数据文件路径
    'output_dir'   : '/content/drive/MyDrive/Qwen-QLoRA-Project/output',     # [必须改] 输出目录
    'epochs'       : 3,     # [可选改] 训练轮数，数据少时用 1~2，数据多时用 3~5
    'lr'           : 1e-4,  # [可选改] 学习率，LoRA 微调通常 1e-4 到 5e-4
    'r'            : 16,    # [可选改] LoRA rank，数据少用 8，数据多用 32
    'alpha'        : 32,    # [可选改] 始终保持 alpha = 2 × r
    'dropout'      : 0.05   # [可选改] 过拟合严重时加大到 0.1，数据多时设 0
}

# [不用改] T4 显存限制，batch_size 只能是 1，用梯度累积补偿
# 等效 batch_size = 1 × 16 = 16
bs, grad_accum = 1, 16
```

#### 第二块：加载 Tokenizer 和模型

```python
# [不用改] 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(CFG['model_id'], trust_remote_code=True)
# pad_token 用于填充不同长度的样本到同一 batch，LLaMA/Qwen 默认没有，用 eos_token 代替
tokenizer.pad_token = tokenizer.eos_token

# [不用改] 4bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 用 4bit 加载，显存减少 4 倍
    bnb_4bit_quant_type='nf4',              # NF4 量化，误差最小的 4bit 方式
    bnb_4bit_compute_dtype=torch.float16,   # 存储 4bit，计算时临时解压成 fp16
    bnb_4bit_use_double_quant=True          # 双重量化，再省约 200~300MB 显存
)

# [不用改] 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    CFG['model_id'],
    quantization_config=bnb_config,
    device_map='auto',          # 自动分配层到 GPU/CPU
    trust_remote_code=True
)
```

#### 第三块：稳定性配置（三行必须写）

```python
# [不用改] 关闭推理时的 KV Cache，训练时用不到，开着白占显存
model.config.use_cache = False

# [不用改] 梯度检查点：不保存所有中间激活值，用重新计算换显存
# 代价：训练速度慢约 20%，收益：显存减少约 50%，T4 上必须开
model.gradient_checkpointing_enable()

# [不用改] 4bit 量化后梯度追踪被打断，这行恢复它
# 不加这行 LoRA 参数不会被训练，Loss 不下降但你不知道原因
model.enable_input_require_grads()

# [不用改] 为 4bit 训练做进一步稳定性处理
model = prepare_model_for_kbit_training(model)
```

#### 第四块：注入 LoRA

```python
# [不用改] 自动嗅探模型里所有线性层的名字
# 这样写的原因：不同模型层名字不同（有的叫 q_proj，有的叫 query）
# 硬编码层名换个模型就报错，动态嗅探对任何模型都通用
cls_types = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, nn.Linear)
target_modules = list({
    name.split('.')[-1]
    for name, module in model.named_modules()
    if isinstance(module, cls_types)
    and name.split('.')[-1] != 'lm_head'  # 排除输出层，风险高
})

# [不用改，参数在 CFG 里改] 将 LoRA 注入模型
model = get_peft_model(model, LoraConfig(
    r=CFG['r'],
    lora_alpha=CFG['alpha'],
    target_modules=target_modules,
    lora_dropout=CFG['dropout'],
    bias='none',          # 不训练 bias，省参数，效果无明显差异
    task_type='CAUSAL_LM'
))

# 打印可训练参数占比，正常应该在 0.5%~2% 左右
model.print_trainable_parameters()
```

#### 第五块：数据加载与格式化

> **数据格式要求**
>
> 你的 train.jsonl 文件每一行必须是一个 JSON 对象，包含三个字段：
> - `instruction`：任务描述（必填）
> - `input`：附加输入内容（没有时填空字符串 `""`）
> - `output`：期望模型输出的答案（必填）
>
> 示例：
> ```json
> {"instruction": "帮我写一个排序函数", "input": "", "output": "def sort(lst): return sorted(lst)"}
> {"instruction": "翻译成英文", "input": "今天天气很好", "output": "The weather is nice today"}
> ```

```python
# [必须改] 数据加载
# 如果是 jsonl 或 json 文件：
dataset = load_dataset('json', data_files=CFG['dataset_path'], split='train')
# 如果是 csv 文件，把 'json' 改成 'csv'
# 如果是 HuggingFace 上的数据集，改成：
# dataset = load_dataset('数据集名字', split='train')

# [必须改] 格式化函数：把你的数据转成 Qwen 能理解的对话格式
# <|im_start|> 和 <|im_end|> 是 Qwen 系列的特殊分隔符，不要改动
def format_prompts(examples):
    texts = []
    for inst, inp, out in zip(
        examples['instruction'],  # [必须改] 换成你的字段名
        examples['input'],        # [必须改] 换成你的字段名，没有这个字段就删掉这行
        examples['output']        # [必须改] 换成你的字段名
    ):
        # 如果你的数据没有 input 字段，把 f'{inst}\n{inp}' 改成 f'{inst}'
        text = f'<|im_start|>user\n{inst}\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>'
        texts.append(text)
    return {'text': texts}

train_dataset = dataset.map(format_prompts, batched=True)
```

> **如果你的数据字段名不叫 instruction/input/output**
>
> 只需要改 format_prompts 函数里 zip() 括号内的字段名，其他代码完全不动。  
> 例如字段名是 question 和 answer：
> ```python
> for q, a in zip(examples['question'], examples['answer']):
>     text = f'<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>'
> ```

#### 第六块：训练参数与执行

```python
# [不用改] 训练参数（已针对 T4 调优，不要随意修改）
training_args = SFTConfig(
    output_dir=CFG['output_dir'],
    per_device_train_batch_size=bs,         # T4 只能是 1，否则 OOM
    gradient_accumulation_steps=grad_accum, # 等效 batch_size=16
    optim='paged_adamw_8bit',               # 8bit 优化器，内存不够时自动换页到 CPU
    logging_steps=10,                       # 每 10 步打印一次 Loss
    save_strategy='epoch',                  # 每个 epoch 保存一次 checkpoint
    learning_rate=CFG['lr'],
    fp16=False,  # 必须 False：T4 不支持 bf16，fp16=True 会触发 bf16 代码路径导致崩溃
    bf16=False,  # 必须 False：同上
    num_train_epochs=CFG['epochs'],
    dataset_text_field='text',              # 告诉 trainer 用哪个字段
    max_length=512,                         # 超过 512 token 的样本会被截断
    packing=False                           # 不把多个短样本打包成一个，避免上下文混乱
)

# [不用改] 创建训练器并启动
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args
)

print('开始训练...')
trainer.train()

# [不用改] 保存 LoRA Adapter（只有几十 MB，不是完整模型）
adapter_path = os.path.join(CFG['output_dir'], 'lora-adapter')
trainer.model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f'Adapter 已保存至: {adapter_path}')
```

#### 进阶技巧：如何应对 Colab 断线（断点续训）

Colab 免费层随时可能断线。由于我们配置了 `save_strategy='epoch'`，每个 epoch 结束时都会自动保存 checkpoint 到 `output_dir`。

断线重连后，重新运行 Cell 1（环境初始化）和 Cell 2 的前五块（不要执行 `trainer.train()`），然后把最后的训练执行部分改成：

```python
# [不用改] 从最近一次 checkpoint 恢复训练，自动找到 output_dir 下最新的 checkpoint
print('开始训练（从断点恢复）...')
trainer.train(resume_from_checkpoint=True)
```

`resume_from_checkpoint=True` 会自动扫描 `output_dir`，找到最新的 checkpoint 目录接着跑，不需要手动指定路径。

### 训练过程中如何判断是否正常

**Loss 正常下降**：从初始值（通常 2~4）持续降低到 0.5~1.5 左右，说明训练正常。

**Loss 不下降**：大概率是数据格式有问题，或者 `enable_input_require_grads()` 没有生效。打印几条 format_prompts 的输出检查格式是否正确。

**Loss 很快降到接近 0**：过拟合，模型在背训练数据。减少 epochs 或增大 dropout。

**显存 OOM**：先确认 `fp16=False` 和 `bf16=False`，然后把 `max_length` 从 512 降到 256。

---

## 第二阶段：权重合并

> **强制要求：运行第二阶段前必须重启会话**
>
> 点击顶部菜单：代码执行程序 -> 重新启动会话
>
> 原因：训练后 GPU 显存和 RAM 都几乎被占满。合并操作需要在 CPU 上加载完整的 float16 模型（约 6GB），内存不释放直接运行必然 OOM。
>
> 重启后先运行一次：`!pip install -q -U transformers peft`  
> 然后再运行下方代码。

### 为什么合并是必要的

训练完成后你有两样东西：原始基座模型（6GB）和 LoRA Adapter（几十 MB）。第三阶段的 llama.cpp 只接受一个完整的模型文件，不理解「基座 + 补丁」这种分离格式，所以必须先合并。

合并的数学本质：对每个被 LoRA 修改的层，计算 `W_merged = W + (alpha/r) × A × B`，然后删除 A 和 B，留下普通的 W_merged。这就是 `merge_and_unload()` 的含义：merge（合并）+ unload（卸载 LoRA 结构）。

### Cell 3：防 OOM 权重合并

```python
import os, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==================== 路径配置（只改这三行）====================
BASE_MODEL  = 'Qwen/Qwen2.5-3B-Instruct'                                     # [必须改] 你用的基座模型
ADAPTER_DIR = '/content/drive/MyDrive/Qwen-QLoRA-Project/output/lora-adapter' # [必须改] 第一阶段保存的 Adapter 路径
MERGED_DIR  = '/content/drive/MyDrive/Qwen-QLoRA-Project/output/merged-fp16'  # [必须改] 合并后的模型保存路径

# [不用改] 临时 offload 目录放在本地盘（比 Drive 快得多）
OFFLOAD_DIR = '/content/offload_cache'
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# [不用改] 以 CPU Offload 模式加载基座模型
# device_map={'': 'cpu'}：强制所有层放 CPU，不碰 GPU
# low_cpu_mem_usage=True：按层逐一加载，防止整个文件同时在内存里
# offload_folder：内存不够时把暂时用不到的层换页到磁盘
# offload_state_dict=True：把 state dict 本身也换页到磁盘，进一步压低内存峰值
print('1. 加载基座模型（按层加载，约需 1~3 分钟）...')
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={'': 'cpu'},
    low_cpu_mem_usage=True,
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,   # 把 state dict 换页到磁盘，对内存吃紧的环境有帮助
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# [不用改] 挂载 Adapter 并执行合并
# PeftModel.from_pretrained 把 A、B 矩阵附着到基座模型对应的层旁边
# merge_and_unload 执行 W = W + (alpha/r)×A×B，然后删除 A 和 B
print('2. 挂载 Adapter 并合并权重...')
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
merged_model = peft_model.merge_and_unload()

# [不用改] 强制释放旧对象内存
# del 只删除变量名，gc.collect() 才真正回收内存
# 必须同时 del 两个变量，因为 peft_model 内部持有对 base_model 的引用
# 不做这步，保存时内存里同时存在旧对象+序列化数据，必然 OOM
print('3. 释放旧对象内存...')
del base_model
del peft_model
gc.collect()

# [不用改] 分片保存合并后的模型
# max_shard_size='1GB'：每次只序列化 1GB，避免整个 6GB 同时在内存里（需要额外 6GB）
# safe_serialization=True：使用更安全的 safetensors 格式
print('4. 分片保存模型（约需 3~5 分钟）...')
merged_model.save_pretrained(
    MERGED_DIR,
    safe_serialization=True,
    max_shard_size='1GB'
)
tokenizer.save_pretrained(MERGED_DIR)
print(f'合并完成！模型位于: {MERGED_DIR}')
```

### 合并成功后的目录结构

合并完成后，MERGED_DIR 目录下应该有以下文件，缺少任何一个后续步骤都可能报错：

```
merged-fp16/
├── model-00001-of-00006.safetensors    <- 模型权重分片（每块约 1GB）
├── model-00002-of-00006.safetensors
├── ...
├── model.safetensors.index.json         <- 索引文件，记录每层在哪个分片里
├── config.json                          <- 模型结构配置
├── tokenizer.json                       <- 词表
├── tokenizer_config.json                <- tokenizer 配置
└── special_tokens_map.json              <- 特殊 token 映射
```

---

## 第三阶段：GGUF 量化

### 为什么需要这一步

合并后的模型是 float16 格式，大小约 6GB，需要显卡才能推理。GGUF 量化做了两件事：

**第一**：把 float16（每参数 16bit）压缩成 Q4_K_M（每参数约 4bit），文件从 6GB 缩小到约 1.8GB。

**第二**：生成的 GGUF 格式专为 CPU 推理优化，配合 llama.cpp 可以在没有显卡的普通电脑上运行，速度比 Python 框架快 5~10 倍。

**为什么是两步（先转 FP16 GGUF，再量化到 Q4）？** 因为 llama.cpp 的量化工具只接受 GGUF 格式作为输入，必须先把 HuggingFace 格式转成 GGUF，再做精度压缩。

### Cell 4：GGUF 量化转换

这个 Cell 用 `%%bash` 运行，整个 Cell 是一个 shell 脚本，不是 Python。不需要重启会话，接着第二阶段的会话直接运行即可。

```bash
%%bash

# ==================== 路径配置（只改这里）====================
STABLE_COMMIT='b4600'  # [可选改] llama.cpp 的稳定版本号，如果这个版本将来出问题再换
MERGED_MODEL_PATH='/content/drive/MyDrive/Qwen-QLoRA-Project/output/merged-fp16'  # [必须改]
GGUF_DIR='/content/drive/MyDrive/Qwen-QLoRA-Project/output/gguf'                   # [必须改]
GGUF_FP16="$GGUF_DIR/qwen2.5-3b-fp16.gguf"    # [可选改] 中间产物文件名
GGUF_Q4="$GGUF_DIR/qwen2.5-3b-q4_k_m.gguf"   # [可选改] 最终产物文件名
# =============================================================

# 创建输出目录；rm -rf 删除旧的 llama.cpp（防止 clone 时因目录已存在而报错）
mkdir -p $GGUF_DIR
rm -rf /content/llama.cpp

echo '1. 克隆 llama.cpp 并切换到稳定版本...'
cd /content
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout $STABLE_COMMIT   # 锁定版本，防止新版本 API 变化导致脚本失效
pip install -q gguf            # convert_hf_to_gguf.py 的 Python 依赖

echo '2. 编译量化工具（CMake 方式，Makefile 已废弃）...'
# cmake -B build：配置阶段，检查编译器和依赖，结果放到 build 目录
# cmake --build build：编译阶段
# --config Release：开启优化，比 Debug 快 3~10 倍
# -j：并行编译，用所有 CPU 核心
# --target llama-quantize：只编译量化工具，不编译其他（节省时间）
cmake -B build && cmake --build build --config Release -j --target llama-quantize

echo '3. 修复 tokenizer_config.json（覆盖 save_pretrained 的已知 Bug）...'
# HuggingFace 的 save_pretrained 有时会把 special tokens 从字典写成列表
# llama.cpp 转换脚本读取时期望字典，看到列表就报错
# 解法：直接从官方仓库下载原版文件覆盖
# [必须改] 如果你换了模型，把 URL 里的模型名改掉
wget -q -O $MERGED_MODEL_PATH/tokenizer_config.json \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/tokenizer_config.json

echo '4. 转换 HuggingFace 格式 -> FP16 GGUF...'
# 必须先转成 fp16 GGUF，才能再量化成 Q4
python convert_hf_to_gguf.py $MERGED_MODEL_PATH --outfile $GGUF_FP16 --outtype f16

echo '5. 执行 Q4_K_M 量化...'
# find 动态查找编译产物路径（不同系统路径不同，动态查找更稳）
QUANTIZE_BIN=$(find build -name llama-quantize -type f | head -n 1)
# 参数顺序：输入文件 输出文件 量化方式
# Q4_K_M：4bit 量化，K-Quant 算法，Medium 精度，性价比最高的方案
$QUANTIZE_BIN $GGUF_FP16 $GGUF_Q4 Q4_K_M

echo "全流程完成！最终模型位于: $GGUF_Q4"
```

### 量化完成后如何使用

**方式一（推荐新手）**：下载 LM Studio，把 .gguf 文件拖进去，直接图形界面聊天。

**方式二（服务器部署）**：使用 llama.cpp 的 llama-cli 或 llama-server 命令行工具直接推理。

---

## 踩坑总结：5 个高发错误

遇到新模型跑不通时，按这个顺序排查。

### 坑 1：TRL 库 API 版本变化

- **报错现象**：`AttributeError: dataset_text_field 或 max_seq_length 参数不存在`
- **根本原因**：TRL 新版本把训练参数从 SFTTrainer 移入了独立的 SFTConfig 对象，参数名也从 `max_seq_length` 改成了 `max_length`
- **解决方法**：把所有训练参数（包括 `dataset_text_field` 和 `max_length`）都放进 SFTConfig，不要放在 SFTTrainer 里

### 坑 2：BF16 幽灵报错

- **报错现象**：`RuntimeError: not implemented for 'BFloat16'`，即使你没有设置 `bf16=True`
- **触发链条**：`fp16=True` -> 启动 GradScaler -> GradScaler 调用底层算子 -> 该算子内有 Qwen 的 bf16 代码路径 -> T4 不支持 bf16 -> 崩溃
- **根本原因**：T4 是老架构，不支持 bf16。Qwen 底层代码里有 bf16 路径，被 GradScaler 间接触发
- **解决方法**：`fp16=False` 且 `bf16=False`，直接关闭 GradScaler，那条危险路径永远不会被触碰

### 坑 3：合并阶段 OOM

- **报错现象**：`Your session crashed after using all available RAM`，或进度卡在 0%
- **根本原因**：训练后内存已满；保存时 HuggingFace 默认把整个模型序列化成一个大文件，需要额外 6GB 内存，超出上限
- **解决方法**：①必须重启会话再运行合并。②保存时加 `max_shard_size='1GB'`。③保存前执行 `del + gc.collect()`

### 坑 4：tokenizer_config.json 结构错误

- **报错现象**：`AttributeError: 'list' object has no attribute 'keys'`，发生在 GGUF 转换步骤
- **根本原因**：`save_pretrained()` 的已知 Bug，有时把 special tokens 的字典结构错误写成列表结构
- **解决方法**：用 wget 从 HuggingFace 官方仓库下载原版 `tokenizer_config.json` 直接覆盖，绕过这个 Bug

### 坑 5：llama.cpp 构建系统变化

- **报错现象**：`Makefile build is deprecated` 或找不到 `llama-quantize` 文件
- **根本原因**：llama.cpp 已废弃 Makefile，全面迁移到 CMake 构建系统
- **解决方法**：使用 `cmake -B build && cmake --build build` 替代 make 命令

---

## 附录 A：换数据集时怎么改

换数据集只需要改两个地方，其他代码完全不动。

**第一处：dataset_path**

```python
CFG = {
    'dataset_path': '/content/drive/MyDrive/你的项目/data/你的文件.jsonl',  # 改这里
    # 其他不动
}
```

**第二处：format_prompts 里的字段名**

```python
# 如果你的数据有 instruction / input / output 三个字段（和默认一样）
# -> 什么都不用改

# 如果你的数据有两个字段，比如 question 和 answer
def format_prompts(examples):
    texts = []
    for q, a in zip(examples['question'], examples['answer']):
        text = f'<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>'
        texts.append(text)
    return {'text': texts}

# 如果你的数据是对话格式（conversations 字段）
def format_prompts(examples):
    texts = []
    for conv in examples['conversations']:
        text = ''
        for turn in conv:
            if turn['role'] == 'user':
                text += f'<|im_start|>user\n{turn["content"]}<|im_end|>\n'
            elif turn['role'] == 'assistant':
                text += f'<|im_start|>assistant\n{turn["content"]}<|im_end|>\n'
        texts.append(text)
    return {'text': texts}

# 如果是 csv 文件，把加载方式改成：
dataset = load_dataset('csv', data_files=CFG['dataset_path'], split='train')
```

---

## 附录 B：换模型时怎么改

换模型需要改四处，都是改字符串，不涉及逻辑变化。

**第一阶段改两处**

```python
# 第一处：CFG 里的 model_id
CFG = {
    'model_id': '新模型的HuggingFace-ID',
    # ...
}

# 第二处：format_prompts 里的对话格式
# 不同模型用不同的特殊 token，必须匹配

# Qwen 系列（当前文档使用的格式）
text = f'<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>'

# LLaMA-3 系列
text = f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{inst}<|eot_id|>'
text += f'<|start_header_id|>assistant<|end_header_id|>\n{out}<|eot_id|>'

# 最安全的方式：用 tokenizer 自带的 apply_chat_template（推荐，自动适配任何模型）
messages = [{'role': 'user', 'content': inst}, {'role': 'assistant', 'content': out}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
```

**第二阶段改一处**

```python
BASE_MODEL = '新模型的HuggingFace-ID'  # 和第一阶段保持一致
```

**第三阶段改一处**

```bash
# wget 下载 tokenizer_config.json 时，URL 里的模型名也要改
wget -q -O $MERGED_MODEL_PATH/tokenizer_config.json \
  https://huggingface.co/新模型的HuggingFace-ID/resolve/main/tokenizer_config.json
```

---

## 附录 C：参数调优速查

| 问题现象 | 调整方向 |
|---------|---------|
| Loss 不下降 | 检查数据格式是否正确；检查 enable_input_require_grads() 是否调用；适当增大 learning_rate |
| Loss 下降但测试效果差（过拟合） | 减少 epochs；增大 dropout 到 0.1；减小 r 到 8；增加数据量 |
| Loss 一直很高（欠拟合） | 增大 epochs；增大 r 到 32 或 64；对应调整 alpha；检查数据质量 |
| 显存 OOM（训练时） | 确认 fp16=False；减小 max_length 到 256；确认 gradient_checkpointing 已开启 |
| 内存 OOM（合并时） | 重启会话后再合并；确认 max_shard_size='1GB'；确认 del + gc.collect() 已执行 |
| 训练速度太慢 | T4 的硬件限制，gradient_checkpointing 牺牲 20% 速度换显存，无法避免 |
| 效果比预期差 | 优先增加数据量和提升数据质量，这比调参数有效得多 |

---

## 附录 D：在 Colab 中快速测试 GGUF 模型

下载 1.8GB 的文件需要时间，可以在 Colab 里当场验证量化模型是否正常。

注意：第三阶段编译时用了 `--target llama-quantize`，只编译了量化工具。测试前需要先补充编译 `llama-cli`。

```bash
%%bash

# 第一步：补充编译 llama-cli（第三阶段只编译了 llama-quantize，没有编译这个）
cd /content/llama.cpp
cmake --build build --config Release -j --target llama-cli

# 第二步：运行测试
GGUF_Q4='/content/drive/MyDrive/Qwen-QLoRA-Project/output/gguf/qwen2.5-3b-q4_k_m.gguf'  # [必须改]

LLAMA_CLI=$(find build -name llama-cli -type f | head -n 1)

$LLAMA_CLI \
  -m $GGUF_Q4 \
  -n 512 \
  --color \
  -p '<|im_start|>system\n你是一个有用的AI助手。<|im_end|>\n<|im_start|>user\n你好，请做个自我介绍。<|im_end|>\n<|im_start|>assistant\n'

# 参数说明：
# -m：模型路径
# -n 512：最多生成 512 个 token
# --color：终端输出带颜色，方便区分输入和输出
# -p：输入的 prompt，使用 Qwen 的对话格式
```

如果模型能正常回复且内容合理，说明整个流程成功，可以放心下载使用。

---

*文档版本：适用于 TRL >= 0.8，transformers >= 4.38，peft >= 0.9，bitsandbytes >= 0.43*  
*如果库版本升级后遇到新问题，优先查看各库的官方 CHANGELOG 或 GitHub Issues。*
