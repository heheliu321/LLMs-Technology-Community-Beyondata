## 《大模型Agent开发实战》（体验课）

# 多模态RAG引擎开发实战

# Part 4.多模态RAG项目开发实战

- 本期公开课四大模块内容

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250828194616411.png" alt="image-20250828194616411" style="zoom:50%;" />

- 【演示】实操项目一：从零到一快速搭建多模态RAG系统

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/74cfd666d005af475500d97302823538_raw.mp4"></video>

- 【演示】实操项目二：企业级多模态RAG系统开发实战

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/27f4b2e749af80e62b1a9e3900e30e3f_raw.mp4"></video>

- 课件&代码&项目源码领取：

  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/f7c49313c41eaeb3a2b3b9e9240d9f1e.png" alt="f7c49313c41eaeb3a2b3b9e9240d9f1e" style="zoom:50%;" />

- 本节目录

[toc]

## 阶段一：多模态RAG项目需求描述、技术栈规划与接口设计

### 1. 项目背景与需求

​	本项目是一个**教学型多模态 RAG（Retrieval-Augmented Generation，检索增强生成）系统**，目标是帮助学习者理解 RAG 系统的完整开发流程。

我们希望实现以下功能：

1. **PDF 文档处理**
   - 用户上传 PDF 文件；
   - 后端完成 **OCR、版面解析、Markdown 转换**；
   - 提供文档解析状态查询与预览。
2. **索引构建**
   - 将解析得到的 Markdown 文档进行**切分（chunking）**；
   - 使用 Embedding 模型（OpenAI Embeddings）将片段向量化；
   - 保存至 **FAISS 向量数据库**，用于后续检索。
3. **对话问答（RAG）**
   - 用户输入问题，系统先在向量数据库中检索相关片段；
   - 将检索结果与用户问题一起交给 LLM 生成答案；
   - 答案中附带引用（citations），方便追溯来源。
   - 支持**流式输出（SSE）**，提升交互体验。
   - 支持**历史对话记忆**（基于 InMemorySaver），并提供清空功能。
4. **健壮性需求**
   - 系统在没有上传文档时也能正常回答（走模型自带知识）；
   - 所有接口有**清晰的 API 约定**，方便前端对接。

- 效果展示

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/62eb94d6683941bcbc7ebfd3b711dbb7.png" alt="62eb94d6683941bcbc7ebfd3b711dbb7" style="zoom:50%;" />

- 完整源码领取：

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250901202309705.png" alt="image-20250901202309705" style="zoom:50%;" />

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/0d489e9d3c29af2d7f319171eac040d0.png" alt="0d489e9d3c29af2d7f319171eac040d0" style="zoom:50%;" />

### 2. 技术栈规划

#### 2.1 **后端技术**

- **FastAPI**：高性能 Python Web 框架，自动生成 Swagger UI。
- **Uvicorn**：ASGI 服务器，支持异步处理。
- **LangChain / LangGraph**：RAG 框架，管理对话状态与检索逻辑。
- **Unstructured / fitz (PyMuPDF)**：PDF 解析、OCR、图片处理。
- **FAISS**：Facebook 开源的向量数据库，用于相似检索。

#### 2.2 **前端技术**

- **Figma**：快速完成 UI 原型设计；
- **React / Next.js**：实现流式 SSE 接口调用、前端展示。

### 2.3 **AI 模型与 API**

- **OpenAI Embeddings**：用于向量化（`text-embedding-3-small` / `large`）；
- **对话模型（LLM）**：支持通用对话（如 DeepSeek-Chat / OpenAI GPT-4）。

#### 2.4 **环境与依赖**

- Python >= 3.9
- 主要依赖：`fastapi`、`uvicorn`、`python-multipart`、`langchain`、`faiss-cpu`、`unstructured`、`pymupdf`、`paddleocr`

### 3. 接口规划

我们采用 **RESTful API** 风格，分为 4 大模块：

1. **健康检查（Health）**
   - `/health`：确认服务正常运行。
2. **PDF 处理（PDF Service）**
   - `/pdf/upload`：上传 PDF 文件；
   - `/pdf/parse`：触发解析任务；
   - `/pdf/status`：查询解析进度；
   - `/pdf/page`：获取 PDF 页图（原始/解析）；
   - `/pdf/chunk`：根据 citationId 获取片段。
3. **索引构建（Index Service）**
   - `/index/build`：构建向量索引；
   - `/index/search`：检索相似片段。
4. **对话（Chat Service）**
   - `/chat`：RAG 聊天（SSE 流式输出，包含 citations）；
   - `/chat/clear`：清空当前会话历史。

### 4. OpenAPI 文件编写

基于以上规划，我们使用 **OpenAPI 3.1.0** 来定义后端接口。

> OpenAPI 的作用：
>
> - 是前后端沟通的“契约”；
> - 能被 FastAPI 自动识别并生成交互式文档；
> - 可生成前端 SDK / TS 类型，避免手写错误。

下面是一份完整的 **`openapi.yaml`**（已简化描述，完整版本可参考项目仓库）：

```yaml
openapi: 3.1.0
info:
  title: RAG Demo API
  version: 1.0.0
servers:
  - url: http://localhost:8001/api/v1

paths:
  /health:
    get:
      summary: 健康检查
      responses:
        "200":
          description: OK

  /pdf/upload:
    post:
      summary: 上传 PDF
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              required: [file]
              properties:
                file: { type: string, format: binary }
      responses:
        "200":
          description: 上传成功

  /pdf/parse:
    post:
      summary: 触发解析
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                fileId: { type: string }
      responses:
        "202": { description: 已接受 }

  /pdf/status:
    get:
      summary: 查询解析状态
      parameters:
        - in: query
          name: fileId
          required: true
          schema: { type: string }
      responses:
        "200": { description: 状态返回 }

  /index/build:
    post:
      summary: 构建索引
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                fileId: { type: string }
      responses:
        "200": { description: 构建完成 }

  /chat:
    post:
      summary: 聊天接口（SSE）
      description: 返回流式回答 + 引用
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                message: { type: string }
                fileId: { type: string }
      responses:
        "200": { description: text/event-stream }

  /chat/clear:
    post:
      summary: 清空会话
      responses:
        "200": { description: 会话已清空 }
```

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250901202606259.png" alt="image-20250901202606259" style="zoom:50%;" />

在开发的第一部分，我们完成了：

1. 明确了系统需求：PDF → 索引 → RAG 问答；
2. 规划了技术栈：FastAPI + LangChain + FAISS + 前端 React；
3. 梳理了接口模块：PDF、索引、聊天；
4. 编写了 `openapi.yaml`，形成前后端开发的“契约”。

**下一步（Part 2）**：我们将基于 OpenAPI 文档，逐步实现 **后端服务（FastAPI + Mock + 真正逻辑）**，并在 Swagger UI 中进行测试。

## 阶段二：后端功能思路规划与Mock功能实现

### 1. 项目结构与职责边界

**目标**：前后端“用接口说话”；后端内部“用服务分层说话”。

```
backend/
├─ app.py                        # 入口与路由（FastAPI）
├─ services/
│  ├─ pdf_service.py             # 上传/解析/页图/可视化
│  ├─ index_service.py           # 切分/向量化/索引/检索
│  └─ rag_service.py             # RAG 检索+生成流（SSE）与会话历史
├─ data/                         # 解析与索引产物（按 fileId 分目录）
│  └─ f_xxx/
│     ├─ original.pdf
│     ├─ output.md
│     ├─ pages/{original|parsed}/page-0001.png
│     └─ index_faiss/{index.faiss,index.pkl}
├─ .env
└─ requirements.txt
```

**为什么这么拆？**

- `app.py` 专注“路由+协议”（HTTP/SSE），**不掺业务细节**。
- `services/*` 可单独测试、复用、替换（比如换别的向量库/LLM）。
- `data/<fileId>/…` 是**教学友好**的本地可观测产物（便于课堂讲解与排错）。

### 2. 从 Mock 到真实现：增量替换的策略

**先 Mock**：接口通、前端能调、边界清晰；
 **再替换**：把 Mock 的“假数据/睡眠”替换成真实调用与产物。

> 这能最大化减少“前后端互相等待”的时间，也让你每步都有可验证成果。

#### 2.1 PDF 管线（/pdf/*）

**Mock 版思路**

- `/pdf/upload` 返回一个临时 `fileId`（UUID），把文件暂存；
- `/pdf/parse` 启动一个**后台任务**（不要阻塞 HTTP 响应），假装解析，写个“进度文件”；
- `/pdf/status` 根据进度文件返回 `parsing/ready`;
- `/pdf/page` 返回占位图/原图。

**逐步替换为真实现**

- 解析任务里接入 `Unstructured + PaddleOCR + PyMuPDF`：
  - 导出 `output.md`、每页 `original.png`、`parsed.png`、抽取图片 `images/`；
  - 把 **页号、bbox、类别**等结构写入中间文件（便于 `/pdf/chunk`/可视化叠框）；
- 进度更新：**按阶段写入**（20/50/80/100），让 `/status` 有“在动”的感觉；
- 页图加载：`/pdf/page` 用 `FileResponse` 或 `StreamingResponse` 回图。

**关键示意**：

```python
# app.py
@app.post("/api/v1/pdf/parse")
def start_parse(req: ParseBody, bg: BackgroundTasks):
    bg.add_task(run_full_parse_pipeline, req.fileId)
    return {"jobId": f"j_{shortid()}"}

# services/pdf_service.py
def run_full_parse_pipeline(file_id: str):
    write_status(file_id, "parsing", 10)
    # 1) OCR/结构解析 -> elements
    # 2) 导出 output.md、images/
    # 3) 渲染 original/parsed/page-*.png
    write_status(file_id, "ready", 100)
```

#### 2.2 索引管线（/index/*）

**Mock 版思路**

- `/index/build` 直接 `time.sleep(1)` 返回 `"chunks": 42`；
- `/index/search` 返回固定三条占位片段。

**真实现替换**

- 构建阶段读取 `data/<fileId>/output.md`，做 “**按标题切分（Header1/2）→ 清洗 → 向量化（OpenAI Embeddings）→ FAISS**”；
- 检索阶段加载本地 FAISS，`similarity_search_with_score(query, k)`。

**关键示意**：

```python
# services/index_service.py
def build_faiss_index(file_id: str):
    md = load_markdown(file_id)
    docs = header_split(md)               # MarkdownHeaderTextSplitter
    vs = FAISS.from_documents(docs, embedding=load_embeddings())
    vs.save_local(index_dir(file_id))
    return {"ok": True, "chunks": len(docs)}

def search_faiss(file_id: str, query: str, k=5):
    vs = FAISS.load_local(index_dir(file_id), load_embeddings(), allow_dangerous_deserialization=True)
    return vs.similarity_search_with_score(query, k=k)
```

**为什么先做“标题切分”？**

- 讲解性强（学生容易理解“结构化”切分）；
- 对课程 PPT/讲义类文档命中率高；
- 之后再引入“递归字符切分”作对照，讨论召回/粒度权衡。

#### 2.3 RAG 聊天（/chat SSE）

**Mock 版思路**

- 原来 `/chat` 返回“伪流”：sleep + 逐句 `token` 事件；
- 先不发 `citation`，只发 `token`/`done`。

**真实现替换**

- 收到 `message + pdfFileId + sessionId`：
  - 如果有 `pdfFileId` 且索引存在：先 `retrieve` → 生成 `citations`；
  - **先发若干 `event: citation`**（前端角标立刻出现）；
  - 组装历史 + 上下文 → 调 LLM 流式生成 → `event: token` 连发；
  - 结尾 `event: done` 带 `{"used_retrieval": true|false}`。
- **无文档也能聊**：当索引缺失或 `pdfFileId` 为空，直接走“通识回答”。

**关键示意**：

```python
# app.py
@app.post("/api/v1/chat")
def chat_sse(req: ChatReq):
    async def gen():
        citations, context = [], ""
        if req.pdfFileId:
            citations, context = await retrieve(req.message, req.pdfFileId)
            for c in citations:
                yield "event: citation\ndata: " + json.dumps(c) + "\n\n"

        async for ev in answer_stream(
            question=req.message,
            citations=citations,
            context_text=context,
            branch="with_context" if context else "no_context",
            session_id=req.sessionId or "default"
        ):
            # token / done
            ...
    return StreamingResponse(gen(), media_type="text/event-stream")
```

#### 2.4 多轮对话（/chat/clear）

**思路**

- 教学场景只需**进程内内存**保存历史（`sessionId → [messages]`）；
- `/chat/clear` 清空该 `sessionId`；
- 重启后自然丢失（即是你要的效果）。

**关键示意**：

```python
# services/rag_service.py
_sessions = defaultdict(list)

def get_history(sid): return _sessions.get(sid, [])
def append_history(sid, role, content): _sessions[sid].append({"role":role,"content":content})
def clear_history(sid): _sessions.pop(sid, None)
```

**为什么不立刻引入数据库？**

- 教学项目**先证明闭环**；
- 学生先理解“多轮上下文对生成的影响”；
- 需要持久化时，再引入 Redis / SQLite 讲“状态存储”。

### 3. 错误处理与返回规范

**目标**：**前端可预测**、**便于排查**。

- 统一错误体：

  ```json
  { "error": "CODE", "message": "人类可读描述" }
  ```

- 常见错误码建议：

  - `FILE_NOT_FOUND`（`fileId` 不存在）
  - `NEED_PARSE_FIRST`（没 `output.md` 就建索引）
  - `INDEX_NOT_FOUND`（索引没建）
  - `PAGE_NOT_FOUND`（页码越界）
  - `OCR_FAILED / PARSE_FAILED / INDEX_BUILD_ERROR`

**示意**：

```python
return JSONResponse({"error":"INDEX_NOT_FOUND","message":"请先构建索引"}, status_code=400)
```

### 4. 性能、可观测性与稳定性

- **进度可见**：解析阶段分阶段更新 `/status`；
- **日志可读**：关键里程碑打印（上传成功、开始 OCR、导出 MD、渲染页图、完成）；
- **幂等**：`/index/build` 再次调用可复用已有索引（返回 `{"ok":true,"reused":true}`）；
- **资源控制**：解析时限制并发，避免 OCR 占满 CPU；
- **超时与回退**：LLM 流式异常时，回退为整段生成 + 手动切片；
- **Windows 兼容**：路径用 `pathlib`；端口冲突换 `8001`；中文文件名注意编码。

## 阶段三：多模态RAG系统后端功能开发与测试

- 后端目录与角色回顾

```
backend/
├─ app.py                    # FastAPI 入口 & 路由（HTTP/SSE 协议层）
├─ services/
│  ├─ pdf_service.py         # PDF 上传/解析/OCR/可视化导出
│  ├─ index_service.py       # Markdown切分、向量化、FAISS索引/检索
│  └─ rag_service.py         # RAG检索+生成、SSE流式输出、多轮会话
├─ data/
│  └─ <fileId>/
│     ├─ original.pdf
│     ├─ output.md
│     ├─ images/                 # Markdown内引用的图片
│     ├─ pages/original/page-*.png
│     ├─ pages/parsed/page-*.png # 叠框可视化
│     └─ index_faiss/{index.faiss,index.pkl}
├─ .env
└─ requirements.txt
```

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250901203312903.png" alt="image-20250901203312903" style="zoom:50%;" />

### 1. `app.py`  路由与协议层功能开发

- **定位**：只负责“HTTP/SSE 协议 + 入参校验 + 调用 services”，**不直接写业务细节**。这样方便单测与替换实现。

核心路由

- `GET /api/v1/health`：可用性探针。
- `POST /api/v1/pdf/upload`：保存文件，返回 `fileId/name/pages`。
- `POST /api/v1/pdf/parse`：启动**后台任务**（`BackgroundTasks`），马上返回 `202` 风格的结果（我们用 `200`/JSON 也行），解析过程在后台跑。
- `GET /api/v1/pdf/status?fileId=`：轮询解析状态（`idle/parsing/ready/error` + 进度）。
- `GET /api/v1/pdf/page?fileId=&page=&type=`：回传 PNG（原始页/叠框页）。
- `GET /api/v1/pdf/chunk?citationId=`：按角标ID回查片段（你若实现了 citation→chunk 的落地）。
- `POST /api/v1/index/build`：对 `data/<fileId>/output.md` 切分→向量化→FAISS 存盘；返回 `chunks`。
- `POST /api/v1/index/search`：Top-K 相似检索，返回 `text/score/metadata`。
- `POST /api/v1/chat`（SSE）：按 RAG 流生成事件：`citation`→`token`→`done`（错误时 `error`）。
- `POST /api/v1/chat/clear`：清空指定 `sessionId` 的内存会话。

设计要点

- **后台任务**：`/pdf/parse` 解析耗时，必须放 `BackgroundTasks`，避免阻塞请求。

- **错误体统一**：用 `JSONResponse({"error":"CODE","message":"..."} , status_code=xxx)`。

- **SSE**：`StreamingResponse(gen(), media_type="text/event-stream")`，事件格式严格：

  ```
  event: token
  data: {"text":"..."}
  
  event: done
  data: {"used_retrieval": true}
  ```

- **跨域**：本地联调需要 `CORSMiddleware`（允许前端端口）。

完整代码：

```python
from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio, time, os, random, string
from typing import Optional, Dict, Any, List
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional

from fastapi import BackgroundTasks
from services.pdf_service import (
    save_upload, run_full_parse_pipeline,
    original_pdf_path, dir_original_pages, dir_parsed_pages, markdown_output
)
from services.index_service import build_faiss_index, search_faiss
from fastapi.responses import StreamingResponse, JSONResponse
from services.rag_service import retrieve, answer_stream, clear_history

app = FastAPI(
    title="九天老师公开课：多模态RAG系统API",
    version="1.0.0",
    description="九天老师公开课《多模态RAG系统开发实战》后端API。"
)

# 允许前端本地联调
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 课堂演示方便，生产请收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"

# ---------------- 内存态存储（教学Mock） ----------------
current_pdf: Dict[str, Any] = {
    "fileId": None,
    "name": None,
    "pages": 0,
    "status": "idle",      # idle | parsing | ready | error
    "progress": 0
}
citations: Dict[str, Dict[str, Any]] = {}   # citationId -> { fileId, page, snippet, bbox, previewUrl }

# ---------------- 工具函数 ----------------
def rid(prefix: str) -> str:
    return f"{prefix}_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

def now_ts() -> int:
    return int(time.time())

def err(code: str, message: str) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message}, "requestId": rid("req"), "ts": now_ts()}

# ---------------- Pydantic 模型（契约） ----------------
class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = None
    pdfFileId: Optional[str] = None

# ---------------- Health ----------------
@app.get(f"{API_PREFIX}/health", tags=["Health"])
async def health():
    return {"ok": True, "version": "1.0.0"}

# ---------------- Chat（SSE，POST 返回 event-stream） ----------------
class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = None
    pdfFileId: Optional[str] = None

@app.post(f"{API_PREFIX}/chat", tags=["Chat"])
async def chat_stream(req: ChatRequest):
    """
    SSE 事件：token | citation | done | error
    """
    async def gen():
        try:
            question = (req.message or "").strip()
            session_id = (req.sessionId or "default").strip()  # 默认单会话
            file_id = (req.pdfFileId or "").strip()

            citations, context_text = [], ""
            branch = "no_context"
            if file_id:
                try:
                    citations, context_text = await retrieve(question, file_id)
                    branch = "with_context" if context_text else "no_context"
                except FileNotFoundError:
                    branch = "no_context"

            # 先推送引用（若有）
            if branch == "with_context" and citations:
                for c in citations:
                    yield "event: citation\n"
                    yield f"data: {c}\n\n"

            # 再推送 token 流（内部会写入历史）
            async for evt in answer_stream(
                question=question,
                citations=citations,
                context_text=context_text,
                branch=branch,
                session_id=session_id
            ):
                if evt["type"] == "token":
                    yield "event: token\n"
                    # 注意：这里确保 data 是合法 JSON 字符串
                    text = evt["data"].replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
                    yield f'data: {{"text":"{text}"}}\n\n'
                elif evt["type"] == "citation":
                    yield "event: citation\n"
                    yield f"data: {evt['data']}\n\n"
                elif evt["type"] == "done":
                    used = "true" if evt["data"].get("used_retrieval") else "false"
                    yield "event: done\n"
                    yield f"data: {{\"used_retrieval\": {used}}}\n\n"

        except Exception as e:
            yield "event: error\n"
            esc = str(e).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
            yield f'data: {{"message":"{esc}"}}\n\n'

    headers = {"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# ---------------- Chat: 清除对话 ----------------
class ClearChatRequest(BaseModel):
    sessionId: Optional[str] = None

@app.post(f"{API_PREFIX}/chat/clear", tags=["Chat"])
async def chat_clear(req: ClearChatRequest):
    sid = (req.sessionId or "default").strip()
    clear_history(sid)
    return {"ok": True, "sessionId": sid, "cleared": True}


# ---------------- PDF: 上传（仅单文件，直接替换） ----------------

current_pdf = {"fileId": None, "name": None, "pages": 0, "status": "idle", "progress": 0}

@app.post(f"{API_PREFIX}/pdf/upload", tags=["PDF"])
async def pdf_upload(file: UploadFile = File(...), replace: Optional[bool] = True):
    if not file:
        return JSONResponse(err("NO_FILE", "缺少文件"), status_code=400)
    # 生成新的 fileId（替换策略：上传即替换）
    fid = rid("f")
    saved = save_upload(fid, await file.read(), file.filename)
    current_pdf.update({**saved, "status": "idle", "progress": 0})
    citations.clear()
    return saved

# ---------------- PDF: 触发解析 ----------------
@app.post(f"{API_PREFIX}/pdf/parse", tags=["PDF"])
async def pdf_parse(payload: Dict[str, Any] = Body(...), bg: BackgroundTasks = None):
    file_id = payload.get("fileId")
    if not current_pdf["fileId"] or current_pdf["fileId"] != file_id:
        return JSONResponse(err("FILE_NOT_FOUND", "未找到该文件"), status_code=400)

    current_pdf["status"] = "parsing"
    current_pdf["progress"] = 5

    def _job():
        try:
            # 20 → 60 → 100 三阶段进度示意
            current_pdf["progress"] = 20
            run_full_parse_pipeline(file_id)   # 真解析
            current_pdf["progress"] = 100
            current_pdf["status"] = "ready"
        except Exception as e:
            current_pdf["status"] = "error"
            current_pdf["progress"] = 0
            print("Parse error:", e)

    if bg is not None:
        bg.add_task(_job)
    else:
        _job()

    return {"jobId": rid("j")}

# ---------------- PDF: 状态 ----------------
@app.get(f"{API_PREFIX}/pdf/status", tags=["PDF"])
async def pdf_status(fileId: str = Query(...)):
    if not current_pdf["fileId"] or current_pdf["fileId"] != fileId:
        return {"status": "idle", "progress": 0}
    resp = {"status": current_pdf["status"], "progress": current_pdf["progress"]}
    if current_pdf["status"] == "error":
        resp["errorMsg"] = "解析失败"
    return resp

# ---------------- PDF: 页面图 ----------------
@app.get(f"{API_PREFIX}/pdf/page", tags=["PDF"])
async def pdf_page(
    fileId: str = Query(...),
    page: int = Query(..., ge=1),
    type: str = Query(..., regex="^(original|parsed)$")
):
    if not current_pdf["fileId"] or current_pdf["fileId"] != fileId:
        return JSONResponse(status_code=404, content=None)

    if current_pdf["status"] != "ready" and type == "parsed":
        # 未解析就请求 parsed 页，按你的契约可以给 400/403；这里保持 204 更温和
        return JSONResponse(status_code=204, content=None)

    base = dir_original_pages(fileId) if type == "original" else dir_parsed_pages(fileId)
    img = base / f"page-{page:04d}.png"
    if not img.exists():
        return JSONResponse(err("PAGE_NOT_FOUND", "页面不存在或未渲染"), status_code=404)
    return FileResponse(str(img), media_type="image/png")

# ---------------- PDF: 图片文件 ----------------
@app.get(f"{API_PREFIX}/pdf/images", tags=["PDF"])
async def pdf_images(
    fileId: str = Query(...),
    imagePath: str = Query(...)
):
    """获取PDF解析后的图片文件"""
    if not current_pdf["fileId"] or current_pdf["fileId"] != fileId:
        return JSONResponse(status_code=404, content=None)

    # 构建图片文件的完整路径
    from services.pdf_service import images_dir
    image_file = images_dir(fileId) / imagePath
    
    if not image_file.exists():
        return JSONResponse(err("IMAGE_NOT_FOUND", "图片文件不存在"), status_code=404)
    
    # 检查文件是否在images目录内（安全考虑）
    try:
        image_file.resolve().relative_to(images_dir(fileId).resolve())
    except ValueError:
        return JSONResponse(err("INVALID_PATH", "无效的图片路径"), status_code=400)
    
    return FileResponse(str(image_file), media_type="image/png")

# ---------------- PDF: 引用片段 ----------------
@app.get(f"{API_PREFIX}/pdf/chunk", tags=["PDF"])
async def pdf_chunk(citationId: str = Query(...)):
    ref = citations.get(citationId)
    if not ref:
        return JSONResponse(err("NOT_FOUND", "无该引用"), status_code=404)
    return ref

class BuildIndexRequest(BaseModel):
    fileId: str

class SearchRequest(BaseModel):
    fileId: str
    query: str
    k: Optional[int] = 5

@app.post(f"{API_PREFIX}/index/build", tags=["Index"])
async def index_build(req: BuildIndexRequest):
    # 可校验：current_pdf["status"] 应为 ready
    if not current_pdf["fileId"] or current_pdf["fileId"] != req.fileId:
        raise HTTPException(status_code=400, detail="FILE_NOT_FOUND_OR_NOT_CURRENT")
    if current_pdf["status"] != "ready":
        raise HTTPException(status_code=409, detail="NEED_PARSE_FIRST")

    out = build_faiss_index(req.fileId)
    if not out.get("ok"):
        return JSONResponse(err(out.get("error", "INDEX_BUILD_ERROR"), "索引构建失败"), status_code=500)
    return {"ok": True, "chunks": out["chunks"]}

@app.post(f"{API_PREFIX}/index/search", tags=["Index"])
async def index_search(req: SearchRequest):
    out = search_faiss(req.fileId, req.query, req.k or 5)
    if not out.get("ok"):
        code = out.get("error", "INDEX_NOT_FOUND")
        return JSONResponse(err(code, "请先构建索引"), status_code=400)
    return out
```

代码解释如下：

1. 导入和基本设置

- **导入模块**: 代码首先导入了大量所需的库，这包括：
  - `fastapi`: FastAPI 框架的核心。
  - `fastapi.responses`: 用于返回不同类型的响应，如流式响应 (`StreamingResponse`)、JSON 响应 (`JSONResponse`) 和文件响应 (`FileResponse`)。
  - `fastapi.middleware.cors`: 用于处理跨域请求，允许前端在不同域名下访问后端。
  - `pydantic`: 用于数据验证和模型定义 (`BaseModel`)，确保请求数据的格式正确。
  - `asyncio`, `time`, `os`, `random`, `string`: Python 的内置库，用于异步编程、时间处理、文件路径操作和字符串生成等。
  - `typing`: 提供类型提示，让代码更易读和健壮。
  - `fastapi.BackgroundTasks`: 用于在不阻塞主线程的情况下运行后台任务，比如 PDF 解析。
- **导入自定义服务**: `services` 目录下的模块包含了核心业务逻辑：
  - `pdf_service`: 处理 PDF 文件的上传、保存和解析（如将页面转换为图片）。
  - `index_service`: 负责构建和搜索 **FAISS 索引**。FAISS 是一个用于高效相似性搜索的库，这里用来检索与用户问题最相关的文档片段。
  - `rag_service`: 包含了 RAG 系统的核心逻辑，如`retrieve`（检索文档片段）和`answer_stream`（流式生成回答）。
- **初始化 FastAPI 应用**:
  - `app = FastAPI(...)`: 创建一个 FastAPI 应用实例。`title`、`version` 和 `description` 用于生成 API 文档（通过 `/docs` 或 `/redoc` 路径访问）。
- **CORS 中间件**:
  - `app.add_middleware(CORSMiddleware, ...)`: 启用 CORS（跨源资源共享），允许来自任何域名的前端访问此 API。`allow_origins=["*"]` 是一个宽泛的设置，在生产环境中通常会收紧以提高安全性。
- **全局变量和工具函数**:
  - `current_pdf`: 一个字典，用于在内存中存储当前正在处理的 PDF 文件的状态信息，包括文件 ID、名称、页数、处理状态 (`idle`, `parsing`, `ready`, `error`) 和进度。这是一个教学示例，实际生产环境会使用数据库。
  - `citations`: 另一个字典，用于存储检索到的文档引用信息。
  - `rid(prefix: str)`: 生成一个带前缀的随机 ID。
  - `now_ts()`: 返回当前 Unix 时间戳。
  - `err(...)`: 一个辅助函数，用于快速创建包含错误码和错误信息的 JSON 响应。

2. API 路由讲解

代码中的每个 `@app.get` 或 `@app.post` 装饰器都定义了一个 API 接口。

**Health 检查 (`/health`)**

- 这是一个简单的 GET 请求，用于检查服务是否正常运行，常用于部署后的健康监控。

**聊天功能 (`/chat`)**

- **`ChatRequest(BaseModel)`**: 定义了请求体的数据模型，包括用户消息 (`message`)、会话 ID (`sessionId`) 和 PDF 文件 ID (`pdfFileId`)。
- **`@app.post(f"{API_PREFIX}/chat", tags=["Chat"])`**:
  - 这个 API 接口以 **POST** 方式接收聊天请求。
  - 它返回一个 **SSE** (Server-Sent Events，服务器发送事件) 的流式响应。这意味着后端会逐步将数据推送到前端，而不需要等待整个响应生成完毕。
  - **核心逻辑**:
    1. 从请求中获取 `question`（用户问题）、`session_id` 和 `file_id`。
    2. 如果提供了 `file_id`，调用 `retrieve` 服务从 PDF 文档中检索与问题相关的片段（即 **RAG 的检索步骤**）。
    3. 根据是否成功检索到上下文 (`branch`)，选择性地执行下一步。
    4. 通过 `answer_stream` 服务，利用检索到的上下文（或直接回答）来生成答案。`answer_stream` 返回一个异步生成器，逐步产生 `token`、`citation` 或 `done` 等不同类型的事件。
    5. 代码将这些事件封装成 SSE 格式 (`event: ...` 和 `data: ...`)，然后通过 `StreamingResponse` 返回给前端。
    6. `try...except` 块确保在发生错误时也能返回一个 `error` 事件，通知前端。

**清除聊天历史 (`/chat/clear`)**

- 这个 POST 接口允许用户清除特定会话的聊天历史，通过调用 `services.rag_service.clear_history` 函数来实现。

3. PDF 文件处理 API

- **`pdf/upload`**:
  - 一个 POST 接口，用于接收用户上传的 PDF 文件。
  - 使用 `UploadFile` 类型来处理文件上传。
  - 调用 `save_upload` 函数将文件保存到本地，并更新 `current_pdf` 状态，将旧文件替换。
- **`pdf/parse`**:
  - 一个 POST 接口，用于触发 PDF 解析流程。
  - **关键点**: 使用 `BackgroundTasks` 来运行解析任务 (`_job`)。这意味着 API 会立即返回响应，而解析任务会在后台异步执行，不会阻塞用户的请求。
  - 在 `_job` 函数中，`run_full_parse_pipeline` 会执行实际的解析工作，并模拟了进度更新。
- **`pdf/status`**:
  - 一个 GET 接口，用于查询当前 PDF 的解析状态和进度。
- **`pdf/page`**:
  - 一个 GET 接口，用于获取 PDF 某一页的图片。
  - 根据 `type` 参数 (`original` 或 `parsed`) 返回不同版本的页面图像。
- **`pdf/images`**:
  - 一个 GET 接口，用于获取 PDF 解析后产生的嵌入图片文件。
  - **安全注意**: 代码中包含了一个重要的安全检查 `image_file.resolve().relative_to(...)`，它确保请求的文件路径不会超出指定的图片目录，防止**路径遍历攻击**。
- **`pdf/chunk`**:
  - 一个 GET 接口，用于根据 `citationId` 获取具体的引用片段信息。

4. 索引构建和搜索 API

- **`index/build`**:
  - 一个 POST 接口，用于触发 **FAISS 索引的构建**。
  - 它首先检查 PDF 文件是否已经解析完毕 (`current_pdf["status"] == "ready"`)，如果不是则返回错误。
  - 调用 `build_faiss_index` 函数来创建索引，这个过程会将 PDF 内容分块并向量化。
- **`index/search`**:
  - 一个 POST 接口，用于对已构建的索引进行搜索。
  - 接收一个 `SearchRequest` 模型，包含 `query`（查询文本）和 `k`（要返回的最相关结果数）。
  - 调用 `search_faiss` 函数来执行实际的向量相似性搜索，并返回匹配的文档片段。

### 2. `services/pdf_service.py`PDF解析功能开发

**定位**：把 PDF 变成 RAG 友好的产物：`output.md` + 页图 + 可视化叠框 + 图片资源。

关键步骤

1. **保存上传文件**
   - 命名为 `data/<fileId>/original.pdf`（`fileId` 由 `upload` 生成）
   - 读取页数用于返回给前端（PyMuPDF `fitz.open`）。
2. **异步解析（后台任务）**
   - 进度管理：`write_status(fileId, "parsing", 10/30/60/100)`
   - `Unstructured.partition_pdf(...)`：
     - `strategy="hi_res"`、`ocr_engine="paddleocr"`、`ocr_languages="chi_sim+eng"`
     - 得到 `elements`（含文本、表格、图片、标题等类别）
   - **导出 Markdown**：
     - 标题转 `# / ##`，表格优先 `text_as_html`→`html2text`；
     - 图片导出到 `images/` 并在 MD 里用相对路径引用。
   - **页图渲染**：
     - `pages/original/page-0001.png`：用 `fitz` 渲染页面；
     - `pages/parsed/page-0001.png`：把 `elements` 的 bbox 按类别上色叠加（matplotlib / PIL）。
3. **状态持久**（可选）
   - 教学场景可以只用内存 `current_pdf`；若想重启恢复，写 `manifest.json`（`status/progress/pages/name`）。

**完整代码如下**

```python
# services/pdf_service.py
from __future__ import annotations
import os, io, math, json
from pathlib import Path
from typing import Dict, Any, List
import fitz
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # 服务器无头
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from langchain_unstructured import UnstructuredLoader
from unstructured.partition.pdf import partition_pdf
from html2text import html2text

from dotenv import load_dotenv
load_dotenv(override=True)

# 统一的根目录：每个 fileId 一个子目录
DATA_ROOT = Path("data")

def workdir(file_id: str) -> Path:
    d = DATA_ROOT / file_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def dir_original_pages(file_id: str) -> Path:
    p = workdir(file_id) / "pages" / "original"
    p.mkdir(parents=True, exist_ok=True); return p

def dir_parsed_pages(file_id: str) -> Path:
    p = workdir(file_id) / "pages" / "parsed"
    p.mkdir(parents=True, exist_ok=True); return p

def original_pdf_path(file_id: str) -> Path:
    return workdir(file_id) / "original.pdf"

def markdown_output(file_id: str) -> Path:
    return workdir(file_id) / "output.md"

def images_dir(file_id: str) -> Path:
    p = workdir(file_id) / "images"
    p.mkdir(parents=True, exist_ok=True); return p

def save_upload(file_id: str, upload_bytes: bytes, filename: str) -> Dict[str, Any]:
    """保存上传的 PDF，并返回页数"""
    pdf_path = original_pdf_path(file_id)
    pdf_path.write_bytes(upload_bytes)
    with fitz.open(pdf_path) as doc:
        pages = doc.page_count
    return {"fileId": file_id, "name": filename, "pages": pages}

def render_original_pages(file_id: str, dpi: int = 144):
    """把原始 PDF 渲染为 PNG，存到 pages/original/"""
    pdf_path = original_pdf_path(file_id)
    out_dir = dir_original_pages(file_id)
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            (out_dir / f"page-{idx:04d}.png").write_bytes(pix.tobytes("png"))

def _plot_boxes_to_ax(ax, pix, segments):
    category_to_color = {
        "Title": "orchid",
        "Image": "forestgreen",
        "Table": "tomato",
    }
    categories = set()
    for seg in segments:
        points = seg["coordinates"]["points"]
        lw = seg["coordinates"]["layout_width"]
        lh = seg["coordinates"]["layout_height"]
        scaled = [(x * pix.width / lw, y * pix.height / lh) for x, y in points]
        color = category_to_color.get(seg.get("category"), "deepskyblue")
        categories.add(seg.get("category", "Text"))
        poly = patches.Polygon(scaled, linewidth=1, edgecolor=color, facecolor="none")
        ax.add_patch(poly)

    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for cat, color in category_to_color.items():
        if cat in categories:
            legend_handles.append(patches.Patch(color=color, label=cat))
    ax.legend(handles=legend_handles, loc="upper right")

def render_parsed_pages_with_boxes(file_id: str, docs_local: List[Dict[str, Any]], dpi: int = 144):
    """
    根据 UnstructuredLoader 的 metadata（含坐标）在原图上叠框，输出到 pages/parsed/
    """
    pdf_path = original_pdf_path(file_id)
    out_dir = dir_parsed_pages(file_id)
    with fitz.open(pdf_path) as doc:
        # 预聚合：按 page_number 分组 segments
        segments_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for d in docs_local:
            meta = d.metadata if hasattr(d, "metadata") else d["metadata"]
            pno = meta.get("page_number")
            if pno is None: continue
            segments_by_page.setdefault(pno, []).append(meta)

        for page_number in range(1, doc.page_count + 1):
            page = doc.load_page(page_number - 1)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(pil)
            ax.axis("off")
            _plot_boxes_to_ax(ax, pix, segments_by_page.get(page_number, []))
            fig.tight_layout()
            fig.savefig(out_dir / f"page-{page_number:04d}.png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)

def unstructured_segments(file_id: str) -> List[Any]:
    """用 UnstructuredLoader 产生高分辨率布局段"""
    pdf_path = str(original_pdf_path(file_id))
    loader = UnstructuredLoader(
        file_path=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        ocr_languages="chi_sim+eng",
        ocr_engine="paddleocr",  # 如果装不上可换成 'auto' 或注释掉
    )
    out = []
    for d in loader.lazy_load():
        out.append(d)
    return out

def pdf_to_markdown(file_id: str):
    pdf_path = str(original_pdf_path(file_id))
    out_md = markdown_output(file_id)
    img_dir = images_dir(file_id)

    elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res",
        ocr_languages="chi_sim+eng",
        ocr_engine="paddleocr"  # 同上
    )

    # 提取图片
    image_map = {}
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            image_map[page_num] = []
            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                img_path = img_dir / f"page{page_num}_img{img_index}.png"
                if pix.n < 5:
                    pix.save(str(img_path))
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(str(img_path))
                image_map[page_num].append(img_path.name)  # 只保存文件名

    md_lines: List[str] = []
    inserted_images = set()
    for el in elements:
        cat = getattr(el, "category", None)
        text = (getattr(el, "text", "") or "").strip()
        meta = getattr(el, "metadata", None)
        page_num = getattr(meta, "page_number", None) if meta else None

        if not text and cat != "Image":
            continue

        if cat == "Title" and text.startswith("- "):
            md_lines.append(text + "\n")
        elif cat == "Title":
            md_lines.append(f"# {text}\n")
        elif cat in ["Header", "Subheader"]:
            md_lines.append(f"## {text}\n")
        elif cat == "Table":
            html = getattr(meta, "text_as_html", None) if meta else None
            if html:
                md_lines.append(html2text(html) + "\n")
            else:
                md_lines.append((text or "") + "\n")
        elif cat == "Image" and page_num:
            for name in image_map.get(page_num, []):
                if (page_num, name) not in inserted_images:
                    md_lines.append(f"![Image](./images/{name})\n")
                    inserted_images.add((page_num, name))
        else:
            md_lines.append(text + "\n")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    return {"markdown": out_md.name, "images_dir": "images"}

def run_full_parse_pipeline(file_id: str) -> Dict[str, Any]:
    """
    完整流程：原始页图渲染 → Unstructured 布局段 → 叠框图 → 输出 Markdown
    返回用于 /status 的统计或元信息
    """
    render_original_pages(file_id)
    docs = unstructured_segments(file_id)
    render_parsed_pages_with_boxes(file_id, docs)
    md_info = pdf_to_markdown(file_id)
    return {"md": md_info["markdown"]}
```

代码解释如下：

1. 核心依赖和配置

- **导入库**:
  - `fitz` (PyMuPDF): 用于处理 PDF 文档的核心库，可以提取文本、图片，并渲染页面为图像。
  - `PIL` (Pillow): 用于图像处理，这里用于将 PyMuPDF 的图像数据转换为 PIL 格式，以便用 Matplotlib 处理。
  - `matplotlib`: 强大的绘图库，在这里主要用来在 PDF 页面渲染图上绘制 bounding box（边界框），以展示解析出的内容区域。
  - `langchain_unstructured` 和 `unstructured`: **核心组件**。这些库负责执行 PDF 的复杂解析，包括布局分析、表格识别、OCR（光学字符识别）以及将内容分段。
  - `html2text`: 将 HTML 表格转换为 Markdown 格式。
  - `dotenv`: 用于从 `.env` 文件加载环境变量。
- **文件系统路径**:
  - `DATA_ROOT = Path("data")`: 定义了所有处理过的数据的根目录。
  - `workdir(file_id)`: 为每个上传的 PDF 文件创建一个独立的子目录，这是管理不同文档数据的好方法。
  - `dir_original_pages`, `dir_parsed_pages`, `original_pdf_path`, `markdown_output`, `images_dir`: 一系列辅助函数，用于生成 PDF 文档在本地文件系统中的特定存储路径。

2. PDF 文件处理函数

- `save_upload(file_id, upload_bytes, filename)`:
  - 这是 API 中 **`/pdf/upload`** 路由调用的函数。
  - 它将上传的二进制文件内容写入到 `data/{file_id}/original.pdf`。
  - 使用 `fitz.open()` 打开文件以快速获取页数，然后返回一个包含文件元信息（ID、名称、页数）的字典。
- `render_original_pages(file_id, dpi)`:
  - 将原始 PDF 的每一页渲染成 PNG 图片。
  - 使用 `fitz.get_pixmap()` 方法将 PDF 页面转换为像素图，并保存到 `pages/original` 文件夹。`dpi`（每英寸点数）参数决定了输出图片的清晰度。
- `_plot_boxes_to_ax(ax, pix, segments)`:
  - 一个辅助函数，用于在 Matplotlib 的绘图轴上绘制边界框。
  - 它根据解析出的段落（`segments`）的坐标信息，在图片上用不同颜色绘制矩形框，来区分标题、图像、表格和普通文本。
  - 这个函数展示了如何利用 `matplotlib.patches.Polygon` 在图像上叠加矢量图形。
- `render_parsed_pages_with_boxes(file_id, docs_local, dpi)`:
  - 这是 `render_original_pages` 的增强版。
  - 它首先将 `unstructured_segments` 函数解析出的数据按页分组。
  - 然后，它遍历每一页，先加载原始页面图片，再使用 `_plot_boxes_to_ax` 函数在图片上叠加绘制解析出的 **内容类型（如文本、表格、图像）的边界框**。
  - 最终将带有这些标记框的图片保存到 `pages/parsed` 文件夹。这些图片对于前端展示解析结果非常有价值。

3. PDF 内容解析函数

- `unstructured_segments(file_id)`:
  - 使用 `langchain_unstructured.UnstructuredLoader` 来加载 PDF。
  - `strategy="hi_res"`: 告诉解析器使用高分辨率策略，这会调用更高级的布局分析模型。
  - `infer_table_structure=True`: 尝试识别 PDF 中的表格结构。
  - `ocr_languages` 和 `ocr_engine`: 指定 OCR 引擎和语言，这里配置了中英文，以支持包含扫描件或图片文本的 PDF。
  - 此函数返回一个包含多个文档片段（Document 对象）的列表，每个片段都带有其内容、元数据（包括坐标）和类型信息。
- `pdf_to_markdown(file_id)`:
  - 这是**多模态 RAG** 系统中将非结构化 PDF 内容转换为结构化文本的关键步骤。
  - 它使用 `unstructured.partition.pdf` 来解析 PDF，这与 `UnstructuredLoader` 类似，但更侧重于**提取结构化元素**。
  - **提取图片**: 它遍历 PDF 页面，提取所有图片并保存到 `images` 目录。
  - **转换为 Markdown**: 它遍历 `partition_pdf` 返回的元素列表，根据元素的 `category`（如 `Title`, `Header`, `Table` 等）将其转换为相应的 Markdown 格式。
    - 例如，`Title` 转换为 `# 标题`，`Table` 转换为 Markdown 表格。
  - `html2text` 被用来将表格的 HTML 表示转换为更易读的 Markdown 格式。
  - 它还将提取出的图片作为 Markdown 链接 (`![Image](./images/...)`) 插入到文档中，从而实现了**多模态**信息的整合。
  - 最终，将所有 Markdown 行连接成一个字符串，并保存到 `output.md` 文件中。

### 3. `services/index_service.py` 切分/向量化/FAISS 索引

**定位**：把 `output.md` 变成“可检索的 chunks + 向量索引”。

核心流程

1. **加载 `.env` & Embeddings**
   - `OpenAIEmbeddings(model="text-embedding-3-small|large", base_url/key from .env)`
   - 你加了 `load_dotenv(override=True)`，确保环境稳定。
2. **Markdown 切分**
   - 用 `MarkdownHeaderTextSplitter` 按 `# / ##` 切段，保留 `metadata`（最好带上 `page_number`）。
   - **二次切分**（可选）：对超长段用字符长度递归切分（更平衡召回）。
3. **构建 & 保存索引**
   - `FAISS.from_documents(docs, embeddings)` → `save_local("data/<fileId>/index_faiss")`。
   - 幂等：若目录已存在，返回 `reused:true`（可选）。
4. **检索**
   - `load_local(..., allow_dangerous_deserialization=True)`
   - `similarity_search_with_score(query, k)` → 返回 `(Document, score)`。

**完整代码如下**

```python
# services/index_service.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv(override=True)

# 复用你已有的数据目录结构
DATA_ROOT = Path("data")

def workdir(file_id: str) -> Path:
    p = DATA_ROOT / file_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def markdown_path(file_id: str) -> Path:
    return workdir(file_id) / "output.md"

def index_dir(file_id: str) -> Path:
    p = workdir(file_id) / "index_faiss"
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_embeddings() -> OpenAIEmbeddings:
    # 读取环境变量；支持你的代理 base_url
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_EMBEDDING_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_EMBEDDING_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(model="text-embedding-3-small", **kwargs)

def split_markdown(md_text: str) -> List[Document]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        # 需要更细可以加 ("###", "Header 3")
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(md_text)
    # 可加一点清洗
    cleaned: List[Document] = []
    for d in docs:
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        # 限制太长的段落，避免向量化出错
        if len(txt) > 8000:
            txt = txt[:8000]
        cleaned.append(Document(page_content=txt, metadata=d.metadata))
    return cleaned

def build_faiss_index(file_id: str) -> Dict[str, Any]:
    md_file = markdown_path(file_id)
    if not md_file.exists():
        return {"ok": False, "error": "MARKDOWN_NOT_FOUND"}
    md_text = md_file.read_text(encoding="utf-8")

    docs = split_markdown(md_text)
    if not docs:
        return {"ok": False, "error": "EMPTY_MD"}

    embeddings = load_embeddings()
    vs = FAISS.from_documents(docs, embedding=embeddings)
    vs.save_local(str(index_dir(file_id)))
    return {"ok": True, "chunks": len(docs)}

def search_faiss(file_id: str, query: str, k: int = 5) -> Dict[str, Any]:
    idx = index_dir(file_id)
    if not (idx / "index.faiss").exists():
        return {"ok": False, "error": "INDEX_NOT_FOUND"}

    embeddings = load_embeddings()
    vs = FAISS.load_local(str(idx), embeddings, allow_dangerous_deserialization=True)
    hits = vs.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in hits:
        results.append({
            "text": doc.page_content,
            "score": float(score),
            "metadata": doc.metadata,
        })
    return {"ok": True, "results": results}

```

代码解释如下：

1. 核心依赖和配置

- **导入库**:
  - `langchain_openai`: 用于调用 OpenAI 的 API，特别是**向量化模型（embedding model）**。
  - `langchain_text_splitters`: 用于将长文档分割成小块，便于检索。这里用的是 `MarkdownHeaderTextSplitter`，可以根据 Markdown 的标题来切分文档。
  - `langchain.docstore.document`: 定义了 `Document` 对象，这是 LangChain 中用于表示文档块的标准格式。
  - `langchain_community.vectorstores.FAISS`: 核心库，实现了高效的相似性搜索和向量存储功能。FAISS（Facebook AI Similarity Search）是一个用于快速、高效地在大规模数据中进行相似性搜索的库。
  - `dotenv`: 从 `.env` 文件加载环境变量，用于配置 OpenAI API 密钥和代理地址。
- **文件路径**:
  - 这部分代码复用了 `pdf_service.py` 中定义的统一数据目录结构。
  - `markdown_path`: 指向由 `pdf_service` 生成的 `output.md` 文件。
  - `index_dir`: 为每个文件创建一个独立的目录，用于存储构建好的 FAISS 索引。
- **`load_embeddings()`**:
  - 这个函数的作用是**加载并配置 OpenAI 的向量化模型**。
  - 它从环境变量中读取 API 密钥和可选的 `base_url`（用于代理），然后初始化 `OpenAIEmbeddings` 对象。
  - `model="text-embedding-3-small"` 指定了使用的具体模型，这是一个性价比很高的向量化模型。

2. 文档分割和向量化

- **`split_markdown(md_text)`**:
  - 这个函数是 RAG 系统中**分块（chunking）**步骤的关键。
  - 它使用 `MarkdownHeaderTextSplitter`，根据 Markdown 的一级和二级标题（`#` 和 `##`）来分割文档。这样做可以确保每个文档块都保留其上下文结构，例如，一个段落会与它所属的子标题关联在一起。
  - 此外，它还包含了一个简单的清洗逻辑：
    - 移除空的内容块。
    - **限制过长的内容块**（例如，超过 8000 个字符）。这是因为过长的输入可能会导致向量化模型处理失败或效率低下。
  - 最终返回一个 `Document` 对象的列表，每个对象都代表一个可用于检索的文档块。
- **`build_faiss_index(file_id)`**:
  - 这是**索引构建**的核心函数，对应于 API 中的 `/index/build` 路由。
  - 它执行以下步骤：
    1. **读取 Markdown 文件**: 从 `data/{file_id}/output.md` 读取文本内容。
    2. **文档分割**: 调用 `split_markdown` 将文本分割成文档块。
    3. **向量化和索引构建**:
       - 调用 `load_embeddings()` 获取向量化模型。
       - 使用 `FAISS.from_documents(docs, embedding=embeddings)` 将所有文档块转换为向量，并构建 FAISS 索引。这个过程会计算每个文档块的**嵌入向量**（embedding vector），并将其存储在 FAISS 数据库中。
    4. **保存索引**: `vs.save_local()` 将构建好的索引和文档数据保存到 `index_faiss` 目录下，以便后续使用。

3. 索引搜索

- **`search_faiss(file_id, query, k)`**:
  - 这是**检索**步骤的核心函数，对应于 API 中的 `/index/search` 路由。
  - 它执行以下步骤：
    1. **加载索引**: 从 `index_dir` 目录加载之前保存的 FAISS 索引。`allow_dangerous_deserialization=True` 是一个安全警告，因为加载的索引可能包含任意 Python 对象，但在这个特定的上下文（本地文件）中是安全的。
    2. **向量化查询**: 调用 `vs.similarity_search_with_score(query, k=k)`。在内部，FAISS 会首先将用户输入的 `query`（查询文本）通过 `load_embeddings()` 函数进行向量化，得到一个查询向量。
    3. **相似性搜索**: FAISS 然后会在其数据库中，**使用高效的算法**（如 LSH、PQ 等）来寻找与查询向量最相似的 `k` 个文档向量。
    4. **返回结果**: 返回一个包含文档内容、相似度分数和元数据（如原始 PDF 的页码）的列表，这些结果将作为上下文传递给语言模型以生成回答。

### 4. `services/rag_service.py` RAG 检索+生成（SSE）与会话历史功能开发

**定位**：提供**一次问答的完整闭环**：
 检索 → 判断是否相关 → 先发 `citation` → 带历史与上下文生成流式回答 → `done`。

关键组成

1. **可配置项**
   - LLM：`init_chat_model("deepseek-chat")`（或其他提供商）。
   - Embedding：与 `index_service` 保持一致。
   - 阈值：`SCORE_TAU_TOP1/MEAN3`（FAISS L2：越小越相似）。
   - K 值：top-k 片段数量（默认 3~5）。
2. **检索与相关性判定**
   - 先做向量检索，得到 `(doc, score)` 列表；
   - 构造 `citations`：`{citation_id, fileId, rank, page?, snippet, score, previewUrl}`；
   - **双信号判定**：
     - 规则：分数阈值（top-1/前3均值）
     - LLM 复核：`GRADE_PROMPT` 让模型判断“检索上下文是否能回答该问”
   - 只要一个为真→用检索上下文；否则走“无上下文回答”。
3. **SSE 输出顺序**
   - 如果走 with_context：**先** `event: citation` *n* 条；
   - **再** 通过 `astream` 流式生成 `event: token`；
   - 结尾 `event: done`，包含 `{"used_retrieval": true|false}`。
   - 若 `astream` 不可用 → 回退整段生成 + 手工切片。
4. **多轮会话（内存态）**
   - `_sessions: Dict[sessionId, List[{"role","content"}]]`
   - 每次 `/chat`：把历史 + 本轮用户拼进消息列表；
   - 生成后把本轮问答追加进历史；
   - `/chat/clear` 清空 `sessionId`；重启即丢失（满足教学目标）。

**完整代码如下**

```python
# services/rag_service.py
from __future__ import annotations
import os, asyncio, textwrap
from typing import List, Dict, Any, Tuple, AsyncGenerator
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- 配置 ----------------
MODEL_NAME = "deepseek-chat"
MODEL_PROVIDER = "deepseek"
TEMPERATURE = 0

EMBED_MODEL = "text-embedding-3-large"
K = 3
# FAISS L2：越小越相似；按你的数据可微调
SCORE_TAU_TOP1 = 0.45
SCORE_TAU_MEAN3 = 0.60

SYSTEM_INSTRUCTION = (
    "You are an MCP technical training assistant. 'MCP' means Model Context Protocol.\n"
    "Prefer using retrieved course materials to answer. If no relevant context is found, "
    "answer from your general knowledge and explicitly mention that no matching course content was found."
)
GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved context to the user's question.\n"
    "Context snippets:\n{context}\n\nQuestion: {question}\n"
    "Return exactly 'yes' if the context is helpful to answer the question, otherwise 'no'."
)
ANSWER_WITH_CONTEXT = (
    "Answer the user's question using the provided context.\n\n"
    "Question:\n{question}\n\nContext:\n{context}\n\n"
    "Write in Markdown. Be concise but complete. If code is relevant, use fenced code blocks.\n"
    "Do not fabricate information not present or entailed by the context."
)
ANSWER_NO_CONTEXT = (
    "No relevant course context was found for the question.\n"
    "Answer from your general knowledge. Be clear and accurate.\n"
    "Question:\n{question}"
)

# ---------------- 模型/向量函数 ----------------
def _get_llm():
    return init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=TEMPERATURE)

def _get_grader():
    return init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=0)

def _get_embeddings():
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_EMBEDDING_BASE_URL") or "https://ai.devtool.tech/proxy/v1",
        model=EMBED_MODEL,
    )

def _vs_dir(file_id: str) -> str:
    return os.path.join("data", file_id, "index_faiss")

def _load_vs(file_id: str) -> FAISS:
    vs_path = _vs_dir(file_id)
    idx_file = os.path.join(vs_path, "index.faiss")
    if not os.path.exists(idx_file):
        raise FileNotFoundError(f"FAISS index not found at {vs_path}; build index first.")
    return FAISS.load_local(vs_path, _get_embeddings(), allow_dangerous_deserialization=True)

def _score_ok(scores: List[float]) -> bool:
    if not scores:
        return False
    top1 = scores[0]
    mean3 = sum(scores[:3]) / min(3, len(scores))
    return (top1 <= SCORE_TAU_TOP1) or (mean3 <= SCORE_TAU_MEAN3)

# ---------------- 主流程：检索 + 判定 + 生成 ----------------
async def retrieve(question: str, file_id: str) -> tuple[list[dict], str]:
    """
    返回 (citations, context_text)
    citations: [{citation_id, fileId, rank, page, snippet, score, previewUrl}]
    context_text: 供 LLM 使用的拼接上下文
    """
    vs = _load_vs(file_id)
    hits = vs.similarity_search_with_score(question, k=K)
    citations = []
    ctx_snippets = []
    scores = []
    for i, (doc, score) in enumerate(hits, start=1):
        snippet_short = (doc.page_content or "").strip()
        if len(snippet_short) > 500:
            snippet_short = snippet_short[:500] + "..."
        page = doc.metadata.get("page") or doc.metadata.get("page_number")
        citations.append({
            "citation_id": f"{file_id}-c{i}",
            "fileId": file_id,
            "rank": i,
            "page": page,
            "snippet": (doc.page_content or "")[:4000],
            "score": float(score),
            "previewUrl": f"/api/v1/pdf/page?fileId={file_id}&page={(page or 1)}&type=original",
        })
        ctx_snippets.append(f"[{i}] {snippet_short}")
        scores.append(float(score))
    context_text = "\n\n".join(ctx_snippets) if ctx_snippets else "(no hits)"

    # 规则 + LLM 复核
    ok_by_score = _score_ok(scores)
    if not ok_by_score:
        grader = _get_grader()
        grade_prompt = GRADE_PROMPT.format(context=context_text, question=question)
        decision = await grader.ainvoke([{"role": "user", "content": grade_prompt}])
        ok_by_llm = "yes" in (decision.content or "").lower()
    else:
        ok_by_llm = True

    branch = "with_context" if ok_by_llm else "no_context"
    return citations, context_text if branch == "with_context" else ""

async def answer_stream(
    question: str,
    citations: list[dict],
    context_text: str,
    branch: str
) -> AsyncGenerator[dict, None]:
    """
    以增量事件的形式产出：
      {"type":"citation", "data": citation_dict}
      {"type":"token", "data": "text chunk"}
      {"type":"done"}
    """
    # 先把 citations 全部发给前端（便于角标立即出现）
    if branch == "with_context" and citations:
        for c in citations:
            yield {"type": "citation", "data": c}

    llm = _get_llm()
    if branch == "with_context" and context_text:
        prompt = ANSWER_WITH_CONTEXT.format(question=question, context=context_text)
    else:
        prompt = ANSWER_NO_CONTEXT.format(question=question)

    # 优先尝试流式（如果模型/提供方不支持，就回退整段再切片）
    try:
        async for chunk in llm.astream([
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]):
            # chunk 是 ChatGenerationChunk；取文本增量
            delta = getattr(chunk, "content", None)
            if delta:
                yield {"type": "token", "data": delta}
    except Exception:
        # 回退：一次性生成，再手动切片 token
        resp = await llm.ainvoke([
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ])
        text = resp.content or ""
        for i in range(0, len(text), 20):
            yield {"type": "token", "data": text[i:i+20]}
            await asyncio.sleep(0.01)

    yield {"type": "done", "data": {"used_retrieval": branch == "with_context"}}
```

代码解释如下：

1. 配置与辅助函数

- **模型与参数配置**:
  - `MODEL_NAME`, `MODEL_PROVIDER`, `TEMPERATURE`: 定义了用于生成答案的大语言模型（LLM）和其参数。这里使用 `deepseek-chat`，并设置 `TEMPERATURE` 为 0，这意味着模型会给出更确定、更少变化的答案。
  - `EMBED_MODEL`: 指定用于向量化的模型。
  - `K`: 指定检索时返回的最相关的文档片段数量（默认为3）。
  - `SCORE_TAU_TOP1`, `SCORE_TAU_MEAN3`: 设定了用于判断检索结果是否相关的**阈值**。L2 距离越小，相似度越高。如果最相关的文档得分低于 `0.45`，或者前三名的平均得分低于 `0.60`，则认为检索结果是相关的。
- **Prompt 模板**:
  - `SYSTEM_INSTRUCTION`: 给 LLM 的系统指令，定义了其角色和行为（例如，作为技术助手，优先使用提供的材料回答）。
  - `GRADE_PROMPT`: 用于**评估**检索到的上下文是否对回答问题有用的 Prompt。这是一个关键的 RAG 步骤，即**检索结果的自评估**。
  - `ANSWER_WITH_CONTEXT`: 如果检索到了相关的上下文，使用此 Prompt 引导 LLM 基于上下文来回答。
  - `ANSWER_NO_CONTEXT`: 如果没有找到相关上下文，使用此 Prompt 告诉 LLM 从其通用知识来回答，并明确说明这一点。
- **辅助函数**:
  - `_get_llm()`, `_get_grader()`, `_get_embeddings()`: 初始化并返回不同目的的 LLM 和嵌入模型实例。
  - `_vs_dir(file_id)`, `_load_vs(file_id)`: 封装了加载 FAISS 向量数据库的逻辑。
  - `_score_ok(scores)`: 这是一个**基于规则的检索结果评估**函数。它检查检索到的文档块的相似度得分是否达到了预设的阈值。

2. 检索与评估 (`retrieve`)

`retrieve` 函数是整个 RAG 流程的起点，它负责从向量数据库中**检索**最相关的文档片段。

1. **加载向量数据库**: `_load_vs(file_id)` 加载指定文件的 FAISS 索引。
2. **执行相似性搜索**: `vs.similarity_search_with_score(question, k=K)` 根据用户问题，在索引中找到最相似的 `k` 个文档块，并返回每个文档块及其相似度得分。
3. **格式化引用**: 遍历检索结果，为每个文档块创建**引用对象（`citations`）**。这些对象包含了文档内容片段 (`snippet`)、分数 (`score`)、页码 (`page`) 和一个用于前端展示的预览 URL (`previewUrl`)。
4. **拼接上下文**: 将检索到的文档块内容拼接成一个单一的字符串 (`context_text`)，供 LLM 使用。
5. **评估检索结果**:
   - **规则评估**: 首先，通过 `_score_ok(scores)` 函数，基于简单的相似度得分阈值来判断检索结果是否相关。
   - **LLM 评估**: 如果规则评估失败（即得分太高，表示不相似），代码会引入一个**额外的 LLM（即 Grader）** 来进行二次评估。它会给 Grader LLM 发送一个 Prompt，让其判断检索到的上下文对回答问题是否有用。这是一种更灵活、更智能的评估方法，可以弥补单纯依赖数值阈值的不足。
   - **决定分支**: 根据规则和 LLM 的评估结果，最终决定是进入 `with_context` 分支（使用检索到的上下文）还是 `no_context` 分支（不使用）。

3. 流式生成答案 (`answer_stream`)

`answer_stream` 函数是 RAG 流程的**生成**部分，它负责将最终的答案以流式（Streaming）形式返回给用户。

1. **确定 Prompt**: 根据 `retrieve` 函数决定的 `branch`（`with_context` 或 `no_context`），选择相应的 Prompt 模板 (`ANSWER_WITH_CONTEXT` 或 `ANSWER_NO_CONTEXT`)。
2. **流式生成**:
   - 代码使用 `llm.astream()` 方法来**异步地、增量地**获取 LLM 的生成内容。
   - 它将系统指令和用户 Prompt 发送给 LLM。
   - 随着 LLM 逐步生成答案，`astream()` 会返回一个一个的 `chunk`。
   - 代码会提取每个 `chunk` 的内容增量（`delta`），并将其封装成一个 `{"type": "token", "data": "..."}` 的事件，通过 `yield` 返回给调用者（即 FastAPI）。
3. **回退机制**:
   - 如果 `astream()` 因任何原因失败（例如，模型提供商不支持流式传输），代码会优雅地**回退**到一次性生成完整答案，然后手动将答案分割成小块，并模拟流式返回。
4. **发送引用**: 在开始生成答案之前，它会先将所有有效的 `citations`（引用）以流事件的形式发送出去，这样前端就可以立即显示文档来源。
5. **完成事件**: 在所有内容都发送完毕后，它会发送一个 `{"type": "done", "data": ...}` 事件，告知前端问答流程已结束，并包含是否使用了检索结果的信息。

功能小结

- `app.py`：**协议层**，把请求分发给业务服务，并负责 SSE/错误/跨域。
- `pdf_service.py`：**解析工厂**，从 PDF 到 Markdown + 页图 + 可视化，异步可观测。
- `index_service.py`：**索引工厂**，切分→向量化→FAISS；提供 Top-K 检索。
- `rag_service.py`：**对话大脑**，“检索→相关性判定→ citations→流式生成→写历史”。

## 阶段四：后端功能验证与前后端功能联调

​	然后即可开启后端接口并进行基本功能测试：

```bash
cd backend
uvicorn app:app --reload --port 8001
```

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250901204256769.png" alt="image-20250901204256769" style="zoom:50%;" />

然后即可在网址：http://127.0.0.1:8002/docs 中进行功能测试：

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250901204409259.png" alt="image-20250901204409259" style="zoom:50%;" />

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250901204517808.png" alt="image-20250901204517808" style="zoom:50%;" />

- 前端功能说明

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250901205300037.png" alt="image-20250901205300037" style="zoom:50%;" />

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250901205332575.png" alt="image-20250901205332575" style="zoom:50%;" />

## 多模态RAG系统完整运行流程

- 完整项目源码领取

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250901204733597.png" alt="image-20250901204733597" style="zoom:50%;" />

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/0d489e9d3c29af2d7f319171eac040d0.png" alt="0d489e9d3c29af2d7f319171eac040d0" style="zoom:50%;" />

- 安装依赖

  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250901204819715.png" alt="image-20250901204819715" style="zoom:50%;" />

  ```bash
  pip install -r requirements.txt
  ```

- 创建环境变量

  需要手动创建.env文件，并输入OpenAI的API-KEY和DeepSeek API-KEY

  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20250901204916773.png" alt="image-20250901204916773" style="zoom:50%;" />

  ```bash
  DEEPSEEK_API_KEY=sk-...
  OPENAI_API_KEY=sk-...
  OPENAI_BASE_URL=https://ai.devtool.tech/proxy/v1
  ```

- 开启后端

  ```bash
  cd backend
  uvicorn app:app --reload --port 8001
  ```

- 开启前端

  然后再打开一个命令行，开启前端：

  ```bash
  # 1. 安装依赖
  npm install
  
  # 2. 启动开发服务器
  npm run dev
  ```

- 进行对话

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/27f4b2e749af80e62b1a9e3900e30e3f_raw.mp4"></video>

---

- 体验课内容节选自[《2025大模型Agent智能体开发实战》(秋招冲刺班)](https://ix9mq.xetslk.com/s/2S2Vpy)完整版付费课程

&emsp;&emsp;体验课时间有限，若想深度学习大模型技术，欢迎大家报名由我主讲的[《2025大模型Agent智能体开发实战》(秋招冲刺班)](https://ix9mq.xetslk.com/s/2S2Vpy)

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/06661cb459aa3e4b655aface404435d.png" alt="06661cb459aa3e4b655aface404435d" style="zoom:15%;" />

**[《2025大模型Agent智能体开发实战》(秋招冲刺班)](https://ix9mq.xetslk.com/s/2S2Vpy)为【100+小时】体系大课，总共20大模块精讲精析，零基础直达大模型企业级应用！**

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/202506172010074.png" alt="a55d48e952ed59f8d93e050594843bc" style="zoom:50%;" />

### 部分课程成果演示

- Dify+DeepSeek搭建智能客服

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/2f1b47f42c65fd59e8d3a83e6cb9f13b_raw.mp4"></video>

- Coze自动图文视频创作流程

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/Coze%E5%8A%A8%E6%80%81%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90%E5%AE%9E%E4%BE%8B.mp4"></video>

- 可视化数据分析Multi-Agent

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/%E5%8F%AF%E8%A7%86%E5%8C%96%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90Multi-Agent%E6%95%88%E6%9E%9C%E6%BC%94%E7%A4%BA%E6%95%88%E6%9E%9C.mp4"></video>

- Ollama 自动化并发请求测试与动态资源监控

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/3.Ollama%20%E8%87%AA%E5%8A%A8%E5%8C%96%E5%B9%B6%E5%8F%91%E8%AF%B7%E6%B1%82%E6%B5%8B%E8%AF%95%E4%B8%8E%E5%8A%A8%E6%80%81%E8%B5%84%E6%BA%90%E7%9B%91%E6%8E%A7.mp4"></video>

- Neo4j并行多线程导入百万级文本方法与实践

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/2.Neo4j%E5%B9%B6%E8%A1%8C%E5%A4%9A%E7%BA%BF%E7%A8%8B%E5%AF%BC%E5%85%A5%E7%99%BE%E4%B8%87%E7%BA%A7%E6%96%87%E6%9C%AC%E6%96%B9%E6%B3%95%E4%B8%8E%E5%AE%9E%E6%88%98%E6%BC%94%E7%A4%BA.mp4"></video>

- 高效微调全自动数据集创建

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/easy_daset_yanshi.mp4"></video>

- MateGen Pro 项目功能演示

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/MG%E6%BC%94%E7%A4%BA%E8%A7%86%E9%A2%91.mp4"></video>

- 智能客服项目展示

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/%E6%99%BA%E8%83%BD%E5%AE%A2%E6%9C%8D%E6%A1%88%E4%BE%8B%E8%A7%86%E9%A2%91.mp4"></video>

- **GraphRAG+多模态文档检索**

<video src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/7%E6%9C%8817%E6%97%A5%281%29%20%E8%BF%9B%E5%BA%A6%E6%9D%A1.mp4"></video>

此外，若是对大模型底层原理感兴趣，也欢迎报名由我和菜菜老师共同主讲的[《2025大模型原理与实战课程》(秋招冲刺班)](https://ix9mq.xetslk.com/s/3AME7R)

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/202506171650709.png" alt="4a11b7807056e9f5b281278c0e37dad" style="zoom:20%;" />

**大模型秋招冲刺班开班特惠进行中，直播间享五折特价+全套SVIP新班特定福利，合购还有更多优惠哦~<span style="color:red;">详细信息扫码添加助教，回复“大模型”，即可领取课程大纲&查看课程详情👇</span>**

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/26449c9c3e90ea66e0af9150ad00e0c6.png" alt="26449c9c3e90ea66e0af9150ad00e0c6" style="zoom:50%;" />

<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/0d489e9d3c29af2d7f319171eac040d0.png" alt="0d489e9d3c29af2d7f319171eac040d0" style="zoom:50%;" />







