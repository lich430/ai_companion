# AI Companion (Guoguo Engine)

这是一个基于 `GLM-4.7` 的角色聊天引擎项目，当前已经不是单纯的 prompt 调用，而是一个包含：

- 角色圣经
- 关系状态机
- 分层记忆
- 语义去重
- 消息分类
- quiet hours / noreply 策略
- 异步 Hook 入站

的完整链路。

## 1. 当前整体架构

项目现在分为 2 条主线：

1. `app.py`
角色引擎核心。
负责状态、记忆、分类、生成、后处理、入队。

2. `api.py`
FastAPI 服务层。
负责把引擎能力暴露成 HTTP 接口，并提供 Hook 异步入口和待发送消息队列接口。

微信发送侧执行器。
不再负责“读未读消息并生成回复”，而是只做两件事：
- 从后端拉取待发送消息
- 用 ADB/uiautomator2 打开对应聊天窗口并发送

## 2. 目录主模块说明

### 核心引擎

- `app.py`
  - `GuoguoEngine` 主类
  - 聊天主流程 `chat()`
  - Hook 入站批处理 `ingest_incoming_messages()`
  - 待发送消息队列 `fetch_pending_replies()`

- `data/guoguo_bible.json`
  - 果果角色圣经
  - 包含人设、关系阶段、营销规则、冲突降级规则

- `engine/prompt_builder.py`
  - 构造 system prompt / user prompt
  - 控制营销信息仅在 trigger 命中时进入 prompt

- `engine/relationship_state_machine.py`
  - 关系阶段流转
  - 根据消息事件更新 `stranger/familiar/close/upset/recovering`

- `engine/memory_manager.py`
  - SQLite 持久化
  - 保存聊天消息、记忆、用户状态、turn_count、主动消息时间

- `engine/memory_summarizer.py`
  - 每 N 轮对最近对话做记忆摘要压缩

- `engine/message_classifier.py`
  - 用户消息分类
  - 输出：
    - `low_context`
    - `hostile`
    - `boring`
    - `need_no_reply`
    - `urgent`

- `engine/text_postprocess.py`
  - 生成后处理
  - 轻微口语化标点松弛
  - 低频 emoji / 文本表情

- `engine/repetition_guard.py`
  - 词面 + 语义双重去重

- `engine/response_generator.py`
  - GLM API 调用层

- `engine/cadence_simulator.py`
  - 把一条回复拆成分段发送计划（cadence）

- `engine/proactive_scheduler.py`
  - 主动消息调度器

### API 层

- `api.py`
  - `/chat`
  - `/chat/sync_user_messages`
  - `/proactive/suggest`
  - `/proactive/mark_sent`
  - `/context/{user_id}`
  - `/users`
  - `/hook/incoming`
  - `/hook/pending`
  - `/metrics/reply_policy`

### 微信发送侧

  - 轮询 `/hook/pending`
  - 通过广播：
    - `com.example.wechatglm.OPEN_CHAT`
  - 打开指定 wxid 聊天窗口
  - 识别聊天页输入框/发送按钮
  - 按 `recommended_delay_ms` 发送消息

## 3. 当前主线流程

### 流程 A：普通 API 聊天

适合 Web / App / 调试。

1. 客户端调用 `POST /chat`
2. `api.py` 调用 `engine.chat(user_id, message)`
3. `GuoguoEngine.chat()` 做这些事：
   - 分类用户消息
   - 写入聊天消息
   - 更新关系状态
   - 写入用户记忆
   - quiet hours / noreply 判定
   - 召回相关记忆
   - 构建 prompt
   - 请求 GLM
   - 做回复后处理
   - 写回 assistant 消息
   - 更新统计
4. API 返回：
   - `type` (`reply` / `noreply`)
   - `reply`
   - `recommended_delay_ms`
   - `reason`
   - `state`
   - `cadence`

### 流程 B：微信 Hook 入站 -> 后端异步处理

这是现在微信链路的主线。

1. `WechatHook.java` 监听到微信数据库里的新文本消息
2. Hook 调用后端 `POST /hook/incoming`
3. `api.py` 先把任务放入内存队列，立即返回
4. 后台 worker 线程 `_incoming_worker()` 做批处理：
   - 合并同一个 `user_id` 的连续消息
   - 调用 `engine.ingest_incoming_messages()`
5. `GuoguoEngine.ingest_incoming_messages()`：
   - 先把多条用户消息全部入库
   - 更新状态和记忆
   - 判断是否 `noreply`
   - 如果需要回复，允许一次生成多条回复
   - 把结果放入 `pending_replies` 队列
7. 对每条 pending item：
   - 用 ADB 广播打开指定 `wxid` 聊天窗口
   - 识别聊天页输入框与发送按钮
   - 等待 `recommended_delay_ms`
   - 分段发送文本

### 流程 C：主动消息

1. 客户端调用 `POST /proactive/suggest`
2. 引擎根据：
   - 最近用户消息时间
   - 关系阶段
   - 冷却期
   - 当前时间
   生成一个 `ProactivePlan`
3. 若允许发送，则返回文本 + cadence
4. 外部系统自行决定是否真的发送

## 4. 当前回复策略

### 4.1 不一定每条都回复

引擎现在支持：

- `type=reply`
- `type=noreply`

常见 `noreply` 原因：

- `quiet_hours`
- `need_no_reply`

### 4.2 quiet hours

默认：

- `02:00-09:00`

该时段默认不回复，除非消息被分类为 `urgent`。

### 4.3 low context

如果消息信息量很低，并且最近没有有效上下文：

- 直接回 `？`
- 并给一个较短的 `recommended_delay_ms`

### 4.4 hostile / boring

- 持续无聊、无建设性、挑衅：可能直接 `noreply`
- hostile 但可降温：回复一条温和降级句，不对骂

### 4.5 营销信息控制

营销信息不会主动进入 prompt。
只有当用户明确问到以下内容时才允许进入生成上下文：

- 怎么玩
- 怎么消费
- 多少钱
- 什么价格
- 包厢价格
- 店名
- 电话
- 地址

对应配置在：

- `data/guoguo_bible.json -> marketing`

## 5. 当前数据持久化

SQLite 文件：

- `memory.db`

已持久化的数据包括：

- 聊天消息
- 用户状态
- turn_count
- 主动消息时间
- 结构化记忆

所以 API 重启后，这些不会丢。

不持久化的主要是：

- `api.py` 进程内的 incoming queue
- `app.py` 进程内的 pending_replies 队列

也就是说：
- 正在排队但尚未处理的 Hook 入站任务，服务重启会丢

## 6. 关键接口

### 健康检查

```http
GET /health
```

### 聊天

```http
POST /chat
```

请求：

```json
{
  "user_id": "u_001",
  "message": "你好",
  "with_cadence": true
}
```

返回：

```json
{
  "type": "reply",
  "reply": "你好呀",
  "recommended_delay_ms": 1200,
  "reason": "reply",
  "state": {
    "stage": "stranger",
    "mood": "neutral"
  },
  "cadence": [
    {"text": "你好呀", "delay_ms": 1200}
  ]
}
```

### 批量同步用户消息（只入库，不生成）

```http
POST /chat/sync_user_messages
```

### 主动消息建议

```http
POST /proactive/suggest
```

### 标记主动消息已发送

```http
POST /proactive/mark_sent
```

### 查看用户上下文

```http
GET /context/{user_id}?limit=20
```

### 查看用户列表

```http
GET /users
```

### Hook 入站

```http
POST /hook/incoming
```

请求：

```json
{
  "username": "wxid_xxx",
  "user_id": "wxid_xxx",
  "message": "哈喽"
}
```

### 拉取待发送消息

```http
GET /hook/pending?limit=5&pop=true
```

### 查看回复策略统计

```http
GET /metrics/reply_policy
```



- 读微信未读消息
- 从聊天页提取对方消息
- 直接调用聊天引擎生成回复

它现在只负责发送：

1. 拉取后端 `/hook/pending`
2. 通过 ADB 广播打开指定用户聊天窗口
3. 识别输入框 / 发送按钮
4. 按推荐延迟发送内容

这意味着“收消息”和“回消息”的链路已经拆开：

- 收消息：Hook + `/hook/incoming`
- 生成回复：`app.py`

## 8. 启动方式

### 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 启动 API

推荐 PowerShell：

```powershell
.\scripts\start_api.ps1
```

这个脚本会：

- 清理旧的 8080 占用进程
- 再启动 `uvicorn api:app --reload`

只清理旧进程：

```powershell
.\scripts\stop_api.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stop_api.ps1
```

手动启动也可以：

```powershell
python -m uvicorn api:app --host 0.0.0.0 --port 8080 --reload
```

### 启动本地 CLI 调试

```powershell
python app.py
```

### 启动微信发送器

```powershell
python app.py
```

## 9. 环境变量

常用 `.env` 配置：

```env
LLM_PROVIDER=glm
GLM_API_KEY=your_glm_api_key_here
GLM_CHAT_MODEL=glm-4.7
GLM_EMBED_MODEL=embedding-3
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://api.openai.com/v1
PREFERRED_REPLY_CHARS=30
MULTI_REPLY_PROBABILITY=0.2
BLOCK_ALL_GROUP_MESSAGES=false
BLOCK_GROUP_IDS=
BLOCK_GROUP_KEYWORDS=
INCOMING_DEDUP_WINDOW_SEC=3
MEMORY_SUMMARY_EVERY_N_TURNS=8
LEXICAL_REPEAT_THRESHOLD=0.72
SEMANTIC_REPEAT_THRESHOLD=0.90
LOCAL_TIMEZONE=Asia/Shanghai
QUIET_HOURS=02:00-09:00
EMOJI_ENABLED=true
TEXT_POSTPROCESS_SEED=20260307
ENGINE_API_BASE=http://127.0.0.1:8080/
```

## 10. 测试

运行：

```powershell
python -m pytest
```

当前覆盖：

- postprocess 标点/emoji 开关
- quiet hours 下默认 noreply / urgent 例外
- low_context + no context -> `？`
- boring / hostile -> noreply

## 11. 现在的核心设计结论

这个项目当前已经不是“模型直连回复器”，而是：

- 角色工程化
- 状态驱动
- 记忆驱动
- 产品策略驱动
- 微信发送链路解耦

主线最重要的函数是：

- `app.py -> GuoguoEngine.chat()`
- `app.py -> GuoguoEngine.ingest_incoming_messages()`
- `api.py -> /hook/incoming`
- `api.py -> /hook/pending`

如果后面继续演进，最值得做的方向是：

1. 把 pending 队列持久化
3. 给 `/metrics/reply_policy` 增加按用户维度的统计

## 12. 项目总结

这是一个面向私聊场景的角色聊天后端，不是单纯的 LLM 调用器。它围绕角色圣经、关系状态机、长期记忆、时间感知、重复控制和回复策略来组织生成过程，并通过 FastAPI 提供聊天、Hook 入站、待发送队列等接口。整体目标是让角色回复更像“持续存在的人”，而不是每轮独立生成的文本。
