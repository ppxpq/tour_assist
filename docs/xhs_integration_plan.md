# 小红书 URL 导入资料库集成方案

## 目标

在现有“资料库”能力上增加一种输入来源：

```text
用户粘贴小红书笔记 URL
-> 系统抓取笔记正文
-> 清洗成旅行资料
-> 写入现有 Chroma 知识库
-> Researcher/Planner 规划时可检索引用
```

这个能力定位为“用户主动提供社媒攻略链接并整理入库”，不做全站搜索和批量爬取。

## xhs 项目判断

本地项目路径：

```text
/Users/heyining/xhs
```

核心文件：

- `api/feed.js`：根据 `note_id` 和 `xsec_token` 获取笔记详情。
- `api/page_comment.js`：获取一级评论，可作为后续增强。
- `config.js`：保存 `b1`、`cookie_a1`、完整 cookie、`baseURL`。

第一版只调 `feed` 基本够用。

`feed` 返回里可用字段包括：

- `note_card.title`
- `note_card.desc`
- `note_card.tag_list`
- `note_card.user.nickname`
- `note_card.interact_info`
- `note_card.image_list`
- `note_card.time`
- `note_card.ip_location`

注意：当前 `feed.js` 和评论接口里 import 了不存在的 `../his/X-S_deprecated.js`。这个 import 没被实际使用，但会导致 Node 模块加载失败，需要在 xhs 项目里删掉该 import 或补齐文件。

## 推荐架构

不要把 `/Users/heyining/xhs` 作为子目录提交进当前项目。

当前项目只保存适配逻辑：

```text
Tour_assist_agent
├─ main.py
├─ core/
│  ├─ travel_service.py
│  ├─ db_manager.py
│  └─ xhs_importer.py        # 新增：调用外部 xhs 脚本、清洗文本
└─ frontend/
```

外部 xhs 项目独立存在：

```text
/Users/heyining/xhs
```

配置用环境变量指定路径：

```env
XHS_PROJECT_PATH=/Users/heyining/xhs
```

## 第一阶段：最小可用版本

### 1. 在 xhs 项目增加一个 CLI 脚本

新增：

```text
/Users/heyining/xhs/scripts/fetch_note.mjs
```

功能：

```bash
node scripts/fetch_note.mjs --note-id xxx --xsec-token xxx
```

输出标准 JSON：

```json
{
  "success": true,
  "source": "xhs",
  "note_id": "...",
  "title": "...",
  "desc": "...",
  "tags": ["无锡", "旅游"],
  "author": "...",
  "interact": {
    "liked_count": "...",
    "collected_count": "...",
    "comment_count": "..."
  },
  "url": "https://www.xiaohongshu.com/explore/..."
}
```

### 2. 当前项目新增后端接口

新增 API：

```http
POST /api/knowledge-base/xhs-url
Content-Type: application/json
```

请求：

```json
{
  "url": "https://www.xiaohongshu.com/explore/xxxx?xsec_token=yyyy",
  "model": "glm-4.5-air"
}
```

后端流程：

```text
解析 URL
-> 提取 note_id / xsec_token
-> subprocess 调用 node scripts/fetch_note.mjs
-> 归一化成 .txt 文本
-> 复用 service.ingest_knowledge()
```

### 3. 入库文本格式

将小红书笔记清洗成类似：

```text
【来源】小红书笔记
【链接】https://www.xiaohongshu.com/explore/...
【标题】...
【作者】...
【标签】无锡、旅游、亲子
【互动】点赞 123，收藏 45，评论 6

【正文】
...

【适合规划参考】
- 用户主动提供的社媒攻略资料。
- 可用于景点推荐、避坑提醒、餐饮选择和行程节奏判断。
```

然后包装为 `UploadedFileData(name="xhs_{note_id}.txt", content=..., content_type="text/plain")`，直接走现有知识库入库。

## 前端改动

资料库面板增加：

```text
小红书链接
[输入框：粘贴笔记 URL]
[导入]
```

导入成功后复用现有知识库状态：

```text
已收纳 N 条资料片段，规划时会一起参考。
```

## 第二阶段增强

### 评论摘要

如果只用 `feed`，能拿正文但没有真实评论反馈。

后续可以加：

- `page_comment` 只拉前 10 条一级评论。
- 清洗出高频词、避坑、排队、人流、交通、餐饮反馈。
- 入库文本追加 `【评论口碑摘要】`。

### Planner 输出增强

Planner prompt 增加要求：

```text
如果知识库中包含小红书/社媒攻略资料，请在行程概览或注意事项中提炼“社媒口碑参考”和“避坑提醒”，但不要声称实时全网数据。
```

## 风险与边界

- cookie 不能放前端，不能提交 Git。
- 不做批量爬取，不做搜索全站。
- 只处理用户主动粘贴的公开笔记链接。
- 抓取失败时要返回清晰错误，例如 cookie 失效、note_id 解析失败、接口限流。
- `xhs/config.js` 当前包含 cookie 相关配置，必须保留在独立项目中，不纳入当前仓库。

## 推荐实现顺序

1. 修 xhs 项目 `feed.js` 的无效 import。
2. 在 xhs 项目新增 `scripts/fetch_note.mjs`。
3. 在当前项目新增 `core/xhs_importer.py`。
4. 在 `main.py` 新增 `/api/knowledge-base/xhs-url`。
5. 前端资料库面板增加 URL 输入和导入按钮。
6. Planner 增加“社媒口碑参考”提示词。
