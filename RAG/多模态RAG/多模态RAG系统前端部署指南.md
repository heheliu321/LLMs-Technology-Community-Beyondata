# 多模态RAG检索系统 - 前端部署指南

这是一个基于React + TypeScript + Tailwind CSS构建的多模态RAG（检索增强生成）系统前端界面。

## 🚀 快速开始

### 1. 本地开发部署

```bash
# 1. 克隆或下载代码到本地
# 将所有文件拷贝到您的项目目录中

# 2. 安装依赖
npm install

# 3. 启动开发服务器
npm run dev
```

开发服务器将在 `http://localhost:3000` 启动。

### 2. 生产环境部署

```bash
# 构建生产版本
npm run build

# 启动生产服务器
npm run start
```

## ⚙️ 环境配置

### 运行模式配置

本系统支持两种运行模式：

#### 🔸 模拟模式（默认）
无需真实后端即可体验所有功能，系统将：
- 自动生成虚拟PDF文档页面
- 模拟流式聊天响应
- 展示完整的引用和解析功能
- 提供真实的交互体验

当 `VITE_API_BASE_URL` 未设置或指向localhost时自动启用。

#### 🔸 真实后端模式
连接到您的实际后端服务：

创建 `.env.local` 文件在项目根目录：

```bash
# .env.local
VITE_API_BASE_URL=https://your-backend-domain.com/api/v1

# 或本地开发环境：
# VITE_API_BASE_URL=http://localhost:8001/api/v1
```

#### 切换模式的方法

1. **使用模拟模式**（推荐用于演示）:
   - 删除或注释掉 `.env.local` 中的 `VITE_API_BASE_URL`
   - 系统将自动启用模拟功能

2. **连接真实后端**:
   - 在 `.env.local` 中设置正确的 `VITE_API_BASE_URL`
   - 确保后端服务正在运行

### 常见后端端口配置

- **FastAPI开发服务器**: 通常使用 `8000` 或 `8001` 端口
- **Flask开发服务器**: 通常使用 `5000` 端口  
- **Express/Node.js**: 通常使用 `3001` 或 `8080` 端口
- **生产环境**: 通常使用 `80` (HTTP) 或 `443` (HTTPS) 端口

## 🎯 功能特性

### 当前支持的功能

✅ **完整的用户界面**
- 现代化深色主题设计
- 响应式两列布局（聊天界面 + PDF面板）
- 流式聊天界面与markdown渲染
- PDF上传、预览和翻页功能
- 实时后端连接状态检查

✅ **智能模拟系统**
- 🎯 完整的PDF上传和解析模拟
- 🤖 智能AI对话响应（支持文档问答）
- 📄 动态PDF页面生成和预览
- 🔗 真实的引用链接和弹窗功能
- ⚡ 流式打字效果和进度指示
- 📊 向量索引构建和搜索模拟

✅ **生产就绪特性**
- TypeScript类型安全
- 错误处理和用户反馈
- 性能优化的组件架构
- 可配置的API连接

### 后端API要求

为了实现完整功能，您的后端需要提供以下API端点：

```
GET  /api/v1/health                    # 健康检查
POST /api/v1/pdf/upload               # PDF文件上传
POST /api/v1/pdf/parse                # 开始PDF解析
GET  /api/v1/pdf/status               # 查询解析状态
GET  /api/v1/pdf/page                 # 获取PDF页面图片
POST /api/v1/index/build              # 构建向量索引
POST /api/v1/index/search             # 搜索向量索引
POST /api/v1/chat                     # SSE流式聊天
GET  /api/v1/pdf/chunk                # 获取引用块详情
POST /api/v1/chat/clear               # 清空聊天会话
```

## 📦 项目结构

```
├── App.tsx                 # 主应用组件
├── components/             # 所有React组件
│   ├── ChatInterface.tsx   # 聊天界面组件
│   ├── Header.tsx          # 顶部导航栏
│   ├── HealthCheck.tsx     # API状态检查
│   ├── MarkdownRenderer.tsx # Markdown渲染器
│   ├── PDFPanel.tsx        # PDF展示面板
│   └── ui/                 # ShadCN UI组件库
├── services/               # API服务层
│   └── api.ts              # 后端API调用封装
└── styles/                 # 样式文件
    └── globals.css         # 全局样式和主题
```

## 🛠️ 开发指南

### 技术栈

- **React 18** - 用户界面框架
- **TypeScript** - 类型安全的JavaScript
- **Tailwind CSS v4** - 原子化CSS框架
- **ShadCN/UI** - 现代化组件库
- **Lucide React** - 图标库
- **Sonner** - Toast通知组件

### 自定义配置

#### 修改主题色彩

编辑 `styles/globals.css` 中的CSS变量：

```css
.dark {
  --primary: #3b82f6;        /* 主色调 */
  --secondary: rgba(51, 65, 85, 0.85); /* 次要色调 */
  --accent: rgba(59, 130, 246, 0.15);  /* 强调色 */
}
```

#### 添加新的API端点

在 `services/api.ts` 中添加新的API函数：

```typescript
export async function yourNewApiCall(): Promise<YourResponse> {
  const response = await fetch(`${API_BASE_URL}/your-endpoint`);
  return response.json();
}
```

## 🚨 常见问题

### Q: 前端启动后显示"API Offline"？

**A**: 这完全正常！系统已自动启用模拟模式，您可以：
1. 🎮 **立即体验**: 上传任意PDF文件体验完整功能
2. 💬 **测试聊天**: 与AI进行对话，支持中英文
3. 📖 **查看文档问答**: 上传PDF后提问相关内容
4. 🔌 **连接真实后端**: 配置 `VITE_API_BASE_URL` 环境变量

### Q: 如何修改后端连接地址？

**A**: 创建 `.env.local` 文件并设置 `VITE_API_BASE_URL` 变量，或直接修改 `services/api.ts` 文件。

### Q: 部署到生产环境需要注意什么？

**A**: 
1. 确保后端API地址配置正确
2. 检查CORS跨域设置
3. 使用HTTPS协议保证安全性
4. 配置合适的错误监控和日志记录

## 🔧 环境变量参考

```bash
# .env.local
VITE_API_BASE_URL=http://localhost:8001/api/v1  # 后端API地址
VITE_APP_TITLE=多模态RAG检索系统                  # 应用标题（可选）
```

## 📝 更新日志

- **v1.0.0** - 初始版本，完整的RAG系统前端界面
- 支持PDF上传、解析、聊天和引用功能
- 深色主题和现代化UI设计
- 完善的错误处理和离线演示模式

---

**作者**: 九天Hector  
**项目**: 多模态RAG检索系统演示界面