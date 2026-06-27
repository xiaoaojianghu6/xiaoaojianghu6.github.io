# Builder 二级页——项目数据
# 编辑此文件即可修改文字、调整章节顺序、增删媒体元素
# 改完后运行: python3 build.py

PROJECTS = {
    "cardioagent": {
        "title": "多模态心脏诊断智能体",
        "subtitle": "MULTI-AGENT · ISAIMS 2025",
        "role": "CO-AUTHOR",
        "date": "2025",
        "hero": "/detail/cardioagent/mv-pc.jpg",
        "hero_alt": "cardioagent 主视觉图",
        "overview": [
            "心血管疾病的精准诊疗太复杂，现有 AI 工具大多是个孤立的黑箱——能给预测，却说不清理由，更别提接进临床工作流。CardioAgent 换了个思路：多个智能体各管一摊，把 ECG、影像、临床指南这些多模态信号嚼碎了再拼起来。",
            "这篇工作是我在上海理工大学HealtIT团队实习时的工作成果，被 ISAIMS 2025（ACM）录用，我是共同作者，负责Agent内一维时序增强、多模态融合和 FAISS 向量检索那块 RAG 闭环。",
        ],
        "sections": [
            {
                "type": "pdf",
                "src": "/detail/cardioagent/cardioagent论文_副本.pdf",
                "title": "CardioAgent · ISAIMS 2025",
            },
            {
                "type": "text",
                "tag": "Unfold",
                "paragraphs": [
                    "心血管疾病诊疗的痛点很明确：现有 AI 大多是孤立的黑箱模型——能吐一个预测分数，却说不出为什么。更麻烦的是，心电图、医学影像、临床指南这些多模态信号各说各话，没有一个框架把它们揉在一起。CardioAgent 想做的，就是搭一个多智能体协作框架——每个 Agent 管一摊，把多模态信号嚼碎了再拼起来，给出可解释、能落地的决策支持。",
                    "技术路线上，多智能体分工协作是骨架，一维时序数据增强让 ECG 信号更鲁棒，FAISS 向量索引搭起 RAG 闭环——让每条建议都挂得上指南证据。我的角色是共同作者，扛了一维时序增强策略、多模态融合机制和 FAISS 向量索引 RAG 闭环三块——把「黑箱预测」往「可解释决策」推了一把。",
                ],
            },
        ],
        "next_slug": "text-restructuring",
        "next_title": "文本逻辑结构化引擎",
    },

    "image-segmentation": {
        "title": "图像/视频分割",
        "subtitle": "VIDEO STREAM SEGMENTATION",
        "role": "RESEARCH INTERN",
        "date": "2025.09至今",
        "hero": "/detail/image-segmentation/tim 分割.jpg",
        "hero_alt": "image-segmentation 主视觉图",
        "overview": [
            "SAM，全称 Segment Anything Model——Meta 搞出来的「万物皆可分割」模型。给它一张图，点一个位置，它就能把那个「东西」的轮廓完整地框出来。但我关心的不是「万物」，是「这一个」——在一段连续视频流里，持续追踪我指定的那个人，不是别人，不是路边的牛或马，是我框出来的 Tim哥。这个就是实例分割。",
        ],
        "sections": [
            {
                "type": "video",
                "src": "/detail/image-segmentation/sam2-native-overlay.mp4",
                "caption": "原生 SAM2 · 人物实例分割。逐帧追踪 Tim——「这一团像素，到底是谁？」",
            },
            {
                "type": "text",
                "tag": "人物实例分割",
                "paragraphs": [
                    "SAM2 的核心能力建立在 Hiera 架构和 Memory Bank 时序记忆之上——它不只是逐帧分割，而是跨帧「记住」目标。Hiera 作为分层视觉编码器，在不同尺度上提取特征；Memory Bank 在时间轴上维护目标的外观与位置记忆，让模型在遮挡、快速运动、光照变化时仍能锁定同一对象。我用的接口极简：仅保留 SAM2.1 原生权重和 BoxPrompt（Bounding Box 提示），在第一帧框出 Tim，后续帧模型自主追踪。",
                    "这一步的意义不仅是「能跑通」——它同时是一次隐式的架构泛化性验证。ReSurgSAM2 基于 SAM2.1 基座构建，其上叠加了 CSTMamba 跨模态融合等增强模块。我在非手术域的日常视频上仅用基座+BoxPrompt 做推理，等于在验证：SAM2.1 原生模块对域外数据的泛化潜力如何？Prompt 接口的通用适配能力如何？这为后续手术分割中的接口兼容性分析埋下了关键的对照基线。",
                ],
            },
            {
                "type": "text",
                "tag": "手术器械分割",
                "paragraphs": [
                    "这项工作的起点是 MICCAI 2025 Early Accepted 论文 ReSurgSAM2 的完整复现。手术器械分割的难度远超日常场景：无影灯下的钳子反光、血雾中模糊的刃口、手腕快速翻转时的运动模糊——每一帧都是一场高难度辨认游戏。我在服务器端覆盖 CUDA 11.7 到 12.2 的版本升级（无 root 权限），从零搭建训练与推理 pipeline，完整跑通了 CSTMamba + CIFS + DLM 架构。",
                    "显存是手术视频推理的硬瓶颈——高分辨率（1920×1080）手术视频在默认配置下峰值达 15GB。我启用了 CPU-offloading 策略，将视频特征卸载到系统内存，把推理显存峰值压缩到 6–7GB，降幅 50–70%，让消费级 GPU 也能跑完整推理。在 Ref-Endovis17 验证集上达到 76.98% J&F / 80.56% Dice / 61.2 FPS。",
                ],
            },
            {
                "type": "image_panel",
                "images": [
                    {
                        "src": "/detail/image-segmentation/指尖创世图和 resurgsam 分割对比图.jpg",
                        "alt": "指尖创世与 ReSurgSAM2 分割对比",
                        "label": "A",
                    },
                    {
                        "src": "/detail/image-segmentation/resurgsam 分割与指尖创世动作很像.png",
                        "alt": "ReSurgSAM2 分割掩码与指尖创世动作",
                        "label": "B",
                    },
                ],
            },
            {
                "type": "text",
                "tag": "缺陷定位与改进",
                "paragraphs": [
                    "复现不是终点，找问题才是。我设计了一组 first_frame 约束基线实验——只用第一帧的标注信息做全视频推理，得到 J&F 80.15%。与完整方法的差距分析定位出两类独立缺陷：早期误识别（模型在第一帧之后的扩散过程中逐渐漂移）和 CLIP 语义混淆（CSTMamba 引入的 CLIP 文本先验过强，导致器械被映射到错误的语义类别）。",
                    "在此基础上，我实现了完整的 PEFT 微调方案：CLIP 高层解冻（释放语义编码器的适应性），object_score 损失加权（加大误分类样本的惩罚），以及 SAM3 风格的 Existence Token（让模型学会「这个目标是否还在画面中」）。首次微调因多变量耦合导致性能不升反降——这恰好验证了缺陷定位的准确性：三个改进维度必须独立调优。目前正在搭建基于 DDP 的多卡分布式训练，设计覆盖学习率、解冻层数、损失权重等维度的 Grid Search 消融实验，系统验证各组件的收敛特征。",
                    "GitHub: xiaoaojianghu6/rss2。HUST | WNLO 算法研究实习生（2025.09至今）。",
                ],
            },
        ],
        "next_slug": "cardioagent",
        "next_title": "多模态心脏诊断智能体",
    },

    "math-modeling": {
        "title": "数学建模",
        "subtitle": "COMPETITION",
        "role": "TEAM LEADER · MODELER",
        "date": "2025–2026",
        "hero": "/detail/math-modeling/mv-pc.jpg",
        "hero_alt": "math-modeling 主视觉图",
        "overview": [
            "2025 这一年，我拿同一套建模思维打了两场仗。国赛对着 NIPT 产前检测拿了个省级一等奖，美赛对着 KiBaM 双井电池混了个 Honorable Mention。",
            "一头是医学统计，一头是物理仿真——八竿子打不着，但内核一样：把现实揉成一堆方程，再让方程替你做决定。",
        ],
        "sections": [
            {
                "type": "text",
                "tag": "Unfold",
                "paragraphs": [
                    "国赛的问题很实际：NIPT 产前检测对所有孕妇「一刀切」，高 BMI 人群准头和效率都打折。我们想的是——能不能给每个孕妇找个最优的检测时点，把异常判定做成可解释的决策框架？技术路线上用了四件套联动：非线性混合效应模型（个体差异当随机效应，解释约 66% 变异）→ 生存分析递归分割定最优时点 → 多变量逻辑回归揭示异质性 → ML 二分类，优先保召回率防漏诊。最终省级一等奖，条件 R²=0.7888，AIC=-4392.56，召回率 77.78%。",
                ],
            },
            {
                "type": "pdf",
                "src": "/detail/math-modeling/国赛论文.pdf",
                "title": "NIPT · 国赛 · 省级一等奖",
            },
            {
                "type": "text",
                "tag": "Unfold",
                "paragraphs": [
                    "美赛这边，问题是「手机还能撑多久」。传统库仑计数压根讲不清「电量焦虑」——电池的倍率容量效应、恢复现象、温度漂移，它一个都解释不了。我们换了个思路：KiBaM 双井模型把可用电荷和束缚电荷分开管，二阶等效电路描瞬态响应，集总热耦合管温度，SEI 老化管衰减——四个物理模型耦合在一起，再上蒙特卡洛 500 次量化不确定性。最终 Honorable Mention，TTE 预测 RMSE=42.3min，R²>0.85，均值 8.2h。",
                ],
            },
            {
                "type": "pdf",
                "src": "/detail/math-modeling/美赛论文.pdf",
                "title": "KiBaM · 美赛 · Honorable Mention",
            },
            {
                "type": "text",
                "tag": "Unfold",
                "paragraphs": [
                    "两场都是建模核心。国赛扛混合效应 + 生存分析 + 逻辑回归三模型联动和稳健性分析；美赛主攻 KiBaM 双井 + 等效电路 + 热耦合 + 老化多模型耦合，外加蒙特卡洛和敏感性分析。一个临床决策的「宁可错杀」，一个物理建模的「追根溯源」——团队里最较真方程的那个人。",
                ],
            },
        ],
        "next_slug": "grab-car",
        "next_title": "抓取型小车",
    },

    "signal-processing": {
        "title": "信号处理(EEG+rPPG)",
        "subtitle": "rPPG · EEG · SIGNAL PROCESSING",
        "role": "INDEPENDENT RESEARCH",
        "date": "2024",
        "hero": "/detail/signal-processing/独立成分分析-用于封面.png",
        "hero_alt": "独立成分分析脑电信号处理(EEG+rPPG)封面",
        "overview": [
            "两个独立信号处理实验，主题相近：从噪声里提取隐藏的生理信息。一个看脸——rPPG 从视频帧里读心率，不接触；一个看脑——EEG 从多通道电信号里分离脑波与伪迹，FFT + ICA 各显神通。",
        ],
        "sections": [
            {
                "type": "image",
                "src": "/detail/signal-processing/rppg 视频识别心率等生理信号.png",
                "alt": "rPPG 信号与心率检测结果",
            },
            {
                "type": "text",
                "tag": "Unfold",
                "paragraphs": [
                    "测心率通常要指夹或电极——接触式、有门槛。rPPG（远程光电容积描记法）换个路子：对着脸录视频，从皮肤血液容积变化造成的微弱色差里反推生理信号。我从零搭了一个 rPPG 基础框架：MediaPipe Face Mesh 定位额头 ROI → 抠绿色通道平均亮度变化提取原始 BVP → 去趋势 + 带通滤波（0.7–4.0Hz）+ 归一化 → 滑动窗口 FFT 算动态心率 → 峰值检测算 RR 间期，输出时域 HRV（RMSSD/SDNN）。",
                    "在 UBFC-rPPG 数据集 52 秒视频上跑通：FFT 估算心率与 ground truth 趋势高度重合（约 100–110 BPM），庞加莱图显示 RR 间期集中在 800–1000ms。算法库 pyVHR 环境冲突，索性从零手写信号提取与分析全流程——踩坑反而是最扎实的部分。GitHub: xiaoaojianghu6/rppg-estimation。",
                ],
            },
            {
                "type": "image",
                "src": "/detail/signal-processing/脑电信号频率提取.png",
                "alt": "脑电信号频率提取结果",
            },
            {
                "type": "text",
                "tag": "Unfold",
                "paragraphs": [
                    "脑电图（EEG）信号是多通道、低信噪比的时序数据，眼动、肌电等伪迹混杂其中——把真正的神经振荡从噪声堆里挑出来，是信号处理的经典难题。我用了两条路线：FFT 频域分析把信号分解为 Delta/Theta/Alpha/Beta/Gamma 五个脑波频带，量化各频带功率分布；ICA 独立成分分析提取 9 个成分，识别并剔除疑似眼电/肌电伪迹（成分 0、1、8），对比处理前后 Beta 波功率变化——干扰显著下降。基于 MNE-Python + SciPy 实现完整流水线。",
                ],
            },
        ],
        "next_slug": "workflow",
        "next_title": "AI 自动化工作流",
    },

    "text-restructuring": {
        "title": "文本逻辑结构化引擎",
        "subtitle": "NLP · TAURI 2 + REACT 19",
        "role": "SOLO DEVELOPER",
        "date": "2025",
        "hero": "/detail/text-restructuring/mv-pc.jpg",
        "hero_alt": "text-restructuring 主视觉图",
        "overview": [
            "写长文的人都有这种焦虑：结构乱没乱？前面说的话和后面矛盾了吗？宕开一笔的伏笔捡回来没有？市面上的工具要么让你手动搭知识图谱，要么 AI 丢给你一个「摘要」——可我要的不是摘要，是全量保留、逐句理清。",
            "于是我自己造了一个轮子：六套语言学理论撑起一条全自动结构化管线。它不概括、不缩写、不替你写——它只做一件事：读你的全部文档，告诉你哪里有问题。",
        ],
        "sections": [
            {
                "type": "image",
                "src": "/detail/text-restructuring/多项目对比1.png",
                "alt": "(A) RST 修辞关系标注",
            },
            {
                "type": "image",
                "src": "/detail/text-restructuring/多项目对比2.png",
                "alt": "(B) 言语行为分类",
            },
            {
                "type": "image",
                "src": "/detail/text-restructuring/多项目对比3.png",
                "alt": "(C) Dung 论证图",
            },
            {
                "type": "text",
                "tag": "引擎内核",
                "paragraphs": [
                    "三张截图来自同一个测试案例。文件夹里有三个文档——项目预算、团队构成、季度费用报告。它们是一个项目经理随手写的，看起来都没什么问题。但引擎读完三份之后，报告了以下问题：",
                    "文件 1 说「团队总人数控制在 10 人以内」，文件 2 列出了 12 名成员——人数超标 20%。文件 2 说「采用敏捷开发模式，每两周一个迭代」，文件 3 却说「决定采用瀑布模型，需求分析已完成」——两个方法论完全互斥。更隐蔽的是预算矛盾：文件 1 规定每人每天 20 元上限 × 10 人 × 90 天 = 18000 元理论天花板，文件 3 报告实际支出 30000 元——即使按 12 人算也只有 21600 元，仍然严重超支。这些矛盾人类逐篇读很容易忽略，因为它们分散在三份不同文档的不同段落中，没有明显标记。引擎在几秒内全部揪出来了。",
                    "这是怎么做到的？底层是六套语言学与逻辑理论的协同管线。SentenceSplitter 分句 → Tagger 用 Mini-Batch LLM 标注每句的修辞角色和语义范围 → GraphUpdater 以增量方式构建跨文档的语义图谱——7 条纯 Python 规则（零 LLM 调用）负责实体链接、层级归属和关系推断，保证图谱的基础结构干净无幻觉。",
                    "图谱建成后，四组件对抗验证系统接管：",
                    "AdversaryScanner——7 条纯 Python 规则扫描逻辑漏洞，不调用任何 LLM，零幻觉成本。",
                    "SemanticRouter——在 Top-5 语义最相近但来自不同文档的语句对上进行双轨 LLM 语义审计，判断是否存在隐含冲突。",
                    "EntityHedgeScanner——三管齐下：跨距离实体模糊检测（同一概念在不同文档中被弱化为不同表述）、数值冲突检测、以及承诺跟踪（文档 A 做出的承诺是否在文档 B/C 中被确认执行）。",
                    "CrossFileConflictDetector——将候选冲突簇批量送交 LLM 一次性审计，一次 API 调用审完全部可疑簇。上述预算-团队-费用的三项矛盾正是这个组件发现的——它先在纯规则层提取所有数值及其上下文，按语义范围聚类，再送 LLM 做跨文件一致性判断。",
                ],
            },
            {
                "type": "text",
                "tag": "工程落地",
                "paragraphs": [
                    "整个系统用 FCR（Fact Coverage Ratio，事实覆盖率）作为客观评估指标——通过字符串锚点匹配衡量重建后的图谱覆盖了原文中多少事实实体。247 项测试全过。打包为 Tauri 2 桌面应用（.app），前端 React 19，后端 Python 用 PyInstaller 捆绑，支持 NVIDIA NIM / DeepSeek API / Ollama（qwen3.5）三种推理后端切换。Token Bucket 线程安全限速器控制 API 成本。",
                    "最难的不是写代码——是让六套理论不打架、乖乖协同。RST 管修辞结构，言语行为理论分 intent，Dung 论证框架管攻防，艾宾浩斯遗忘曲线做图谱节点衰减，AGM 信念修正处理新旧信息冲突，因果图配合 Allen 时序逻辑建因果链。每一套理论都有自己的学术体系，把它们揉进一条统一的管线，既不能削足适履，也不能各自为政。这是个语言学工程化的项目，不是套壳的 ChatBot。",
                ],
            },
        ],
        "next_slug": "williamnotes",
        "next_title": "WilliamNotes（数字分身决策系统）",
    },

    "williamnotes": {
        "title": "WilliamNotes（数字分身决策系统）",
        "subtitle": "PERSONAL KNOWLEDGE ENGINE",
        "role": "SOLO DEVELOPER",
        "date": "2024至今",
        "hero": "/detail/williamnotes/mv-pc.jpg",
        "hero_alt": "williamnotes 主视觉图",
        "overview": [
            "LLM-wiki 有一个根本缺陷：AI 往 Wiki 里写，人偶尔看一看，但没有反馈闭环。每次新对话，AI 重新犯同样的错——因为它不知道上次你纠正了什么。这不是记忆问题，是架构问题。",
            "WilliamNotes 的起点就是这个洞察：一个会自己运转、会从人的反馈中学习、会用规则约束自己的个人信息中枢——不是另一个 AI 笔记工具，而是一个受控的、可进化的数字分身。",
        ],
        "sections": [
            {
                "type": "text",
                "tag": "规则制衡架构",
                "paragraphs": [
                    "系统的核心不是「让 AI 变聪明」，而是「让 AI 变可靠」。我设计了 7 个互锁的控制文件，每个有不同的职责和权限边界，它们互相检查、互相约束：",
                    "运行法则——系统的宪法。定义了 AI 的 5 级权限矩阵（全自主 / 条件自主 / 人工门禁 / 绝对禁止 / 只读）和全部工作流规范。每一层权限有明确的读写边界：00 收件箱 AI 可自主处理，10 系统内核任何改动必须走缓冲提案经人工审批，40 不动产（日记、经历、素材）的正文 AI 绝对不可触碰。",
                    "数据结构 Schema——用 YAML 定义了 14 种笔记类型，每种类型有强制的 required_yaml 字段和受控标签体系（domains / roles / maturity / meta 四个维度共 28 个值）。AI 不能自由发明分类标签——这防止了 LLM 维护 Wiki 时最常见的「标签通胀」问题。",
                    "AI 行为校准——这是整个系统最关键的差异化组件。人的每一次纠正（通过 / 驳回 / 手动修改 / 原因说明）被 append 进校准文件。当同类偏差积累 ≥2 次，consolidate 机制将它们聚合成一条 active rule，自动注入后续 AI 会话。超过 90 天未被触发，decay 机制将其归档为历史参考，不再主动加载。这是一个带遗忘曲线的持久化反馈闭环——LLM-wiki 永远做不到这一点，因为它的每个对话都是独立上下文。",
                    "价值观（500 字硬上限）和战略规划（800 字硬上限）——最高决策依据。新原则须先完整写入客观知识层，经过实践验证确实影响决策后，才能在压缩旧条目或替换弱条目的前提下浓缩进内核。两级蒸馏机制防止了「每看一篇好文章就改一次人生方向」的知识浮躁。",
                    "文风规范——8 项风格特征 + 17 篇跨体裁 few-shot 样本 + AI-ese 黑名单。每次 AI 输出后做漂移检测：与原始样本比对，丢失频率超阈值的签名表达将触发文风规范的自我修正提案。写作风格维护是一个闭环，不是一张静态清单。",
                ],
            },
            {
                "type": "text",
                "tag": "系统何以有效",
                "paragraphs": [
                    "技术栈基于 Obsidian + Claude AI，Python 脚本做 Apple Notes 同步，自动化三大工作流：Ingest（9 步摄入管线：扫描→分类→多跳检索→价值判断→双向链接→MOC 同步→内核进化检查）、Commit（4 步合并流程：扫描→按 change_type 执行→删除已处理源→追加反馈记录）、Lint（周期性完整性扫描：孤节点→未关联人员→元数据缺失→跨区链接完整性→重构建议）。",
                    "为什么这套规则系统比 LLM-wiki 有效？三件事：",
                    "第一，不可变锚点。40 不动产层的所有原始数据——日记、经历、素材——AI 可以读、可以学、可以提取实体，但绝对不能改。所有衍生知识向下可溯源，向上经蒸馏。这解决了 LLM-wiki 最致命的「改写漂移」——AI 读了一遍你的日记，下次再写就歪了。",
                    "第二，多道防线。hooks（OS 级别 PreToolUse + Stop 拦截）+ 权限矩阵（workflow 级别）+ Schema 验证（数据级别）+ Deny 规则（永不降级，即使开启 bypassPermissions）——四层防火墙让系统失控的概率降到极限。",
                    "第三，共进化。6 个独立反馈循环（内核两级蒸馏 / 人-AI 反馈三原语 / 文风漂移自检 / 交互式提案遗忘检测 / 周期性 Lint / Schema 自治）让系统不是一台装完就老化的机器，而是一个会持续校准自己的有机体。既是开发者也是重度用户——每天靠它活着。",
                ],
            },
        ],
        "next_slug": "math-modeling",
        "next_title": "数学建模",
    },

    "workflow": {
        "title": "AI 自动化工作流",
        "subtitle": "LLM ENGINEERING · DIFY",
        "role": "LLM ENGINEER",
        "date": "2025",
        "hero": "/detail/workflow/mv-pc.jpg",
        "hero_alt": "workflow 主视觉图",
        "overview": [
            "LLM 单独用是玩具，串进工作流才是生产力。这个仓库记的是我拿 Dify + Docker 本地搭 AI Agent 的全套实践——从最朴素的 RAG，一路玩到 Text-to-SQL 和 MCP。",
            "每个应用都本地跑通验证过，不是 PPT 里的概念图。",
        ],
        "sections": [
            {
                "type": "image_panel",
                "images": [
                    {
                        "src": "/detail/workflow/高德MCP.png",
                        "alt": "高德地图 MCP 工具接入示意",
                        "label": "MCP",
                        "caption": "本地 Ollama + Dify 工作流 + 高德 MCP——把 LLM 接进真实世界的工具链",
                    },
                ],
            },
            {
                "type": "text",
                "tag": "Unfold",
                "paragraphs": [
                    "LLM 单独用是玩具，知识陈旧还会「胡说八道」。串进工作流、挂上外部工具和知识库，才变成靠谱的生产力。我用 Dify + Docker 在本地搭了一套 AI Agent 实践场：基础 RAG 配医学指南知识库，知识检索节点加 Prompt 约束专家角色；Text-to-SQL 用自然语言查 PostgreSQL，Antv/Echarts 自动出图；MCP 高德工具调用，再把工作流本身发布为 MCP Server；高阶 RAG 调 PubMed 拉文献。",
                    "全部 Docker 本地部署，一套可复现的 Dify 工作流范式：RAG 让 Agent 回答有据可依，Text-to-SQL 把数据库变成聊天对象，MCP 让工作流反向成为可被调用的工具。边学边搭，每个节点都自己踩过坑验证过，沉淀成可复用的学习笔记。",
                ],
            },
        ],
        "next_slug": "image-segmentation",
        "next_title": "图像/视频分割",
    },

    "grab-car": {
        "title": "抓取型小车",
        "subtitle": "HARDWARE · EMBEDDED SYSTEMS",
        "role": "STUDENT PROJECT",
        "date": "2024",
        "hero": "/detail/grab-car/mv-pc.jpg",
        "hero_alt": "grab-car 主视觉图",
        "overview": [
            "机械与控制的入门课设——从一张白纸到一辆会自己抓东西的小车。这是我第一次把课本上的运动学和电路图，变成能在地上跑、能听我遥控的铁疙瘩。",
            "从 SolidWorks 画第一根轴开始，到 PS2 手柄遥控整车联调结束——走完了一个完整硬件项目的每一道工序。",
        ],
        "sections": [
            {
                "type": "image_panel",
                "images": [
                    {
                        "src": "/detail/grab-car/小车结构设计图.JPG",
                        "alt": "小车结构设计图",
                        "label": "Structure",
                        "caption": "SolidWorks 三维建模——从原理图到装配体的完整机械设计",
                    },
                ],
            },
            {
                "type": "video",
                "src": "/detail/grab-car/小车视频.mp4",
                "caption": "整车联调——PS2 手柄遥控抓取演示",
            },
            {
                "type": "text",
                "tag": "完整实现过程",
                "paragraphs": [
                    "机械设计阶段：在 SolidWorks 中完成整车三维建模——底盘、齿轮传动组、抓取机械臂的完整装配体。所有零部件从图纸开始：齿轮和抓手臂的三维模型导出 STL，用 3D 打印机逐层堆出来；底盘和支撑结构选用亚克力板，在车床上切削加工、钻孔攻丝。第一次体会到「设计-制造-装配」的尺寸链——图纸上差了 0.5mm，实物就是装不上。",
                    "电控系统阶段：Arduino 主板作为控制核心。直流马达通过 L298N 驱动模块做 PWM 调速——占空比映射到车速，写了一套带刹车逻辑的差速转向控制（左 / 右轮独立 PWM 通道，原地转向用差动反转）。舵机控制抓取机械臂的三个自由度——底部旋转、大臂俯仰、爪子开合——每个舵机的角度范围经过实物标定，避开结构限位。",
                    "遥控通信阶段：HC-05 蓝牙模块串口透传，自己写了 PS2 手柄的通信协议解析代码——摇杆模拟量映射到车速 PWM 占空比和舵机角度，按键映射到爪子开合和急停。手柄数据帧解析在 Arduino 端完成，16ms 刷新周期保证控制的实时性。烧录代码到主板，上电自检，整车联调。",
                    "这个项目不追求创新——它追求完整。从机械到电子到软件，从建模到加工到装配到调试，每一步都是实打实的工程训练。SolidWorks / 3D 打印 / 车床加工 / Arduino (C/C++) / PWM 伺服控制 / 蓝牙串口通信 / PS2 协议——这就是一个本科生能做出来的东西。",
                ],
            },
        ],
        "next_slug": "signal-processing",
        "next_title": "信号处理(EEG+rPPG)",
    },

}