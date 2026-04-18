// Node-oriented editable pro deck builder.
// Run this after editing SLIDES, SOURCES, and layout functions.
// The init script installs a sibling node_modules/@oai/artifact-tool package link
// and package.json with type=module for shell-run eval builders. Run with the
// Node executable from Codex workspace dependencies or the platform-appropriate
// command emitted by the init script.
// Do not use pnpm exec from the repo root or any Node binary whose module
// lookup cannot resolve the builder's sibling node_modules/@oai/artifact-tool.

const fs = await import("node:fs/promises");
const path = await import("node:path");
const { Presentation, PresentationFile } = await import("@oai/artifact-tool");

const W = 1280;
const H = 720;

const DECK_ID = "or-tools-project-deck";
const OUT_DIR = "/Users/sandmwong/Desktop/OR_tools/outputs/or_tools_presentation";
const REF_DIR = "/Users/sandmwong/Desktop/OR_tools/outputs/or_tools_presentation/reference-images";
const SCRATCH_DIR = path.resolve(process.env.PPTX_SCRATCH_DIR || path.join("tmp", "slides", DECK_ID));
const PREVIEW_DIR = path.join(SCRATCH_DIR, "preview");
const VERIFICATION_DIR = path.join(SCRATCH_DIR, "verification");
const INSPECT_PATH = path.join(SCRATCH_DIR, "inspect.ndjson");
const MAX_RENDER_VERIFY_LOOPS = 3;

const INK = "#101214";
const GRAPHITE = "#30363A";
const MUTED = "#687076";
const PAPER = "#F7F4ED";
const PAPER_96 = "#F7F4EDF5";
const WHITE = "#FFFFFF";
const ACCENT = "#27C47D";
const ACCENT_DARK = "#116B49";
const GOLD = "#D7A83D";
const CORAL = "#E86F5B";
const TRANSPARENT = "#00000000";

const TITLE_FACE = "Caladea";
const BODY_FACE = "Lato";
const MONO_FACE = "Aptos Mono";

const FALLBACK_PLATE_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=";

const SOURCES = {
  repo: "Repository structure and workflow summary from README.md in /Users/sandmwong/Desktop/OR_tools.",
  model: "Optimization logic, scenarios, and algorithm setup from src/main_optimizer/dark_store_optimizer_with_plans.py.",
  prep: "Data cleaning and parameter construction from analysis/data_prep/dark_store_data_prep.py.",
  main: "Primary rolling-window optimization results from results/main_optimizer/summary.csv.",
  whole: "Whole-order replay results from results/evaluation/whole_order/basket_summary.csv.",
  partial: "Partial-order replay results from results/evaluation/partial_order/basket_summary.csv.",
  benchmark: "Benchmark comparison against hybrid_v2 from results/literature/benchmark_vs_hybrid_v2/benchmark_whole_summary.csv and benchmark_partial_summary.csv.",
  data: "Sample counts derived from data/raw/scanner_data.csv and data/processed/sku_params.csv.",
};

const SLIDES = [
  {
    kicker: "PROJECT PRESENTATION",
    title: "即时零售前置仓 SKU 选品与库存配置联合优化研究",
    subtitle: "基于真实 scanner data 的建模、求解、订单回放评估与管理启示",
    moment: "核心主线：业务痛点 → 文献定位 → 联合优化 → 订单回放 → 管理建议",
    notes: "封面页先说明问题不是单独的补货，也不是单独的选品，而是在容量受限的前置仓中同时决定卖什么和备多少。接下来会沿着背景、文献、模型、实验、结果和管理启示这条主线展开。",
    sources: ["repo", "model"],
  },
  {
    kicker: "BACKGROUND",
    title: "研究背景：前置仓运营面临服务与容量的双重矛盾",
    subtitle: "消费者希望更快送达和更完整供给，但前置仓不可能覆盖全部 SKU 与库存深度。",
    cards: [
      ["业务压力", "即时零售强调分钟级配送、低缺货和足够丰富的商品覆盖，前置仓直接承接用户体验。"],
      ["资源约束", "仓容、重量、陈列位和运营能力都有限，SKU 过多会挤占空间并提高复杂度。"],
      ["决策挑战", "SKU 太少会丢失需求，SKU 太多又会带来滞销和管理成本，因此必须做结构化取舍。"],
    ],
    notes: "这页要把业务矛盾讲清楚。建议用一句话概括：前置仓不是追求商品越多越好，而是在有限资源下权衡选品广度与备货深度。",
    sources: ["repo", "model"],
  },
  {
    kicker: "QUESTION",
    title: "研究问题：前置仓到底该卖什么，以及每个 SKU 备多少",
    subtitle: "本文关注在不确定需求与容量约束下，如何联合决定 SKU 上架与库存配置。",
    cards: [
      ["核心决策", "决定 SKU 是否进入前置仓 assortment，同时决定进入后的备货数量。"],
      ["评价目标", "在满足体积和重量约束的前提下，提高服务水平并兼顾经济性与求解效率。"],
      ["验证方式", "不仅看聚合 line fill rate，还回到真实订单层面做整单与部分履约回放。"],
    ],
    notes: "强调这是一个联合优化问题。可以点出两层验证：一层是模型内部的聚合指标，一层是真实订单 replay 的业务指标。",
    sources: ["model", "whole", "partial"],
  },
  {
    kicker: "SIGNIFICANCE",
    title: "研究意义：把学术问题和业务落地连接起来",
    subtitle: "本文希望建立一个既可建模求解、又能回到履约现实的研究框架。",
    metrics: [
      ["联合优化", "选品与库存同框考虑", "避免将 assortment 和 stocking depth 割裂处理"],
      ["真实数据", "scanner transaction data", "参数估计与实验验证都基于真实订单数据"],
      ["回放检验", "订单级 replay", "检验聚合 fill rate 与真实履约表现之间的偏差"],
    ],
    notes: "这页从理论和实践两个角度讲意义。理论上是联合框架和真实回放；实践上是能给前置仓选品与补货提供量化依据。",
    sources: ["repo", "model", "whole", "partial"],
  },
  {
    kicker: "LITERATURE",
    title: "文献综述：现有研究主要沿三条主线展开",
    subtitle: "选品优化、库存优化和联合优化与替代需求建模构成了本文的文献背景。",
    cards: [
      ["选品研究", "assortment planning 关注利润、覆盖和吸引力，但通常弱化库存深度与履约过程。"],
      ["库存研究", "库存与补货研究强调缺货和持有成本，但常在既定 SKU 集合上展开，较少处理 assortment 决策。"],
      ["联合与替代", "更接近前置仓实践，但往往需要篮子级数据、效用模型或完整替代矩阵，落地门槛较高。"],
    ],
    notes: "不要把这一页讲成论文堆砌。重点是分类并指出各自最相关的切入点，以及与前置仓情境的距离。",
    sources: ["repo", "model", "benchmark"],
  },
  {
    kicker: "CONTRIBUTION",
    title: "文献不足与本文贡献",
    subtitle: "本文把研究空白具体落在联合决策、真实数据实验和订单级评估三个方面。",
    metrics: [
      ["Gap 1", "选品和库存常被拆开", "本文用统一变量体系表示是否上架与备货量"],
      ["Gap 2", "替代与评估门槛较高", "本文加入文献 hybrid baseline 作为比较参考"],
      ["Gap 3", "缺少真实订单层验证", "本文设计整单与部分履约两类订单回放评估"],
    ],
    notes: "这一页是承上启下的关键。把贡献说得具体，不要泛泛地说‘做了模型’。要让听众明确本文增量在哪里。",
    sources: ["model", "whole", "partial", "benchmark"],
  },
  {
    kicker: "FRAMEWORK",
    title: "研究框架：从原始订单到管理启示形成完整闭环",
    subtitle: "项目组织成一条清晰的实验流水线，而不是零散脚本集合。",
    cards: [
      ["数据处理", "原始 scanner data 经过清洗、候选 SKU 筛选和需求矩阵构造，生成建模输入。"],
      ["优化与评估", "主优化器生成 rolling-window 方案，再用真实订单 replay 检验整单与部分履约表现。"],
      ["对比与提炼", "在文献 hybrid_v2 基线下做 benchmark，对结果进行解释并提炼管理启示。"],
    ],
    notes: "这页最好配合口头把代码仓库结构映射成研究流程，帮助听众建立全局认知。",
    sources: ["repo", "prep", "model", "benchmark"],
  },
  {
    kicker: "SETTING",
    title: "业务场景与建模思路",
    subtitle: "通过一个库存量变量同时编码‘是否入选 assortment’和‘备货深度’。",
    cards: [
      ["决策对象", "若 q_i = 0，则 SKU 不进入前置仓；若 q_i > 0，则该 SKU 入仓并占用容量资源。"],
      ["现实要素", "模型同时考虑随机需求、体积与重量约束、持有成本、缺货损失和上架固定成本。"],
      ["优化目标", "在容量受限的前提下，平衡服务水平、库存风险和经营成本，而不是只追求 SKU 数量。"],
    ],
    notes: "强调这里的变量设计很重要，因为它把 assortment 和 inventory sizing 统一到了同一个框架里。",
    sources: ["model"],
  },
  {
    kicker: "MODEL",
    title: "联合优化模型构建",
    subtitle: "目标函数围绕期望净收益展开，约束则体现前置仓资源边界。",
    cards: [
      ["目标函数", "最大化销售收益减去库存持有成本、缺货损失和上架固定成本的期望净收益。"],
      ["核心约束", "总体积不超过 V，总重量不超过 W，同时备货量非负并受上界约束。"],
      ["评价输出", "模型内部输出 line fill rate、平均缺货概率、备货方案和算法运行时间。"],
    ],
    notes: "这一页不用铺满公式，重点是把目标和约束的业务含义讲清楚。需要时可以在口头补充 E[min(D,q)] 的含义。",
    sources: ["model"],
  },
  {
    kicker: "DEMAND",
    title: "需求建模与参数估计",
    subtitle: "先从日销售序列估计需求分布，再构造收益、成本和容量代理参数。",
    metrics: [
      ["500", "候选 SKU", "经过活跃天数和总销量筛选后保留的高频 SKU 数量"],
      ["345", "NegBin SKU", "use_negbin=1，说明过度离散较明显的 SKU 数量"],
      ["0.60", "平均日需求", "sku_params.csv 中 mu_daily 的均值，反映长尾需求特征"],
    ],
    notes: "这里要说明参数不是主观设定，而是来自真实数据估计。若某些成本和体积重量缺少直接观测，则采用价格相关代理值。",
    sources: ["prep", "data"],
  },
  {
    kicker: "METHODS",
    title: "求解方法设计：从快速启发式到精确优化基准",
    subtitle: "本文比较了三类算法，并为每类算法构造 nominal 与 robust 两个版本。",
    cards: [
      ["Greedy", "基于单位资源价值密度的启发式方法，优点是速度极快、实现简单、可解释性强。"],
      ["DP-Lagrangian", "通过拉格朗日松弛和动态规划取得效果与效率之间的折中。"],
      ["CP-SAT", "利用 OR-Tools 的整数规划求解能力作为精确优化基准，并和 robust 版本一起比较。"],
    ],
    notes: "不要只说算法名称，要明确三类算法各自代表怎样的求解策略，以及为什么三者都值得比较。",
    sources: ["model"],
  },
  {
    kicker: "EXPERIMENT",
    title: "滚动窗口实验设计",
    subtitle: "rolling-window 设计用于避免结果依赖某一段特定时间区间。",
    cards: [
      ["时间结构", "每次用前 8 周作为训练窗口，再用后 1 周作为测试窗口，并按周向前滚动。"],
      ["场景设计", "baseline、high_volatility 与 large_scale 三类场景分别对应标准、高波动和大规模紧容量环境。"],
      ["比较目标", "对比不同算法在多个窗口上的服务水平、缺货风险和运行时间，而不是只看单次结果。"],
    ],
    notes: "这页建议配合口头强调 rolling-window 的稳健性，因为单一 train-test 切分容易受偶然时期影响。",
    sources: ["model", "main"],
  },
  {
    kicker: "DATA",
    title: "数据来源与清洗规则",
    subtitle: "原始数据来自真实 scanner transaction records，并经过多步清洗后进入建模流程。",
    cards: [
      ["原始字段", "日期、订单号、SKU、销量、销售额和品类构成了研究的核心交易信息。"],
      ["清洗步骤", "去除非整数销量、异常大单和异常单价记录，尽量保证订单和 SKU 级统计的可靠性。"],
      ["样本筛选", "以活跃天数不少于 30 天、总销量不少于 50 件筛出稳定候选 SKU，再构造日需求矩阵。"],
    ],
    notes: "这里可以强调数据清洗不是附属工作，而是决定参数估计和后续结果可信度的关键环节。",
    sources: ["prep"],
  },
  {
    kicker: "DATA FACTS",
    title: "描述性统计：需求异质性和订单结构特征都很明显",
    subtitle: "数据本身已经说明，仅靠简单均值补货或单一 fill rate 指标并不足够。",
    metrics: [
      ["131,706", "原始交易行", "scanner_data.csv 中的记录数量"],
      ["64,682", "订单数", "原始订单号唯一值数量，覆盖完整年度交易"],
      ["2016", "全年样本", "2016-01-02 至 2016-12-31，共 365 天需求轨迹"],
    ],
    notes: "讲这页时把样本规模和全年覆盖范围点出来，再说明 SKU 需求长尾和多 SKU 篮子的存在，支撑后续分布建模和 replay 设计。",
    sources: ["data", "prep"],
  },
  {
    kicker: "RESULTS",
    title: "主结果：多场景下算法表现存在明显差异",
    subtitle: "从 rolling-window 汇总结果看，DP-Lagrangian 通常带来更高 LFR，而 Greedy 速度最快。",
    metrics: [
      ["0.204", "最佳 baseline LFR", "baseline 场景下 DP-Lagrangian 的平均 lfr_mean"],
      ["17.136s", "DP 平均耗时", "baseline 场景下 DP-Lagrangian 的 time_mean"],
      ["0.0004s", "Greedy 平均耗时", "baseline 场景下 Greedy 的 time_mean，体现极高速度优势"],
    ],
    notes: "这里先讲结论再讲比较逻辑。DP 在 baseline 和 high volatility 场景下的 LFR 都更高，但 Greedy 的运行速度几乎是瞬时，这对部署很重要。",
    sources: ["main"],
  },
  {
    kicker: "REPLAY",
    title: "订单回放评估：聚合指标与真实履约体验并不相同",
    subtitle: "将优化方案放回真实订单序列后，可以看到整单逻辑比聚合 line fill rate 更严苛。",
    cards: [
      ["整单回放", "baseline 下表现最好的 dp_nominal，basket_lfr_mean 仅为 0.0416，basket_ofr_mean 为 0.0754。"],
      ["部分履约", "同一方案在 partial replay 下的 basket_lfr_mean 上升到 0.1211，说明履约规则对指标有显著影响。"],
      ["核心含义", "若只看聚合 line fill rate，可能高估真实订单层面的履约能力，尤其在多 SKU 订单环境下更明显。"],
    ],
    notes: "这页是项目亮点之一。要把 aggregated LFR 与 basket replay 指标的差异解释成管理上更贴近顾客体验的评估要求。",
    sources: ["whole", "partial"],
  },
  {
    kicker: "BENCHMARK",
    title: "文献基线比较：hybrid_v2 在共同窗口上提供了重要参照",
    subtitle: "benchmark 比较说明联合优化框架与替代感知基线在不同评估口径下各有侧重。",
    cards: [
      ["whole-order", "baseline benchmark 中，hybrid_v2 的 basket_lfr_mean 为 0.0632，basket_ofr_mean 为 0.1071，高于主模型方案。"],
      ["partial-order", "在 partial benchmark 中，hybrid_v2 的 basket_lfr_mean 为 0.1244，仍略高于 baseline 下的主模型 replay 表现。"],
      ["解释角度", "hybrid_v2 显式引入替代感知 sizing，提示前置仓问题中需求替代结构值得进一步深入建模。"],
    ],
    notes: "这一页不要把 benchmark 讲成输赢，而是把它作为理解模型边界和替代效应价值的参照系。",
    sources: ["benchmark"],
  },
  {
    kicker: "DISCUSSION",
    title: "结果讨论：几项关键发现如何理解",
    subtitle: "结果不是简单地说明某个算法最好，而是揭示了不同指标和场景下的权衡关系。",
    metrics: [
      ["发现 1", "联合优化有效", "相比直觉式分开决策，统一建模能系统处理容量与成本约束"],
      ["发现 2", "鲁棒性有场景价值", "高波动环境下 robust 思路更强调防错，而不是盲目追求均值最优"],
      ["发现 3", "回放不可省略", "真实订单级 replay 揭示了聚合指标无法直接替代的履约差异"],
    ],
    notes: "把前面几页结果收束成三四条规律。这里的重点是解释机制和权衡，而不是重复数值。",
    sources: ["main", "whole", "partial", "benchmark"],
  },
  {
    kicker: "IMPLICATIONS",
    title: "管理启示：前置仓优化不能只看 SKU 数量或平均需求",
    subtitle: "研究结果可以转化为选品、备货和评估机制上的操作建议。",
    cards: [
      ["结构优化", "前置仓不应追求 SKU 越多越好，而应重视容量约束下的 assortment 结构与库存深度配置。"],
      ["稳健备货", "高波动 SKU 不宜仅按均值需求配置库存，应考虑更稳健的安全偏移或鲁棒策略。"],
      ["评估升级", "企业评估前置仓方案时不能只看 line fill rate，还应纳入整单满足率和订单 replay。"],
    ],
    notes: "这一页要尽量用业务语言。可以把它讲成三条管理动作：优化结构、提高稳健性、升级评估指标。",
    sources: ["main", "whole", "partial"],
  },
  {
    kicker: "CLOSING",
    title: "研究结论、局限与未来工作",
    subtitle: "本文建立了一个可解释、可运行、可回到订单现实检验的前置仓联合优化框架。",
    cards: [
      ["研究结论", "项目完成了从数据处理、联合建模、算法比较到订单回放和 benchmark 的完整闭环。"],
      ["研究局限", "替代效应仍采用近似构造，部分成本和容量参数依赖代理值，外生变量尚未纳入。"],
      ["未来工作", "可进一步扩展到动态补货、多仓协同、显式消费者替代模型和在线学习决策。"],
    ],
    notes: "最后一页先稳稳总结你完成了什么，再主动承认局限，最后给出清晰的扩展方向，形成完整收束。",
    sources: ["repo", "model", "benchmark"],
  },
];

const inspectRecords = [];

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  if (!bytes.byteLength) {
    throw new Error(`Image file is empty: ${imagePath}`);
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function normalizeImageConfig(config) {
  if (!config.path) {
    return config;
  }
  const { path: imagePath, ...rest } = config;
  return {
    ...rest,
    blob: await readImageBlob(imagePath),
  };
}

async function ensureDirs() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const obsoleteFinalArtifacts = [
    "preview",
    "verification",
    "inspect.ndjson",
    ["presentation", "proto.json"].join("_"),
    ["quality", "report.json"].join("_"),
  ];
  for (const obsolete of obsoleteFinalArtifacts) {
    await fs.rm(path.join(OUT_DIR, obsolete), { recursive: true, force: true });
  }
  await fs.mkdir(SCRATCH_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(VERIFICATION_DIR, { recursive: true });
}

function lineConfig(fill = TRANSPARENT, width = 0) {
  return { style: "solid", fill, width };
}

function recordShape(slideNo, shape, role, shapeType, x, y, w, h) {
  if (!slideNo) return;
  inspectRecords.push({
    kind: "shape",
    slide: slideNo,
    id: shape?.id || `slide-${slideNo}-${role}-${inspectRecords.length + 1}`,
    role,
    shapeType,
    bbox: [x, y, w, h],
  });
}

function addShape(slide, geometry, x, y, w, h, fill = TRANSPARENT, line = TRANSPARENT, lineWidth = 0, meta = {}) {
  const shape = slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: lineConfig(line, lineWidth),
  });
  recordShape(meta.slideNo, shape, meta.role || geometry, geometry, x, y, w, h);
  return shape;
}

function normalizeText(text) {
  if (Array.isArray(text)) {
    return text.map((item) => String(item ?? "")).join("\n");
  }
  return String(text ?? "");
}

function textLineCount(text) {
  const value = normalizeText(text);
  if (!value.trim()) {
    return 0;
  }
  return Math.max(1, value.split(/\n/).length);
}

function requiredTextHeight(text, fontSize, lineHeight = 1.18, minHeight = 8) {
  const lines = textLineCount(text);
  if (lines === 0) {
    return minHeight;
  }
  return Math.max(minHeight, lines * fontSize * lineHeight);
}

function assertTextFits(text, boxHeight, fontSize, role = "text") {
  const required = requiredTextHeight(text, fontSize);
  const tolerance = Math.max(2, fontSize * 0.08);
  if (normalizeText(text).trim() && boxHeight + tolerance < required) {
    throw new Error(
      `${role} text box is too short: height=${boxHeight.toFixed(1)}, required>=${required.toFixed(1)}, ` +
        `lines=${textLineCount(text)}, fontSize=${fontSize}, text=${JSON.stringify(normalizeText(text).slice(0, 90))}`,
    );
  }
}

function wrapText(text, widthChars) {
  const words = normalizeText(text).split(/\s+/).filter(Boolean);
  const lines = [];
  let current = "";
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length > widthChars && current) {
      lines.push(current);
      current = word;
    } else {
      current = next;
    }
  }
  if (current) {
    lines.push(current);
  }
  return lines.join("\n");
}

function recordText(slideNo, shape, role, text, x, y, w, h) {
  const value = normalizeText(text);
  inspectRecords.push({
    kind: "textbox",
    slide: slideNo,
    id: shape?.id || `slide-${slideNo}-${role}-${inspectRecords.length + 1}`,
    role,
    text: value,
    textPreview: value.replace(/\n/g, " | ").slice(0, 180),
    textChars: value.length,
    textLines: textLineCount(value),
    bbox: [x, y, w, h],
  });
}

function recordImage(slideNo, image, role, imagePath, x, y, w, h) {
  inspectRecords.push({
    kind: "image",
    slide: slideNo,
    id: image?.id || `slide-${slideNo}-${role}-${inspectRecords.length + 1}`,
    role,
    path: imagePath,
    bbox: [x, y, w, h],
  });
}

function applyTextStyle(box, text, size, color, bold, face, align, valign, autoFit, listStyle) {
  box.text = text;
  box.text.fontSize = size;
  box.text.color = color;
  box.text.bold = Boolean(bold);
  box.text.alignment = align;
  box.text.verticalAlignment = valign;
  box.text.typeface = face;
  box.text.insets = { left: 0, right: 0, top: 0, bottom: 0 };
  if (autoFit) {
    box.text.autoFit = autoFit;
  }
  if (listStyle) {
    box.text.style = "list";
  }
}

function addText(
  slide,
  slideNo,
  text,
  x,
  y,
  w,
  h,
  {
    size = 22,
    color = INK,
    bold = false,
    face = BODY_FACE,
    align = "left",
    valign = "top",
    fill = TRANSPARENT,
    line = TRANSPARENT,
    lineWidth = 0,
    autoFit = null,
    listStyle = false,
    checkFit = true,
    role = "text",
  } = {},
) {
  if (!checkFit && textLineCount(text) > 1) {
    throw new Error("checkFit=false is only allowed for single-line headers, footers, and captions.");
  }
  if (checkFit) {
    assertTextFits(text, h, size, role);
  }
  const box = addShape(slide, "rect", x, y, w, h, fill, line, lineWidth);
  applyTextStyle(box, text, size, color, bold, face, align, valign, autoFit, listStyle);
  recordText(slideNo, box, role, text, x, y, w, h);
  return box;
}

async function addImage(slide, slideNo, config, position, role, sourcePath = null) {
  const image = slide.images.add(await normalizeImageConfig(config));
  image.position = position;
  recordImage(slideNo, image, role, sourcePath || config.path || config.uri || "inline-data-url", position.left, position.top, position.width, position.height);
  return image;
}

async function addPlate(slide, slideNo, opacityPanel = false) {
  slide.background.fill = PAPER;
  const platePath = path.join(REF_DIR, `slide-${String(slideNo).padStart(2, "0")}.png`);
  if (await pathExists(platePath)) {
    await addImage(
      slide,
      slideNo,
      { path: platePath, fit: "cover", alt: `Text-free art-direction plate for slide ${slideNo}` },
      { left: 0, top: 0, width: W, height: H },
      "art plate",
      platePath,
    );
  } else {
    await addImage(
      slide,
      slideNo,
      { dataUrl: FALLBACK_PLATE_DATA_URL, fit: "cover", alt: `Fallback blank art plate for slide ${slideNo}` },
      { left: 0, top: 0, width: W, height: H },
      "fallback art plate",
      "fallback-data-url",
    );
  }
  if (opacityPanel) {
    addShape(slide, "rect", 0, 0, W, H, "#FFFFFFB8", TRANSPARENT, 0, { slideNo, role: "plate readability overlay" });
  }
}

function addHeader(slide, slideNo, kicker, idx, total) {
  addText(slide, slideNo, String(kicker || "").toUpperCase(), 64, 34, 430, 24, {
    size: 13,
    color: ACCENT_DARK,
    bold: true,
    face: MONO_FACE,
    checkFit: false,
    role: "header",
  });
  addText(slide, slideNo, `${String(idx).padStart(2, "0")} / ${String(total).padStart(2, "0")}`, 1114, 34, 104, 24, {
    size: 13,
    color: ACCENT_DARK,
    bold: true,
    face: MONO_FACE,
    align: "right",
    checkFit: false,
    role: "header",
  });
  addShape(slide, "rect", 64, 64, 1152, 2, INK, TRANSPARENT, 0, { slideNo, role: "header rule" });
  addShape(slide, "ellipse", 57, 57, 16, 16, ACCENT, INK, 2, { slideNo, role: "header marker" });
}

function addTitleBlock(slide, slideNo, title, subtitle = null, x = 64, y = 86, w = 780, dark = false) {
  const titleColor = dark ? PAPER : INK;
  const bodyColor = dark ? PAPER : GRAPHITE;
  addText(slide, slideNo, title, x, y, w, 142, {
    size: 40,
    color: titleColor,
    bold: true,
    face: TITLE_FACE,
    role: "title",
  });
  if (subtitle) {
    addText(slide, slideNo, subtitle, x + 2, y + 148, Math.min(w, 720), 70, {
      size: 19,
      color: bodyColor,
      face: BODY_FACE,
      role: "subtitle",
    });
  }
}

function addIconBadge(slide, slideNo, x, y, accent = ACCENT, kind = "signal") {
  addShape(slide, "ellipse", x, y, 54, 54, PAPER_96, INK, 1.2, { slideNo, role: "icon badge" });
  if (kind === "flow") {
    addShape(slide, "ellipse", x + 13, y + 18, 10, 10, accent, INK, 1, { slideNo, role: "icon glyph" });
    addShape(slide, "ellipse", x + 31, y + 27, 10, 10, accent, INK, 1, { slideNo, role: "icon glyph" });
    addShape(slide, "rect", x + 22, y + 25, 19, 3, INK, TRANSPARENT, 0, { slideNo, role: "icon glyph" });
  } else if (kind === "layers") {
    addShape(slide, "roundRect", x + 13, y + 15, 26, 13, accent, INK, 1, { slideNo, role: "icon glyph" });
    addShape(slide, "roundRect", x + 18, y + 24, 26, 13, GOLD, INK, 1, { slideNo, role: "icon glyph" });
    addShape(slide, "roundRect", x + 23, y + 33, 20, 10, CORAL, INK, 1, { slideNo, role: "icon glyph" });
  } else {
    addShape(slide, "rect", x + 16, y + 29, 6, 12, accent, TRANSPARENT, 0, { slideNo, role: "icon glyph" });
    addShape(slide, "rect", x + 25, y + 21, 6, 20, accent, TRANSPARENT, 0, { slideNo, role: "icon glyph" });
    addShape(slide, "rect", x + 34, y + 14, 6, 27, accent, TRANSPARENT, 0, { slideNo, role: "icon glyph" });
  }
}

function addCard(slide, slideNo, x, y, w, h, label, body, { accent = ACCENT, fill = PAPER_96, line = INK, iconKind = "signal" } = {}) {
  if (h < 156) {
    throw new Error(`Card is too short for editable pro-deck copy: height=${h.toFixed(1)}, minimum=156.`);
  }
  addShape(slide, "roundRect", x, y, w, h, fill, line, 1.2, { slideNo, role: `card panel: ${label}` });
  addShape(slide, "rect", x, y, 8, h, accent, TRANSPARENT, 0, { slideNo, role: `card accent: ${label}` });
  addIconBadge(slide, slideNo, x + 22, y + 24, accent, iconKind);
  addText(slide, slideNo, label, x + 88, y + 22, w - 108, 28, {
    size: 15,
    color: ACCENT_DARK,
    bold: true,
    face: MONO_FACE,
    role: "card label",
  });
  const wrapped = wrapText(body, Math.max(28, Math.floor(w / 13)));
  const bodyY = y + 86;
  const bodyH = h - (bodyY - y) - 22;
  if (bodyH < 54) {
    throw new Error(`Card body area is too short: height=${bodyH.toFixed(1)}, cardHeight=${h.toFixed(1)}, label=${JSON.stringify(label)}.`);
  }
  addText(slide, slideNo, wrapped, x + 24, bodyY, w - 48, bodyH, {
    size: 14,
    color: INK,
    face: BODY_FACE,
    role: `card body: ${label}`,
  });
}

function addMetricCard(slide, slideNo, x, y, w, h, metric, label, note = null, accent = ACCENT) {
  if (h < 132) {
    throw new Error(`Metric card is too short for editable pro-deck copy: height=${h.toFixed(1)}, minimum=132.`);
  }
  addShape(slide, "roundRect", x, y, w, h, PAPER_96, INK, 1.2, { slideNo, role: `metric panel: ${label}` });
  addShape(slide, "rect", x, y, w, 7, accent, TRANSPARENT, 0, { slideNo, role: `metric accent: ${label}` });
  addText(slide, slideNo, metric, x + 22, y + 24, w - 44, 54, {
    size: 34,
    color: INK,
    bold: true,
    face: TITLE_FACE,
    role: "metric value",
  });
  addText(slide, slideNo, label, x + 24, y + 82, w - 48, 38, {
    size: 16,
    color: GRAPHITE,
    face: BODY_FACE,
    role: "metric label",
  });
  if (note) {
    addText(slide, slideNo, note, x + 24, y + h - 42, w - 48, 22, {
      size: 10,
      color: MUTED,
      face: BODY_FACE,
      role: "metric note",
    });
  }
}

function addNotes(slide, body, sourceKeys) {
  const sourceLines = (sourceKeys || []).map((key) => `- ${SOURCES[key] || key}`).join("\n");
  slide.speakerNotes.setText(`${body || ""}\n\n[Sources]\n${sourceLines}`);
}

function addReferenceCaption(slide, slideNo) {
  return null;
}

async function slideCover(presentation) {
  const slideNo = 1;
  const data = SLIDES[0];
  const slide = presentation.slides.add();
  await addPlate(slide, slideNo);
  addShape(slide, "rect", 0, 0, W, H, "#FFFFFFCC", TRANSPARENT, 0, { slideNo, role: "cover contrast overlay" });
  addShape(slide, "rect", 64, 86, 7, 455, ACCENT, TRANSPARENT, 0, { slideNo, role: "cover accent rule" });
  addText(slide, slideNo, data.kicker, 86, 88, 520, 26, {
    size: 13,
    color: ACCENT_DARK,
    bold: true,
    face: MONO_FACE,
    role: "kicker",
  });
  addText(slide, slideNo, data.title, 82, 130, 785, 184, {
    size: 48,
    color: INK,
    bold: true,
    face: TITLE_FACE,
    role: "cover title",
  });
  addText(slide, slideNo, data.subtitle, 86, 326, 610, 86, {
    size: 20,
    color: GRAPHITE,
    face: BODY_FACE,
    role: "cover subtitle",
  });
  addShape(slide, "roundRect", 86, 456, 390, 92, PAPER_96, INK, 1.2, { slideNo, role: "cover moment panel" });
  addText(slide, slideNo, data.moment || "Replace with core idea", 112, 478, 336, 40, {
    size: 23,
    color: INK,
    bold: true,
    face: TITLE_FACE,
    role: "cover moment",
  });
  addReferenceCaption(slide, slideNo);
  addNotes(slide, data.notes, data.sources);
}

async function slideCards(presentation, idx) {
  const data = SLIDES[idx - 1];
  const slide = presentation.slides.add();
  await addPlate(slide, idx);
  addShape(slide, "rect", 0, 0, W, H, "#FFFFFFB8", TRANSPARENT, 0, { slideNo: idx, role: "content contrast overlay" });
  addHeader(slide, idx, data.kicker, idx, SLIDES.length);
  addTitleBlock(slide, idx, data.title, data.subtitle, 64, 86, 760);
  const cards = data.cards?.length
    ? data.cards
    : [
        ["Replace", "Add a specific, sourced point for this slide."],
        ["Author", "Use native PowerPoint chart objects for charts; use deterministic geometry for cards and callouts."],
        ["Verify", "Render previews, inspect them at readable size, and fix actionable layout issues within 3 total render loops."],
      ];
  const cols = Math.min(3, cards.length);
  const cardW = (1114 - (cols - 1) * 24) / cols;
  const iconKinds = ["signal", "flow", "layers"];
  for (let cardIdx = 0; cardIdx < cols; cardIdx += 1) {
    const [label, body] = cards[cardIdx];
    const x = 84 + cardIdx * (cardW + 24);
    addCard(slide, idx, x, 410, cardW, 194, label, body, { iconKind: iconKinds[cardIdx % iconKinds.length] });
  }
  addReferenceCaption(slide, idx);
  addNotes(slide, data.notes, data.sources);
}

async function slideMetrics(presentation, idx) {
  const data = SLIDES[idx - 1];
  const slide = presentation.slides.add();
  await addPlate(slide, idx);
  addShape(slide, "rect", 0, 0, W, H, "#FFFFFFBD", TRANSPARENT, 0, { slideNo: idx, role: "metrics contrast overlay" });
  addHeader(slide, idx, data.kicker, idx, SLIDES.length);
  addTitleBlock(slide, idx, data.title, data.subtitle, 64, 86, 700);
  const metrics = data.metrics || [
    ["00", "Replace metric", "Source"],
    ["00", "Replace metric", "Source"],
    ["00", "Replace metric", "Source"],
  ];
  const accents = [ACCENT, GOLD, CORAL];
  for (let metricIdx = 0; metricIdx < Math.min(3, metrics.length); metricIdx += 1) {
    const [metric, label, note] = metrics[metricIdx];
    addMetricCard(slide, idx, 92 + metricIdx * 370, 404, 330, 174, metric, label, note, accents[metricIdx % accents.length]);
  }
  addReferenceCaption(slide, idx);
  addNotes(slide, data.notes, data.sources);
}

async function createDeck() {
  await ensureDirs();
  if (!SLIDES.length) {
    throw new Error("SLIDES must contain at least one slide.");
  }
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  await slideCover(presentation);
  for (let idx = 2; idx <= SLIDES.length; idx += 1) {
    const data = SLIDES[idx - 1];
    if (data.metrics) {
      await slideMetrics(presentation, idx);
    } else {
      await slideCards(presentation, idx);
    }
  }
  return presentation;
}

async function saveBlobToFile(blob, filePath) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await fs.writeFile(filePath, bytes);
}

async function writeInspectArtifact(presentation) {
  inspectRecords.unshift({
    kind: "deck",
    id: DECK_ID,
    slideCount: presentation.slides.count,
    slideSize: { width: W, height: H },
  });
  presentation.slides.items.forEach((slide, index) => {
    inspectRecords.splice(index + 1, 0, {
      kind: "slide",
      slide: index + 1,
      id: slide?.id || `slide-${index + 1}`,
    });
  });
  const lines = inspectRecords.map((record) => JSON.stringify(record)).join("\n") + "\n";
  await fs.writeFile(INSPECT_PATH, lines, "utf8");
}

async function currentRenderLoopCount() {
  const logPath = path.join(VERIFICATION_DIR, "render_verify_loops.ndjson");
  if (!(await pathExists(logPath))) return 0;
  const previous = await fs.readFile(logPath, "utf8");
  return previous.split(/\r?\n/).filter((line) => line.trim()).length;
}

async function nextRenderLoopNumber() {
  return (await currentRenderLoopCount()) + 1;
}

async function appendRenderVerifyLoop(presentation, previewPaths, pptxPath) {
  const logPath = path.join(VERIFICATION_DIR, "render_verify_loops.ndjson");
  const priorCount = await currentRenderLoopCount();
  const record = {
    kind: "render_verify_loop",
    deckId: DECK_ID,
    loop: priorCount + 1,
    maxLoops: MAX_RENDER_VERIFY_LOOPS,
    capReached: priorCount + 1 >= MAX_RENDER_VERIFY_LOOPS,
    timestamp: new Date().toISOString(),
    slideCount: presentation.slides.count,
    previewCount: previewPaths.length,
    previewDir: PREVIEW_DIR,
    inspectPath: INSPECT_PATH,
    pptxPath,
  };
  await fs.appendFile(logPath, JSON.stringify(record) + "\n", "utf8");
  return record;
}

async function verifyAndExport(presentation) {
  await ensureDirs();
  const nextLoop = await nextRenderLoopNumber();
  if (nextLoop > MAX_RENDER_VERIFY_LOOPS) {
    throw new Error(
      `Render/verify/fix loop cap reached: ${MAX_RENDER_VERIFY_LOOPS} total renders are allowed. ` +
        "Do not rerender; note any remaining visual issues in the final response.",
    );
  }
  await writeInspectArtifact(presentation);
  const previewPaths = [];
  for (let idx = 0; idx < presentation.slides.items.length; idx += 1) {
    const slide = presentation.slides.items[idx];
    const preview = await presentation.export({ slide, format: "png", scale: 1 });
    const previewPath = path.join(PREVIEW_DIR, `slide-${String(idx + 1).padStart(2, "0")}.png`);
    await saveBlobToFile(preview, previewPath);
    previewPaths.push(previewPath);
  }
  const pptxBlob = await PresentationFile.exportPptx(presentation);
  const pptxPath = path.join(OUT_DIR, "output.pptx");
  await pptxBlob.save(pptxPath);
  const loopRecord = await appendRenderVerifyLoop(presentation, previewPaths, pptxPath);
  return { pptxPath, loopRecord };
}

const presentation = await createDeck();
const result = await verifyAndExport(presentation);
console.log(result.pptxPath);
