"""
前置仓SKU选品与库存优化 —— 数据清洗与需求参数估计
========================================================
输入: scanner_data.csv
输出:
  - sku_params.csv          : 每个SKU的需求参数与成本代理值（送入优化模型）
  - daily_demand.csv        : 每日SKU需求矩阵（用于SAA）
  - weekly_demand.csv       : 每周SKU需求矩阵（备用粒度）
  - cleaning_report.txt     : 数据清洗过程记录
  - figures/                : 探索性分析图表
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import warnings, os
warnings.filterwarnings('ignore')

ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
RAW_SCANNER_FILE = ROOT_DIR / "data" / "raw" / "scanner_data.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "analysis" / "data_prep"
FIGURES_DIR = REPORT_DIR / "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

# ─────────────────────────────────────────
# 1. 读取原始数据
# ─────────────────────────────────────────
log("=" * 60)
log("STEP 1: 读取原始数据")
log("=" * 60)

df_raw = pd.read_csv(RAW_SCANNER_FILE)
df_raw['Date'] = pd.to_datetime(df_raw['Date'], dayfirst=True)
df_raw['unit_price'] = df_raw['Sales_Amount'] / df_raw['Quantity']

log(f"原始行数: {len(df_raw):,}")
log(f"时间范围: {df_raw['Date'].min().date()} ~ {df_raw['Date'].max().date()}")
log(f"原始SKU数: {df_raw['SKU'].nunique():,}")
log(f"原始品类数: {df_raw['SKU_Category'].nunique():,}")
log(f"原始订单数: {df_raw['Transaction_ID'].nunique():,}")

# ─────────────────────────────────────────
# 2. 数据清洗
# ─────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 2: 数据清洗")
log("=" * 60)

df = df_raw.copy()
n0 = len(df)

# 2a. 去除非整数销售量（称重商品）
mask_int = df['Quantity'] == df['Quantity'].round()
n_nonint = (~mask_int).sum()
df = df[mask_int].copy()
df['Quantity'] = df['Quantity'].astype(int)
log(f"[清洗] 去除非整数销售量: -{n_nonint} 行 (占比 {n_nonint/n0:.2%})")

# 2b. 去除异常大单（99.9分位数以上，视为批发或录入错误）
q999 = df['Quantity'].quantile(0.999)
mask_outlier = df['Quantity'] <= q999
n_outlier = (~mask_outlier).sum()
df = df[mask_outlier].copy()
log(f"[清洗] 去除大单异常值 (>{q999:.0f}件): -{n_outlier} 行")

# 2c. 去除单价异常（单价为0或极大值，视为数据错误）
mask_price = (df['unit_price'] > 0) & (df['unit_price'] < df['unit_price'].quantile(0.999))
n_price = (~mask_price).sum()
df = df[mask_price].copy()
log(f"[清洗] 去除单价异常: -{n_price} 行")

log(f"\n清洗后行数: {len(df):,} (保留 {len(df)/n0:.2%})")
log(f"清洗后SKU数: {df['SKU'].nunique():,}")
log(f"清洗后订单数: {df['Transaction_ID'].nunique():,}")

# ─────────────────────────────────────────
# 3. 筛选高频SKU（前置仓候选集）
# ─────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 3: 筛选候选SKU")
log("=" * 60)

# 统计每个SKU的出现天数与总销量
sku_stats = df.groupby('SKU').agg(
    total_qty   = ('Quantity', 'sum'),
    active_days = ('Date', 'nunique'),
    avg_price   = ('unit_price', 'median'),
    category    = ('SKU_Category', 'first')
).reset_index()

# 筛选条件：
#   - 出现天数 >= 30天（避免季节性冷门SKU主导模型）
#   - 总销量 >= 50件（确保需求参数估计有足够样本）
mask_sku = (sku_stats['active_days'] >= 30) & (sku_stats['total_qty'] >= 50)
sku_selected = sku_stats[mask_sku].copy()

log(f"筛选条件: 出现天数>=30 且 总销量>=50")
log(f"筛选前SKU数: {len(sku_stats):,}")
log(f"筛选后SKU数: {len(sku_selected):,}")
log(f"覆盖销量占比: {sku_selected['total_qty'].sum() / sku_stats['total_qty'].sum():.2%}")

# 取高频Top 500（若超过500则按总销量排序截取，模拟前置仓容量有限的选品逻辑）
N_TARGET = 500
if len(sku_selected) > N_TARGET:
    sku_selected = sku_selected.nlargest(N_TARGET, 'total_qty')
    log(f"进一步截取Top {N_TARGET} 高销量SKU")

selected_skus = set(sku_selected['SKU'])
df = df[df['SKU'].isin(selected_skus)].copy()
log(f"\n最终候选SKU数: {len(selected_skus)}")

# ─────────────────────────────────────────
# 4. 构建每日需求矩阵
# ─────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 4: 构建需求矩阵")
log("=" * 60)

# 按日汇总每个SKU的销量（未售出的天记为0）
daily_agg = df.groupby(['Date', 'SKU'])['Quantity'].sum().reset_index()
daily_agg.columns = ['Date', 'SKU', 'demand']

# 构建完整日期×SKU矩阵（补0）
all_dates = pd.date_range(df['Date'].min(), df['Date'].max(), freq='D')
all_skus  = list(selected_skus)
idx = pd.MultiIndex.from_product([all_dates, all_skus], names=['Date', 'SKU'])
daily_matrix = daily_agg.set_index(['Date', 'SKU']).reindex(idx, fill_value=0).reset_index()

# 透视为宽表（行=日期，列=SKU）
daily_wide = daily_matrix.pivot(index='Date', columns='SKU', values='demand').fillna(0)
log(f"每日需求矩阵: {daily_wide.shape[0]} 天 × {daily_wide.shape[1]} SKU")

# 同样构建每周需求矩阵
weekly_wide = daily_wide.resample('W').sum()
log(f"每周需求矩阵: {weekly_wide.shape[0]} 周 × {weekly_wide.shape[1]} SKU")

# ─────────────────────────────────────────
# 5. 需求分布参数估计（Poisson & 负二项）
# ─────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 5: 需求分布参数估计")
log("=" * 60)

def fit_negbin(data):
    """用矩估计法拟合负二项分布参数 (mu, r)"""
    mu  = data.mean()
    var = data.var()
    if var <= mu or mu == 0:
        # 方差<=均值时退化为Poisson
        return mu, np.inf
    r = mu**2 / (var - mu)   # 离散度参数，越小波动越大
    return mu, r

def poisson_gof(data, mu):
    """Poisson拟合优度（卡方检验p值）"""
    if mu == 0:
        return 0.0
    obs_counts = np.bincount(data.astype(int))
    max_k = min(len(obs_counts)-1, 20)
    obs = obs_counts[:max_k+1]
    exp = np.array([stats.poisson.pmf(k, mu) * len(data) for k in range(max_k+1)])
    exp[-1] += len(data) * (1 - stats.poisson.cdf(max_k-1, mu))
    # 合并期望<5的格子
    mask = exp >= 5
    if mask.sum() < 2:
        return np.nan
    try:
        chi2, p = stats.chisquare(obs[mask], exp[mask])
        return p
    except:
        return np.nan

sku_params_list = []

for sku in all_skus:
    d = daily_wide[sku].values.astype(float)

    mu, r         = fit_negbin(d)
    overdispersion = (d.var() / d.mean()) if d.mean() > 0 else 1.0  # >1 表示过度离散
    poisson_p      = poisson_gof(d, mu)
    use_negbin     = (overdispersion > 1.3) and (r < 20)   # 判断是否应用NB分布

    # 价格代理（中位数单价）
    avg_price = sku_selected.loc[sku_selected['SKU'] == sku, 'avg_price'].values[0]
    category  = sku_selected.loc[sku_selected['SKU'] == sku, 'category'].values[0]
    total_qty = sku_selected.loc[sku_selected['SKU'] == sku, 'total_qty'].values[0]

    # 成本参数代理（基于单价）
    # u_i: 单位收益 = 单价（简化，假设毛利率50%）
    # h_i: 滞销成本 = 单价 × 15%（资金占用+损耗）
    # b_i: 缺货成本 = 单价 × 30%（体验损失+转单成本，通常高于h_i）
    # f_i: 上架固定成本 = 固定值（管理复杂度，按品类统一设定）
    u_i = avg_price * 0.50
    h_i = avg_price * 0.15
    b_i = avg_price * 0.30
    f_i = 1.0   # 可在敏感性分析中调整

    sku_params_list.append({
        'SKU':           sku,
        'Category':      category,
        'mu_daily':      round(mu, 4),
        'r_negbin':      round(r, 4) if r != np.inf else 9999,
        'overdispersion': round(overdispersion, 4),
        'poisson_p':     round(poisson_p, 4) if not np.isnan(poisson_p) else -1,
        'use_negbin':    int(use_negbin),
        'total_qty':     int(total_qty),
        'avg_price':     round(avg_price, 4),
        'u_i':           round(u_i, 4),
        'h_i':           round(h_i, 4),
        'b_i':           round(b_i, 4),
        'f_i':           f_i,
        # 体积/重量：前置仓场景无真实数据，用单价分层模拟
        # 实践中可按品类替换为真实测量值
        'v_i':           round(1.0 + avg_price / 20, 2),   # 单位体积（升，粗代理）
        'w_i':           round(0.2 + avg_price / 50, 2),   # 单位重量（kg，粗代理）
    })

sku_params = pd.DataFrame(sku_params_list)

log(f"完成 {len(sku_params)} 个SKU的参数估计")
log(f"建议使用负二项分布的SKU数: {sku_params['use_negbin'].sum()} ({sku_params['use_negbin'].mean():.1%})")
log(f"过度离散系数分布 (均值/中位数): {sku_params['overdispersion'].mean():.2f} / {sku_params['overdispersion'].median():.2f}")
log(f"\n参数摘要:")
log(sku_params[['mu_daily','r_negbin','overdispersion','avg_price','u_i','h_i','b_i']].describe().round(3).to_string())

# ─────────────────────────────────────────
# 6. 篮子结构分析（整单满足率 vs 行满足率）
# ─────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 6: 篮子结构分析")
log("=" * 60)

basket = df[df['SKU'].isin(selected_skus)].groupby('Transaction_ID')['SKU'].count()
log(f"平均篮子大小: {basket.mean():.2f} SKU/订单")
log(f"篮子大小分布:")
log(basket.value_counts().sort_index().head(10).to_string())

# 篮子大小为1的比例（这类订单行满足率=整单满足率，无误差）
pct_size1 = (basket == 1).mean()
log(f"\n单SKU订单比例: {pct_size1:.2%} → 这部分订单两个指标完全一致")
log(f"多SKU订单比例: {1-pct_size1:.2%} → 这部分存在代理误差，需在论文中量化")

# ─────────────────────────────────────────
# 7. 图表输出
# ─────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 7: 生成图表")
log("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Dark Store Data Exploration', fontsize=14, fontweight='bold')

# 7a. 每日总销量时序
ax = axes[0, 0]
daily_total = daily_wide.sum(axis=1)
ax.plot(daily_total.index, daily_total.values, linewidth=0.8, color='steelblue', alpha=0.8)
ax.set_title('Daily Total Demand')
ax.set_xlabel('Date'); ax.set_ylabel('Units Sold')
ax.tick_params(axis='x', rotation=30)

# 7b. SKU需求均值分布（log scale）
ax = axes[0, 1]
ax.hist(np.log1p(sku_params['mu_daily']), bins=40, color='steelblue', alpha=0.8, edgecolor='white')
ax.set_title('SKU Daily Demand Mean Distribution (log scale)')
ax.set_xlabel('log(1 + mu)'); ax.set_ylabel('SKU Count')

# 7c. 过度离散系数分布
ax = axes[0, 2]
disp_clip = sku_params['overdispersion'].clip(upper=10)
ax.hist(disp_clip, bins=40, color='coral', alpha=0.8, edgecolor='white')
ax.axvline(1.3, color='red', linestyle='--', linewidth=1.5, label='NegBin threshold (1.3)')
ax.set_title('Overdispersion Index by SKU')
ax.set_xlabel('Var/Mean (clipped at 10)'); ax.set_ylabel('SKU Count')
ax.legend(fontsize=8)

# 7d. 篮子大小分布
ax = axes[1, 0]
bc = basket.value_counts().sort_index()
bc_plot = bc[bc.index <= 8]
ax.bar(bc_plot.index, bc_plot.values, color='steelblue', alpha=0.8, edgecolor='white')
ax.set_title('Basket Size Distribution')
ax.set_xlabel('SKUs per Order'); ax.set_ylabel('Transaction Count')

# 7e. 单价分布（log scale）
ax = axes[1, 1]
ax.hist(np.log1p(sku_params['avg_price']), bins=40, color='mediumseagreen', alpha=0.8, edgecolor='white')
ax.set_title('SKU Unit Price Distribution (log scale)')
ax.set_xlabel('log(1 + price)'); ax.set_ylabel('SKU Count')

# 7f. Poisson vs NegBin占比
ax = axes[1, 2]
labels = ['Poisson\n(use_negbin=0)', 'NegBin\n(use_negbin=1)']
counts = [len(sku_params) - sku_params['use_negbin'].sum(), sku_params['use_negbin'].sum()]
colors = ['steelblue', 'coral']
bars = ax.bar(labels, counts, color=colors, alpha=0.85, edgecolor='white')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{count}\n({count/len(sku_params):.1%})', ha='center', va='bottom', fontsize=10)
ax.set_title('Recommended Demand Distribution')
ax.set_ylabel('SKU Count')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'data_exploration.png', dpi=150, bbox_inches='tight')
plt.close()
log(f"已保存: {FIGURES_DIR / 'data_exploration.png'}")

# ─────────────────────────────────────────
# 8. 保存输出文件
# ─────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 8: 保存输出文件")
log("=" * 60)

sku_params.to_csv(PROCESSED_DIR / 'sku_params.csv', index=False)
daily_wide.to_csv(PROCESSED_DIR / 'daily_demand.csv')
weekly_wide.to_csv(PROCESSED_DIR / 'weekly_demand.csv')

log(f"sku_params.csv    : {len(sku_params)} 行 × {len(sku_params.columns)} 列")
log(f"daily_demand.csv  : {daily_wide.shape[0]} 天 × {daily_wide.shape[1]} SKU")
log(f"weekly_demand.csv : {weekly_wide.shape[0]} 周 × {weekly_wide.shape[1]} SKU")

with open(REPORT_DIR / 'cleaning_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
log(f"cleaning_report.txt : 已保存到 {REPORT_DIR / 'cleaning_report.txt'}")

log("\n" + "=" * 60)
log("全部完成！")
log("=" * 60)
