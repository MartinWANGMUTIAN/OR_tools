"""
DeFiGuard 复现版（修正）
修正点：
  1. 库换成 DGL（论文原版 DGL 0.9.1），去掉 PyTorch Geometric 依赖
  2. DBaccount 结构对齐论文：{address: is_verified_bool}，不在库里的地址直接当 EOA
  3. 训练协议对齐论文：balanced sampling（PMA:non-PMA = 1:1），train_size=100，epoch=100
  4. 修复 dummy data 的浅拷贝问题（改用 copy.deepcopy）

依赖安装：
  pip install torch dgl -f https://data.dgl.ai/wheels/repo.html
  pip install scikit-learn
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv, SumPooling
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class TraceRecord:
    caller: str
    callee: str
    value: float          # native token 数量


@dataclass
class EventRecord:
    token_address: str
    from_addr: str
    to_addr: str
    value: float          # ERC20 数量


@dataclass
class ParsedTransaction:
    traces: List[TraceRecord] = field(default_factory=list)
    events: List[EventRecord] = field(default_factory=list)


# ============================================================
# 2. Transaction Parser
#    对齐论文 Section V-A：
#    - 提取 value > 0 的 call trace（native asset transfer）
#    - 提取 Transfer event（ERC20 asset transfer）
#    - 没有任何 asset transfer 的交易不向后传递（此处直接返回空）
# ============================================================

class TransactionParser:
    def parse(self, tx: Dict[str, Any]) -> ParsedTransaction:
        traces: List[TraceRecord] = []
        events: List[EventRecord] = []

        for tr in tx.get("call_traces", []):
            value = float(tr.get("value", 0))
            if value > 0:
                traces.append(TraceRecord(
                    caller=tr["caller"].lower(),
                    callee=tr["callee"].lower(),
                    value=value,
                ))

        for ev in tx.get("event_logs", []):
            if ev.get("event_name") == "Transfer":
                value = float(ev.get("value", 0))
                if value > 0:
                    events.append(EventRecord(
                        token_address=ev["token_address"].lower(),
                        from_addr=ev["from"].lower(),
                        to_addr=ev["to"].lower(),
                        value=value,
                    ))

        return ParsedTransaction(traces=traces, events=events)


# ============================================================
# 3. Graph Builder
#    对齐论文 Section V-B：
#
#    DBaccount 结构（修正）：
#      key   = CA 地址（小写）
#      value = bool，True 表示 verified（透明合约），False 表示 unverified（不透明合约）
#      不在库里的地址 → EOA
#
#    Node type 编码：
#      Opaque CA      → [1, 0, 0]   (is_contract=True, verified=False)
#      Transparent CA → [0, 1, 0]   (is_contract=True, verified=True)
#      EOA            → [0, 0, 1]   (不在 DBaccount 里)
#
#    特征维度：3 + 2 + 2 + 1 = 8
# ============================================================

class GraphBuilder:
    def __init__(self, db_account: Optional[Dict[str, bool]] = None):
        # 修正：value 只存 bool（是否 verified），不再存 is_contract
        self.db_account: Dict[str, bool] = db_account or {}

    def build_dgl_graph(
        self,
        parsed_tx: ParsedTransaction,
        label: Optional[int] = None,
    ) -> dgl.DGLGraph:
        nodes, edges, edge_assets, edge_amounts = self._construct_graph(parsed_tx)
        node_features = self._extract_features(nodes, edges, edge_assets, edge_amounts)

        node_list = sorted(nodes)
        node_to_idx = {addr: i for i, addr in enumerate(node_list)}

        src_ids, dst_ids = [], []
        for sender, receiver in edges:
            src_ids.append(node_to_idx[sender])
            dst_ids.append(node_to_idx[receiver])

        if src_ids:
            g = dgl.graph((src_ids, dst_ids))
        else:
            # 孤立节点图（无边）
            g = dgl.graph(([], []))
            g.add_nodes(len(node_list))

        x = torch.tensor(
            [node_features[n] for n in node_list], dtype=torch.float
        )
        g.ndata["feat"] = x

        if label is not None:
            g.label = torch.tensor([label], dtype=torch.long)
        else:
            g.label = None

        return g

    # ---- 内部方法 ----

    def _construct_graph(
        self, parsed_tx: ParsedTransaction
    ) -> Tuple[set, List[Tuple[str, str]], List[str], List[float]]:
        nodes: set = set()
        edges: List[Tuple[str, str]] = []
        edge_assets: List[str] = []
        edge_amounts: List[float] = []

        for tr in parsed_tx.traces:
            nodes.add(tr.caller)
            nodes.add(tr.callee)
            edges.append((tr.caller, tr.callee))
            edge_assets.append("native")
            edge_amounts.append(tr.value)

        for ev in parsed_tx.events:
            nodes.add(ev.from_addr)
            nodes.add(ev.to_addr)
            edges.append((ev.from_addr, ev.to_addr))
            edge_assets.append(ev.token_address)
            edge_amounts.append(ev.value)

        return nodes, edges, edge_assets, edge_amounts

    def _extract_features(
        self,
        nodes: set,
        edges: List[Tuple[str, str]],
        edge_assets: List[str],
        edge_amounts: List[float],
    ) -> Dict[str, List[float]]:

        # ---------- Feature 1: Node Type ----------
        # 修正：db_account[addr] 只是 bool（verified 与否）
        # 不在库里 → EOA
        node_type: Dict[str, List[float]] = {}
        for n in nodes:
            if n in self.db_account:
                if self.db_account[n] is False:
                    node_type[n] = [1.0, 0.0, 0.0]   # Opaque CA
                else:
                    node_type[n] = [0.0, 1.0, 0.0]   # Transparent CA
            else:
                node_type[n] = [0.0, 0.0, 1.0]        # EOA

        # ---------- Feature 2: Transfer Frequency ----------
        in_cnt: Dict[str, int] = defaultdict(int)
        out_cnt: Dict[str, int] = defaultdict(int)
        for sender, receiver in edges:
            out_cnt[sender] += 1
            in_cnt[receiver] += 1

        max_in = max(in_cnt.values(), default=1)
        max_out = max(out_cnt.values(), default=1)

        freq: Dict[str, List[float]] = {
            n: [in_cnt[n] / max_in, out_cnt[n] / max_out]
            for n in nodes
        }

        # ---------- Feature 3: Transfer Diversity ----------
        in_assets: Dict[str, List[str]] = defaultdict(list)
        out_assets: Dict[str, List[str]] = defaultdict(list)
        for (sender, receiver), asset in zip(edges, edge_assets):
            out_assets[sender].append(asset)
            in_assets[receiver].append(asset)

        max_in_div = max((len(set(v)) for v in in_assets.values()), default=1)
        max_out_div = max((len(set(v)) for v in out_assets.values()), default=1)

        diversity: Dict[str, List[float]] = {
            n: [
                len(set(in_assets[n])) / max_in_div,
                len(set(out_assets[n])) / max_out_div,
            ]
            for n in nodes
        }

        # ---------- Feature 4: Profit Score ----------
        max_amount: Dict[str, float] = defaultdict(float)
        for asset, amount in zip(edge_assets, edge_amounts):
            if amount > max_amount[asset]:
                max_amount[asset] = amount

        profit: Dict[str, float] = defaultdict(float)
        for (sender, receiver), asset, amount in zip(edges, edge_assets, edge_amounts):
            denom = max_amount[asset] if max_amount[asset] > 0 else 1.0
            normed = amount / denom
            profit[sender] -= normed
            profit[receiver] += normed

        # ---------- 拼接 ----------
        return {
            n: node_type[n] + freq[n] + diversity[n] + [profit[n]]
            for n in nodes
        }


# ============================================================
# 4. Graph Classifier (GraphSAGE，使用 DGL)
#    对齐论文：2层，hidden=16，output=2，ReLU，Adam
# ============================================================

class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 16, out_dim: int = 2):
        super().__init__()
        # DGL SAGEConv 默认 aggregator_type='mean'，与论文一致
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggregator_type="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type="mean")
        self.pool = SumPooling()          # graph-level pooling
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(g, feat))
        h = F.relu(self.conv2(g, h))
        hg = self.pool(g, h)              # (batch_size, hidden_dim)
        return self.lin(hg)


# ============================================================
# 5. Balanced DataLoader（对齐论文 train_size=100 的 1:1 采样）
# ============================================================

def balanced_batch(
    pma_graphs: List[dgl.DGLGraph],
    non_pma_graphs: List[dgl.DGLGraph],
    train_size: int,
    batch_size: int = 16,
) -> List[List[dgl.DGLGraph]]:
    """
    从 PMA / non-PMA 各采 train_size 条，1:1 混合后分 mini-batch。
    对齐论文：train_size=100 → 200 条/epoch。
    """
    pma_sample = random.choices(pma_graphs, k=train_size)
    non_pma_sample = random.choices(non_pma_graphs, k=train_size)
    combined = pma_sample + non_pma_sample
    random.shuffle(combined)
    return [combined[i: i + batch_size] for i in range(0, len(combined), batch_size)]


# ============================================================
# 6. 训练与评估
# ============================================================

def train_one_epoch(
    model: nn.Module,
    pma_graphs: List[dgl.DGLGraph],
    non_pma_graphs: List[dgl.DGLGraph],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_size: int = 100,
    batch_size: int = 16,
) -> float:
    model.train()
    total_loss = 0.0
    batches = balanced_batch(pma_graphs, non_pma_graphs, train_size, batch_size)

    for batch in batches:
        gs = dgl.batch(batch).to(device)
        feats = gs.ndata["feat"]
        labels = torch.cat([g.label for g in batch]).to(device)

        optimizer.zero_grad()
        logits = model(gs, feats)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(batches), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    graphs: List[dgl.DGLGraph],
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, float]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for i in range(0, len(graphs), batch_size):
        batch = graphs[i: i + batch_size]
        gs = dgl.batch(batch).to(device)
        feats = gs.ndata["feat"]
        labels = torch.cat([g.label for g in batch])

        logits = model(gs, feats)
        prob = F.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)

        y_true.extend(labels.tolist())
        y_pred.extend(pred.cpu().tolist())
        y_prob.extend(prob.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "TPR": tpr, "FPR": fpr, "AUC": auc}


def train_model(
    pma_train: List[dgl.DGLGraph],
    non_pma_train: List[dgl.DGLGraph],
    test_graphs: List[dgl.DGLGraph],
    epochs: int = 100,
    train_size: int = 100,
    lr: float = 1e-3,
) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGEClassifier(in_dim=8, hidden_dim=16, out_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, pma_train, non_pma_train, optimizer, device, train_size
        )
        if epoch % 10 == 0 or epoch == 1:
            metrics = evaluate(model, test_graphs, device)
            print(
                f"[Epoch {epoch:03d}] loss={loss:.4f}  "
                f"acc={metrics['accuracy']:.4f}  "
                f"TPR={metrics['TPR']:.4f}  "
                f"FPR={metrics['FPR']:.4f}  "
                f"AUC={metrics['AUC']:.4f}"
            )

    return model


# ============================================================
# 7. 推断（单笔交易）
# ============================================================

@torch.no_grad()
def predict_single(
    model: nn.Module, graph: dgl.DGLGraph
) -> Tuple[int, float]:
    device = next(model.parameters()).device
    model.eval()
    g = graph.to(device)
    feat = g.ndata["feat"]
    logits = model(g, feat)
    prob = F.softmax(logits, dim=1)[0, 1].item()
    pred = int(logits.argmax(dim=1)[0].item())
    return pred, prob


# ============================================================
# 8. 示例数据
#    修正：用 copy.deepcopy 避免浅拷贝共享内嵌 list 的问题
# ============================================================

def make_dummy_db() -> Dict[str, bool]:
    # 修正：value 只是 bool（verified 与否），不再有 is_contract 字段
    return {
        "0xcontracta": False,   # Opaque CA（unverified）
        "0xcontractb": True,    # Transparent CA（verified）
        "0xrouter":    True,    # Transparent CA（verified）
    }


_MALICIOUS_TEMPLATE = {
    "call_traces": [
        {"caller": "0xUser1", "callee": "0xRouter",    "value": 10},
        {"caller": "0xRouter",    "callee": "0xContractA", "value": 8},
        {"caller": "0xContractA", "callee": "0xRouter",    "value": 9},
        {"caller": "0xRouter",    "callee": "0xUser1",     "value": 11},
    ],
    "event_logs": [
        {"event_name": "Transfer", "token_address": "0xUSDT", "from": "0xUser1",     "to": "0xRouter",    "value": 1000},
        {"event_name": "Transfer", "token_address": "0xWBTC", "from": "0xRouter",    "to": "0xContractA", "value": 2},
        {"event_name": "Transfer", "token_address": "0xUSDT", "from": "0xContractA", "to": "0xRouter",    "value": 1300},
        {"event_name": "Transfer", "token_address": "0xWETH", "from": "0xRouter",    "to": "0xUser1",     "value": 5},
    ],
    "label": 1,
}

_BENIGN_TEMPLATE = {
    "call_traces": [
        {"caller": "0xUser2", "callee": "0xRouter", "value": 3},
    ],
    "event_logs": [
        {"event_name": "Transfer", "token_address": "0xUSDC", "from": "0xUser2",  "to": "0xRouter", "value": 500},
        {"event_name": "Transfer", "token_address": "0xWETH", "from": "0xRouter", "to": "0xUser2",  "value": 1},
    ],
    "label": 0,
}


def make_dummy_transactions(n: int = 150) -> List[Dict[str, Any]]:
    """修正：使用 deepcopy，避免不同 dict 实例共享同一 list 引用。"""
    txs = []
    for _ in range(n):
        txs.append(copy.deepcopy(_MALICIOUS_TEMPLATE))
        txs.append(copy.deepcopy(_BENIGN_TEMPLATE))
    random.shuffle(txs)
    return txs


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    parser_obj = TransactionParser()
    # 修正：传入修正后的 DBaccount（value 为 bool）
    builder = GraphBuilder(db_account=make_dummy_db())

    txs = make_dummy_transactions(n=150)

    pma_graphs: List[dgl.DGLGraph] = []
    non_pma_graphs: List[dgl.DGLGraph] = []

    for tx in txs:
        parsed = parser_obj.parse(tx)
        g = builder.build_dgl_graph(parsed, label=tx["label"])
        if tx["label"] == 1:
            pma_graphs.append(g)
        else:
            non_pma_graphs.append(g)

    # 修正：按论文协议分割（保留一部分做测试，其余用于 balanced sampling）
    # 论文 train_size=100 意味着每 epoch 从训练集各采 100 条；
    # 这里粗糙地把 20% 留作测试集。
    def split(lst: List, ratio: float = 0.8):
        k = int(len(lst) * ratio)
        return lst[:k], lst[k:]

    pma_train,     pma_test     = split(pma_graphs)
    non_pma_train, non_pma_test = split(non_pma_graphs)
    test_graphs = pma_test + non_pma_test
    random.shuffle(test_graphs)

    print(f"PMA train={len(pma_train)}, non-PMA train={len(non_pma_train)}, test={len(test_graphs)}")

    # 修正：对齐论文 epoch=100，train_size=100
    model = train_model(
        pma_train, non_pma_train, test_graphs,
        epochs=100, train_size=min(100, len(pma_train)), lr=1e-3,
    )

    # 单笔推断示例
    sample_tx = {
        "call_traces": [
            {"caller": "0xUser3",    "callee": "0xRouter",    "value": 7},
            {"caller": "0xRouter",    "callee": "0xContractA", "value": 6},
            {"caller": "0xContractA", "callee": "0xRouter",    "value": 7.5},
        ],
        "event_logs": [
            {"event_name": "Transfer", "token_address": "0xUSDT", "from": "0xUser3",     "to": "0xRouter",    "value": 800},
            {"event_name": "Transfer", "token_address": "0xWBTC", "from": "0xRouter",    "to": "0xContractA", "value": 1.2},
            {"event_name": "Transfer", "token_address": "0xUSDT", "from": "0xContractA", "to": "0xRouter",    "value": 950},
        ],
    }

    parsed = parser_obj.parse(sample_tx)
    g = builder.build_dgl_graph(parsed)
    pred, prob = predict_single(model, g)
    print(f"\nSingle transaction → pred={pred} (1=PMA, 0=non-PMA), prob={prob:.4f}")


if __name__ == "__main__":
    main()
