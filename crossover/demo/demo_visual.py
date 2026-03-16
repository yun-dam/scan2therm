"""
CrossOver Visual Demo — Text-to-Scene Cross-Modal Retrieval
Encodes text descriptions, retrieves matching 3D scenes from pre-computed
databases, and generates visual plots (saved as PNG).
"""
import os
import os.path as osp
import sys
import random
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import timedelta

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
os.environ["PYTHONPATH"] = osp.join(osp.dirname(__file__), '..', 'third_party', 'BLIP') \
    + ":" + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'third_party', 'BLIP'))

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from model.scene_crossover import SceneCrossOverModel
from util import torch_util
from common.load_utils import load_npz_as_dict

# ── Queries ──────────────────────────────────────────────────────────────────
TEXT_QUERIES = [
    "a large living room with a sofa and a coffee table near the window",
    "a small kitchen with white cabinets and a microwave",
    "a bathroom with a bathtub and a sink",
    "a bedroom with a double bed and nightstands",
    "an office room with a desk, chair and computer monitor",
    "a dining room with a wooden table and four chairs",
    "a conference room with a long table and many chairs",
    "a hallway with a bookshelf and a coat rack",
]

QUERY_LABELS = [
    "Living Room",
    "Kitchen",
    "Bathroom",
    "Bedroom",
    "Office",
    "Dining Room",
    "Conference",
    "Hallway",
]

DB_NAMES = ["scannet", "scan3r", "multiscan", "arkitscenes"]
DB_DISPLAY = ["ScanNet", "3RScan", "MultiScan", "ARKitScenes"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device="cuda"):
    init_kw = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    acc = Accelerator(kwargs_handlers=[init_kw])
    args = argparse.Namespace(
        input_dim_3d=512, input_dim_2d=1536, input_dim_1d=768, out_dim=768
    )
    model = SceneCrossOverModel(args, acc.device)
    model = acc.prepare(model)
    model.eval()
    model.to(acc.device)
    torch_util.load_weights(model, ckpt_path, acc.device)
    return model


def encode_texts(model, texts):
    """Return (N, 768) tensor of text embeddings — one call per query."""
    embeds = []
    for t in texts:
        with torch.no_grad():
            e = model.encode_1d([t]).cpu()   # model takes list, returns (1, 768)
        embeds.append(e)
    return torch.cat(embeds, dim=0)           # (N, 768)


def load_database(db_path, modality="referral"):
    """Return (M, 768) embeddings, mask, scan_ids."""
    data = load_npz_as_dict(db_path)
    scenes = data["scene"]
    if isinstance(scenes, np.ndarray):
        scenes = scenes.tolist()
    embeds = np.array([s["scene_embeds"][modality].reshape(-1) for s in scenes])
    masks  = np.array([s["masks"][modality] for s in scenes])
    ids    = [s["scan_id"] for s in scenes]
    return torch.from_numpy(embeds), torch.from_numpy(masks).bool(), ids


def retrieve(query_embed, db_embeds, db_masks, db_ids, top_k=5):
    """Return top-k (scan_id, score) pairs."""
    valid = db_embeds[db_masks]
    valid_ids = [db_ids[i] for i, m in enumerate(db_masks) if m]
    sim = torch.softmax(query_embed.unsqueeze(0) @ valid.t(), dim=-1).squeeze(0)
    topk_vals, topk_idx = sim.topk(min(top_k, len(valid)))
    return [(valid_ids[i], topk_vals[j].item())
            for j, i in enumerate(topk_idx.tolist())]


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_similarity_matrix(embeds, labels, out_path):
    """Cosine similarity heatmap between all text queries."""
    normed = embeds / embeds.norm(dim=-1, keepdim=True)
    sim = (normed @ normed.t()).numpy()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sim, cmap="YlOrRd", vmin=sim.min(), vmax=sim.max())
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if sim[i, j] > 0.6 * sim.max() else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")
    ax.set_title("CrossOver Text Embedding — Pairwise Cosine Similarity", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_retrieval_results(all_results, labels, out_path):
    """Bar charts: top-5 retrieval scores per query across all databases."""
    n_queries = len(labels)
    n_dbs = len(DB_NAMES)

    fig, axes = plt.subplots(n_queries, n_dbs, figsize=(5 * n_dbs, 3 * n_queries))
    if n_queries == 1:
        axes = axes[np.newaxis, :]

    for qi in range(n_queries):
        for di in range(n_dbs):
            ax = axes[qi, di]
            results = all_results[qi][di]
            if not results:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{labels[qi]} → {DB_DISPLAY[di]}", fontsize=9)
                continue
            ids   = [r[0][:16] for r in results]
            scores = [r[1] for r in results]
            colors = plt.cm.viridis(np.linspace(0.8, 0.3, len(scores)))
            bars = ax.barh(range(len(ids)), scores, color=colors)
            ax.set_yticks(range(len(ids)))
            ax.set_yticklabels(ids, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0, max(scores) * 1.15)
            for bar, s in zip(bars, scores):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                        f"{s:.4f}", va="center", fontsize=7)
            if qi == 0:
                ax.set_title(DB_DISPLAY[di], fontsize=11, fontweight="bold")
            if di == 0:
                ax.set_ylabel(labels[qi], fontsize=10, fontweight="bold")

    fig.suptitle("CrossOver Text→Scene Retrieval  (top-5 per database)", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_cross_modal_retrieval(model, embeds_dir, labels, text_embeds, out_path):
    """For a single database (ScanNet), compare text→point vs text→rgb vs text→referral."""
    modalities = ["point", "rgb", "referral"]
    mod_display = ["Point Cloud", "RGB Images", "Text (Referral)"]
    db_path = osp.join(embeds_dir, "embed_scannet.npz")

    n_q = len(labels)
    fig, axes = plt.subplots(n_q, len(modalities), figsize=(5 * len(modalities), 3 * n_q))
    if n_q == 1:
        axes = axes[np.newaxis, :]

    for mi, mod in enumerate(modalities):
        db_emb, db_mask, db_ids = load_database(db_path, mod)
        for qi in range(n_q):
            ax = axes[qi, mi]
            results = retrieve(text_embeds[qi], db_emb, db_mask, db_ids, top_k=5)
            ids    = [r[0][:16] for r in results]
            scores = [r[1] for r in results]
            colors = plt.cm.plasma(np.linspace(0.8, 0.3, len(scores)))
            bars = ax.barh(range(len(ids)), scores, color=colors)
            ax.set_yticks(range(len(ids)))
            ax.set_yticklabels(ids, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0, max(scores) * 1.15)
            for bar, s in zip(bars, scores):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                        f"{s:.4f}", va="center", fontsize=7)
            if qi == 0:
                ax.set_title(f"Target: {mod_display[mi]}", fontsize=11, fontweight="bold")
            if mi == 0:
                ax.set_ylabel(labels[qi], fontsize=10, fontweight="bold")

    fig.suptitle("Cross-Modal Retrieval on ScanNet — Text Query → Different Target Modalities",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_tsne(text_embeds, text_labels, embeds_dir, out_path):
    """t-SNE of text query embeddings mixed with database scene embeddings."""
    from sklearn.manifold import TSNE

    # Collect referral embeddings from all databases (subsample for speed)
    all_embs = [text_embeds.numpy()]
    all_labels = list(text_labels)
    all_types = ["query"] * len(text_labels)

    for di, db_name in enumerate(DB_NAMES):
        db_path = osp.join(embeds_dir, f"embed_{db_name}.npz")
        db_emb, db_mask, db_ids = load_database(db_path, "referral")
        valid = db_emb[db_mask].numpy()
        # subsample to keep plot readable
        n_sample = min(80, len(valid))
        idx = np.random.choice(len(valid), n_sample, replace=False)
        all_embs.append(valid[idx])
        all_labels += [DB_DISPLAY[di]] * n_sample
        all_types += [DB_DISPLAY[di]] * n_sample

    X = np.concatenate(all_embs, axis=0)
    X_norm = X / np.linalg.norm(X, axis=-1, keepdims=True)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    Z = tsne.fit_transform(X_norm)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Database points
    cmap = {"ScanNet": "#1f77b4", "3RScan": "#ff7f0e", "MultiScan": "#2ca02c", "ARKitScenes": "#9467bd"}
    for db_disp, color in cmap.items():
        mask = np.array([t == db_disp for t in all_types])
        if mask.sum() > 0:
            ax.scatter(Z[mask, 0], Z[mask, 1], c=color, alpha=0.4, s=25, label=db_disp)

    # Query points — larger, with annotation
    q_mask = np.array([t == "query" for t in all_types])
    qZ = Z[q_mask]
    ax.scatter(qZ[:, 0], qZ[:, 1], c="red", s=150, marker="*", zorder=5,
               edgecolors="black", linewidths=0.5, label="Text Queries")
    for i, lbl in enumerate(text_labels):
        ax.annotate(lbl, (qZ[i, 0], qZ[i, 1]), fontsize=8, fontweight="bold",
                    xytext=(8, 8), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

    ax.legend(fontsize=10, loc="best")
    ax.set_title("t-SNE of CrossOver Embedding Space\n(Text queries ★ vs Database scene embeddings)",
                 fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CrossOver Visual Demo")
    parser.add_argument("--ckpt", type=str,
                        default="checkpoints/scene_crossover_scannet+scan3r.pth")
    parser.add_argument("--embeds_dir", type=str,
                        default="demo_data/release_embeds")
    parser.add_argument("--out_dir", type=str, default="demo_output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # ── 1. Load model ────────────────────────────────────────────────────────
    print("\n[1/5] Loading CrossOver model …")
    model = load_model(args.ckpt)

    # ── 2. Encode text queries ───────────────────────────────────────────────
    print("[2/5] Encoding text queries …")
    text_embeds = encode_texts(model, TEXT_QUERIES)
    print(f"  Encoded {len(TEXT_QUERIES)} queries → {text_embeds.shape}")

    # ── 3. Similarity matrix ─────────────────────────────────────────────────
    print("[3/5] Plotting text similarity matrix …")
    plot_similarity_matrix(text_embeds, QUERY_LABELS,
                           osp.join(args.out_dir, "1_text_similarity_matrix.png"))

    # ── 4. Text → Scene retrieval across all databases ───────────────────────
    print("[4/5] Running text→scene retrieval across 4 databases …")
    all_results = []  # [query_idx][db_idx] → [(scan_id, score), ...]
    for qi, query in enumerate(TEXT_QUERIES):
        query_results = []
        for di, db_name in enumerate(DB_NAMES):
            db_path = osp.join(args.embeds_dir, f"embed_{db_name}.npz")
            db_emb, db_mask, db_ids = load_database(db_path, "referral")
            results = retrieve(text_embeds[qi], db_emb, db_mask, db_ids, top_k=5)
            query_results.append(results)
            print(f"  {QUERY_LABELS[qi]:15s} → {DB_DISPLAY[di]:12s}  "
                  f"top-1: {results[0][0] if results else 'N/A'}")
        all_results.append(query_results)

    plot_retrieval_results(all_results, QUERY_LABELS,
                           osp.join(args.out_dir, "2_retrieval_results.png"))

    # ── 5. Cross-modal comparison (ScanNet only) ─────────────────────────────
    print("[5/5] Cross-modal retrieval comparison (text→point / text→rgb / text→referral) …")
    plot_cross_modal_retrieval(model, args.embeds_dir, QUERY_LABELS, text_embeds,
                               osp.join(args.out_dir, "3_cross_modal_retrieval.png"))

    # ── 6. t-SNE embedding space ─────────────────────────────────────────────
    print("[Bonus] t-SNE embedding space visualisation …")
    plot_tsne(text_embeds, QUERY_LABELS, args.embeds_dir,
              osp.join(args.out_dir, "4_tsne_embedding_space.png"))

    print(f"\n{'='*60}")
    print(f"  All plots saved to:  {osp.abspath(args.out_dir)}/")
    print(f"    1_text_similarity_matrix.png   — query-vs-query cosine sim")
    print(f"    2_retrieval_results.png        — top-5 scenes per query × DB")
    print(f"    3_cross_modal_retrieval.png    — text→point/rgb/referral")
    print(f"    4_tsne_embedding_space.png     — embedding space map")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
