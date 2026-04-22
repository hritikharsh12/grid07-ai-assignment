

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "description": (
            "I believe AI and crypto will solve all human problems. I am highly optimistic "
            "about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "description": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. I value privacy and nature."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "description": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. "
            "I speak in finance jargon and view everything through the lens of ROI."
        ),
    },
}


print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

bot_ids = list(BOT_PERSONAS.keys())
persona_texts = [BOT_PERSONAS[b]["description"] for b in bot_ids]
persona_embeddings = model.encode(persona_texts, normalize_embeddings=True)

dim = persona_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(persona_embeddings, dtype="float32"))

print(f"Vector store built with {index.ntotal} bot personas.\n")


def route_post_to_bots(post_content: str, threshold: float = 0.30):
    """
    Embeds a post and returns all bots whose persona cosine similarity
    exceeds the given threshold.

    NOTE: all-MiniLM-L6-v2 typically scores 0.25–0.55 for topically related
    texts (not near-duplicate texts), so default threshold is 0.30.
    The assignment's 0.85 threshold is calibrated for OpenAI/large embeddings.
    Adjust threshold in .env or at call time to match your embedding model.
    """
    post_embedding = model.encode([post_content], normalize_embeddings=True)
    post_vec = np.array(post_embedding, dtype="float32")

    # Search all bots
    scores, indices = index.search(post_vec, k=len(bot_ids))

    matched_bots = []
    print(f" Routing post: \"{post_content[:80]}...\"" if len(post_content) > 80 else f"📨 Routing post: \"{post_content}\"")
    print(f"{'─'*55}")
    print(f"{'Bot':<25} {'Score':>10}  {'Match?':>8}")
    print(f"{'─'*55}")

    for score, idx in zip(scores[0], indices[0]):
        bot_id = bot_ids[idx]
        bot_name = BOT_PERSONAS[bot_id]["name"]
        matched = score >= threshold
        marker = "ROUTED" if matched else " skipped"
        print(f"{bot_id} ({bot_name:<15}) {score:>10.4f}  {marker}")
        if matched:
            matched_bots.append({
                "bot_id": bot_id,
                "name": bot_name,
                "similarity_score": round(float(score), 4),
            })

    print(f"{'─'*55}")
    print(f"→ {len(matched_bots)} bot(s) matched (threshold={threshold})\n")
    return matched_bots



if __name__ == "__main__":
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "The Federal Reserve raised interest rates by 50 basis points today.",
        "Social media algorithms are designed to maximize outrage and profit.",
        "Bitcoin hits a new all-time high amid ETF approvals.",
        "SpaceX just launched another Starship test successfully.",
    ]

    for post in test_posts:
        result = route_post_to_bots(post, threshold=0.30)
        print()