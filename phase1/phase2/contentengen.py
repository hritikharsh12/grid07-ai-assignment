

import os
import json
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)


MOCK_NEWS_DB = {
    "crypto":     "Bitcoin surges past $100K as SEC approves spot ETF; altcoins rally 30%.",
    "ai":         "OpenAI's GPT-5 demo causes panic among junior devs; automation wave accelerating.",
    "spacex":     "SpaceX Starship successfully completes orbital test; Musk eyes 2026 Mars mission.",
    "market":     "S&P 500 hits record high; Fed signals two rate cuts in 2025.",
    "regulation": "EU AI Act enforcement begins; OpenAI, Google face compliance audits.",
    "privacy":    "Meta fined €1.2B for data transfers; users flee to decentralised platforms.",
    "climate":    "Record-breaking heatwaves drive ESG investing surge; oil majors under pressure.",
    "default":    "Tech stocks rally as inflation cools; AI sector leads gains.",
}


@tool
def mock_searxng_search(query: str) -> str:
    """Simulates a SearxNG web search and returns relevant mock news headlines."""
    query_lower = query.lower()
    for keyword, headline in MOCK_NEWS_DB.items():
        if keyword in query_lower:
            return headline
    return MOCK_NEWS_DB["default"]



BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "system_prompt": (
            "You are Bot A, an ultra-optimistic Tech Maximalist. You believe AI, crypto, and "
            "space exploration will solve all human problems. You idolise Elon Musk. "
            "You dismiss regulatory concerns as fear-mongering. Your tone is excited and aggressive."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "system_prompt": (
            "You are Bot B, a Doomer and Skeptic. You believe late-stage capitalism and tech "
            "monopolies are destroying society. You are critical of AI, social media, and billionaires. "
            "You value privacy and nature. Your tone is cynical and alarmed."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "system_prompt": (
            "You are Bot C, a Finance Bro. You ONLY care about markets, interest rates, trading "
            "algorithms, and ROI. You speak exclusively in finance jargon. You see everything "
            "as a trade opportunity. Your tone is sharp and transactional."
        ),
    },
}


class PostState(TypedDict):
    bot_id: str
    persona_prompt: str
    search_query: str
    search_results: str
    final_output: dict  # { bot_id, topic, post_content }



def decide_search(state: PostState) -> PostState:
    """LLM picks today's topic and formats a search query based on persona."""
    print("[Node 1] Deciding search topic...")

    response = llm.invoke([
        SystemMessage(content=state["persona_prompt"]),
        HumanMessage(content=(
            "You want to post something opinionated today. "
            "Decide on ONE topic you care about and return ONLY a short 2–5 word search query. "
            "No explanation. Just the search query."
        )),
    ])
    query = response.content.strip().strip('"')
    print(f"   → Search query: \"{query}\"")
    return {**state, "search_query": query}



def web_search(state: PostState) -> PostState:
    """Calls the mock SearxNG tool with the decided query."""
    print("🔍 [Node 2] Running mock web search...")
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"   → Results: {results}")
    return {**state, "search_results": results}



def draft_post(state: PostState) -> PostState:
    """LLM drafts a 280-char opinionated post and returns strict JSON."""
    print("  [Node 3] Drafting post...")

    prompt = f"""
You are drafting a social media post based on this news:
"{state['search_results']}"

Rules:
1. Stay 100% in character with your persona.
2. Post must be under 280 characters.
3. Be opinionated, not neutral.
4. Return ONLY a valid JSON object — no markdown, no explanation, no backticks.
   Format: {{"bot_id": "{state['bot_id']}", "topic": "<2-4 word topic>", "post_content": "<your post>"}}
"""

    response = llm.invoke([
        SystemMessage(content=state["persona_prompt"]),
        HumanMessage(content=prompt),
    ])

    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        parsed["bot_id"] = state["bot_id"]
        print(f"   → JSON output: {json.dumps(parsed, indent=2)}")
    except json.JSONDecodeError:
        print(f" JSON parse failed, raw output: {raw}")
        parsed = {
            "bot_id": state["bot_id"],
            "topic": state["search_query"],
            "post_content": raw[:280],
        }

    return {**state, "final_output": parsed}



def build_content_graph():
    graph = StateGraph(PostState)
    graph.add_node("decide_search", decide_search)
    graph.add_node("web_search", web_search)
    graph.add_node("draft_post", draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)

    return graph.compile()



if __name__ == "__main__":
    app = build_content_graph()

    for bot_id, persona in BOT_PERSONAS.items():
        print(f"\n{'='*60}")
        print(f" Running content engine for: {bot_id} ({persona['name']})")
        print(f"{'='*60}")

        initial_state: PostState = {
            "bot_id": bot_id,
            "persona_prompt": persona["system_prompt"],
            "search_query": "",
            "search_results": "",
            "final_output": {},
        }

        result = app.invoke(initial_state)

        print(f"\n FINAL OUTPUT:")
        print(json.dumps(result["final_output"], indent=2))