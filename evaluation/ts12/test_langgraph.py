"""TS-12 Framework-Test: LangGraph - Zustandsgraph mit bedingter Kante (Primary -> Critic -> ggf. zurück zum Primary)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as fw


def setup(cfg):
    from typing import TypedDict
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    def llm(model):
        return ChatOpenAI(model=model, base_url=cfg["base_url"], api_key="ollama",
                          temperature=cfg["temperature"], max_tokens=cfg["max_tokens"])

    primary, critic = llm(cfg["primary"]), llm(cfg["critic"])

    class State(TypedDict):
        answer: str
        critique: str
        verdict: str
        rounds: int

    def primary_node(s: State) -> dict:
        prompt = fw.primary_prompt() if not s["answer"] else fw.revision_prompt(s["answer"], s["critique"])
        msg = primary.invoke([SystemMessage(fw.PRIMARY_SYSTEM), HumanMessage(prompt)])
        return {"answer": msg.content, "rounds": s["rounds"] + (1 if s["answer"] else 0)}

    def critic_node(s: State) -> dict:
        msg = critic.invoke([SystemMessage(fw.CRITIC_SYSTEM), HumanMessage(fw.critic_prompt(s["answer"]))])
        verdict, critique = fw.parse_verdict(msg.content)
        return {"verdict": verdict, "critique": critique}

    def route(s: State) -> str:       # Schleife nur bei FEHLER und solange Korrekturrunden übrig sind
        return "primary" if s["verdict"] == "FEHLER" and s["rounds"] < fw.MAX_KORREKTUREN else END

    g = StateGraph(State)
    g.add_node("primary", primary_node)
    g.add_node("critic", critic_node)
    g.add_edge(START, "primary")
    g.add_edge("primary", "critic")
    g.add_conditional_edges("critic", route, ["primary", END])
    app = g.compile()

    def run():
        s = app.invoke({"answer": "", "critique": "", "verdict": "", "rounds": 0})
        return {"final_answer": s["answer"], "korrekturrunden": s["rounds"], "finales_urteil": s["verdict"]}

    return run


if __name__ == "__main__":
    fw.run_benchmark("LangGraph", setup, __file__)