"""TS-12 Framework-Test: Haystack - Pipeline aus Komponenten mit Rückschleife (BranchJoiner + entscheidende Komponente)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")      # Offline-Betrieb: keine Telemetrie
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as fw


def setup(cfg):
    from haystack import Pipeline, component
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.joiners import BranchJoiner
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret

    def generator(model):
        return OpenAIChatGenerator(api_key=Secret.from_token("ollama"), model=model, api_base_url=cfg["base_url"],
                                   generation_kwargs={"temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"]})

    @component
    class MessageBuilder:
        def __init__(self, system: str):
            self.system = system

        @component.output_types(messages=list[ChatMessage])
        def run(self, prompt: str):
            return {"messages": [ChatMessage.from_system(self.system), ChatMessage.from_user(prompt)]}

    @component
    class CriticMessageBuilder:
        @component.output_types(messages=list[ChatMessage])
        def run(self, replies: list[ChatMessage]):
            return {"messages": [ChatMessage.from_system(fw.CRITIC_SYSTEM),
                                 ChatMessage.from_user(fw.critic_prompt(replies[0].text))]}

    @component
    class Review:        # entscheidet: Schleife (revise_prompt) oder Ende (final)
        def __init__(self):
            self.rounds = 0

        @component.output_types(revise_prompt=str, final=dict)
        def run(self, answer: list[ChatMessage], critic: list[ChatMessage]):
            text = answer[0].text
            verdict, critique = fw.parse_verdict(critic[0].text)
            if verdict == "FEHLER" and self.rounds < fw.MAX_KORREKTUREN:
                self.rounds += 1
                return {"revise_prompt": fw.revision_prompt(text, critique)}
            return {"final": {"answer": text, "rounds": self.rounds, "verdict": verdict}}

    review = Review()
    p = Pipeline(max_runs_per_component=10)
    p.add_component("prompt_in", BranchJoiner(str))
    p.add_component("primary_builder", MessageBuilder(fw.PRIMARY_SYSTEM))
    p.add_component("primary", generator(cfg["primary"]))
    p.add_component("critic_builder", CriticMessageBuilder())
    p.add_component("critic", generator(cfg["critic"]))
    p.add_component("review", review)
    p.connect("prompt_in.value", "primary_builder.prompt")
    p.connect("primary_builder.messages", "primary.messages")
    p.connect("primary.replies", "critic_builder.replies")
    p.connect("primary.replies", "review.answer")
    p.connect("critic_builder.messages", "critic.messages")
    p.connect("critic.replies", "review.critic")
    p.connect("review.revise_prompt", "prompt_in.value")           # Rückschleife

    def run():
        review.rounds = 0
        out = p.run({"prompt_in": {"value": fw.primary_prompt()}})["review"]["final"]
        return {"final_answer": out["answer"], "korrekturrunden": out["rounds"], "finales_urteil": out["verdict"]}

    return run


if __name__ == "__main__":
    fw.run_benchmark("Haystack", setup, __file__)