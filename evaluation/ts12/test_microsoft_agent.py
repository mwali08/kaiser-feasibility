"""TS-12 Framework-Test: Microsoft Agent Framework - Agents als Executors in einem Workflow-Graphen mit Rückkante."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as fw


def setup(cfg):
    import asyncio
    from dataclasses import dataclass

    from agent_framework import Agent, Executor, WorkflowBuilder, WorkflowContext, handler
    from agent_framework.openai import OpenAIChatCompletionClient

    def make_agent(model, name, instructions):     # Chat-Completions-Client (Ollama /v1), nicht die Responses-API
        client = OpenAIChatCompletionClient(model=model, base_url=cfg["base_url"], api_key="ollama")
        return Agent(client=client, name=name, instructions=instructions,
                     default_options={"temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"]})

    primary = make_agent(cfg["primary"], "primary", fw.PRIMARY_SYSTEM)
    critic = make_agent(cfg["critic"], "critic", fw.CRITIC_SYSTEM)

    @dataclass
    class Draft:                 # Nachricht Primary -> Critic
        answer: str
        rounds: int

    @dataclass
    class Revision:              # Nachricht Critic -> Primary
        answer: str
        critique: str
        rounds: int

    @dataclass
    class Result:                # Endergebnis (Workflow-Ausgabe)
        answer: str
        rounds: int
        verdict: str

    class PrimaryExecutor(Executor):
        def __init__(self):
            super().__init__(id="primary_executor")

        @handler
        async def answer(self, question: str, ctx: WorkflowContext[Draft]) -> None:
            r = await primary.run(fw.primary_prompt())
            await ctx.send_message(Draft(r.text, 0))

        @handler
        async def revise(self, rev: Revision, ctx: WorkflowContext[Draft]) -> None:
            r = await primary.run(fw.revision_prompt(rev.answer, rev.critique))
            await ctx.send_message(Draft(r.text, rev.rounds + 1))

    class CriticExecutor(Executor):
        def __init__(self):
            super().__init__(id="critic_executor")

        @handler
        async def review(self, draft: Draft, ctx: WorkflowContext[Revision, Result]) -> None:
            r = await critic.run(fw.critic_prompt(draft.answer))
            verdict, critique = fw.parse_verdict(r.text)
            if verdict == "FEHLER" and draft.rounds < fw.MAX_KORREKTUREN:
                await ctx.send_message(Revision(draft.answer, critique, draft.rounds))      # Rückkante -> Schleife
            else:
                await ctx.yield_output(Result(draft.answer, draft.rounds, verdict))

    p, c = PrimaryExecutor(), CriticExecutor()
    workflow = (WorkflowBuilder(start_executor=p, output_from=[c])
                .add_edge(p, c)
                .add_edge(c, p, condition=lambda m: isinstance(m, Revision))
                .build())

    def run():
        result = asyncio.run(workflow.run(cfg["frage"]))
        out = result.get_outputs()[0]
        return {"final_answer": out.answer, "korrekturrunden": out.rounds, "finales_urteil": out.verdict}

    return run


if __name__ == "__main__":
    fw.run_benchmark("Microsoft Agent Framework", setup, __file__)