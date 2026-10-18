"""TS-12 Framework-Test: CrewAI - Flow (Start/Listen/Router mit Schleife) steuert zwei Agenten, die als Crew-Tasks laufen."""
import os
import sys
from pathlib import Path

# Offline-Betrieb: Telemetrie/Tracing vor dem Import abschalten
for k, v in (("CREWAI_DISABLE_TELEMETRY", "true"), ("OTEL_SDK_DISABLED", "true"), ("CREWAI_TRACING_ENABLED", "false")):
    os.environ.setdefault(k, v)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as fw


def setup(cfg):
    from crewai import LLM, Agent, Crew, Task
    from crewai.flow.flow import Flow, listen, or_, router, start
    from pydantic import BaseModel

    def llm(model):      # "openai/"-Präfix = OpenAI-kompatibler Endpunkt (Ollama /v1)
        return LLM(model=f"openai/{model}", base_url=cfg["base_url"], api_key="ollama",
                   temperature=cfg["temperature"], max_tokens=cfg["max_tokens"])

    primary = Agent(role="Primary Actor", goal="Fragen sachlich und korrekt auf Deutsch beantworten",
                    backstory=fw.PRIMARY_SYSTEM, llm=llm(cfg["primary"]), allow_delegation=False, verbose=False)
    critic = Agent(role="Critic", goal="Antworten auf sachliche Fehler prüfen",
                   backstory=fw.CRITIC_SYSTEM, llm=llm(cfg["critic"]), allow_delegation=False, verbose=False)

    def run_task(agent, description):
        task = Task(description=description, expected_output="Die Antwort als Text.", agent=agent)
        return Crew(agents=[agent], tasks=[task], verbose=False).kickoff().raw

    class State(BaseModel):
        answer: str = ""
        critique: str = ""
        verdict: str = ""
        rounds: int = 0

    class ReviewFlow(Flow[State]):
        @start()
        def draft(self):
            self.state.answer = run_task(primary, fw.primary_prompt())

        @listen(or_(draft, "pruefen"))
        def review(self):
            verdict, critique = fw.parse_verdict(run_task(critic, fw.critic_prompt(self.state.answer)))
            self.state.verdict, self.state.critique = verdict, critique

        @router(review)
        def decide(self):
            return "korrigieren" if self.state.verdict == "FEHLER" and self.state.rounds < fw.MAX_KORREKTUREN else "fertig"

        @listen("korrigieren")
        def ueberarbeiten(self):
            self.state.answer = run_task(primary, fw.revision_prompt(self.state.answer, self.state.critique))
            self.state.rounds += 1

        @router(ueberarbeiten)
        def zurueck_zum_critic(self):       # Schleife: Router-Label löst review erneut aus (direktes Zurückhören ist verboten)
            return "pruefen"

    def run():
        flow = ReviewFlow()
        flow.kickoff()
        s = flow.state
        return {"final_answer": s.answer, "korrekturrunden": s.rounds, "finales_urteil": s.verdict}

    return run


if __name__ == "__main__":
    fw.run_benchmark("CrewAI", setup, __file__)