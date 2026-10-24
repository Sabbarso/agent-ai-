"""General ReAct agent as a LangGraph state machine.

decide (LLM picks a tool or answers, JSON validated by Pydantic) -> act (run tool, record
trace) -> decide ... -> finalize (forced answer if max_iterations is reached, then
validate_citations against the arXiv ids seen in tool observations).

The LLM is driven through `LLMClient.structured`, so the agent works with any
OpenAI-compatible provider, even without native tool calling.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rpe.agent.tools import describe_tools
from rpe.llm import prompts
from rpe.llm.client import LLMClient, LLMError
from rpe.llm.schemas import AgentDecision
from rpe.rag.citations import ARXIV_ID_PATTERN, normalize_id, validate_citations

_ID_RE = re.compile(rf"(?<![\d.]){ARXIV_ID_PATTERN}(?![\d])")
FINAL = "final_answer"


class AgentStep(BaseModel):
    iteration: int
    thought: str = ""
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""


class AgentResult(BaseModel):
    question: str
    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    hit_max_iterations: bool = False


class AgentState(TypedDict, total=False):
    question: str
    messages: list[dict[str, str]]
    steps: list[dict[str, Any]]
    iterations: int
    pending: dict[str, Any] | None
    answer: str | None
    seen_ids: list[str]
    hit_max: bool


def summarize_observation(obs: str, max_chars: int = 200) -> str:
    try:
        data = json.loads(obs)
    except json.JSONDecodeError:
        return obs[:max_chars]
    if isinstance(data, list):
        titles = [d.get("title", "") for d in data if isinstance(d, dict)][:3]
        return f"{len(data)} items" + (f": {'; '.join(titles)}" if titles else "")
    if isinstance(data, dict) and "error" in data:
        return f"error: {data['error']}"[:max_chars]
    return obs[:max_chars]


def build_agent(tools: list[BaseTool], llm: LLMClient, max_iterations: int = 6) -> Any:
    """Compile the LangGraph ReAct agent."""
    tool_map = {t.name: t for t in tools}
    system = prompts.AGENT_SYSTEM.format(tools=describe_tools(tools), max_iterations=max_iterations)

    def decide(state: AgentState) -> AgentState:
        if state.get("iterations", 0) >= max_iterations:
            return {"pending": None, "hit_max": True}
        try:
            d = llm.structured(state["messages"], AgentDecision, system=system)
        except LLMError as exc:
            return {
                "pending": None,
                "answer": None,
                "hit_max": False,
                "messages": state["messages"] + [{"role": "user", "content": f"error {exc}"}],
            }
        if d.action == FINAL or not d.action:
            return {"pending": None, "answer": d.answer or d.thought}
        return {"pending": d.model_dump()}

    def act(state: AgentState) -> AgentState:
        d = state["pending"] or {}
        name, args = d.get("action", ""), d.get("args") or {}
        it = state.get("iterations", 0) + 1
        if name in tool_map:
            try:
                obs = str(tool_map[name].invoke(args))
            except Exception as exc:  # noqa: BLE001 - invalid args from the LLM
                obs = json.dumps({"error": f"invalid arguments for {name}: {exc}"})
        else:
            obs = json.dumps({"error": f"unknown tool {name!r}; available: {sorted(tool_map)}"})
        seen = list(
            dict.fromkeys(
                state.get("seen_ids", []) + [normalize_id(m) for m in _ID_RE.findall(obs)]
            )
        )
        step = AgentStep(
            iteration=it,
            thought=d.get("thought", ""),
            tool=name,
            args=args,
            result_summary=summarize_observation(obs),
        ).model_dump()
        messages = state["messages"] + [
            {"role": "assistant", "content": json.dumps(d, ensure_ascii=False)},
            {
                "role": "user",
                "content": prompts.AGENT_OBSERVATION.format(tool=name, observation=obs),
            },
        ]
        return {
            "messages": messages,
            "steps": state.get("steps", []) + [step],
            "iterations": it,
            "seen_ids": seen,
            "pending": None,
        }

    def finalize(state: AgentState) -> AgentState:
        answer = state.get("answer")
        if not answer:
            msgs = state["messages"] + [
                {
                    "role": "user",
                    "content": "Maximum number of tool calls reached. Give the final answer now "
                    f"with action '{FINAL}', using only the observations above.",
                }
            ]
            try:
                answer = llm.structured(msgs, AgentDecision, system=system).answer
            except LLMError:
                answer = ""
        return {"answer": answer or "I could not find an answer with the available tools."}

    def route(state: AgentState) -> str:
        return "act" if state.get("pending") else "finalize"

    g = StateGraph(AgentState)
    g.add_node("decide", decide)
    g.add_node("act", act)
    g.add_node("finalize", finalize)
    g.add_edge(START, "decide")
    g.add_conditional_edges("decide", route, {"act": "act", "finalize": "finalize"})
    g.add_edge("act", "decide")
    g.add_edge("finalize", END)
    return g.compile()


def run_agent(
    question: str, tools: list[BaseTool], llm: LLMClient, max_iterations: int = 6
) -> AgentResult:
    """Run the agent and validate the citations of its final answer."""
    agent = build_agent(tools, llm, max_iterations)
    state = agent.invoke(
        {
            "question": question,
            "messages": [{"role": "user", "content": question}],
            "steps": [],
            "iterations": 0,
            "seen_ids": [],
        },
        {"recursion_limit": 2 * max_iterations + 5},
    )
    rep = validate_citations(state.get("answer") or "", state.get("seen_ids", []))
    return AgentResult(
        question=question,
        answer=rep.text.strip(),
        steps=[AgentStep(**s) for s in state.get("steps", [])],
        citations=rep.valid,
        invalid_citations=rep.invalid,
        hit_max_iterations=bool(state.get("hit_max")),
    )
