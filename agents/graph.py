from langgraph.graph import END, StateGraph

from agents.planner_node import planner_agent
from agents.research_node import researcher_agent
from agents.router_node import router_agent
from agents.ticket_node import ticket_agent
from agents.state import TravelState


def _route_after_router(state: TravelState) -> list[str]:
    """Route requests after intent parsing.
    Ticket lookup is only entered for explicit ticket-query intent.
    """
    intent = (state.get("intent") or "").strip().lower()

    if intent == "need_ticket":
        return ["ticket_agent"]

    if intent in {"need_plan", "need_answer"}:
        return ["researcher"]

    # 其他意图（general_chat, other, need_more_info）直接结束
    return [END]


def _route_after_researcher(state: TravelState) -> str:
    """Skip planner for direct answers and continue for trip planning."""
    intent = (state.get("intent") or "").strip().lower()

    if intent == "need_plan":
        return "planner"

    return END


def _route_after_ticket(state: TravelState) -> str:
    """Ticket queries are terminal responses."""
    return END


def build_travel_graph():
    """Build the travel workflow graph."""
    workflow = StateGraph(TravelState)

    workflow.add_node("router", router_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("ticket_agent", ticket_agent)

    workflow.set_entry_point("router")

    # 条件边：router 根据意图决定执行路径
    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "ticket_agent": "ticket_agent",
            "researcher": "researcher",
            END: END,
        },
    )

    # researcher 完成后，need_plan 继续到 planner
    workflow.add_conditional_edges(
        "researcher",
        _route_after_researcher,
        {
            "planner": "planner",
            END: END,
        },
    )

    # ticket_agent 完成后直接结束
    workflow.add_conditional_edges(
        "ticket_agent",
        _route_after_ticket,
        {
            END: END,
        },
    )

    # planner 完成后到 END
    workflow.add_edge("planner", END)

    return workflow.compile()
