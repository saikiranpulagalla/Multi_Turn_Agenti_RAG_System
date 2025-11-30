"""
Planner agent: turns synthesis output into an action plan (study plan, checklist, reminders).
Optionally interacts with Execution Agent for scheduling.
"""
from core.llm_client import call_llm
from core.utils.logging_utils import trace_event

class PlannerAgent:
    def __init__(self, memory_service=None):
        self.mem = memory_service

    def create_plan(self, summary_text: str, goal: str = "Create a 7-day study plan"):
        prompt = f"""
        Based on the following summary, create a clear step-by-step plan with timelines, priority, and
        a checklist. Summary:
        {summary_text}

        Output JSON with fields: plan_title, days: [{'{'}day:int, tasks:[str]{'}'}], total_hours_estimate, priorities.
        """
        resp = call_llm(prompt, temperature=0.0, max_tokens=400)
        trace_event("planner", "plan_generated", {"goal": goal})
        # return raw response for now (caller can parse JSON)
        return {"raw_plan": resp}
