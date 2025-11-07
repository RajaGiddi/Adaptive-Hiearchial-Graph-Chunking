# save as debug_adk_sig.py and run: python debug_adk_sig.py

import inspect
from google.adk.agents import LlmAgent

agent = LlmAgent(
    model="gemini-2.5-flash",
    name="inspect",
    description="inspect",
    instruction="inspect",
    tools=[],
)

print("AGENT TYPE:", type(agent))

# 1) Show the actual run_live implementation
try:
    src = inspect.getsource(agent.run_live)
    print("\n=== run_live source ===")
    print(src)
except Exception as e:
    print("could not get source for run_live:", e)

# 2) Show the *private* impl it is calling
try:
    src = inspect.getsource(agent._run_live_impl)
    print("\n=== _run_live_impl source ===")
    print(src)
except Exception as e:
    print("could not get source for _run_live_impl:", e)

# 3) Show the type of the first param of _run_live_impl
sig = inspect.signature(agent._run_live_impl)
print("\n=== _run_live_impl signature ===")
print(sig)

# 4) Show any ADK "State"/"Context" types sitting next to it
import google.adk.agents as agents_mod
print("\n=== dir(google.adk.agents) ===")
print([n for n in dir(agents_mod) if "State" in n or "Context" in n or "Invocation" in n])

# 5) If there is a base state class on the agent, show its type
for attr in dir(agent):
    if "state" in attr.lower():
        print("STATE-LIKE ATTR:", attr, "->", getattr(agent, attr))
