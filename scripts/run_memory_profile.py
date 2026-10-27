import asyncio
import tracemalloc
import time
from zelyr.agents.orchestrator_agent import OrchestratorAgent
from zelyr.config import get_config

async def profile_memory():
    print("[*] Starting Zelyr Memory Leak Profiling...")
    tracemalloc.start()
    
    agent = OrchestratorAgent()
    config = get_config()
    
    # Take initial snapshot
    snapshot1 = tracemalloc.take_snapshot()
    
    # Run 50 iterations of orchestrator instantiation and basic planning
    print("[*] Running 50 iterations of Orchestrator planning...")
    for i in range(50):
        # We don't execute the full loop to avoid LLM calls in this basic test,
        # just the plan creation and memory parsing
        plan = agent._create_plan(f"Task {i}")
        _ = agent._build_messages(f"Task {i}", plan, [], f"session_{i}")
        
    snapshot2 = tracemalloc.take_snapshot()
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("[*] Top 10 memory differences:")
    for stat in top_stats[:10]:
        print(stat)
        
    print("[*] Profiling complete.")

if __name__ == "__main__":
    asyncio.run(profile_memory())
