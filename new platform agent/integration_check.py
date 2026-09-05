from core.env import MockPlatformEnv
from core.interfaces import PerceptionModule, PlanningModule, PolicyModule, PerceptionOutput, Plan, Action
from member4_llm_rag.orchestrator import run_episode
from member4_llm_rag.mission import RAGMissionInterpreter


class P(PerceptionModule):
    def process(self, frame, history):
        return PerceptionOutput(entities=[], player_pose=None, predicted_trajectories={}, confidence=0.0)


class Pl(PlanningModule):
    def plan(self, state, guidance=None):
        return Plan(waypoints=[state.goal], committed_action=Action.RIGHT, risk_score=0.0)


class Po(PolicyModule):
    def act(self, state, plan=None, perception=None):
        return Action.RIGHT


env = MockPlatformEnv()
final_state = run_episode(
    env, P(), Pl(), Po(), RAGMissionInterpreter(),
    mission_text="reach the goal, avoid hazards",
    guidance_refresh_every=30, max_steps=60,
)
print("final tick:", final_state.tick)
