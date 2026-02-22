
# episode_metrics.py
from collections import deque
import numpy as np
import builtins

#CC log episode metrics
if not hasattr(builtins, "_EP_METRICS_STATE"):
    builtins._EP_METRICS_STATE = {
        "EP_BUF": deque(maxlen=10000),
        "SUM": None,
        "STEPS": 0,
    }

EP_BUF = builtins._EP_METRICS_STATE["EP_BUF"]  # shared across all imports


def on_env_step(rew_vec, term, trunc, reason=None, agent_id=None):
    st = builtins._EP_METRICS_STATE
    if "AGENTS" not in st:
        st["AGENTS"] = {}

    #CC init this agent’s record if not seen before
    rec = st["AGENTS"].setdefault(agent_id, {"SUM": 0.0, "STEPS": 0})

    #CC accumulate reward and step count
    rec["SUM"] += float(np.sum(rew_vec))
    rec["STEPS"] += 1

    #CC when this agent terminates, push entry + reset it
    if bool(term or trunc):
        entry = {
            "agent_id": agent_id,
            "return": rec["SUM"],
            "steps": rec["STEPS"],
            "reason": reason or "unknown",
        }

        EP_BUF.append(entry)
        del st["AGENTS"][agent_id]


































    # v = np.asarray(rew_vec, dtype=np.float32)
    #
    # if st["SUM"] is None:
    #     st["SUM"] = np.zeros_like(v)
    #     st["STEPS"] = 0
    #     st["ALIVE_STEPS"] = np.zeros_like(v, dtype=np.int32)
    #
    # st["SUM"] += v
    # st["STEPS"] += 1
    # if alive_steps is not None:
    #     st["ALIVE_STEPS"] += np.asarray(alive_steps, dtype=np.int32)
    #
    # if bool(term or trunc):
    #     entry = {
    #         "returns": st["SUM"].copy(),
    #         "steps": int(st["STEPS"]),
    #         "alive_steps": st["ALIVE_STEPS"].copy() if st.get("ALIVE_STEPS") is not None else None,
    #         "reason": reason or "unknown",
    #     }
    #     print(f"[EP_MET PUSH] reason={entry['reason']} steps={entry['steps']} "
    #           f"alive_steps={entry['alive_steps']} returns={entry['returns'].tolist()}")
    #     EP_BUF.append(entry)
    #
    # if bool(term or trunc):
    #     EP_BUF.append({"returns": st["SUM"].copy(), "steps": int(st["STEPS"])})
    #     #print("[EP_END] returns", st["SUM"].tolist(), "steps", st["STEPS"])
    #     st["SUM"] = None
    #     st["STEPS"] = 0
    #     st["ALIVE_STEPS"] = None



