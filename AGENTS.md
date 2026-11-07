# Movens-Sim

Read agent.md before changes. Only reconstruct the initial static table, apple, plate and ego camera. No robot policies, grasp trajectories or training. Movens-Interact is a read-only source; never change or stop its other jobs. Keep real and simulation image identities explicit. API credentials come from an external dotenv file/environment, never source control or logs. Use .venv/bin/python -B. Preserve every rejected/failed candidate and do not weaken batch thresholds or replace frozen episodes. Static display and physical diagnostics are separate artifacts and statuses.
