# Implementation state

Project: /root/autodl-tmp/Movens-Sim. Source: /root/autodl-tmp/Movens-Interact, read only.
Initial audit found active reconstructed batch outputs/appletoplate_tableup005mm_20261105T020624Z and locally available raw ego video at revision d89c126a713c6632432a607c12661546ff4d6ea9. Selection uses complete registered scene inputs, never grasp success. Independent Python 3.12 environment, MuJoCo 3.7.0.

Credential path /root/autodl-tmp/api.env is external. Its legacy `api` variable is mapped to OPENAI_API_KEY only in process memory. No credential values may appear here.

## Current API model

User requested `gpt-6.1-terra`. `configs/batch_v2.json` now selects that exact API ID; `scripts/movens-sim` defaults to this configuration for new work in runs/batch_v2. Explicit --config overrides the launcher default. Historical batch_v1 config/results still record gpt-4.1-2025-04-14; resume them with `scripts/movens-sim run --config configs/batch_v1.json --resume --gpt`. The Python module/console entry retains its legacy default, so use the main shell launcher or pass the desired --config explicitly. Model availability and one read-only image/strict-JSON compatibility call are recorded in docs/model_switch_terra_check.json. No scene optimization is part of this model switch.

## Completed first batch

Frozen distinct episodes 0,1,2,3,4, all processed in runs/batch_v1/episode_XXXXXX. Each has portable static and physical XML/state/assets, actual original real frame0, pixel masks, baseline/optimized/overlay images, ego_static.mp4 (5s,30FPS,640x480), settle_diagnostic.mp4 (1s gravity,30FPS), per-object metrics, candidate histories and provenance. Input=5, processed=5, numeric image gate=4, full visual qualified=0, physical qualified=0, review=5. All are analysis candidates; do not label them real capture or validated real dynamics. Source plate remains fixed and this limitation is explicit.

Main report runs/batch_v1/index.html, results.csv, overview.jpg and docs/first_batch_report.md. Actual Responses API with gpt-4.1-2025-04-14 verified for all five. v1 semantic error retained/rejected; diagnostic_v2 checked on first two and frozen for last three, all five valid. GPT suggestions never override local/final qualification.

Commands for historical results: scripts/movens-sim run --config configs/batch_v1.json --resume --gpt; scripts/movens-sim report --config configs/batch_v1.json. New work: scripts/movens-sim run --config configs/batch_v2.json --episode 0 --gpt. --local-only, --dry-run, --stage and bounded --concurrency1..4 are implemented. Initial batch ran sequentially; concurrency>1 not stress-tested.

Current maintenance version 0.1.1. Original numeric/physics execution source preserved in runs/batch_v1/code_executed_v0.1.0. Deterministic, explicit per-episode compatibility upgrades preserve original identities/receipts in compatibility/ and allow checked resume with current code. Do not silently adopt incompatible source/config/prompt changes. Tests pass; delivery_checks.json records independent load/render/video validation. MuJoCo3.7 MjSpec fails to preserve one source flex bookkeeping case; physical XML uses verified direct MjModel compilation, static display uses MjSpec.

Known limits: red/yellow apple appearance and cloth wrinkles not reconstructed; table extents ambiguous and sometimes at bounds; inherited post-settle soft geometry produces poor rest behavior after layout changes. No robot trajectories, policies, model training, mass/friction/contact fitting or DA3 deployment. Next work should address these explicit visual/support limitations in a new version/run, preserving the five frozen episodes and all failure evidence.

Additional plate-support ray audit sampled center plus four radii (65 rays), comparing inherited convex collision pieces against the visual concave plate. Maximum mismatch ~7.26mm, recorded per sample in plate_support_surface.json and summarized in plate_surface_review.json/result.json. Do not describe the plate collision interior as verified from its visual mesh.
