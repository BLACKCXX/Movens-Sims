# Mount + joint object reconstruction v3

The new formal workflow prepares real frame 0 and the observed-joint mount prior, fits both object XY/absolute size/orientation jointly, requests a bounded Terra geometry review, executes supported repairs through local MCP, then fits appearance only after numeric/support and actual Terra geometry approval. It exports and reloads before the terminal read-only review. Static display, initial support, dynamic rest, numeric quality, Terra quality and processing completion are separate statuses.

## Run one episode

```bash
scripts/movens-sim run --config configs/mount_joint_v3.json --episode 0 --terra
scripts/movens-sim run --config configs/mount_joint_v3.json --episode 0 --terra --resume
scripts/movens-sim run --config configs/mount_joint_v3.json --episode 0 --local-only --dry-run
```

The default config's allowed list is only episode 0, with concurrency 1. No automatic batch execution is permitted. Output is `runs/mount_joint_v3/episode_000000`. The current model is exactly `gpt-6.1-terra`; credentials are read from the external dotenv/environment and never embedded in requests or logs. Missing credentials prepare requests/offline fixtures but cannot approve geometry or unlock formal appearance.

## Provenance and camera

The actual previous selection chooses `batch_final__oak_document_approximation__fovy_42p0`, not a hard-coded guess from the task text. FK is recomputed using observed frame-0 joints, the pinned official G1 waist chain and d435 installation reference. The OAK mount orientation is explicitly approximate. World camera position is approximately `[0.0484931, 0.00930084, 1.186277] m`, down angle `39.1107 deg`, nominal cloth clearance `0.409161 m`, vertical FOV `42 deg`; root, hardware identity, principal point and optical offset are not measured calibration.

Video/data revision is `d89c126a713c6632432a607c12661546ff4d6ea9`. The decoded full frame is 640×480; there is no additional crop or distortion correction. Frames 1/3/6 are checked with observed joints and tracked image features, not used as extra static layouts or claimed triangulation. Masks retain explicit identity/review records and the top 20% remains evaluated. Crops use the union window of real/render masks at the same pixel scale.

## Scene and search

Immutable original object assets are imported into the selected Mount environment; the inherited expanded table is explicitly frozen, not re-estimated. `SceneBackend` is the only editing implementation. Every scale recompiles fresh source MJCF, clones meshes used by the target, scales visual/collision/local offsets and inertials, and reconstructs names/data/renderers. Repeating `scale=1.2` is idempotent. Uniform scaling preserves source density with mass s³ and inertia s⁵. Positive bounded `plate.depth_ratio` is an additional evidence-gated vertical profile transformation; it is not an arbitrary geometry generator. Source mass/contact assumptions remain unmeasured.

XY spans the original physical support patch. Z is solved against actual saved visible triangles using collision vertices and face centroids, then visual gap and object intersections are checked. This is an effective geometric support model; the original flex state is separately used in a free-object gravity diagnostic. Plate collision/visual interior discrepancies documented in the old audit remain a limitation.

Search uses dispersed Sobol projection starts, bounded least-squares projection moments and multistart normalized Powell refinement with rendered silhouettes. Both scales are present from the first numeric stage. Geometry loss averages the two normalized object objectives: IoU, symmetric edge, center, log area and axes; background/top remain separately monitored. Inner plate RGB edges are reported at low confidence, not fabricated geometric annotations. Plate yaw stays fixed; limited tilt/depth need evidence. Independent camera hypotheses require shared two-object AND global residuals, keep FOV/table fixed and retain the fixed-Mount control.

New v3 tolerances are declared before the run: both IoU≥0.90, center/edge≤3px, area/axes≤5%, support gap≤2mm, penetration/intersection≤1mm. Nondegradation is checked per object and whole image; final global contradictions cannot be overridden by a good silhouette. These are engineering targets, not measured truth, and do not rewrite the old experiment's thresholds or conclusions.

## Task budget and review

The task group `pipeline_refactor_ep000000_v1` shares one SQLite ledger across all processes and run roots. Limits are 5 Terra requests, 4 geometry rounds, 1200 actual optimization renders (600 initial + 400 repairs + 200 appearance), 5000 projection residuals and 1800 active optimization seconds. Counters reserve before execution. Rollback does not roll them back. Outstanding time leases and requests remain conservatively charged on interruption. SDK implicit retries are disabled and the transport blocks unexpected additional HTTP sends. Every failed/timeout attempt has its own record. The last model slot is always terminal read-only.

The completed pilot used 225 renders, 2025 projection residuals and 3 model attempts (2 validated responses and 1 interrupted/unknown response). The first executed router failed to authorize an evidence-backed plate tilt fit, so the second geometry reservation produced no new renders. This is retained as an implementation failure, not presented as a failed physical camera hypothesis. The router and pre-fit permission guards were fixed and regression-tested. An explicit read-only code adoption verified identical frozen RGB and kept the original source snapshot/ledger. No geometry changed after terminal review. The only retry reviewed the identical frozen candidate. The original segmented optimizer timing was conservatively supplemented with omitted solver/cache/checkpoint overhead from durable round timestamps: 58.43s kernel time became 82.27s full accounted time, with no refunds or cap increase. Current timers wrap complete searches, enforce deadlines and avoid nested double counting.

API waits, static video output, immutable reload checks and diagnostic output are separately timed/counted. Cache identity includes the full scene/assets/params/camera, image size, observation, evaluator and renderer settings; hits do not consume another real render. Multiple views/frames are separate render evaluations when used to choose a candidate. No unbudgeted optimization is hidden behind previews.

Terra returns a strict diagnostic/finite repair plan; local gates retain final authority. Tools use the same backend and limits. The actual stdio server exposes inspect, pose/absolute scale update, joint fit, supported depth/tilt, appearance, independent camera hypothesis, evaluate, checkpoint and restore. It returns capability or stage errors rather than substituting operations.

Appearance uses the supported Phong lighting/material controls, texture color mixture and albedo, with geometry locked. PBR roughness is not advertised as effective in this renderer. Unresolved geometry or missing real review blocks this stage. Exported XML/state/assets and hashes bind final review and video; no subsequent visual parameter changes are allowed.

## Validation and artifacts

Tests cover real geometry behavior, repeated scale, shared assets, local support, rollback, phase permissions, pixel-identical portable reload, irreversible limits, failed SDK requests and resumption. Offline fixtures are only under test temporary directories. The real trial's `terra/`, `mcp_tool_calls.jsonl`, `budget_ledger.json` and `commands.jsonl` are the evidence for actual calls.

Deliverables include `comparison.png` (real / old batch / selected Mount-layout / joint geometry / final appearance or explicit blocked label), overlays, matching local crops, camera and object parameters/dimensions, raw metrics, candidate history, portable static and physics models, 150-frame 640×480 30FPS static video, separate 1-second free-body settle video, reload checks, final review, HTML/Markdown report and source/config/prompt snapshots. Consult the actual result rather than assuming implementation implies visual or physical success.

Historical batch and head-camera outputs stay unchanged. A new code/config identity cannot silently adopt old completed receipts; old snapshots remain available for exact legacy execution. The current pilot does not stress-test multi-episode concurrency or establish measured camera hardware or real dynamics.

## Frozen replay and package

```bash
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -B scripts/render_frozen_v3.py --run runs/mount_joint_v3/episode_000000 --output-directory runs/mount_joint_v3/frozen_replay
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -B scripts/check_v3_delivery.py --run runs/mount_joint_v3/episode_000000
```

The replay loads only the delivered portable XML/assets/state, performs no integration, fitting or API request, and compares the resulting pixels with the reviewed image. The delivery checker also verifies the original-source and protected-batch hashes on this server. Packaging includes current code, exact dependency pins, all episode evidence except search-image cache, portable scenes and the conserved task ledger; no credential file is included.
