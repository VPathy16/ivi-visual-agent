"""Run the IVI visual agent on the AndroidWorld benchmark.

This mirrors ``android_world/minimal_task_runner.py`` but swaps in
:class:`IviVisualAgent`. It requires ``android_world`` installed and a running
Android emulator (see the README in this directory). Example::

    python -m ivi_agent.integrations.androidworld.run_benchmark \
        --config config.json --console-port 5554 --n-tasks 20

Omit ``--task`` to sample random tasks, or pass a task name to run one.
"""

from __future__ import annotations

import argparse
import json
import random


def main() -> int:
    parser = argparse.ArgumentParser(description="Run IVI visual agent on AndroidWorld")
    parser.add_argument("--config", help="IVI agent config JSON (model, grounding_mode, ...)")
    parser.add_argument("--console-port", type=int, default=5554, help="Emulator console port")
    parser.add_argument("--adb-path", default=None, help="Path to adb (optional)")
    parser.add_argument("--task", default=None, help="Specific task name; omit to sample")
    parser.add_argument("--n-tasks", type=int, default=1, help="How many tasks to run")
    parser.add_argument("--max-steps-scale", type=float, default=10.0,
                        help="Steps allowed per task = ceil(task.complexity * scale)")
    parser.add_argument("--emulator-setup", action="store_true",
                        help="Run AndroidWorld's one-time app setup on the emulator")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Imported here so --help works without android_world installed.
    from android_world import registry
    from android_world.env import env_launcher

    from ivi_agent.integrations.androidworld import IviVisualAgent

    if args.seed is not None:
        random.seed(args.seed)

    launch_kwargs = {"console_port": args.console_port, "emulator_setup": args.emulator_setup}
    if args.adb_path:
        launch_kwargs["adb_path"] = args.adb_path
    env = env_launcher.load_and_setup_env(**launch_kwargs)

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

    if args.task:
        if args.task not in aw_registry:
            raise SystemExit(f"Task {args.task!r} not found in registry.")
        task_types = [aw_registry[args.task]]
    else:
        task_types = random.sample(
            list(aw_registry.values()), k=min(args.n_tasks, len(aw_registry))
        )

    agent = IviVisualAgent(env, config_path=args.config)

    results = []
    passed = 0
    try:
        for task_type in task_types:
            params = task_type.generate_random_params()
            task = task_type(params)
            env.reset(go_home=True)
            task.initialize_task(env)
            agent.reset(go_home=False)
            max_steps = max(1, int(task.complexity * args.max_steps_scale))
            agent.set_max_steps(max_steps)

            print(f"\n=== {task_type.__name__} ===\nGoal: {task.goal}")
            is_done = False
            for _ in range(max_steps):
                response = agent.step(task.goal)
                if response.done:
                    is_done = True
                    break
            success = bool(is_done and task.is_successful(env) == 1)
            passed += int(success)
            print("Result:", "PASS" if success else "FAIL")
            results.append({"task": task_type.__name__, "goal": task.goal, "success": success})
    finally:
        env.close()

    total = len(results)
    summary = {
        "passed": passed,
        "total": total,
        "success_rate": round(passed / total, 4) if total else 0.0,
        "results": results,
    }
    print("\n" + json.dumps(summary, indent=2))
    return 0 if total and passed == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
