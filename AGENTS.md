# Contributor guidance

- When you touch Python modules under `scripts/`, keep the existing docstrings up to date and add context for any new helper functions you introduce.
- Document any new user-facing commands, configuration options, or data dependencies in `README.md` so downstream analysts can reproduce your steps.
- The project does not include automated tests yet. If you change matching logic, describe the manual validation or spot checks you performed in your PR summary.
- When you complete a task from `PLANNING.md`, update the checklist by marking it as done so future contributors have up-to-date context.
