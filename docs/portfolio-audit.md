# Final portfolio-readiness audit

Audit date: 2026-10-26. Scope: this Customer-Churn-Intelligence folder only.
No Git repository was initialized; no commit, remote, push or deployment was performed.

## Verified results

| Check | Actual result |
|---|---|
| Clean Windows environment | Created `.venv-audit` with Python 3.12, without shared site packages; installed requirements successfully |
| Complete clean-environment suite | **39 passed in 16.68 seconds** |
| Complete existing-environment suite | **39 passed in 18.95 seconds** |
| Dependency checks | `pip check` passed in both environments |
| README install command | Existing `.venv` installation from requirements succeeded |
| Streamlit launch | Clean-environment server on a free localhost port: health `ok`, root HTTP 200; audit process stopped afterwards |
| Dashboard behavior | AppTest covered all five pages, three model selections, four SQL views, empty filters and model-filter independence |
| Evaluation snapshots | Fresh clean-environment holdout metrics and AP match saved CSVs to numerical tolerance |
| Markdown links | All relative Markdown links resolve to existing files |
| Credential scan | No credential-pattern matches or `.env`, secrets.toml, PEM/key files found in project content |
| Placeholder scan | No TODO, TBD, FIXME, lorem ipsum or coming-soon markers found |
| Dataset | SHA-256 unchanged: `548b2b5276f789d545f671ed59836a5c3b5a56c72d02c060731b010cb4f843fe` |

The credential scan excluded environments/caches and checked common token/private-key/assignment
patterns; it is not a proof that every possible secret format is absent. Inspect staged content
before publication. No external service links are required; the only external-style README URL
is the local dashboard, available when its server is running.

## Changes made during this audit

- Reorganized README into features, technology stack, project structure, Windows setup, dataset,
  evaluation, screenshots, tests, repository hygiene, limitations and future improvements.
- Preserved detailed modeling analysis in `docs/model-review.md` and linked it from README.
- Expanded `.gitignore` for environment variants, secrets, local databases/sidecars and logs.
- Allowlisted only intentional synthetic evaluation CSV/JSON outputs under `artifacts/`.
- Added four SQL-page regression tests and corrected a formatting inconsistency in report code.
- Left generator, sample data, thresholds and model configuration unchanged.

## Publication exclusions and generated artifacts

The existing `artifacts/churn.sqlite` is a reproducible synthetic local database and is ignored,
along with SQLite journals/WAL/SHM files. The UI uses in-memory SQLite. Test XML and launch logs
are ignored. Both `.venv` and `.venv-audit`, bytecode and pytest caches are ignored.
Explicitly allowlisted CSV/JSON reports contain synthetic metrics, fold indices and provenance;
they are suitable as reproducibility snapshots. The audit environment remains local and ignored.

## Limits of verification

The literal `py -3.12` launcher command cannot run on this machine because `py` is absent from
PATH. Equivalent environment creation was verified with the installed Python executable. The
README explains the Python-launcher prerequisite and full-path fallback. Existing `.venv` commands
work without activation. Launch smoke testing used a free port so it would not disturb the existing
8501 server; the README documents port selection.

AppTest verifies server-side rendering and widget behavior, not a fresh browser visual pass at
every viewport. Synthetic provenance, train-only OOF threshold comparisons and fixed-cutoff test
metrics are explicit. Low recall, limited ranking performance and non-causal recommendations are
reported honestly; this is portfolio-ready as a demonstration, not a production churn system.

## Remaining checklist

- [ ] Add genuine dashboard screenshots if desired before publishing; no image placeholders exist.
- [ ] Decide whether to add an open-source license; no license was chosen on the owner's behalf.
- [ ] Review staged files, then initialize/commit and optionally connect an empty GitHub repository.
- [ ] Optional follow-up: fully lock transitive dependencies and add CI. These do not block local use.

No unresolved application or automated-test failures were found.
