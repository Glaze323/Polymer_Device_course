# AI error ledger - G23 L1

| Date / phase | Tool + task | AI claim or output | Test/evidence that exposed the error | Correction and physical reason |
|---|---|---|---|---|
| 2026-09-21 / L1 | OpenAI Codex, numerical validation | The first pipeline required the 20,001-point numerical integral for n_F to agree with the infinite-upper-limit closed form within a relative tolerance of 2e-8. | The clean pipeline run failed at the exact G23 operating point with a relative difference of 3.10292e-7. | The validation tolerance was changed to 1e-6. The calculation still uses the guide's 20,001-point trapezoidal integration from E_F to 2.5 eV. A small difference from the analytic expression is expected because the numerical calculation is discretised and has a finite upper limit, whereas the closed form assumes an effectively infinite upper limit. The corrected tolerance remains much tighter than the precision needed for the reported L1 values. |
