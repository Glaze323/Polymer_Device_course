# G23 L1 carrier counting

This folder contains the reproducible L1 analysis for the regular G23 dataset.

Run from the repository root with Git and Python on `PATH`:

```text
python G23_L1/reproduce.py
```

The script verifies the supplied L1 file checksums, performs the required 20,001-point numerical integrations, checks the reference ratios and writes all checkpoint outputs. It also creates `G23_L1_Checkpoint.zip` containing the four files required by the practical methods guide.

Required Python packages: `numpy` and `matplotlib`.

If Git is installed but not available on `PATH`, set the `GIT_EXE` environment variable to the full path of `git.exe` before running the command.

For a managed build whose source commit is supplied externally, set `PIPELINE_COMMIT` to that full 40-character commit SHA. The value is validated before it is written to `results.json` and `commit_hash.txt`.
