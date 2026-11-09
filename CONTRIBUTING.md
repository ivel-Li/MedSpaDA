# Contributing

Please describe the problem, the changed behavior, and how you checked it in
each pull request. Keep model changes separate from data preprocessing or
evaluation changes so that experiments remain comparable.

Run the CPU suite before submitting:

```bash
python -m unittest discover -s tests -v
```

For changes to native SAM3 integration, also run a small CUDA smoke test on
licensed data and report the dependency versions. Do not commit images,
patient lists, checkpoints, credentials, or generated experiment artifacts.
