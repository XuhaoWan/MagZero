# Data directory

Keep large datasets outside version control. A typical local layout is:

```text
data/
├── raw/          # original .pt HeteroData files or source exports
└── processed/    # derived splits, caches, or analysis outputs
```

The repository `.gitignore` excludes these directories by default.
