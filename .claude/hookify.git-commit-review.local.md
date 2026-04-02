---
name: git-commit-review
enabled: true
event: bash
pattern: git\s+commit
action: warn
---

**Before committing — have you verified the following?**

- Did you cross-check all changed `.tex` files for correctness?
- Did you recompile the PDF and confirm the output looks right?
- Are there any stray `\\raggedbottom`, `[htbp]`, or other formatting issues?

If yes on all counts, proceed. Otherwise, compile first and check the PDF.
