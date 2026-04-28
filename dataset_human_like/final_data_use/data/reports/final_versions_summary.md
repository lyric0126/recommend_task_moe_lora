# Final Versions Summary

## Comparison

| version | pseudo users | interactions | confidence | global | recommended use | default |
| --- | ---: | ---: | --- | ---: | --- | --- |
| random | 1542 | 290161 | `{'random': 1542}` | 0.395468 | weakest random/V0 baseline | False |
| v1 | 1751 | 481974 | `{'loose': 1746, 'medium': 5}` | 0.453148 | conservative structured baseline | False |
| v2 | 5253 | 4841259 | `{'loose': 5253}` | 0.316953 | wide coverage degraded comparison set | False |
| v2fix_strict | 1211 | 215057 | `{'strict': 1211}` | 0.680195 | core high-quality set | False |
| v2fix_all | 1542 | 298777 | `{'medium': 331, 'strict': 1211}` | 0.680195 | main release / default recommended version | True |

Recommended mapping:

- baseline: `random` / `v0`
- conservative structured baseline: `v1`
- wide coverage degraded comparison: `v2`
- core high-quality version: `v2fix_strict`
- main release/default: `v2fix_all`
