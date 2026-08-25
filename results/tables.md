<!-- results -->
| Arm | Accuracy | 95% CI | p50 | p95 | Prompt tokens | Retrieval hit | Grounded |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `base` | **56.8%** | [53.8, 59.9] | 47 ms | 55 ms | 113 | — | — |
| `rag-external` | **56.7%** | [53.6, 59.8] | 180 ms | 202 ms | 738 | 0.454 | 0.635 |
| `rag-parity` | **67.0%** | [64.1, 69.9] | 161 ms | 181 ms | 656 | 0.570 | 0.725 |

<!-- headline -->
| Arm | Accuracy | 95% CI | p50 | p95 | Prompt tokens | Retrieval hit | Grounded |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `base` | **56.8%** | [53.8, 59.9] | 47 ms | 55 ms | 113 | — | — |
| `rag-external` | **56.7%** | [53.6, 59.8] | 180 ms | 202 ms | 738 | 0.454 | 0.635 |

<!-- comparisons -->
| Comparison | Δ accuracy | Discordant (A/B) | p | Verdict |
| --- | ---: | :---: | ---: | --- |
| `base` vs `rag-external` | +0.001 | 121/120 | 1.0000 | not significant |
| `base` vs `rag-parity` | -0.102 | 88/190 | <0.0001 | **significant** |
| `rag-external` vs `rag-parity` | -0.103 | 75/178 | <0.0001 | **significant** |

<!-- per_subject -->
| Subject | n | `base` | `rag-external` | `rag-parity` |
| --- | ---: | ---: | ---: | ---: |
| Anatomy | 56 | 55.4% | 53.6% | 76.8% |
| Biochemistry | 41 | 70.7% | 68.3% | 95.1% |
| Dental | 313 | 50.2% | 48.6% | 51.1% |
| Gynaecology & Obstetrics | 54 | 63.0% | 53.7% | 68.5% |
| Medicine | 71 | 59.2% | 67.6% | 76.1% |
| Pathology | 80 | 75.0% | 80.0% | 83.8% |
| Pediatrics | 56 | 55.4% | 58.9% | 76.8% |
| Pharmacology | 58 | 62.1% | 70.7% | 81.0% |
| Physiology | 41 | 65.9% | 65.9% | 73.2% |
| Social & Preventive Medicine | 31 | 41.9% | 35.5% | 61.3% |
| Surgery | 89 | 51.7% | 51.7% | 64.0% |

<!-- cost -->
| Query volume | `base` | `rag-external` | `rag-parity` |
| ---: | ---: | ---: | ---: |
| 100 | $0.005 | $3.383 | $0.737 |
| 1,000 | $0.005 | $0.356 | $0.090 |
| 10,000 | $0.005 | $0.054 | $0.025 |
| 100,000 | $0.005 | $0.023 | $0.019 |
| 1,000,000 | $0.005 | $0.020 | $0.018 |
| 10,000,000 | $0.005 | $0.020 | $0.018 |

<!-- provenance -->
Generated from `results/runs/` by `make report`. Test split `9aac1bc01a70dcb6…` (n=1000), git `d51fa273daef`, NVIDIA A40, torch 2.11.0+cu128.
