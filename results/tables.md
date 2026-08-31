<!-- results -->
| Arm | Accuracy | 95% CI | p50 | p95 | Prompt tokens | Retrieval hit | Grounded |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `base` | **56.8%** | [53.8, 59.9] | 103 ms | 118 ms | 113 | — | — |
| `qlora-rag-parity` | **71.1%** | [68.3, 73.9] | 171 ms | 193 ms | 656 | 0.570 | 0.723 |
| `qlora-rag` | **61.4%** | [58.4, 64.4] | 179 ms | 195 ms | 738 | 0.454 | 0.628 |
| `qlora` | **62.9%** | [60.0, 65.9] | 102 ms | 112 ms | 113 | — | — |
| `rag-external` | **56.7%** | [53.6, 59.8] | 179 ms | 195 ms | 738 | 0.454 | 0.635 |
| `rag-parity` | **67.0%** | [64.1, 69.9] | 171 ms | 193 ms | 656 | 0.570 | 0.725 |

<!-- headline -->
| Arm | Accuracy | 95% CI | p50 | p95 | Prompt tokens | Retrieval hit | Grounded |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `base` | **56.8%** | [53.8, 59.9] | 103 ms | 118 ms | 113 | — | — |
| `qlora-rag` | **61.4%** | [58.4, 64.4] | 179 ms | 195 ms | 738 | 0.454 | 0.628 |
| `qlora` | **62.9%** | [60.0, 65.9] | 102 ms | 112 ms | 113 | — | — |
| `rag-external` | **56.7%** | [53.6, 59.8] | 179 ms | 195 ms | 738 | 0.454 | 0.635 |

<!-- comparisons -->
| Comparison | Δ accuracy | Discordant (A/B) | p | Verdict |
| --- | ---: | :---: | ---: | --- |
| `base` vs `qlora-rag-parity` | -0.143 | 69/212 | <0.0001 | **significant** |
| `base` vs `qlora-rag` | -0.046 | 96/142 | 0.0035 | **significant** |
| `base` vs `qlora` | -0.061 | 58/119 | <0.0001 | **significant** |
| `base` vs `rag-external` | +0.001 | 121/120 | 1.0000 | not significant |
| `base` vs `rag-parity` | -0.102 | 88/190 | <0.0001 | **significant** |
| `qlora-rag-parity` vs `qlora-rag` | +0.097 | 171/74 | <0.0001 | **significant** |
| `qlora-rag-parity` vs `qlora` | +0.082 | 153/71 | <0.0001 | **significant** |
| `qlora-rag-parity` vs `rag-external` | +0.144 | 217/73 | <0.0001 | **significant** |
| `qlora-rag-parity` vs `rag-parity` | +0.041 | 98/57 | 0.0013 | **significant** |
| `qlora-rag` vs `qlora` | -0.015 | 97/112 | 0.3328 | not significant |
| `qlora-rag` vs `rag-external` | +0.047 | 102/55 | 0.0002 | **significant** |
| `qlora-rag` vs `rag-parity` | -0.056 | 119/175 | 0.0013 | **significant** |
| `qlora` vs `rag-external` | +0.062 | 165/103 | 0.0002 | **significant** |
| `qlora` vs `rag-parity` | -0.041 | 117/158 | 0.0159 | **significant** |
| `rag-external` vs `rag-parity` | -0.103 | 75/178 | <0.0001 | **significant** |

<!-- per_subject -->
| Subject | n | `base` | `qlora-rag-parity` | `qlora-rag` | `qlora` | `rag-external` | `rag-parity` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anatomy | 56 | 55.4% | 89.3% | 60.7% | 67.9% | 53.6% | 76.8% |
| Biochemistry | 41 | 70.7% | 92.7% | 73.2% | 75.6% | 68.3% | 95.1% |
| Dental | 313 | 50.2% | 59.4% | 55.0% | 55.0% | 48.6% | 51.1% |
| Gynaecology & Obstetrics | 54 | 63.0% | 70.4% | 59.3% | 72.2% | 53.7% | 68.5% |
| Medicine | 71 | 59.2% | 77.5% | 71.8% | 69.0% | 67.6% | 76.1% |
| Pathology | 80 | 75.0% | 80.0% | 77.5% | 73.8% | 80.0% | 83.8% |
| Pediatrics | 56 | 55.4% | 76.8% | 57.1% | 64.3% | 58.9% | 76.8% |
| Pharmacology | 58 | 62.1% | 84.5% | 75.9% | 72.4% | 70.7% | 81.0% |
| Physiology | 41 | 65.9% | 75.6% | 68.3% | 65.9% | 65.9% | 73.2% |
| Social & Preventive Medicine | 31 | 41.9% | 64.5% | 38.7% | 48.4% | 35.5% | 61.3% |
| Surgery | 89 | 51.7% | 65.2% | 59.6% | 56.2% | 51.7% | 64.0% |

<!-- cost -->
| Query volume | `base` | `qlora-rag-parity` | `qlora-rag` | `qlora` | `rag-external` | `rag-parity` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | $0.011 | $20.976 | $23.621 | $20.249 | $3.383 | $0.738 |
| 1,000 | $0.011 | $2.115 | $2.380 | $2.035 | $0.356 | $0.091 |
| 10,000 | $0.011 | $0.229 | $0.256 | $0.214 | $0.053 | $0.026 |
| 100,000 | $0.011 | $0.040 | $0.043 | $0.032 | $0.023 | $0.020 |
| 1,000,000 | $0.011 | $0.021 | $0.022 | $0.013 | $0.020 | $0.019 |
| 10,000,000 | $0.011 | $0.019 | $0.020 | $0.012 | $0.020 | $0.019 |

<!-- provenance -->
Generated from `results/runs/` by `make report`. Test split `9aac1bc01a70dcb6…` (n=1000), git `f8dceb7868a0`, NVIDIA A40, torch 2.11.0+cu128.
