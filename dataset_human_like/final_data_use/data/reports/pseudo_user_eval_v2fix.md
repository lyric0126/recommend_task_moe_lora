# Pseudo User Evaluation V2fix

## Comparison

| method | global | semantic | temporal | behavior | pseudo_users | confidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Random Mix V1 | 0.267952 | 0.210693 | 0.146181 | 0.699011 | - | - |
| V1 | 0.453148 | 0.248588 | 0.767702 | 0.820482 | 1751 | - |
| V2 | 0.316953 | 0.101169 | 0.583215 | 0.832109 | 5253 | {'loose': 5253} |
| V2-fix Random Mix | 0.395468 | 0.4664 | 0.199901 | 0.698657 | - | - |
| V2-fix | 0.680195 | 0.470436 | 0.774082 | 0.871076 | 1542 | {'strict': 1211, 'medium': 331} |

V2-fix better than V2: `True`
V2-fix close to or above V1: `True`
Full better than semantic-only: `True`
Full minus activity+temporal: `0.010795`
Coverage: `{'3': 1231, '4': 311}`
Ablation: `{"activity_temporal": 0.663218, "full": 0.674013, "semantic_only": 0.470436}`
Match reuse rate: `0.61141`
