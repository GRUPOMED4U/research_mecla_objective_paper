**Baseline**:

Number of entities with mutually exclusive tags that overlap:

|**Pair of labels**                 | **Number of entities**|
|-----------------------------------|-----------------------------|
|('HER2_NEGATIVO', 'HER2_POSITIVO') | 108                         |
|('RP_NEGATIVO', 'RP_POSITIVO')     | 49                          |
|('RE_NEGATIVO', 'RE_POSITIVO')     | 18                          |

---

**BCE + MECLA**:

Number of entities with mutually exclusive tags that overlap:

|**Pair of labels**                 | **Number of entities**|
|-----------------------------------|-----------------------------|
|('HER2_NEGATIVO', 'HER2_POSITIVO') | 44 (59% reduction)          |
|('RP_NEGATIVO', 'RP_POSITIVO')     | 14 (71% reduction)          |
|('RE_NEGATIVO', 'RE_POSITIVO')     | 9  (50% reduction)          |

---

**BCE + Pairwise MECLA**:

Number of entities with mutually exclusive tags that overlap:

|**Pair of labels**                 | **Number of entities**|
|-----------------------------------|-----------------------------|
|('HER2_NEGATIVO', 'HER2_POSITIVO') | 35 (68% reduction)          |
|('RP_NEGATIVO', 'RP_POSITIVO')     | 24 (51% reduction)          |
|('RE_NEGATIVO', 'RE_POSITIVO')     | 15 (17% reduction)          |

---

**BCE + Grouped Sofmax**:

Number of entities with mutually exclusive tags that overlap:

|**Pair of labels**                 | **Number of entities**|
|-----------------------------------|-----------------------------|
|('HER2_NEGATIVO', 'HER2_POSITIVO') | 0 (100% reduction)          |
|('RP_NEGATIVO', 'RP_POSITIVO')     | 0 (100% reduction)          |
|('RE_NEGATIVO', 'RE_POSITIVO')     | 0 (100% reduction)          |

---

**BCE + Grouped Sofmax as penalty**:

Number of entities with mutually exclusive tags that overlap:

|**Pair of labels**                 | **Number of entities**|
|-----------------------------------|-----------------------------|
|('HER2_NEGATIVO', 'HER2_POSITIVO') | 0 (100% reduction)          |
|('RP_NEGATIVO', 'RP_POSITIVO')     | 0 (100% reduction)          |
|('RE_NEGATIVO', 'RE_POSITIVO')     | 0 (100% reduction)          |