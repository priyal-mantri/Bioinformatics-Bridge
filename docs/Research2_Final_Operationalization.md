# Research 2 — Final Literature-to-NHANES Operationalization Framework

**Status: Final operationalization baseline**

This document converts the literature-to-NHANES audit into a variable-level framework for Research 2. The classifications describe what each NHANES variable can operationalize; they do not claim that a variable measures an individual's Ayurvedic constitution.

## Evidence tiers

- **A — Strongly supported:** direct empirical finding in the source literature measuring the same parameter in a corresponding Ayurvedic subgroup.
- **B — Reasonable but indirect:** biologically relevant proxy for a system-level phenotype, but not a direct measurement of the Ayurvedic construct or lifelong Prakriti.
- **C — Weak / speculative:** plausible physiological dimension without direct empirical Dosha-specific evidence, or with substantial interpretive limitations.
- **D — Unsupported / unavailable:** no suitable NHANES proxy exists, or the domain/variable was excluded from the feature set.

## Final variable-level mapping

| System | NHANES variable | What it measures | Literature rationale | Final status | Key limitation |
|---|---|---|---|---|---|
| Body composition | BMXBMI | Body Mass Index | Body-size / weight phenotype | B — indirect | Current body mass is influenced by age, diet, activity and disease; not a direct measure of lifelong Prakriti. |
| Body composition | BMXWAIST | Waist circumference | Abdominal/body-composition phenotype | B — indirect | Plausible phenotype proxy, but not established as a Dosha biomarker. |
| Body composition | DXDTOBMD | Total-body bone mineral density | Structural body phenotype | C — exploratory | No direct Dosha-specific BMD association was established in the audited sources. |
| Body composition | DXDTOPF | Total-body percent fat | Body-composition phenotype | B — indirect | Measures current adiposity, not constitutional body type. |
| Body composition | DXDTOLE | Total-body lean mass | Body-composition / muscularity phenotype | B — indirect | Useful physiological dimension, but no direct numerical Dosha mapping. |
| Cardiovascular / autonomic | BPXPLS | Resting pulse rate | Movement / pace and autonomic cardiovascular phenotype | B — indirect | Single examination pulse is affected by acute anxiety, caffeine, fitness and temperature. |
| Cardiovascular / autonomic | Avg_Systolic_BP | Average systolic blood pressure | Vascular / cardiovascular state | B — indirect | Audited literature did not establish a clear baseline BP difference across healthy Prakriti groups. |
| Cardiovascular / autonomic | Avg_Diastolic_BP | Average diastolic blood pressure | Vascular / cardiovascular state | B — indirect | Medication, age and current health can strongly affect the measurement. |
| Glucose / metabolic | LBXGLU | Blood glucose | Glucose handling / energy metabolism | B — indirect | Current glycemic physiology, not subjective digestive capacity or direct PGM1 activity. |
| Glucose / metabolic | LBXIN | Insulin | Endocrine glucose regulation | B — indirect | Physiological proxy, not direct appetite, digestion or lifelong constitutional type. |
| Lipid metabolism | LBXTC | Total cholesterol | Reported lipid profile differences | A — direct, male-specific | Higher baseline TC was reported in Kapha males; female differences were absent in the cited cohort. |
| Lipid metabolism | LBDHDD | HDL cholesterol | Reported lipid profile differences | A — direct, male-specific | Lower HDL was reported in Kapha males; do not generalize as universal. |
| Lipid metabolism | LBXSTR | Triglycerides | Reported lipid profile differences | A — direct, male-specific | Higher TG was reported in Kapha males; cited sample was small and restricted to healthy young adults. |
| Hepatic / biliary | LBXSATSI | ALT | Liver biochemical phenotype | A — direct, male-specific | Higher ALT/SGPT was reported in Kapha males; retain sex/population qualification. |
| Hepatic / biliary | LBXSTB | Total bilirubin | Biliary / pigmentation-related physiology | B — indirect | Functional alignment exists, but cited empirical source did not show baseline bilirubin elevation. |
| Hepatic / biliary | LBXSAL | Albumin | Hepatic protein synthesis / systemic protein status | C — weak/speculative | General clinical relevance does not establish a Dosha-specific association. |
| Hepatic / biliary | LBXSTP | Total protein | Systemic protein status | C — weak/speculative | No direct Dosha-specific association was established. |
| Renal / waste clearance | LBXSUA | Serum uric acid | Renal/metabolic waste phenotype | A — direct, male-specific | Higher uric acid was reported in Kapha males. |
| Renal / waste clearance | LBXSCR | Creatinine | Renal filtration / waste clearance | B — indirect | Renal marker, but no baseline creatinine difference was established across Prakritis. |
| Renal / waste clearance | LBXSBU | Blood urea nitrogen | Nitrogenous waste clearance | B — indirect | Physiologically relevant, but no direct Dosha-specific baseline association. |
| Electrolytes / minerals | LBXSPH | Serum phosphorus | Mineral physiology | B — indirect, female-specific | Higher phosphorus was reported in Pitta females, with substantial intragroup variance. |
| Electrolytes / minerals | LBXSCA | Serum calcium | Mineral / electrolyte homeostasis | C — weak/speculative | Tightly homeostatically regulated; no established Dosha-specific association. |
| Electrolytes / minerals | LBXSNASI | Serum sodium | Extracellular electrolyte homeostasis | C — weak/speculative | No established Dosha-specific association; strong homeostatic regulation. |
| Electrolytes / minerals | LBXSKSI | Serum potassium | Intracellular electrolyte physiology | C — weak/speculative | No established Dosha-specific association; not a Dosha biomarker. |

## Unsupported domains

| Domain | Reason |
|---|---|
| **Skin & hair** | No suitable NHANES measure for skin hydration, sebum or hair quality. |
| **Hematology & coagulation** | CBC and prothrombin-time findings exist in the literature audit but were excluded from the Research 2 feature set. |
| **Climate / thermoregulation** | No suitable NHANES measure of subjective temperature preference or core thermoregulatory response. |
| **Cognition / memory / behavior** | No appropriate NHANES psychometric measures for learning speed, memory dynamics or communication style. |
| **Genomics / transcriptomics** | Research 2 NHANES data do not contain the genomic, transcriptomic or SNP measurements discussed in the literature. |

## Derived and contextual variables

**Excluded from primary clustering:** `HOMA_IR`, `TC_HDL_ratio`, and `TG_HDL_ratio`. They remain available for downstream cluster profiling and interpretation.

**Contextual / sensitivity variables:** `RIDAGEYR`, `RIAGENDR`, and `RIDRETH3` are not primary clustering features.

## Non-negotiable interpretation rules

1. NHANES variables are continuous physiological dimensions, not 'Dosha biomarkers'.
2. Feature selection was based on biological-system coverage independently of expected Dosha labels, preserving anti-circularity.
3. An A-level result means the cited literature measured the same parameter in a specified subgroup; it does not mean the NHANES variable identifies an individual's Prakriti.
4. NHANES is cross-sectional. Measurements can reflect age, diet, physical activity, medication, disease and acute state.
5. Sex-specific evidence must remain sex-specific. Male-derived lipid, ALT and uric-acid findings should not be presented as universal rules.
6. HOMA_IR, TC_HDL_ratio and TG_HDL_ratio are retained for downstream interpretation but excluded from primary clustering to avoid mathematical collinearity.
7. Age, sex and race/ethnicity are contextual variables for filtering and sensitivity analysis, not primary clustering features.
8. Post-hoc characterization may compare cluster profiles with literature-derived phenotype patterns; it must not retroactively label clusters as Vata, Pitta or Kapha.

## Final analytical logic

**Ayurvedic literature → reported phenotype characteristics → independently selected measurable NHANES dimensions → unsupervised clustering → post-hoc multidimensional phenotype characterization**

The clustering step does not receive Vata/Pitta/Kapha labels. The interpretation step asks whether observed cluster profiles are compatible with, partially resemble, or differ from literature-described phenotype characteristics.

## Main limitations

1. **Lifelong Prakriti vs current physiological state:** NHANES is cross-sectional.
2. **Sex-specific evidence:** several strong biochemical associations were observed in males; female results often showed greater intragroup variance.
3. **Population mismatch:** cited empirical studies used restricted Indian populations, including healthy young adults and, in some analyses, single-Dosha-dominant participants.
4. **Proxy limitations:** a physiological variable can represent a measurable dimension without being equivalent to the traditional construct.
5. **No biomarker claim:** no individual NHANES feature should be called a Vata, Pitta or Kapha biomarker.

## Source basis

- `operationalization_audit.pdf`
- `operational_phenotype_framework.pdf`
- `ayurvedic_phenotype_extraction.pdf` / `ayurvedic_phenotype_framework_table.pdf`
- `Prakriti.pdf`
- Official NHANES 2017–2018 codebooks, including `BIOPRO_J.pdf`
- `variable_selection_analysis.md`

## Locking statement

This table should be treated as the **locked operationalization baseline** for downstream Research 2 analysis unless a new source is added and the mapping is explicitly re-audited. Changes should be documented rather than silently made during phenotype characterization.
