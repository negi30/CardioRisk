# UCI Heart Disease Dataset (Cleveland Subset)

## Dataset Description

The Heart Disease dataset originates from the **UCI Machine Learning Repository** and is one of the standard benchmark datasets in medical machine learning. This project uses the **Cleveland database**, which contains clinical records of patients undergoing evaluation for suspected coronary artery disease.

- **Repository ID**: 45
- **URL**: [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Total Records**: 303 patients
- **Total Input Features**: 13 clinical attributes
- **Target Feature**: `num` (angiographic disease status)

---

## Target Definition & Binary Conversion

In the original UCI dataset, `num` takes an integer value from `0` to `4`:
- `0`: No heart disease (angiographic diameter narrowing < 50%) — **164 patients (54.1%)**
- `1, 2, 3, 4`: Presence of coronary heart disease (angiographic diameter narrowing > 50% across 1 to 4 vessels) — **139 patients (45.9%)**

Following standard experimental literature, this project converts the target into a **binary clinical risk classification**:
$$\text{target} = \begin{cases} 0 & \text{if } num = 0 \text{ (No Heart Disease)} \\ 1 & \text{if } num > 0 \text{ (Heart Disease Present)} \end{cases}$$

---

## Feature Dictionary

| Feature | Type | Unit / Levels | Clinical Interpretation |
| :--- | :--- | :--- | :--- |
| **`age`** | Numerical | Years (29–77) | Patient chronological age |
| **`sex`** | Categorical | `0`: Female, `1`: Male | Biological sex |
| **`cp`** | Categorical | `1`: Typical Angina<br>`2`: Atypical Angina<br>`3`: Non-anginal Pain<br>`4`: Asymptomatic | Chest pain type reported during triage |
| **`trestbps`** | Numerical | mm Hg (94–200) | Resting systolic blood pressure on hospital admission |
| **`chol`** | Numerical | mg/dl (126–564) | Serum cholesterol measurement |
| **`fbs`** | Categorical | `0`: $\le 120$ mg/dl, `1`: $> 120$ mg/dl | Fasting blood sugar indicator |
| **`restecg`** | Categorical | `0`: Normal<br>`1`: ST-T wave abnormality<br>`2`: Left ventricular hypertrophy (Estes' criteria) | Resting 12-lead electrocardiographic evaluation |
| **`thalach`** | Numerical | bpm (71–202) | Maximum heart rate achieved during exercise stress test |
| **`exang`** | Categorical | `0`: No, `1`: Yes | Exercise-induced angina |
| **`oldpeak`** | Numerical | mm (0.0–6.2) | ST depression induced by exercise relative to rest |
| **`slope`** | Categorical | `1`: Upsloping, `2`: Flat, `3`: Downsloping | Slope of peak exercise ST segment |
| **`ca`** | Numerical | 0, 1, 2, 3 | Number of major coronary vessels colored by fluoroscopy (4 missing values imputed via median) |
| **`thal`** | Categorical | `3.0`: Normal<br>`6.0`: Fixed defect<br>`7.0`: Reversible defect | Thallium scintigraphy nuclear stress test results (2 missing values imputed via mode) |

---

## Citation

```bibtex
@misc{uci_heart_disease_45,
  author       = {Janosi, Andras and Steinbrunn, William and Pfisterer, Matthias and Detrano, Robert},
  title        = {{Heart Disease}},
  year         = {1988},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C52P4X}
}
```
