# Reporte de Comparación: Seleccionados vs No Seleccionados
**Población Total:** N=48 | **Seleccionados (Muestra):** n=20 | **No Seleccionados:** n=28

## 1. Comparación de Variables Numéricas (Edad y Score de Riesgo)

| Variable | Seleccionados (n=20) <br> Media (DE) | No Seleccionados (n=28) <br> Media (DE) | SMD | p-value (t-test) |
| :--- | :---: | :---: | :---: | :---: |
| **Edad** | 7.40 (1.23) | 7.50 (1.17) | -0.083 | 0.7785 |
| **Score_Riesgo** | 3.90 (1.68) | 4.36 (1.97) | -0.250 | 0.3920 |


## 2. Comparación de Variables Categóricas (Sociodemográficas y Riesgos)

| Variable | Categoría | Seleccionados (n=20) <br> N (%) | No Seleccionados (n=28) <br> N (%) | SMD | p-value (Chi2 / Fisher) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Sexo** | femenino | 7 (35.0%) | 16 (57.1%) | -0.456 | 0.2221 |
|  | masculino | 13 (65.0%) | 12 (42.9%) | +0.456 |  |
| | | | | | |
| **Sector** | Centro A | 6 (30.0%) | 5 (17.9%) | +0.288 | 0.3379 (Chi2, ⚠️ celdas < 5) |
|  | Centro B | 4 (20.0%) | 4 (14.3%) | +0.152 |  |
|  | Centro C | 2 (10.0%) | 3 (10.7%) | -0.023 |  |
|  | Norte A | 0 (0.0%) | 3 (10.7%) | -0.490 |  |
|  | Norte B | 1 (5.0%) | 6 (21.4%) | -0.500 |  |
|  | Sur | 7 (35.0%) | 7 (25.0%) | +0.220 |  |
| | | | | | |
| **Es_Expuesto** | NO | 9 (45.0%) | 8 (28.6%) | +0.346 | 0.3858 |
|  | SI | 11 (55.0%) | 20 (71.4%) | -0.346 |  |
| | | | | | |
| **Riesgo_Pb** | NO | 3 (15.0%) | 3 (10.7%) | +0.128 | 0.6830 (Fisher) |
|  | SI | 17 (85.0%) | 25 (89.3%) | -0.128 |  |
| | | | | | |
| **Riesgo_Hg** | NO | 12 (60.0%) | 0 (0.0%) | +1.732 | 0.0000 (Fisher) |
|  | SI | 8 (40.0%) | 28 (100.0%) | -1.732 |  |
| | | | | | |
| **Riesgo_Cd** | NO | 5 (25.0%) | 5 (17.9%) | +0.175 | 0.8101 (Chi2, ⚠️ celdas < 5) |
|  | SI | 15 (75.0%) | 23 (82.1%) | -0.175 |  |


## 3. Exploración de Variabilidad de Riesgo en el Grupo Seleccionado (n=20)

Para realizar regresiones estadísticas de manera robusta en etapas posteriores, los predictores (variables de riesgo de metales) no deben ser constantes en la muestra analítica.

| Variable de Riesgo | Cantidad de 'SI' | Cantidad de 'NO' | ¿Tiene Variabilidad? | Estado / Alerta en Muestra |
| :--- | :---: | :---: | :---: | :--- |
| `Riesgo_Pb` | 17 | 3 | Sí | ⚠️ **Baja variabilidad** (sólo 3 observaciones en una categoría). Puede causar problemas de convergencia en modelos logísticos. |
| `Riesgo_Hg` | 8 | 12 | Sí | 🟢 **Variabilidad adecuada**. |
| `Riesgo_Cd` | 15 | 5 | Sí | 🟢 **Variabilidad adecuada**. |


## 4. Diagnóstico de Sesgo de Selección

El sesgo de selección se evalúa examinando si hay desequilibrios significativos entre el grupo seleccionado y el no seleccionado. Generalmente:
* $|SMD| > 0.1$ indica desequilibrio leve.
* $|SMD| > 0.25$ indica desequilibrio importante (posible sesgo).

⚠️ **Posible sesgo de selección detectado en las siguientes variables:**

| Variable | SMD Máximo | Nivel de Desequilibrio |
| :--- | :---: | :--- |
| `Riesgo_Hg` | 1.732 | Importante ($>0.25$) |
| `Sector` | 0.500 | Importante ($>0.25$) |
| `Sexo` | 0.456 | Importante ($>0.25$) |
| `Es_Expuesto` | 0.346 | Importante ($>0.25$) |
| `Score_Riesgo` | 0.250 | Leve ($>0.10$) |
| `Riesgo_Cd` | 0.175 | Leve ($>0.10$) |
| `Riesgo_Pb` | 0.128 | Leve ($>0.10$) |


### Recomendaciones Metodológicas:
1. **Reportar estos desequilibrios:** Es mandatorio documentar que los dos grupos difieren en estas variables.
2. **Ajuste en análisis posteriores:** Al modelar la asociación de los metales, se debe controlar por las covariables que muestran desequilibrio (ej. en regresiones multivariadas) para reducir el sesgo de confusión.

### Alertas de Variabilidad para Modelos de Regresión:
* `Riesgo_Pb` tiene desbalance severo (sólo 3 observaciones de la minoría).