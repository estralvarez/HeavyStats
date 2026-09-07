# Algoritmo de riesgo de exposición

Este documento describe cómo funciona la función `calculate_risk` ubicada en `app/services/patient_service.py`.

## 1. Objetivo

La función evalúa si un paciente presenta señales de riesgo por exposición a metales pesados, especialmente plomo (Pb), mercurio (Hg) y cadmio (Cd), a partir de:

- zonas de exposición,
- síntomas reportados,
- hábitos de salud,
- hábitos alimenticios,
- y otras variables cualitativas del registro.

La salida no pretende ser una medición clínica exacta, sino una regla empírica de priorización para análisis epidemiológicos y muestreo estratificado.

---

## 2. Entradas del algoritmo

La función consume un objeto `p` que normalmente representa un paciente con relaciones como:

- `p.zonas_exposicion`
- `p.antecedentes_patologicos`
- `p.habitos_de_salud`
- `p.habitos_alimenticios`

### Variables principales

- `talleres`: lista de talleres o actividades asociadas a la zona de exposición.
- `industrias`: lista de industrias relevantes.
- `lugares`: lugares cercanos a la exposición.
- `sintomas`: lista de síntomas reportados.
- `fuma`: hábito de fumar.
- `techo`: uso de techos relacionados con riesgo.
- `bombillos`: uso de bombillos/situaciones de riesgo.
- `joyeria`: uso de joyería relacionada a factores de riesgo.
- `agua`: fuentes de agua reportadas.
- `info_alimenticia`: JSON con hábitos alimenticios, por ejemplo frecuencia de consumo de pescado.

---

## 3. Normalización de respuestas

Antes de comparar la información, el algoritmo convierte respuestas cualitativas a un formato uniforme:

- `normalize_yes(value)`: devuelve `True` si el valor se interpreta como afirmativo.
- Acepta variantes como:
  - `si`, `sí`, `yes`, `y`

Esto permite tratar respuestas inconsistentes o con tildes/formatos distintos de una forma homogénea.

También se usa `match_any(pattern, text)`, que busca patrones textuales con expresiones regulares para detectar palabras clave dentro de cadenas largas.

---

## 4. Detección de exposición

El algoritmo define:

- `has_talleres`
- `has_industrias`
- `has_lugares`

y luego calcula:

- `es_expuesto = has_talleres or has_industrias or has_lugares`

Esto significa que el paciente se considera expuesto si reporta al menos una zona, industria o lugar de riesgo no nulo.

Se excluyen valores del tipo:

- `"Ninguna de las anteriores"`

para evitar falsos positivos.

---

## 5. Detección de riesgos por metal

El algoritmo genera tres indicadores booleanos:

### 5.1 `riesgo_Pb` (plomo)
Se activa si se detecta exposición a fuentes o síntomas asociados a plomo, por ejemplo:

- talleres como mecánica, baterías, herrería, cerámica, latonería,
- lugares como vertederos, ríos, canales, estaciones de gasolina o imprentas,
- industrias como fábricas de pintura o metales,
- síntomas como pérdida de peso, pérdida de apetito, náuseas, fatiga o irritabilidad.

### 5.2 `riesgo_Hg` (mercurio)
Se activa solo si el paciente está expuesto y además presenta uno de estos indicadores:

- cerámica o alfarería,
- vertederos, ríos, canales o áreas similares,
- fábricas de pinturas, metales o químicos,
- síntomas neurológicos como irritabilidad, apatía, falta de concentración o náuseas.
- consumo frecuente de pescado (diario o varias veces por semana).

### 5.3 `riesgo_Cd` (cadmio)
Se activa si se detecta cualquiera de estas señales:

- síntomas respiratorios o cutáneos,
- fumar,
- usar joyería de riesgo,
- estar cerca de talleres o lugares con riesgo,
- trabajar o vivir cerca de industrias contaminantes.

---

## 6. Cálculo del score empírico

El score cuantitativo se construye sumando puntos según la presencia de indicadores cualitativos.

### Puntuación actual

```python
score_riesgo = 0
if es_expuesto:
    score_riesgo += 3
if sintomas and any(x for x in sintomas if x and x != "Ninguna de las anteriores"):
    score_riesgo += 1
if any(x for x in agua if x == "Agua de pozo profundo"):
    score_riesgo += 1
if es_expuesto and es_pescados_diario_o_frec:
    score_riesgo += 1
if normalize_yes(fuma):
    score_riesgo += 1
if normalize_yes(techo):
    score_riesgo += 1
if normalize_yes(bombillos):
    score_riesgo += 0
```

### Interpretación

- `es_expuesto`: suma 3 puntos.
- síntomas presentes: suma 1 punto.
- agua de pozo profundo: suma 1 punto.
- consumo diario/frecuente de pescado: suma 1 punto si también hay exposición.
- fumar: suma 1 punto.
- techo: suma 1 punto.
- bombillos: no suma puntos por ahora (`+0`), porque se considera un indicador débil o no decisivo en la regla actual.

### ¿Por qué no puede quedar en 0 cuando hay riesgo?

Aunque la suma puede producir `0` en algunos casos, el algoritmo incluye una regla defensiva:

```python
riesgo_detectado = (
    es_expuesto or riesgo_pb or riesgo_hg or riesgo_cd or bool(sintomas) or normalize_yes(fuma)
    or normalize_yes(joyeria) or any(x for x in agua if x == "Agua de pozo profundo")
    or es_pescados_diario_o_frec
)
if riesgo_detectado and score_riesgo == 0:
    score_riesgo = 1
```

Esto evita que un paciente con señales claras de riesgo quede categorizado como si no tuviera riesgo de ninguna clase. En otras palabras, la regla empírica exige que cualquier evidencia detectable se traduzca al menos en un nivel mínimo de riesgo.

---

## 7. Clasificación del nivel de riesgo

Después del cálculo de score se aplica una clasificación:

```python
def classify_score(score):
    if score >= 8:
        return "alto"
    if score >= 4:
        return "medio"
    if score > 0:
        return "bajo"
    return "sin_riesgo"
```

### Rangos

- `alto`: score >= 8
- `medio`: score >= 4
- `bajo`: score > 0
- `sin_riesgo`: score == 0

Este nivel no equivale a una prueba diagnóstica, sino a una escala de priorización para análisis posterior y selección de muestra.

---

## 8. Cómo se usa en práctica

La salida final del algoritmo es un diccionario como este:

```python
{
    "es_expuesto": True,
    "riesgo_Pb": True,
    "riesgo_Hg": False,
    "riesgo_Cd": True,
    "score_riesgo": 6,
    "nivel_riesgo": "medio",
}
```

Esto permite:

- filtrar pacientes por exposición,
- priorizar casos con mayor sospecha,
- estratificar la muestra por riesgo,
- comparar el riesgo empírico con niveles reales de metales en sangre.

---

## 9. Consideraciones importantes

1. Es una regla empírica y cualitativa.
   No reemplaza resultados clínicos ni mediciones de laboratorio.

2. El score funciona como variable cuantitativa derivada de indicadores cualitativos.
   Eso permite ordenar y contrastar riesgo sin convertir respuestas binarias en diagnóstico médico.

3. El diseño de muestreo debe considerar desequilibrios estructurales.
   En este proyecto se excluyen o ajustan variables como sexo y sector para no introducir sesgos artificiales en la muestra final.

4. El nivel de riesgo debe entenderse como una priorización analítica, no como una clasificación definitiva del paciente.

---

## 10. Resumen

El algoritmo combina:

- presencia de exposición ambiental,
- síntomas compatibles,
- hábitos de riesgo,
- y factores sociodemográficos relevantes,

para producir:

- indicadores booleanos por metal,
- un score cuantitativo,
- y una clasificación final de riesgo (`bajo`, `medio`, `alto`).

Esto lo convierte en una herramienta útil para estratificación, análisis y selección de casos, siempre bajo la lógica de que el riesgo debe detectarse incluso cuando la información es parcial o cualitativa.
