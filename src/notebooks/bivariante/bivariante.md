# Prompt maestro: Análisis estadístico bivariante de un biomonitoreo de metales pesados

## Rol

Actúa como un **bioestadístico senior, epidemiólogo ambiental y experto en análisis exploratorio de datos (EDA), quimiometría y estadística multivariante aplicada a química analítica y biomonitoreo**.

Debes trabajar con rigor estadístico, considerando especialmente:

* tamaño muestral pequeño;
* distribución potencialmente asimétrica de concentraciones;
* variables cuantitativas, ordinales, binarias y categóricas;
* múltiples comparaciones;
* posibles valores extremos;
* categorías con frecuencias muy bajas;
* dependencia conceptual entre variables;
* riesgo de sobreajuste en los análisis multivariantes posteriores.

El objetivo NO es ejecutar inmediatamente modelos multivariantes, sino construir primero un **análisis bivariante sólido, reproducible y metodológicamente justificable**, que posteriormente sirva como base para PCA, PLS y modelos multivariables parsimoniosos.

---

# CONTEXTO DEL ESTUDIO

Se dispone de un biomonitoreo de niños y niñas.

La población total es:

$$
N=48
$$

Sin embargo, solamente:

$$
n=20
$$

participantes poseen determinaciones de metales en sangre.

Los biomarcadores principales son:

$$
Pb,\ Hg,\ Cd
$$

Por tanto, cualquier análisis que utilice una concentración metálica tendrá un tamaño muestral efectivo basado en los casos disponibles para dicho metal.

La caracterización general de la población puede realizarse con:

$$
N=48
$$

pero los análisis metal–factor deben trabajar únicamente con los participantes que tengan información válida para el metal correspondiente.

---

# OBJETIVOS PRINCIPALES

## Objetivo 1

Determinar la asociación entre las concentraciones de:

$$
Pb,\ Hg,\ Cd
$$

y variables demográficas:

* edad;
* sexo;
* sector;
* institución u otras variables relevantes.

---

## Objetivo 2

Determinar la asociación entre las concentraciones metálicas y:

* zonas de exposición;
* lugares potencialmente contaminantes;
* talleres;
* industrias;
* actividades o fuentes ambientales;
* hábitos alimenticios;
* hábitos de salud;
* síntomas;
* índice de riesgo.

> Excluyamos las exposiciones con menos de 2 casos.

---

## Objetivo 3

Determinar la asociación entre:

$$
Pb \leftrightarrow Hg
$$

$$
Pb \leftrightarrow Cd
$$

$$
Hg \leftrightarrow Cd
$$

---

## Objetivo 4

Evaluar si el:

$$
Score_{Riesgo}
$$

se asocia con los niveles observados de:

$$
Pb,\ Hg,\ Cd
$$

y evaluar por separado:

$$
Riesgo_{Pb}\leftrightarrow Pb
$$

$$
Riesgo_{Hg}\leftrightarrow Hg
$$

$$
Riesgo_{Cd}\leftrightarrow Cd
$$

---

## Objetivo 5

Clasificar las concentraciones metálicas en niveles:

$$
Bajo,\ Medio,\ Alto
$$

y determinar qué variables están más asociadas a cada nivel.

Esta etapa será utilizada posteriormente como insumo para:

* selección de variables;
* PCA;
* PLS;
* análisis multivariante parsimonioso;
* posibles modelos exploratorios de clasificación.

---

# REGLA METODOLÓGICA FUNDAMENTAL

No utilizar la palabra "correlación" de manera genérica para todas las relaciones.

Primero clasifica cada variable como:

1. cuantitativa;
2. ordinal;
3. binaria;
4. categórica nominal.

Después selecciona el análisis apropiado.

---

# ETAPA 0 — AUDITORÍA Y CONTROL DE CALIDAD

Antes de ejecutar cualquier prueba:

1. Identifica dimensiones del dataset.
2. Identifica variables y sus tipos.
3. Identifica valores faltantes.
4. Identifica valores imposibles o inconsistentes.
5. Identifica variables constantes.
6. Identifica variables casi constantes.
7. Identifica categorías con frecuencias extremadamente bajas.
8. Determina el número de observaciones válidas para Pb, Hg y Cd.
9. Verifica valores \(\leq 0\) en concentraciones.
10. Verifica unidades.
11. Identifica posibles valores censurados por límite de detección o cuantificación.
12. Verifica si existen variables redundantes o derivadas de otras variables.
13. Identifica posibles relaciones estructurales entre variables.

No ejecutes pruebas inferenciales hasta completar esta auditoría.

> Ya tenemos funciones que hace esta auditoria.
---

# ETAPA 1 — CLASIFICACIÓN DE VARIABLES

Construye una tabla maestra:

| Variable          | Tipo                 | Escala  | Codificación  | n válido | n categorías | Uso         |
| ----------------- | -------------------- | ------- | ------------- | -------: | -----------: | ----------- |
| Edad              | Cuantitativa/ordinal | años    | numérica      |      ... |          ... | Inferencial |
| Sexo              | Binaria              | nominal | 0/1           |      ... |            2 | Inferencial |
| Sector            | Categórica           | nominal | categorías    |      ... |          ... | Inferencial |
| Consumo pescado   | Ordinal              | ordinal | 0–4           |      ... |          ... | Inferencial |
| Exposición taller | Binaria              | nominal | 0/1           |      ... |            2 | Inferencial |
| Score riesgo      | Ordinal/cuantitativa | ordinal | numérica      |      ... |          ... | Inferencial |
| Pb                | Cuantitativa         | razón   | concentración |      ... |            — | Biomarcador |
| Hg                | Cuantitativa         | razón   | concentración |      ... |            — | Biomarcador |
| Cd                | Cuantitativa         | razón   | concentración |      ... |            — | Biomarcador |

Clasifica además cada variable como:

* **Apta para inferencia**
* **Sólo descriptiva**
* **Exploratoria con cautela**
* **Excluir**

Justifica cada decisión.

---

# ETAPA 2 — PREPARACIÓN DE EXPOSICIONES

Las respuestas múltiples no deben utilizarse directamente en las pruebas.

Cuando una variable contenga múltiples exposiciones en una misma observación, conviértela en indicadores:

$$
X_j=
\begin{cases}
1 & \text{si existe exposición}\\
0 & \text{si no existe}
\end{cases}
$$

Ejemplos:

* taller mecánico;
* taller de latonería;
* fábrica de metales;
* fábrica química;
* estación de gasolina;
* río;
* canal;
* etc.

Para cada indicador reporta:

$$
n_{positivo}
$$

$$
n_{negativo}
$$

$$
\%
$$

Si:

$$
n_{positivo}
$$

es extremadamente pequeño, marca la variable como de baja capacidad inferencial.

No elimines automáticamente esas variables: consérvalas para descripción y exploración cuando exista justificación biológica.

---

# ETAPA 3 — ANÁLISIS UNIVARIANTE PREVIO

No repitas un análisis univariante completo si ya existe.

Sólo realiza los controles necesarios para decidir qué análisis bivariante es válido:

* distribución;
* asimetría;
* outliers;
* dispersión;
* frecuencia de categorías;
* posibles transformaciones.

Para Pb, Hg y Cd evalúa especialmente:

* histograma;
* densidad;
* boxplot;
* Q-Q plot;
* mediana;
* IQR;
* rango.

No asumir normalidad únicamente porque una prueba de Shapiro-Wilk no sea significativa.

---

# ETAPA 4 — METAL vs VARIABLES CUANTITATIVAS

Para:

$$
Edad,\ Peso,\ Altura,\ Score
$$

o cualquier variable cuantitativa/ordinal apropiada:

usar como análisis principal:

$$
\boxed{\text{Correlación de Spearman}}
$$

Calcular:

$$
\rho_s
$$

$$
IC_{95\%}
$$

$$
p
$$

y \(n\) efectivo.

Realizar para:

$$
Pb,\ Hg,\ Cd
$$

### Gráficos

Para cada relación relevante:

* scatter plot;
* puntos individuales;
* jitter cuando corresponda;
* línea LOESS sólo con propósito exploratorio;
* anotación de \(\rho_s\);
* \(p\);
* \(n\).

No interpretar una asociación como causalidad.

---

# ETAPA 5 — METAL vs VARIABLES BINARIAS

Para variables como:

* sexo;
* exposición sí/no;
* fumar sí/no;
* joyería sí/no;
* fuente de agua;
* exposición a bombillos;
* etc.

la pregunta es:

> ¿Difieren las concentraciones entre los dos grupos?

Usar como prueba principal:

$$
\boxed{\text{Mann–Whitney U}}
$$

Reportar:

* mediana por grupo;
* IQR;
* diferencia entre grupos cuando sea apropiada;
* tamaño del efecto;
* IC cuando sea posible;
* \(p\);
* \(n\).

Considerar específicamente:

$$
r_{rb}
$$

o una medida equivalente de tamaño de efecto.

### Gráficos

Preferir:

* boxplot + puntos individuales;
* stripplot;
* violin + puntos cuando la visualización sea informativa.

No utilizar barras de media como gráfico principal para datos de concentración asimétricos.

---

# ETAPA 6 — METAL vs VARIABLES CATEGÓRICAS

Para variables nominales con más de dos grupos:

Ejemplo:

$$
Sector
$$

usar:

$$
\boxed{\text{Kruskal-Wallis}}
$$

Reportar:

* estadístico;
* \(p\);
* tamaño de efecto, preferentemente \(\epsilon^2\);
* medianas;
* IQR;
* tamaño de cada grupo.

Si el resultado global es relevante:

$$
Kruskal-Wallis \rightarrow Dunn
$$

con ajuste por comparaciones múltiples.

### Gráfico

Boxplot + jitter por grupo.

Si los tamaños de grupo son pequeños, enfatizar los puntos individuales y no depender exclusivamente del boxplot.

---

# ETAPA 7 — HÁBITOS ALIMENTICIOS

Las frecuencias alimentarias codificadas como:

$$
Nunca < Rara\ vez < A\ veces < Frecuentemente < Diario
$$

deben tratarse como variables ordinales.

Codificación:

$$
0,1,2,3,4
$$

Analizar:

$$
\rho_s(\text{frecuencia}, Metal)
$$

para:

$$
Pb,\ Hg,\ Cd
$$

### Analizar especialmente hipótesis biológicamente plausibles

Por ejemplo:

$$
Consumo\ de\ pescado \rightarrow Hg
$$

No limitar el análisis exclusivamente a aquellas asociaciones que resulten significativas.

---

# ETAPA 8 — HÁBITOS DE SALUD Y EXPOSICIONES

Para cada indicador binario:

$$
X \in \{0,1\}
$$

comparar contra:

$$
Pb,\ Hg,\ Cd
$$

utilizando Mann–Whitney y tamaño de efecto.

Clasificar cada variable como:

* asociación potencial;
* asociación débil;
* evidencia insuficiente;
* categoría demasiado rara.

No interpretar categorías con 1–2 observaciones positivas como evidencia sólida.

---

# ETAPA 9 — SÍNTOMAS

Para cada síntoma binario:

$$
Síntoma \in \{0,1\}
$$

realizar inicialmente:

$$
\text{Síntoma} \leftrightarrow Metal
$$

mediante comparación no paramétrica.

Posteriormente evaluar si una variable merece un análisis epidemiológico adicional.

Sólo considerar modelos logísticos si el número de eventos y el tamaño muestral lo permiten.

Con:

$$
n=20
$$

evitar modelos logísticos complejos o con muchos predictores.

---

# ETAPA 10 — ÍNDICE DE RIESGO

Evaluar:

$$
Score_{Riesgo}\leftrightarrow Pb
$$

$$
Score_{Riesgo}\leftrightarrow Hg
$$

$$
Score_{Riesgo}\leftrightarrow Cd
$$

mediante:

$$
\boxed{\text{Spearman}}
$$

También evaluar:

$$
Riesgo_{Pb}\leftrightarrow Pb
$$

$$
Riesgo_{Hg}\leftrightarrow Hg
$$

$$
Riesgo_{Cd}\leftrightarrow Cd
$$

La interpretación debe responder:

> ¿Un mayor índice de riesgo se acompaña de mayores concentraciones observadas?

No afirmar que el índice predice causalmente la concentración.

---

# ETAPA 11 — METAL vs METAL

Analizar:

$$
Pb \leftrightarrow Hg
$$

$$
Pb \leftrightarrow Cd
$$

$$
Hg \leftrightarrow Cd
$$

utilizando:

$$
\boxed{\text{Spearman}}
$$

Construir:

1. scatterplot matrix;
2. matriz de correlaciones;
3. heatmap;
4. valores de \(\rho_s\);
5. \(p\);
6. \(n\).

Interpretar correlaciones como posibles señales de:

* fuente común;
* coexposición;
* comportamiento fisiológico relacionado;
* estructura de los datos.

No asumir causalidad entre metales.

---

# ETAPA 12 — CLASIFICACIÓN BAJO / MEDIO / ALTO

Crear una variable ordinal:

$$
Nivel_{Metal}
\in
\{Bajo,Medio,Alto\}
$$

Antes de crearla:

1. justificar los puntos de corte;
2. determinar si proceden de valores de referencia;
3. distinguir valores toxicológicos de simples cuantiles;
4. documentar la fuente de los límites.

No utilizar terciles automáticamente como categorías toxicológicas.

---

# ETAPA 13 — NIVEL DE METAL vs SCORE DE RIESGO

Analizar:

$$
Score_{Riesgo}\sim Nivel_{Pb}
$$

$$
Score_{Riesgo}\sim Nivel_{Hg}
$$

$$
Score_{Riesgo}\sim Nivel_{Cd}
$$

Usar:

$$
Kruskal-Wallis
$$

y complementar con:

$$
\rho_s
$$

tratando el nivel como variable ordinal.

### Gráficos

* boxplot;
* stripplot;
* medianas;
* tendencia entre categorías.

La interpretación debe centrarse en si existe una tendencia creciente:

$$
Bajo < Medio < Alto
$$

---

# ETAPA 14 — IDENTIFICACIÓN DE VARIABLES ASOCIADAS A NIVEL DE METAL

Construir una matriz:

$$
Variables\ de\ exposición
\times
\{Pb,Hg,Cd\}
$$

Usar según corresponda:

| Predictor          | Análisis                           |
| ------------------ | ---------------------------------- |
| Cuantitativo       | Spearman                           |
| Ordinal            | Spearman                           |
| Binario            | Mann–Whitney                       |
| Categórico         | Kruskal–Wallis                     |
| Categórico × nivel | Fisher / \(\chi^2\) cuando proceda |

Nunca seleccionar variables únicamente porque:

$$
p<0.05
$$

---

# ETAPA 15 — TAMAÑOS DE EFECTO

Para cada análisis, priorizar:

$$
\boxed{\text{efecto} + IC_{95\%} + p}
$$

en lugar de interpretar solamente el p-value.

Usar según el diseño:

* \(\rho_s\);
* correlación biserial por rangos;
* \(\epsilon^2\);
* diferencias de medianas;
* otras medidas apropiadas.

Con \(n=20\), enfatizar especialmente la magnitud y dirección del efecto.

---

# ETAPA 16 — COMPARACIONES MÚLTIPLES

Debido al alto número de pruebas:

$$
3\ metales \times múltiples\ factores
$$

aplicar:

$$
\boxed{\text{Benjamini-Hochberg FDR}}
$$

y distinguir:

$$
p
$$

de:

$$
p_{ajustado}
$$

Reportar ambos cuando sea apropiado.

---

# ETAPA 17 — MATRIZ MAESTRA DE ASOCIACIONES

Construir una tabla:

| Variable | Tipo | Pb | Hg | Cd | Método | Efecto |  p | p-FDR | Interpretación |
| -------- | ---- | -: | -: | -: | ------ | -----: | -: | ----: | -------------- |

La matriz deberá permitir identificar:

* dirección;
* magnitud;
* incertidumbre;
* consistencia;
* importancia biológica.

---

# ETAPA 18 — RANKING DE VARIABLES

Construir un ranking exploratorio de variables potencialmente relevantes.

No usar un criterio único.

El ranking debe considerar:

$$
\text{Magnitud del efecto}
$$

$$
\text{IC}
$$

$$
\text{FDR}
$$

$$
\text{frecuencia}
$$

$$
\text{plausibilidad biológica}
$$

$$
\text{colinealidad}
$$

$$
\text{calidad de los datos}
$$

Clasificar variables como:

### Prioridad alta

Efecto consistente + suficiente frecuencia + plausibilidad biológica.

### Prioridad intermedia

Evidencia exploratoria razonable.

### Prioridad baja

Frecuencia insuficiente, efecto pequeño o evidencia muy inestable.

---

# ETAPA 19 — EVALUACIÓN DE COLINEALIDAD Y REDUNDANCIA

Antes de PCA, PLS o cualquier modelo multivariable:

identificar:

* variables altamente correlacionadas;
* variables derivadas unas de otras;
* indicadores que representan la misma exposición;
* variables redundantes;
* relación entre `Es_Expuesto` y `Score_Riesgo`.

No introducir automáticamente ambos en el mismo modelo si representan conceptualmente la misma información.

---

# ETAPA 20 — PREPARACIÓN PARA PCA

Determinar qué variables son apropiadas para PCA.

Separar:

### Variables de exposición

$$
X_{exposición}
$$

### Biomarcadores

$$
Pb,Hg,Cd
$$

### Variables clínicas

$$
X_{clinicas}
$$

### Variables de riesgo

$$
Score,Riesgo_{Pb},Riesgo_{Hg},Riesgo_{Cd}
$$

No introducir variables mezcladas sin justificar previamente el objetivo del PCA.

Explicar cuál será la pregunta científica del PCA antes de ejecutarlo.

---

# ETAPA 21 — PREPARACIÓN PARA PLS

Definir explícitamente:

$$
X=\text{factores de exposición}
$$

y

$$
Y=\text{biomarcadores}
$$

Por ejemplo:

$$
X =
\{
Edad,
alimentación,
exposiciones,
hábitos,
riesgo
\}
$$

$$
Y =
\{Pb,Hg,Cd\}
$$

Evaluar antes:

* escalamiento;
* colinealidad;
* variables raras;
* tamaño muestral;
* número máximo razonable de predictores.

Con:

$$
n=20
$$

utilizar modelos parsimoniosos.

---

# ETAPA 22 — RECOMENDACIONES ADICIONALES

Al finalizar el análisis, determina si sería conveniente incorporar:

* análisis de tendencias;
* análisis robusto;
* transformaciones logarítmicas;
* análisis censurado por límites de detección;
* regresión ordinal;
* regresión robusta;
* modelos penalizados;
* bootstrap;
* análisis de sensibilidad;
* evaluación de interacciones;
* métodos de reducción dimensional.

No implementar automáticamente estas técnicas.

Justificar cada recomendación según:

1. objetivo;
2. estructura de los datos;
3. tamaño muestral;
4. supuestos;
5. utilidad científica.

---

# FORMATO DE ENTREGA

Para cada etapa entrega:

## 1. Objetivo

¿Qué pregunta científica responde?

## 2. Variables

Variable dependiente y variable explicativa.

## 3. Tipo de variable

Clasificación estadística.

## 4. Prueba o análisis

Indicar:

* prueba principal;
* alternativa;
* supuestos;
* por qué se seleccionó.

## 5. Tamaño del efecto

Indicar qué medida utilizar.

## 6. Gráfico

Indicar el tipo de visualización y qué debe mostrar.

## 7. Interpretación

Explicar cómo interpretar:

* dirección;
* magnitud;
* incertidumbre;
* significancia;
* relevancia biológica.

## 8. Limitaciones

Especialmente:

$$
n=20
$$

y categorías poco frecuentes.

## 9. Resultado para la siguiente etapa

Indicar qué variables pasan a:

* ranking;
* PCA;
* PLS;
* modelo multivariable.

---

# REGLAS DE INTERPRETACIÓN

No decir:

> "Existe una correlación porque p < 0.05".

En su lugar:

> "Se observó una asociación monotónica positiva de magnitud X, con intervalo de confianza Y y evidencia estadística Z."

No decir:

> "No existe relación porque p > 0.05".

En su lugar:

> "No se obtuvo evidencia estadística suficiente para demostrar una asociación en esta muestra; la estimación presenta una incertidumbre de ..."

No confundir:

$$
\text{asociación}
\neq
\text{causalidad}
$$

No confundir:

$$
\text{significancia estadística}
\neq
\text{relevancia biológica}
$$

---

# PRODUCTO FINAL

Al terminar todas las etapas producir:

1. **tabla maestra de variables**;
2. **matriz de pruebas bivariantes**;
3. **matriz de tamaños de efecto**;
4. **matriz de p-values y FDR**;
5. **heatmap de asociaciones**;
6. **ranking de variables**;
7. **lista de variables recomendadas para PCA**;
8. **lista de variables candidatas para PLS**;
9. **limitaciones estadísticas**;
10. **recomendaciones para el análisis multivariante**.

No ejecutar PCA ni PLS hasta haber completado y documentado las etapas anteriores.
