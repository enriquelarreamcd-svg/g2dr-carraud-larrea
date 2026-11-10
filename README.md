# g2dr-carraud-larrea
“¿Qué efecto tiene la aplicación de técnicas de corrección del data drifting sobre variables monetarias en el desempeño del modelo” ?
# Problema #02 – DR Data Drifting (R + LightGBM)

Efecto de corregir el *data drifting* de las variables monetarias sobre el desempeño de un modelo
LightGBM. Workflow Analista Jr, competencia Kaggle `labo-1-virtual-2026-analista-jr`, grupo A.

## Pregunta

¿Qué efecto tiene corregir el drift de las variables monetarias (IPC, tipo de cambio u otros métodos)
sobre el desempeño del modelo, frente a no tratarlo?


## Hipotesis experimental
En la Argentina, las variables monetarias, juegan un papel fundamental al momento de tomar decisiones a futuro; por lo cual los modelos deben de tener en consideración las repercusiones de estas variables en Baja+1 y Baja+2

## Hipotesis particulares
Corregir el data drift (diferencia en la distribución de los datos de validación/testeo respecto a los de entrenamiento) por inflación (IPC General del INDEC), por tipo de cambio Oficial vs Blue.
Se experimentará, además, con nuevas variables dentro del experimento, como ser tasas internacionales o Riesgo País.
Ponderar con un mayor peso a los momentos temporales más cercanos al mes a predecir asumiendo una mayor probabilidad de similitud. 

## Sesgo cognitivo
Nuestro sesgo en particular, se remonta a la comparabilidad de las variables monetarias expuestas al proceso inflacionario de la República Argentina; lo cual nos lleva a preferir trabajar con variables reales, o tratadas para que una mejor representación de su exposición al paso del tiempo o bien su real poder adquisitivo.

## Diseño

- Entrenamiento 201901–202105 → validación **202107** → reentrenamiento 201901–202107 → predicción **202109**.
- Positivos: `BAJA+1` y `BAJA+2` (en la función de ganancia, `BAJA+2` = 975000).
- LightGBM binario, `max_bin` 31, `learning_rate` 0.03, early stopping 200, hasta 2048 iteraciones.
  Grid Search de 50 combinaciones (hojas, `min_data_in_leaf`, `feature_fraction`) elegido por **AUC**.
- 10 semillas por configuración (la primera, `165463`, es la primigenia):
  `165463 329969 454253 593899 980249 100183 250027 400031 550049 700057`.
- Ensamble = promedio de probabilidades de las 10 semillas.

## Resultados de NEW201 (9 configuraciones × 10 semillas, AUC en 202107)

| N° | Corrección | Variables | AUC medio | sd |
|----|------------|-----------|-----------|----|
| 01 | rank_cero_fijo | intra-mes | **0.934861** | 0.000616 |
| 05 | rank_cero_fijo | + k-means como variable | 0.933777 | 0.001128 |
| 06 | deflación (IPC) | + k-means como variable | 0.933287 | 0.000672 |
| 04 | deflación ÷ dólar blue | intra-mes | 0.933002 | 0.000584 |
| 08 | deflación ÷ dólar blue | + k-means como variable | 0.932835 | 0.001250 |
| 07 | deflación ÷ dólar oficial | + k-means como variable | 0.932574 | 0.000668 |
| 09 | rank_cero_fijo | + k-means como drift | 0.932278 | 0.000717 |
| 02 | deflación (IPC) | intra-mes | 0.932152 | 0.000819 |
| 03 | deflación ÷ dólar oficial | intra-mes | 0.932061 | 0.000958 |

- La 01 gana en 9–10 de 10 semillas contra cada una de las otras; Wilcoxon no pareado con Holm: p entre 0.0015 y 0.021.
- K-means empeora con `rank_cero_fijo` y solo ayuda con la deflación por IPC.
- Ensamble 01 + 05: AUC 0.936242 (la 01 sola: 0.935758).
- **Limitaciones:** un solo mes de validación (202107); el AUC no es la ganancia; elegir y evaluar en el mismo
  mes es optimista. 

## Contenido

| Carpeta | Qué hay |
|---------|---------|
`z620_WorkFlow_06_DR_grupoA_semillerio`  Este notebook corre un metodo de drifting por vez (PARAM$DR$metodo) y evalua con 10 semillas.|
| `new201/` | `NEW201_drift_10semillas` (corre las 9 configuraciones), `NEW201_wilcoxon_y_ensambles.R`, `NEW201_presentacion_final` (tablas y gráficos) |
| `new202/` | `NEW202_mejoras_sobre_la_01`: lag3/lag6, medias móviles, grid ampliado y todo junto, comparados contra la 01 con Wilcoxon. **Sin resultados todavía** |
Los `.R` son la fuente; los `.ipynb` se generan con `tools/build_ipynb.js` (kernel R).

## Cómo correrlo

1. Requisitos: R con `data.table`, `lightgbm`, `R.utils`, `yaml`.
2. El dataset (`analistajr_competencia_2026.csv.gz`) **no está en el repo**; copiarlo a la carpeta de datasets.
3. Ajustar las rutas al principio de cada script (`PARAM$carpeta_exp`, `PARAM$carpeta_dataset`; por defecto
   `/content/buckets/b1/exp` y `/content/datasets`).
4. Orden: `NEW201_drift_10semillas` → `NEW201_wilcoxon_y_ensambles.R` → `NEW201_presentacion_final`.
5. **Cada experimento va en una carpeta nueva.** Para NEW202 conviene poner `PARAM$experimento <- "NEW203"`
   (no renombrar carpetas) y dejar `prueba_rapida <- FALSE`; el modo prueba usa una fracción de los clientes y
   sus resultados no son comparables.

