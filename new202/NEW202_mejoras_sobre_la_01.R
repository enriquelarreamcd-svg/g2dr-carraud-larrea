# %% [markdown]
# # NEW202 · Mejoras sobre la configuración 01
# ## Problema #02 · DR Data Drifting · Grupo A
#
# Todo el código está en R. Parte de **la 01 de NEW201** (`01_rank_cero_fijo_intrames`) como base y prueba, **de a un cambio por vez**, si mejora con 10 semillas. Cada variante se compara contra la 01 con el mismo Wilcoxon de NEW201.
#
# | config | cambio respecto de la 01 | grid |
# |---|---|---|
# | `11_lag36` | agrega lag3 y lag6 (y sus deltas) | original (50 combinaciones) |
# | `12_medias_moviles` | agrega medias móviles de 3 y 6 meses (y la diferencia contra la media) | original |
# | `13_grid_ampliado` | mismas variables que la 01 | **ampliado** (75 combinaciones) |
# | `14_todo_junto` | lag3/lag6 + medias móviles | ampliado |
#
# **Por qué separar los cambios:** si se cambia todo junto no se sabe qué fue lo que ayudó. Las 11, 12 y 13 cambian una sola cosa; la 14 es la combinación, por si suman.
#
# **Grid ampliado:** en NEW201, cuatro configuraciones eligieron `min_data_in_leaf` = 2048 (el borde del grid) y la 01 eligió `feature_fraction` = 0.8 (el máximo). El grid ampliado llega a `min_data_in_leaf` = 8192 y `feature_fraction` = 1.0, y deja de probar los valores chicos que nunca ganaron.
#
# **Control:** la 01 **no se vuelve a correr**: se lee de `exp/NEW201` (AUC por semilla y probabilidades guardadas). Las variantes usan las mismas 10 semillas y el mismo preprocesamiento.
#
# **Salida:** `exp/NEW202/` (una carpeta por configuración, igual que en NEW201). Nada se envía a Kaggle; solo se generan los archivos.
#
# **Tiempo:** cada configuración es un Grid Search + 10 modelos de validación + 10 modelos finales. Con más variables y más combinaciones tarda bastante más que las ~50 minutos de NEW201 (orientativo: varias horas por configuración las 11, 12 y 14; no está medido). Si se corta, se vuelve a correr la celda de corrida y saltea las configuraciones terminadas. En `PARAM$configs_a_correr` se elige cuáles correr.
#
# **Antes de la corrida larga:** probar con `PARAM$prueba_rapida <- TRUE`.

# %%
format(Sys.time(), "%a %b %d %X %Y")

# %%
# limpio la memoria
rm(list=ls(all.names=TRUE))
gc(full=TRUE, verbose=FALSE)

# %%
require("data.table")

if( !require("R.utils")) install.packages("R.utils")
require("R.utils")

if( !require("lightgbm")) install.packages("lightgbm")
require("lightgbm")

if( !require("yaml")) install.packages("yaml")
require("yaml")

# %% [markdown]
# ## Parametros

# %%
PARAM <- list()

# Cada corrida nueva va en una carpeta distinta. Si ya existe una carpeta con resultados
# viejos o de prueba, cambiar el nombre aca (por ejemplo "NEW203") en vez de renombrar la vieja.
PARAM$experimento <- "NEW202"
PARAM$control_experimento <- "NEW201"             # de donde se lee la 01
PARAM$dataset <- "analistajr_competencia_2026.csv.gz"

# TRUE = corrida chica (5% de los clientes, grids de pocas combinaciones, 2 semillas)
#  para verificar que todo funciona; guarda en NEW202_prueba, nunca en NEW202
PARAM$prueba_rapida <- FALSE

# la primera semilla es la semilla primigenia (las mismas de NEW201)
PARAM$semillas <- c(165463, 329969, 454253, 593899, 980249,
                    100183, 250027, 400031, 550049, 700057)
PARAM$semilla_primigenia <- PARAM$semillas[1]

PARAM$bootstrap_B <- 200      # remuestreos de clientes para el ensamble

elegir_carpeta <- function(candidatas, archivo= NULL) {
  ok <- vapply(candidatas, function(d) {
    dir.exists(d) && (is.null(archivo) || file.exists(file.path(d, archivo)))
  }, logical(1))
  if( any(ok) ) candidatas[ok][1] else NA_character_
}

PARAM$carpeta_exp <- elegir_carpeta(c(
  "/content/buckets/b1/exp", "/home/ds/buckets/b1/exp", "exp", "../exp"))

PARAM$carpeta_dataset <- elegir_carpeta(c(
  "/content/datasets", "/home/ds/datasets", "datasets", "../datasets", "."), PARAM$dataset)

# si no las encuentra, escribirlas a mano:
# PARAM$carpeta_exp <- "/ruta/a/exp"
# PARAM$carpeta_dataset <- "/ruta/a/datasets"

if( is.na(PARAM$carpeta_exp) )
  stop("No encuentro la carpeta exp (editar PARAM$carpeta_exp)")
if( is.na(PARAM$carpeta_dataset) )
  stop("No encuentro el dataset ", PARAM$dataset, " (editar PARAM$carpeta_dataset)")

if( PARAM$prueba_rapida ) {
  PARAM$experimento <- paste0(PARAM$experimento, "_prueba")
  PARAM$control_experimento <- paste0(PARAM$control_experimento, "_prueba")
  PARAM$semillas <- PARAM$semillas[1:2]
  PARAM$bootstrap_B <- 20
}

dir_exp <- file.path(PARAM$carpeta_exp, PARAM$experimento)
dir_control <- file.path(PARAM$carpeta_exp, PARAM$control_experimento)
dir.create(dir_exp, showWarnings= FALSE, recursive= TRUE)

cfg_control <- "01_rank_cero_fijo_intrames"

cat("exp:", dir_exp, "\ncontrol:", dir_control, "\ndataset:", PARAM$carpeta_dataset, "\n")

# %% [markdown]
# #### Configuraciones a correr
# Cada una es la 01 (rank_cero_fijo intra-mes) más el cambio indicado.

# %%
tb_config <- data.table(
  config= c("11_lag36", "12_medias_moviles", "13_grid_ampliado", "14_todo_junto"),
  lags_largos= c(TRUE, FALSE, FALSE, TRUE),
  medias_moviles= c(FALSE, TRUE, FALSE, TRUE),
  grid= c("original", "original", "ampliado", "ampliado")
)

# cuales correr (para correr solo algunas: PARAM$configs_a_correr <- c("11_lag36"))
PARAM$configs_a_correr <- tb_config$config

fwrite(tb_config, file= file.path(dir_exp, "configuraciones.txt"), sep="\t")
tb_config

# %% [markdown]
# ### Preprocesamiento (igual que en NEW201)
# #### CA Catastrophe Analysis
# Metodo Machine Learning: se asigna NA a los valores de los meses rotos.

# %%
AsignarNA_campomeses <- function(dataset, pcampo, pmeses) {
  if( pcampo %in% colnames(dataset) ) {
    dataset[ foto_mes %in% pmeses, (pcampo) := NA ]
  }
}

# campo = meses en los que esta roto
CA_lista <- list(
  active_quarter = c(202006),
  internet = c(202006),
  mrentabilidad = c(201905, 201910, 202006),
  mrentabilidad_annual = c(201905, 201910, 202006),
  mcomisiones = c(201905, 201910, 202006),
  mactivos_margen = c(201905, 201910, 202006),
  mpasivos_margen = c(201905, 201910, 202006),
  mcuentas_saldo = c(202006),
  ctarjeta_debito_transacciones = c(202006),
  mautoservicio = c(202006),
  ctarjeta_visa_transacciones = c(202006),
  mtarjeta_visa_consumo = c(202006),
  ctarjeta_master_transacciones = c(202006),
  mtarjeta_master_consumo = c(202006),
  ctarjeta_visa_debitos_automaticos = c(201904),
  mttarjeta_visa_debitos_automaticos = c(201904),
  ccajeros_propios_descuentos = c(201910, 202002, 202006, 202009, 202010, 202102),
  mcajeros_propios_descuentos = c(201910, 202002, 202006, 202009, 202010, 202102),
  ctarjeta_visa_descuentos = c(201910, 202002, 202006, 202009, 202010, 202102),
  mtarjeta_visa_descuentos = c(201910, 202002, 202006, 202009, 202010, 202102),
  ctarjeta_master_descuentos = c(201910, 202002, 202006, 202009, 202010, 202102),
  mtarjeta_master_descuentos = c(201910, 202002, 202006, 202009, 202010, 202102),
  ccomisiones_otras = c(201905, 201910, 202006),
  mcomisiones_otras = c(201905, 201910, 202006),
  cextraccion_autoservicio = c(202006),
  mextraccion_autoservicio = c(202006),
  ccheques_depositados = c(202006),
  mcheques_depositados = c(202006),
  ccheques_emitidos = c(202006),
  mcheques_emitidos = c(202006),
  ccheques_depositados_rechazados = c(202006),
  mcheques_depositados_rechazados = c(202006),
  ccheques_emitidos_rechazados = c(202006),
  mcheques_emitidos_rechazados = c(202006),
  tcallcenter = c(202006),
  ccallcenter_transacciones = c(202006),
  thomebanking = c(202006),
  chomebanking_transacciones = c(201910, 202006),
  ccajas_transacciones = c(202006),
  ccajas_consultas = c(202006),
  ccajas_depositos = c(202006, 202105),
  ccajas_extracciones = c(202006),
  ccajas_otras = c(202006),
  catm_trx = c(202006),
  matm = c(202006),
  catm_trx_other = c(202006),
  matm_other = c(202006)
)

Corregir_Rotas <- function(dataset) {
  gc(verbose= FALSE)
  for( campo in names(CA_lista) )
    AsignarNA_campomeses(dataset, campo, CA_lista[[campo]])
  invisible(NULL)
}

# %%
leer_dataset_base <- function() {

  archivo <- file.path(PARAM$carpeta_dataset, PARAM$dataset)
  if( !file.exists(archivo) )
    stop("No encuentro el dataset: ", archivo, "  (editar PARAM$carpeta_dataset)")

  dataset <- fread(archivo)

  if( PARAM$prueba_rapida )
    dataset <- dataset[ numero_de_cliente %% 20 == 0 ]

  setorder(dataset, numero_de_cliente, foto_mes)
  Corregir_Rotas(dataset)

  # por como estan armados los nombres, estos son los campos monetarios
  campos_monetarios <- colnames(dataset)
  campos_monetarios <- campos_monetarios[ campos_monetarios %like%
    "^(m|Visa_m|Master_m|vm_m)" ]

  dataset[, (campos_monetarios) := lapply(.SD, as.numeric),
    .SDcols= campos_monetarios ]

  list(dataset= dataset, campos_monetarios= campos_monetarios)
}

# %% [markdown]
# #### DR Data Drifting: rank_cero_fijo (intra-mes)
# El cero se transforma en cero; los positivos se rankean por su lado y los negativos por el suyo, dentro de cada mes.

# %%
drift_rank_cero_fijo <- function(dataset, campos, claves_grupo= "foto_mes") {

  cat("inicio drift_rank_cero_fijo()\n")
  for( campo in campos ) {
    nuevo <- paste0(campo, "_rank")
    dataset[, (nuevo) := NA_real_ ]

    dataset[ get(campo) == 0, (nuevo) := 0 ]
    dataset[ get(campo) > 0, (nuevo) :=
      frank(get(campo), ties.method= "random") / .N, by= claves_grupo ]

    dataset[ get(campo) < 0, (nuevo) :=
      -frank(-get(campo), ties.method= "random") / .N, by= claves_grupo ]

    dataset[, (campo) := NULL ]
  }
  cat("fin drift_rank_cero_fijo()\n")
}

# %% [markdown]
# ### Feature Engineering
# #### FE intra-mes (igual que la 01)

# %%
FE_intra_mes <- function(dataset) {

  dataset[, kmes := foto_mes %% 100 ]

  if( all(c("mpayroll", "cliente_edad") %in% colnames(dataset)) ) {
    dataset[, mpayroll_sobre_edad := mpayroll / cliente_edad ]
  } else if( all(c("mpayroll_rank", "cliente_edad") %in% colnames(dataset)) ) {
    dataset[, mpayroll_sobre_edad := mpayroll_rank / cliente_edad ]
  }
  invisible(NULL)
}

# %% [markdown]
# #### FE historico
# **Base (igual que la 01):** lag1, lag2, delta1 y delta2 de todos los campos.
#
# **`lags_largos`:** agrega lag3, lag6, delta3 y delta6.
#
# **`medias_moviles`:** agrega la media movil de 3 y de 6 meses (incluye el mes actual) y la diferencia entre el valor y su media (`_dma3`, `_dma6`). Se usa la diferencia y no el cociente porque, con los ranks, hay muchos ceros y el cociente es inestable.
#
# Las variantes nuevas no se calculan para `kmes` (el numero de mes no tiene historia con sentido). Si a un cliente le faltan meses, los lags se arman por posicion dentro del cliente, igual que en la 01.

# %%
FE_historico <- function(dataset, lags_largos= FALSE, medias_moviles= FALSE) {

  # todo es lagueable, menos la primary key y la clase
  cols_lagueables <- setdiff(
    colnames(dataset),
    c("numero_de_cliente", "foto_mes", "clase_ternaria")
  )

  # ---- base: identica a la 01 de NEW201
  dataset[,
    paste0(cols_lagueables, "_lag1") := shift(.SD, 1, NA, "lag"),
    by= numero_de_cliente,
    .SDcols= cols_lagueables
  ]

  dataset[,
    paste0(cols_lagueables, "_lag2") := shift(.SD, 2, NA, "lag"),
    by= numero_de_cliente,
    .SDcols= cols_lagueables
  ]

  for( vcol in cols_lagueables ) {
    dataset[, paste0(vcol, "_delta1") := get(vcol) - get(paste0(vcol, "_lag1")) ]
    dataset[, paste0(vcol, "_delta2") := get(vcol) - get(paste0(vcol, "_lag2")) ]
  }

  cols_hist <- setdiff(cols_lagueables, "kmes")

  # ---- variante: lags de orden 3 y 6
  if( lags_largos ) {
    for( k in c(3, 6) ) {
      dataset[,
        paste0(cols_hist, "_lag", k) := shift(.SD, k, NA, "lag"),
        by= numero_de_cliente,
        .SDcols= cols_hist
      ]
      for( vcol in cols_hist )
        dataset[, paste0(vcol, "_delta", k) := get(vcol) - get(paste0(vcol, "_lag", k)) ]
    }
  }

  # ---- variante: medias moviles de 3 y 6 meses
  if( medias_moviles ) {
    for( k in c(3, 6) ) {
      dataset[,
        paste0(cols_hist, "_ma", k) := frollmean(.SD, k, align= "right", algo= "exact"),
        by= numero_de_cliente,
        .SDcols= cols_hist
      ]
      for( vcol in cols_hist )
        dataset[, paste0(vcol, "_dma", k) := get(vcol) - get(paste0(vcol, "_ma", k)) ]
    }
  }

  invisible(NULL)
}

# %% [markdown]
# ### Modelado
# #### Training strategy (igual que NEW201)

# %%
PARAM$trainingstrategy$validate <- c(202107)

PARAM$trainingstrategy$training <- c(
  201901, 201902, 201903, 201904, 201905, 201906,
  201907, 201908, 201909, 201910, 201911, 201912,
  202001, 202002, 202003, 202004, 202005, 202006,
  202007, 202008, 202009, 202010, 202011, 202012,
  202101, 202102, 202103, 202104, 202105
)

PARAM$trainingstrategy$final_train <- c(
  PARAM$trainingstrategy$training, 202106, 202107
)

PARAM$trainingstrategy$future <- c(202109)
PARAM$trainingstrategy$training_pct <- 1.0
PARAM$trainingstrategy$positivos <- c( "BAJA+1", "BAJA+2" )

# %% [markdown]
# #### Parametros de LightGBM y grids
# Se optimiza **AUC** (no la ganancia), igual que en NEW201.
#
# * `original`: el de NEW201, 5 x 5 x 2 = 50 combinaciones.
# * `ampliado`: 5 x 5 x 3 = 75 combinaciones, con `min_data_in_leaf` hasta 8192 y `feature_fraction` hasta 1.0.

# %%
PARAM$lgbm$param_fijos <- list(
  objective= "binary",
  metric= "auc",
  first_metric_only= TRUE,
  boost_from_average= TRUE,
  feature_pre_filter= FALSE,
  verbosity= -100,
  force_row_wise= TRUE,
  seed= PARAM$semilla_primigenia,
  max_bin= 31,
  learning_rate= 0.03,
  feature_fraction= 0.5,
  num_iterations= 2048,       # valor grande, lo limita early_stopping_rounds
  early_stopping_rounds= 200,
  num_leaves= 64,
  min_data_in_leaf= 128
)

PARAM$lgbm$grids <- list(
  original= CJ(
    num_leaves= c(64, 128, 256, 384, 512),
    min_data_in_leaf= c(64, 256, 512, 1024, 2048),
    feature_fraction= c(0.5, 0.8)
  ),
  ampliado= CJ(
    num_leaves= c(128, 256, 384, 512, 768),
    min_data_in_leaf= c(512, 1024, 2048, 4096, 8192),
    feature_fraction= c(0.5, 0.8, 1.0)
  )
)

if( PARAM$prueba_rapida ) {
  PARAM$lgbm$param_fijos$num_iterations <- 100
  PARAM$lgbm$param_fijos$early_stopping_rounds <- 20
  PARAM$lgbm$grids <- list(
    original= CJ(num_leaves= c(64, 128), min_data_in_leaf= c(512), feature_fraction= c(0.5)),
    ampliado= CJ(num_leaves= c(64, 128), min_data_in_leaf= c(512, 1024), feature_fraction= c(0.5))
  )
}

sapply(PARAM$lgbm$grids, nrow)

# %%
# devuelve la AUC en validate del modelo entrenado y la cantidad de iteraciones
Estimar_AUC_lightgbm <- function(x, dtrain, dvalidate) {

  # x pisa (o agrega) a param_fijos
  param_completo <- modifyList(PARAM$lgbm$param_fijos, x)

  modelo_train <- lgb.train(
    data= dtrain,
    valids= list(valid = dvalidate),
    eval= "auc",
    param= param_completo,
    verbose= -100
  )

  AUC <- modelo_train$record_evals$valid$auc$eval[[modelo_train$best_iter]]

  message(format(Sys.time(), "%a %b %d %X %Y  "),
    toString(x),
    " niter ", modelo_train$best_iter,
    " AUC ", AUC
  )

  niter <- modelo_train$best_iter
  rm(modelo_train)
  gc(full= TRUE, verbose= FALSE)

  return( list(AUC, niter) )
}

# AUC por ranks (Mann-Whitney), sin paquetes extra
auc_ranks <- function(prob, y) {
  r <- frank(prob, ties.method= "average")
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# %% [markdown]
# #### Kaggle
# Solo se **generan** los archivos del ensamble de las 10 semillas (cortes de 1600 a 2600); no se sube nada.

# %%
PARAM$kaggle$cortes <- seq(1600, 2600, by= 100)

# cantidad de campos de la 01 en NEW201, para verificar que el preprocesamiento coincide
campos_control <- NA_integer_
archivo_resumen_control <- file.path(dir_control, cfg_control, "resumen_config.txt")
if( file.exists(archivo_resumen_control) )
  campos_control <- fread(archivo_resumen_control)$cantidad_campos[1]

# Una corrida terminada solo vale si se hizo con el diseno actual: las mismas semillas y
# el mismo tamano de grid. Una prueba rapida (pocas semillas y pocas combinaciones) no cumple.
config_coincide_con_diseno <- function(config, grid) {
  f_grid <- file.path(dir_exp, config, "tb_grid_search.txt")
  f_auc <- file.path(dir_exp, config, "auc_validacion_por_semilla.txt")

  # si un archivo no existe o no se puede abrir (por ejemplo una carpeta vieja o renombrada
  # en el bucket), la corrida NO coincide con el diseno
  tryCatch({
    file.exists(f_grid) && file.exists(f_auc) &&
      nrow(fread(f_grid)) == nrow(PARAM$lgbm$grids[[grid]]) &&
      nrow(fread(f_auc)) == length(PARAM$semillas)
  }, error= function(e) FALSE)
}

campos_control

# %% [markdown]
# ## Corrida de una configuracion
# Hace todo el pipeline de una configuracion y libera la memoria al terminar.

# %%
correr_config <- function(cfg) {

  dir_cfg <- file.path(dir_exp, cfg$config)
  if( file.exists(file.path(dir_cfg, "config_terminada.txt")) ) {
    if( !config_coincide_con_diseno(cfg$config, cfg$grid) )
      stop("La carpeta ", dir_cfg, " tiene una corrida terminada que NO coincide con el diseno ",
        "(otra cantidad de semillas o de combinaciones del grid, por ejemplo una prueba rapida). ",
        "Renombrarla o moverla y volver a correr; no se pisa.")
    message("Ya terminada, la salteo: ", cfg$config)
    return(invisible("saltada"))
  }

  for( carpeta_sub in c("validacion", "prediccion", "kaggle") )
    dir.create(file.path(dir_cfg, carpeta_sub), showWarnings= FALSE, recursive= TRUE)

  t_inicio <- Sys.time()
  message("==== INICIO ", cfg$config, "   ", format(t_inicio))

  # ---- 1. dataset, catastrophe analysis, drift y feature engineering
  base <- leer_dataset_base()
  dataset <- base$dataset
  campos_monetarios <- base$campos_monetarios
  rm(base)

  set.seed(PARAM$semilla_primigenia)   # los empates del rank son al azar
  drift_rank_cero_fijo(dataset, campos_monetarios)

  FE_intra_mes(dataset)
  FE_historico(dataset, lags_largos= cfg$lags_largos, medias_moviles= cfg$medias_moviles)
  gc(verbose= FALSE)

  # ---- 2. training strategy
  dataset[, clase01 := ifelse(clase_ternaria %in% PARAM$trainingstrategy$positivos, 1L, 0L) ]

  campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01", "azar"))
  message("campos buenos: ", length(campos_buenos))

  # sin variables nuevas, tienen que ser los mismos campos que la 01
  if( !cfg$lags_largos && !cfg$medias_moviles && !is.na(campos_control) &&
      length(campos_buenos) != campos_control )
    message("ATENCION: la 01 de NEW201 tuvo ", campos_control, " campos y esta tiene ",
      length(campos_buenos), ": revisar que el preprocesamiento coincida")

  set.seed(PARAM$semilla_primigenia, kind= "L'Ecuyer-CMRG")
  dataset[, azar := runif(nrow(dataset)) ]

  dataset[, fold_train := foto_mes %in% PARAM$trainingstrategy$training &
    (clase_ternaria %in% c("BAJA+1", "BAJA+2") |
     azar < PARAM$trainingstrategy$training_pct) ]

  dtrain <- lgb.Dataset(
    data= data.matrix(dataset[fold_train == TRUE, campos_buenos, with= FALSE]),
    label= dataset[fold_train == TRUE, clase01],
    free_raw_data= TRUE
  )

  dvalidate <- lgb.Dataset(
    data= data.matrix(dataset[foto_mes %in% PARAM$trainingstrategy$validate, campos_buenos, with= FALSE]),
    label= dataset[foto_mes %in% PARAM$trainingstrategy$validate, clase01],
    free_raw_data= TRUE
  )

  # datos de validation en crudo, para predecir con cada semilla
  tb_val <- dataset[foto_mes %in% PARAM$trainingstrategy$validate,
    list(numero_de_cliente, clase_ternaria, clase01)]
  mat_val <- data.matrix(dataset[foto_mes %in% PARAM$trainingstrategy$validate,
    campos_buenos, with= FALSE])

  # ---- 3. grid search (semilla primigenia)
  tb_grid <- copy(PARAM$lgbm$grids[[cfg$grid]])
  message(format(Sys.time()), "  grid ", cfg$grid, ": ", nrow(tb_grid), " combinaciones")
  tb_grid[, c("AUC", "num_iterations") :=
    Estimar_AUC_lightgbm(.SD, dtrain, dvalidate), by= 1:nrow(tb_grid) ]

  setorder(tb_grid, -AUC)
  fwrite(tb_grid, file= file.path(dir_cfg, "tb_grid_search.txt"), sep= "\t")

  auc_grid <- tb_grid[1, AUC]
  mejores <- as.list(tb_grid[1])
  mejores$AUC <- NULL     # queda num_leaves, min_data_in_leaf, feature_fraction, num_iterations

  param_base <- modifyList(PARAM$lgbm$param_fijos, mejores)
  param_base$early_stopping_rounds <- NULL

  # ---- 4. un modelo de validacion por semilla (entrena en training, predice 202107)
  tb_auc <- data.table(semilla= numeric(0), auc_validacion= numeric(0))
  prob_val <- matrix(NA_real_, nrow= nrow(tb_val), ncol= length(PARAM$semillas))

  for( j in seq_along(PARAM$semillas) ) {
    semilla <- PARAM$semillas[j]
    param_s <- param_base
    param_s$seed <- semilla

    modelo <- lgb.train(data= dtrain, params= param_s, verbose= -100)
    prob_val[, j] <- predict(modelo, mat_val)
    rm(modelo)
    gc(full= TRUE, verbose= FALSE)

    auc_s <- auc_ranks(prob_val[, j], tb_val$clase01)
    tb_auc <- rbind(tb_auc, data.table(semilla= semilla, auc_validacion= auc_s))

    fwrite(
      data.table(tb_val[, list(numero_de_cliente, clase_ternaria)], prob= prob_val[, j]),
      file= file.path(dir_cfg, "validacion", paste0("val_", semilla, ".txt.gz")),
      sep= "\t", compress= "gzip"
    )
    message(format(Sys.time()), "  semilla ", semilla, "  AUC validacion ", round(auc_s, 6))
  }

  auc_ensemble <- auc_ranks(rowMeans(prob_val), tb_val$clase01)
  fwrite(tb_auc, file= file.path(dir_cfg, "auc_validacion_por_semilla.txt"), sep= "\t")

  rm(dtrain, dvalidate, mat_val, prob_val, tb_val)
  gc(full= TRUE, verbose= FALSE)

  # ---- 5. modelo final por semilla (entrena en final_train, predice 202109)
  dataset[, fold_final_train := foto_mes %in% PARAM$trainingstrategy$final_train ]

  dfinal_train <- lgb.Dataset(
    data= data.matrix(dataset[fold_final_train == TRUE, campos_buenos, with= FALSE]),
    label= dataset[fold_final_train == TRUE, clase01],
    free_raw_data= TRUE
  )

  tb_futuro <- dataset[foto_mes %in% PARAM$trainingstrategy$future, list(numero_de_cliente)]
  mat_futuro <- data.matrix(dataset[foto_mes %in% PARAM$trainingstrategy$future,
    campos_buenos, with= FALSE])

  rm(dataset)
  gc(full= TRUE, verbose= FALSE)

  prob_fut <- matrix(NA_real_, nrow= nrow(tb_futuro), ncol= length(PARAM$semillas))

  for( j in seq_along(PARAM$semillas) ) {
    semilla <- PARAM$semillas[j]
    param_s <- param_base
    param_s$seed <- semilla

    final_model <- lgb.train(data= dfinal_train, params= param_s, verbose= -100)
    prob_fut[, j] <- predict(final_model, mat_futuro)

    fwrite(
      data.table(numero_de_cliente= tb_futuro$numero_de_cliente, prob= prob_fut[, j]),
      file= file.path(dir_cfg, "prediccion", paste0("pred_", semilla, ".txt.gz")),
      sep= "\t", compress= "gzip"
    )

    # importancia de variables solo con la semilla primigenia
    if( j == 1 )
      fwrite(as.data.table(lgb.importance(final_model)),
        file= file.path(dir_cfg, "impo.txt"), sep= "\t")

    rm(final_model)
    gc(full= TRUE, verbose= FALSE)
    message(format(Sys.time()), "  modelo final semilla ", semilla)
  }

  # ---- 6. ensamble de las semillas y archivos de Kaggle (solo se generan)
  tb_ens <- data.table(numero_de_cliente= tb_futuro$numero_de_cliente, prob= rowMeans(prob_fut))
  fwrite(tb_ens, file= file.path(dir_cfg, "prediccion_ensemble.txt.gz"),
    sep= "\t", compress= "gzip")

  setorder(tb_ens, -prob)

  for( envios in PARAM$kaggle$cortes ) {
    n_marcar <- min(envios, nrow(tb_ens))
    tb_ens[, Predicted := 0L ]
    tb_ens[seq_len(n_marcar), Predicted := 1L ]

    archivo_kaggle <- file.path(dir_cfg, "kaggle", paste0("KA_", cfg$config, "_", envios, ".csv"))
    fwrite(tb_ens[, list(numero_de_cliente, Predicted)], file= archivo_kaggle, sep= ",")
  }

  # ---- 7. resumen y marca de configuracion terminada
  tb_resumen <- data.table(
    config= cfg$config,
    lags_largos= cfg$lags_largos,
    medias_moviles= cfg$medias_moviles,
    grid= cfg$grid,
    cantidad_campos= length(campos_buenos),
    auc_grid_mejor= auc_grid,
    num_leaves= mejores$num_leaves,
    min_data_in_leaf= mejores$min_data_in_leaf,
    feature_fraction= mejores$feature_fraction,
    num_iterations= mejores$num_iterations,
    auc_val_media= mean(tb_auc$auc_validacion),
    auc_val_sd= sd(tb_auc$auc_validacion),
    auc_val_ensemble= auc_ensemble,
    minutos= as.numeric(difftime(Sys.time(), t_inicio, units= "mins"))
  )
  fwrite(tb_resumen, file= file.path(dir_cfg, "resumen_config.txt"), sep= "\t")

  write_yaml(
    c(PARAM[c("experimento", "control_experimento", "dataset", "semillas", "semilla_primigenia",
              "trainingstrategy", "kaggle")],
      list(config= as.list(cfg),
           lgbm_fijos= PARAM$lgbm$param_fijos,
           mejores_hiperparametros= mejores)),
    file= file.path(dir_cfg, "PARAM.yml")
  )

  writeLines(format(Sys.time()), file.path(dir_cfg, "config_terminada.txt"))
  message("==== FIN ", cfg$config, "   ",
    round(tb_resumen$minutos, 1), " minutos   AUC val media ",
    round(tb_resumen$auc_val_media, 6))

  rm(dfinal_train, mat_futuro, prob_fut, tb_ens, tb_futuro)
  gc(full= TRUE, verbose= FALSE)
  invisible("ok")
}

# %% [markdown]
# ## Corrida
# Si una configuracion falla, guarda `error_<config>.txt` y sigue con la siguiente. Si se corta, se vuelve a correr esta celda: saltea las terminadas.

# %%
format(Sys.time(), "%a %b %d %X %Y")

write_yaml(PARAM[c("experimento", "control_experimento", "dataset", "semillas", "semilla_primigenia",
                   "trainingstrategy", "kaggle")],
  file= file.path(dir_exp, "PARAM_global.yml"))

tb_run <- tb_config[ config %in% PARAM$configs_a_correr ]

for( i in seq_len(nrow(tb_run)) ) {
  cfg <- tb_run[i]

  tryCatch(
    correr_config(cfg),
    error= function(e) {
      message("ERROR en ", cfg$config, ": ", conditionMessage(e))
      writeLines(conditionMessage(e), file.path(dir_exp, paste0("error_", cfg$config, ".txt")))
      gc(full= TRUE, verbose= FALSE)
    }
  )
}

format(Sys.time(), "%a %b %d %X %Y")

# %% [markdown]
# ## Comparacion contra la 01
# Cada variante contra la 01 de NEW201 (mismas 10 semillas). El test **principal es el no pareado** (rank-sum): cada configuracion tiene otras variables e hiperparametros, asi que una misma semilla no es "el mismo azar". El pareado se muestra como referencia. Holm corrige por la cantidad de variantes comparadas.
#
# **Regla para decidir:** una variante reemplaza a la 01 si la diferencia es positiva y el p no pareado ajustado por Holm es menor que 0.05.
#
# **Control de validez:** solo entran a la comparacion las corridas hechas con el diseno actual (las 10 semillas y el grid completo de su configuracion). Una corrida de prueba rapida, o con otro grid, se excluye y el cuaderno avisa con "ATENCION". Tambien en la celda de corrida: si una carpeta ya terminada no coincide con el diseno, la configuracion falla con un mensaje en vez de saltearse en silencio (hay que renombrar esa carpeta).

# %%
terminadas <- tb_config$config[
  file.exists(file.path(dir_exp, tb_config$config, "config_terminada.txt")) ]

hay_control <- file.exists(file.path(dir_control, cfg_control, "auc_validacion_por_semilla.txt"))

# solo entran a la comparacion las corridas hechas con el diseno actual
if( length(terminadas) > 0 ) {
  valida <- vapply(terminadas, function(cfg) {
    config_coincide_con_diseno(cfg, tb_config[config == cfg, grid])
  }, logical(1))
  if( any(!valida) )
    message("ATENCION: se EXCLUYEN de la comparacion porque no coinciden con el diseno ",
      "(otra cantidad de semillas o de combinaciones del grid, por ejemplo una prueba rapida): ",
      paste(terminadas[!valida], collapse= ", "))
  terminadas <- terminadas[valida]
}

# la 01 de NEW201 tiene que tener las mismas semillas
if( hay_control ) {
  n_sem_control <- nrow(fread(file.path(dir_control, cfg_control, "auc_validacion_por_semilla.txt")))
  if( n_sem_control != length(PARAM$semillas) ) {
    message("ATENCION: la 01 de NEW201 tiene ", n_sem_control, " semillas y aca se usan ",
      length(PARAM$semillas), ": no se compara")
    hay_control <- FALSE
  }
}

dir_fig <- file.path(dir_exp, "figuras")
dir.create(dir_fig, showWarnings= FALSE)

if( length(terminadas) == 0 ) message("Todavia no hay configuraciones terminadas en ", dir_exp)
if( !hay_control ) message("No encuentro la 01 en ", dir_control, ": se saltea la comparacion contra la 01")

redondear <- function(tb, d= 6) {
  x <- copy(tb)
  for( j in names(x) ) if( is.numeric(x[[j]]) ) set(x, j= j, value= round(x[[j]], d))
  x
}

# %%
if( length(terminadas) > 0 && hay_control ) {

  tb_resumen_todas <- rbindlist(c(
    list( fread(file.path(dir_control, cfg_control, "resumen_config.txt")) ),
    lapply(terminadas, function(cfg) fread(file.path(dir_exp, cfg, "resumen_config.txt")))
  ), fill= TRUE)

  setorder(tb_resumen_todas, -auc_val_media)

  tb_ver <- redondear(tb_resumen_todas[, list(
    config, grid, lags_largos, medias_moviles, cantidad_campos,
    auc_val_media, auc_val_sd, auc_val_ensemble, auc_grid_mejor,
    num_leaves, min_data_in_leaf, feature_fraction, num_iterations, minutos
  )], 6)

  fwrite(tb_ver, file.path(dir_exp, "resumen_comparacion.txt"), sep= "\t")
  tb_ver
}

# %% [markdown]
# ### Wilcoxon de cada variante contra la 01

# %%
if( length(terminadas) > 0 && hay_control ) {

  tb_semillas <- rbindlist(lapply(c(cfg_control, terminadas), function(cfg) {
    carpeta <- if( cfg == cfg_control ) dir_control else dir_exp
    x <- fread(file.path(carpeta, cfg, "auc_validacion_por_semilla.txt"))
    x[, config := cfg ]
    x
  }))

  a <- tb_semillas[config == cfg_control][order(semilla), auc_validacion]

  # si el test no se puede calcular (por ejemplo, resultados identicos) devuelve NA
  p_wilcoxon <- function(x, y, pareado) {
    tryCatch(
      suppressWarnings(wilcox.test(x, y, paired= pareado, exact= FALSE)$p.value),
      error= function(e) NA_real_
    )
  }

  tb_wil <- rbindlist(lapply(terminadas, function(cfg) {
    b <- tb_semillas[config == cfg][order(semilla), auc_validacion]
    data.table(
      config= cfg,
      diferencia_vs_01= mean(b) - mean(a),
      semillas_que_mejoran= sum(b > a),
      semillas= length(b),
      p_no_pareado= p_wilcoxon(b, a, FALSE),
      p_pareado= p_wilcoxon(b, a, TRUE)
    )
  }))

  tb_wil[, p_no_pareado_holm := p.adjust(p_no_pareado, method= "holm") ]
  tb_wil[, p_pareado_holm := p.adjust(p_pareado, method= "holm") ]
  tb_wil[, veredicto := ifelse(!is.na(p_no_pareado_holm) & p_no_pareado_holm < 0.05,
    ifelse(diferencia_vs_01 > 0, "mejora a la 01", "empeora a la 01"),
    "sin diferencia detectable") ]
  setorder(tb_wil, -diferencia_vs_01)

  tb_wil_ver <- redondear(tb_wil, 6)
  fwrite(tb_wil_ver, file.path(dir_exp, "wilcoxon_contra_la_01.txt"), sep= "\t")
  tb_wil_ver
}

# %%
if( length(terminadas) > 0 && hay_control ) {

  grafico_semillas <- function() {
    g <- copy(tb_semillas)
    orden <- tb_semillas[, list(m= mean(auc_validacion)), by= config][order(m), config]
    g[, cfg := factor(config, levels= orden)]

    par(mar= c(4, 17, 3, 1))
    boxplot(auc_validacion ~ cfg, data= g, horizontal= TRUE, las= 1, cex.axis= 0.85,
      col= "grey92", outline= FALSE, ylab= "",
      xlab= "AUC en validación (202107)",
      main= "AUC por semilla: la 01 contra las variantes")
    stripchart(auc_validacion ~ cfg, data= g, vertical= FALSE, method= "jitter",
      pch= 16, cex= 0.8, col= "steelblue", add= TRUE)
  }

  options(repr.plot.width= 11, repr.plot.height= 5)
  grafico_semillas()

  png(file.path(dir_fig, "auc_por_semilla_variantes.png"), width= 1500, height= 750, res= 130)
  grafico_semillas()
  invisible(dev.off())
}

# %% [markdown]
# ## Ensambles
# Cada configuracion ya es un ensamble de sus 10 semillas. Aca se promedian los **ranks** de varias configuraciones (la 01 y la 05 de NEW201 y las nuevas) y se mide el AUC en 202107 con las probabilidades de validacion guardadas.
#
# Para cada combinacion se calcula la **diferencia contra la 01 sola** con un bootstrap por clientes (los mismos clientes remuestreados para las dos). El intervalo mide cuanto cambia el resultado por los clientes que hay en 202107; **no** mide que pasaria en otro mes.
#
# Ojo: combinaciones elegidas mirando el mismo mes de validacion dan una mejora optimista.
#
# Las fuentes se alinean por `numero_de_cliente`. Antes de calcular, el cuaderno imprime cuantos clientes tiene cada fuente y de que tipo es el id. Si no son exactamente los mismos usa solo los comunes y avisa; si casi no coinciden, frena (señal de que NEW201 y NEW202 no usaron el mismo dataset).

# %%
ens_val <- function(dir_cfg) {
  archivos <- list.files(file.path(dir_cfg, "validacion"), full.names= TRUE)
  x <- rbindlist(lapply(archivos, fread))
  x <- x[, list(prob= mean(prob)), by= list(numero_de_cliente, clase_ternaria)]
  x[, clase01 := as.integer(clase_ternaria %in% c("BAJA+1", "BAJA+2")) ]
  x[, list(numero_de_cliente, clase01, prob)]
}

# Deja a todas las fuentes con los mismos clientes y en el mismo orden (se alinea por
# numero_de_cliente, no por posicion) y calcula el rank de cada una sobre esos clientes.
# Muestra cuantos clientes tiene cada fuente; si casi no coinciden, frena con un mensaje.
alinear_clientes <- function(lista, umbral= 0.99) {

  tb_diag <- data.table(
    fuente= names(lista),
    clientes= vapply(lista, nrow, integer(1)),
    tipo_del_id= vapply(lista, function(x) class(x$numero_de_cliente)[1], character(1)),
    ids_repetidos= vapply(lista, function(x) sum(duplicated(x$numero_de_cliente)), integer(1))
  )
  print(tb_diag)

  if( any(tb_diag$ids_repetidos > 0) )
    stop("Hay clientes repetidos en alguna fuente (ver la tabla): no se puede alinear")

  comunes <- Reduce(intersect, lapply(lista, function(x) x$numero_de_cliente))
  cat("clientes comunes a todas las fuentes:", length(comunes), "\n")

  # se compara contra la fuente con MAS clientes: si una pierde muchos, es otro dataset
  if( length(comunes) < umbral * max(tb_diag$clientes) )
    stop("Las fuentes no usan los mismos clientes (comunes: ", length(comunes),
      ", maximo en una fuente: ", max(tb_diag$clientes),
      "). Revisar que NEW201 y NEW202 hayan usado el mismo dataset (ver la tabla).")

  if( length(comunes) < max(tb_diag$clientes) )
    message("AVISO: se usan solo los ", length(comunes), " clientes comunes a todas las fuentes")

  lapply(lista, function(x) {
    x <- x[ numero_de_cliente %in% comunes ]
    setorder(x, numero_de_cliente)
    x[, rk := frank(prob) / .N ]
    x
  })
}

fuentes <- c(
  "01" = file.path(dir_control, "01_rank_cero_fijo_intrames"),
  "05" = file.path(dir_control, "05_rank_cero_fijo_kmeans_feature")
)
for( cfg in terminadas ) fuentes[ sub("_.*", "", cfg) ] <- file.path(dir_exp, cfg)

fuentes <- fuentes[ dir.exists(file.path(fuentes, "validacion")) ]
hay_ensambles <- "01" %in% names(fuentes) && length(fuentes) >= 2

if( !hay_ensambles ) message("No hay probabilidades de validacion suficientes: se saltea la seccion de ensambles")

# %%
if( hay_ensambles ) {

  lista <- lapply(fuentes, ens_val)
  names(lista) <- names(fuentes)
  lista <- alinear_clientes(lista)

  # la clase de cada cliente tiene que ser la misma en todas las fuentes
  for( x in lista ) {
    if( !identical(as.integer(x$clase01), as.integer(lista[[1]]$clase01)) )
      stop("La clase de validacion de los clientes no coincide entre fuentes: revisar el dataset")
  }
  y <- lista[[1]]$clase01

  nuevas <- sub("_.*", "", terminadas)
  nuevas <- nuevas[ nuevas %in% names(lista) ]

  combos <- list("01 sola" = "01")
  if( "05" %in% names(lista) ) combos[["01 + 05"]] <- c("01", "05")
  for( id in nuevas ) {
    combos[[paste("01 +", id)]] <- c("01", id)
    if( "05" %in% names(lista) ) combos[[paste("01 + 05 +", id)]] <- c("01", "05", id)
  }
  if( length(nuevas) > 1 ) combos[["01 + todas las nuevas"]] <- c("01", nuevas)

  scores <- lapply(combos, function(ids) {
    rowMeans(sapply(lista[ids], function(x) x$rk), na.rm= FALSE)
  })

  B <- PARAM$bootstrap_B
  n <- length(y)
  M <- matrix(NA_real_, nrow= B, ncol= length(scores))
  colnames(M) <- names(scores)

  set.seed(PARAM$semilla_primigenia)
  for( i in seq_len(B) ) {
    idx <- sample.int(n, n, replace= TRUE)
    M[i, ] <- vapply(scores, function(s) auc_ranks(s[idx], y[idx]), numeric(1))
  }
  dif <- M - M[, "01 sola"]

  tb_ens <- data.table(
    combinacion= names(scores),
    auc_ensemble= vapply(scores, function(s) auc_ranks(s, y), numeric(1)),
    dif_vs_01_sola= NA_real_,
    ic95_inferior= NA_real_,
    ic95_superior= NA_real_,
    prob_mejora= NA_real_
  )
  for( j in seq_along(scores) ) {
    d <- dif[, j]
    set(tb_ens, j, "dif_vs_01_sola", tb_ens$auc_ensemble[j] - tb_ens$auc_ensemble[1])
    set(tb_ens, j, "ic95_inferior", unname(quantile(d, 0.025)))
    set(tb_ens, j, "ic95_superior", unname(quantile(d, 0.975)))
    set(tb_ens, j, "prob_mejora", mean(d > 0))
  }

  tb_ens_ver <- redondear(tb_ens, 6)
  fwrite(tb_ens_ver, file.path(dir_exp, "ensambles_vs_01_sola.txt"), sep= "\t")
  tb_ens_ver
}

# %% [markdown]
# ## Ensamble final para 202109
# Promedia los ranks de las predicciones de 202109 (el ensamble de las 10 semillas de cada configuracion) y deja la prediccion y los archivos con los cortes de 1600 a 2600. **No se sube nada a Kaggle.**
#
# Por defecto es **01 + 05** de NEW201. Si en la comparacion de arriba alguna variante nueva resulta mejor, agregarla a la lista (formato `"carpeta/configuracion"` relativo a la carpeta exp).

# %%
PARAM$ensemble_final <- c(
  paste0(PARAM$control_experimento, "/01_rank_cero_fijo_intrames"),
  paste0(PARAM$control_experimento, "/05_rank_cero_fijo_kmeans_feature")
  # , paste0(PARAM$experimento, "/11_lag36")
)

archivos_pred <- file.path(PARAM$carpeta_exp, PARAM$ensemble_final, "prediccion_ensemble.txt.gz")

if( all(file.exists(archivos_pred)) ) {

  lista_pred <- lapply(archivos_pred, function(f) fread(f)[, list(numero_de_cliente, prob)])
  names(lista_pred) <- PARAM$ensemble_final
  lista_pred <- alinear_clientes(lista_pred)

  tb_final <- data.table(
    numero_de_cliente= lista_pred[[1]]$numero_de_cliente,
    prob= rowMeans(sapply(lista_pred, function(x) x$rk))
  )

  dir_final <- file.path(dir_exp, "ensemble_final")
  dir.create(file.path(dir_final, "kaggle"), showWarnings= FALSE, recursive= TRUE)

  fwrite(tb_final, file.path(dir_final, "prediccion_ensemble_final.txt.gz"),
    sep= "\t", compress= "gzip")
  writeLines(PARAM$ensemble_final, file.path(dir_final, "componentes.txt"))

  setorder(tb_final, -prob)
  for( envios in PARAM$kaggle$cortes ) {
    n_marcar <- min(envios, nrow(tb_final))
    tb_final[, Predicted := 0L ]
    tb_final[seq_len(n_marcar), Predicted := 1L ]
    fwrite(tb_final[, list(numero_de_cliente, Predicted)],
      file= file.path(dir_final, "kaggle", paste0("KA_ensemble_final_", envios, ".csv")), sep= ",")
  }

  message("Ensamble final guardado en ", dir_final, " (componentes: ", length(archivos_pred), ")")

} else {
  message("Faltan predicciones de 202109 para el ensamble final:\n",
    paste(archivos_pred[ !file.exists(archivos_pred) ], collapse= "\n"))
}

# %%
list.files(dir_exp)
