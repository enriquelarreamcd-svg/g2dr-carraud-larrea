# %% [markdown]
# # NEW201 · Data drifting con 10 semillas: intra-mes vs k-means
#
# Basado en `z620_WorkFlow_01_junior_grupoA`. Todo el código está en R.
#
# **Métodos de drift**: `rank_cero_fijo`, `deflacion` (IPC), `deflacion_usd_oficial` y `deflacion_usd_blue`.
#
# **Variantes** (9 configuraciones en total):
# * `intrames`: solo el feature engineering intra-mes del cuaderno base, sin k-means.
# * `kmeans_feature`: lo anterior + el cluster de k-means y la distancia a su centroide como variables.
# * `kmeans_drift`: solo para `rank_cero_fijo`; el rank se calcula dentro de cada cluster de cada mes (con las deflaciones no tiene sentido, porque el índice es igual para todos los clientes).
#
# **K-means**: se ajusta una sola vez sobre el rank intra-mes (libre de drift) de las variables monetarias, usando solo los meses de training. Después cada cliente de cada mes se asigna al centroide más cercano, así el cluster significa lo mismo en todos los meses y es idéntico para todos los métodos de drift.
#
# **Por configuración**: Grid Search completo (50 combinaciones, semilla primigenia) y después, para cada una de las 10 semillas, un modelo de validación (202107) y un modelo final (predice 202109). Se guardan las probabilidades, el AUC, el ensamble de las 10 semillas y los archivos para Kaggle.
#
# **Salida**: `exp/NEW201/<configuracion>/` y los resúmenes en `exp/NEW201/`.

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

PARAM$experimento <- "NEW201"
PARAM$dataset <- "analistajr_competencia_2026.csv.gz"

# TRUE = corrida chica (5% de los clientes, grid de 2 combinaciones, 2 semillas)
#  para verificar que todo funciona; guarda en NEW201_prueba, nunca en NEW201
PARAM$prueba_rapida <- FALSE

# la primera semilla es la semilla primigenia
PARAM$semillas <- c(165463, 329969, 454253, 593899, 980249,
                    100183, 250027, 400031, 550049, 700057)
PARAM$semilla_primigenia <- PARAM$semillas[1]

PARAM$carpeta_exp <- "/content/buckets/b1/exp"
PARAM$carpeta_dataset <- "/content/datasets"

PARAM$drift_metodos <- c("rank_cero_fijo", "deflacion",
                         "deflacion_usd_oficial", "deflacion_usd_blue")

# k-means: se ajusta sobre una muestra de clientes por cada mes de training
PARAM$kmeans$k <- 8
PARAM$kmeans$muestra_por_mes <- 4000
PARAM$kmeans$semilla <- PARAM$semilla_primigenia

if( PARAM$prueba_rapida ) {
  PARAM$experimento <- paste0(PARAM$experimento, "_prueba")
  PARAM$semillas <- PARAM$semillas[1:2]
  PARAM$kmeans$muestra_por_mes <- 1000
}

# %% [markdown]
# #### Carpeta del Experimento
# Cada configuracion tiene su propia subcarpeta. Si una configuracion ya termino (tiene `config_terminada.txt`) se saltea, asi la corrida se puede retomar sin pisar nada.

# %%
dir_exp <- file.path(PARAM$carpeta_exp, PARAM$experimento)
dir.create(dir_exp, showWarnings=FALSE, recursive=TRUE)
dir_exp

# %% [markdown]
# #### Configuraciones a correr

# %%
tb_config <- rbindlist(list(
  data.table(variante= "intrames",       dr= PARAM$drift_metodos),
  data.table(variante= "kmeans_feature", dr= PARAM$drift_metodos),
  data.table(variante= "kmeans_drift",   dr= "rank_cero_fijo")
))
tb_config[, config := sprintf("%02d_%s_%s", .I, dr, variante)]

fwrite(tb_config, file= file.path(dir_exp, "configuraciones.txt"), sep="\t")
tb_config

# %% [markdown]
# ### Preprocesamiento
# #### CA Catastrophe Analysis
# Igual que el cuaderno base: metodo Machine Learning, se asigna NA a los valores de los meses rotos.

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

# %% [markdown]
# #### Lectura del dataset + CA
# Las variables monetarias se pasan a numerico para que ninguna deflacion se trunque si alguna columna viene como entera.

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
# ### DR Data Drifting
# Indices calculados por alumnos (momento 1.0 = 31-dic-2020).

# %%
vfoto_mes <- c(
  201901, 201902, 201903, 201904, 201905, 201906,
  201907, 201908, 201909, 201910, 201911, 201912,
  202001, 202002, 202003, 202004, 202005, 202006,
  202007, 202008, 202009, 202010, 202011, 202012,
  202101, 202102, 202103, 202104, 202105, 202106,
  202107, 202108, 202109
)

vIPC <- c(
  1.9903030878, 1.9174403544, 1.8296186587,
  1.7728862972, 1.7212488323, 1.6776304408,
  1.6431248196, 1.5814483345, 1.4947526791,
  1.4484037589, 1.3913580777, 1.3404220402,
  1.3154288912, 1.2921698342, 1.2472681797,
  1.2300475145, 1.2118694724, 1.1881073259,
  1.1693969743, 1.1375456949, 1.1065619600,
  1.0681100000, 1.0370000000, 1.0000000000,
  0.9680542110, 0.9344152616, 0.8882274350,
  0.8532444140, 0.8251880213, 0.8003763543,
  0.7763107219, 0.7566381305, 0.7289384687
)

vdolar_blue <- c(
   39.045455,  38.402500,  41.639474,
   44.274737,  46.095455,  45.063333,
   43.983333,  54.842857,  61.059524,
   65.545455,  66.750000,  72.368421,
   77.477273,  78.191667,  82.434211,
  101.087500, 126.236842, 125.857143,
  130.782609, 133.400000, 137.954545,
  170.619048, 160.400000, 153.052632,
  157.900000, 149.380952, 143.615385,
  146.250000, 153.550000, 162.000000,
  178.478261, 180.878788, 184.357143
)

vdolar_oficial <- c(
   38.430000,  39.428000,  42.542105,
   44.354211,  46.088636,  44.955000,
   43.751429,  54.650476,  58.790000,
   61.403182,  63.012632,  63.011579,
   62.983636,  63.580556,  65.200000,
   67.872000,  70.047895,  72.520952,
   75.324286,  77.488500,  79.430909,
   83.134762,  85.484737,  88.181667,
   91.474000,  93.997778,  96.635909,
   98.526000,  99.613158, 100.619048,
  101.619048, 102.569048, 103.781818
)

tb_indices <- data.table(
  foto_mes = vfoto_mes,
  IPC = vIPC,
  dolar_blue = vdolar_blue,
  dolar_oficial = vdolar_oficial
)

stopifnot( nrow(tb_indices) == 33 )

# %%
# deflacion: multiplica por el IPC
# usd oficial / usd blue: divide por el valor del dolar
drift_por_indice <- function(dataset, campos, indice, operacion) {

  cat("inicio drift", indice, operacion, "\n")
  factor <- tb_indices[[indice]][ match(dataset$foto_mes, tb_indices$foto_mes) ]
  stopifnot( !anyNA(factor) )   # todos los foto_mes tienen que tener indice

  for( campo in campos ) {
    if( operacion == "*" ) {
      set(dataset, j= campo, value= dataset[[campo]] * factor)
    } else {
      set(dataset, j= campo, value= dataset[[campo]] / factor)
    }
  }
  cat("fin drift", indice, "\n")
}

# %%
# El cero se transforma en cero
# los positivos se rankean por su lado, los negativos por el suyo
# claves_grupo: "foto_mes" = rank intra-mes
#   c("foto_mes","kmeans_cluster") = rank dentro de cada cluster de cada mes
drift_rank_cero_fijo <- function(dataset, campos, claves_grupo= "foto_mes") {

  cat("inicio drift_rank_cero_fijo()\n")
  for( campo in campos ) {
    cat(campo, " ")
    nuevo <- paste0(campo, "_rank")
    dataset[, (nuevo) := NA_real_ ]

    dataset[ get(campo) == 0, (nuevo) := 0 ]
    dataset[ get(campo) > 0, (nuevo) :=
      frank(get(campo), ties.method= "random") / .N, by= claves_grupo ]

    dataset[ get(campo) < 0, (nuevo) :=
      -frank(-get(campo), ties.method= "random") / .N, by= claves_grupo ]

    dataset[, (campo) := NULL ]
  }
  cat("\nfin drift_rank_cero_fijo()\n")
}

# %% [markdown]
# ### K-means intra-mes
# Representacion: rank de 0 a 1 dentro del mes de cada campo monetario (mas edad y antiguedad si existen). Es libre de drift, por eso los clusters no dependen del metodo de drift que se pruebe. Los NA toman 0.5.
#
# Los centroides se ajustan una sola vez con una muestra de los meses de training; despues se asigna **todo** el dataset al centroide mas cercano. Se guarda en `kmeans_asignacion.txt.gz`.

# %%
rank_mes <- function(x) {
  n <- sum(!is.na(x))
  if( n < 2 ) return( rep(0.5, length(x)) )
  r <- (frank(x, ties.method= "average", na.last= "keep") - 1) / (n - 1)
  r[ is.na(r) ] <- 0.5
  r
}

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

# %%
asignar_kmeans_global <- function() {

  archivo <- file.path(dir_exp, "kmeans_asignacion.txt.gz")
  if( file.exists(archivo) ) {
    message("k-means ya calculado: ", archivo)
    return(invisible(TRUE))
  }

  message(format(Sys.time()), "  calculando k-means (una sola vez)")
  base <- leer_dataset_base()
  dataset <- base$dataset
  campos_km <- c( base$campos_monetarios,
    intersect(c("cliente_edad", "cliente_antiguedad"), colnames(dataset)) )
  meses <- sort( unique(dataset$foto_mes) )

  # matriz de ranks intra-mes de un mes (filas = clientes, en el orden del dataset)
  matriz_rank_mes <- function(mes_k) {
    datos_mes <- dataset[ foto_mes == mes_k, campos_km, with= FALSE ]
    matrix( unlist(lapply(datos_mes, rank_mes), use.names= FALSE), nrow= nrow(datos_mes) )
  }

  # 1. ajuste de los centroides con una muestra de los meses de training
  set.seed(PARAM$kmeans$semilla)
  lista_fit <- list()
  for( mes_k in intersect(meses, PARAM$trainingstrategy$training) ) {
    X <- matriz_rank_mes(mes_k)
    idx <- sample.int(nrow(X), min(nrow(X), PARAM$kmeans$muestra_por_mes))
    lista_fit[[length(lista_fit) + 1]] <- X[idx, , drop= FALSE]
  }
  Xfit <- do.call(rbind, lista_fit)
  rm(lista_fit)

  km <- kmeans(Xfit, centers= PARAM$kmeans$k, iter.max= 100, nstart= 1)
  centros <- km$centers
  centros2 <- rowSums(centros^2)
  rm(Xfit, km)

  # 2. asigno todos los clientes de todos los meses al centroide mas cercano
  lista_asig <- vector("list", length(meses))
  for( i in seq_along(meses) ) {
    mes_k <- meses[i]
    ids <- dataset[ foto_mes == mes_k, numero_de_cliente ]
    X <- matriz_rank_mes(mes_k)

    D2 <- outer(rowSums(X^2), centros2, "+") - 2 * (X %*% t(centros))
    cl <- max.col(-D2, ties.method= "first")
    dist <- sqrt( pmax(D2[cbind(seq_len(nrow(D2)), cl)], 0) )

    lista_asig[[i]] <- data.table(
      numero_de_cliente = ids,
      foto_mes = mes_k,
      kmeans_cluster = cl,
      kmeans_dist = dist
    )
  }
  tb_km <- rbindlist(lista_asig)

  fwrite(tb_km, file= archivo, sep= "\t", compress= "gzip")

  # tamanio de cada cluster por mes, para ver que sean estables
  tb_tam <- dcast(tb_km[, .N, by= list(foto_mes, kmeans_cluster)],
    foto_mes ~ kmeans_cluster, value.var= "N", fill= 0)
  fwrite(tb_tam, file= file.path(dir_exp, "kmeans_tamanios.txt"), sep= "\t")
  fwrite(as.data.table(centros), file= file.path(dir_exp, "kmeans_centroides.txt"), sep= "\t")

  message(format(Sys.time()), "  k-means listo")
  invisible(TRUE)
}

# %% [markdown]
# ### Feature Engineering
# #### FE intra-mes
# Para los metodos de rank, `mpayroll` ya no existe (queda `mpayroll_rank`), asi que `mpayroll_sobre_edad` se arma con esa columna. Asi el feature existe en las 4 variantes de drift y la comparacion es pareja.

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
# #### FEhist: lags y deltas de orden 1 y 2

# %%
FE_historico <- function(dataset) {

  # todo es lagueable, menos la primary key y la clase
  cols_lagueables <- setdiff(
    colnames(dataset),
    c("numero_de_cliente", "foto_mes", "clase_ternaria")
  )

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
  invisible(NULL)
}

# %% [markdown]
# ### Modelado
# #### Parametros de LightGBM
# Igual que el cuaderno base. Se optimiza **AUC** (no la ganancia). El Grid Search es el completo: 5 x 5 x 2 = 50 combinaciones.

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

PARAM$lgbm$grid <- CJ(
  num_leaves= c(64, 128, 256, 384, 512),
  min_data_in_leaf= c(64, 256, 512, 1024, 2048),
  feature_fraction= c(0.5, 0.8)
)

if( PARAM$prueba_rapida ) {
  PARAM$lgbm$param_fijos$num_iterations <- 100
  PARAM$lgbm$param_fijos$early_stopping_rounds <- 20
  PARAM$lgbm$grid <- CJ(
    num_leaves= c(64, 128),
    min_data_in_leaf= c(512),
    feature_fraction= c(0.5)
  )
}

nrow(PARAM$lgbm$grid)

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
# ### Kaggle
# Por cada configuracion se generan los archivos del ensamble de las 10 semillas, con cortes fijos de 1600 a 2600. **No se suben solos** (`submit = FALSE`): Kaggle acepta 100 envios por dia y entre las 9 configuraciones hay 99 archivos.

# %%
PARAM$kaggle$competencia <- "labo-1-virtual-2026-analista-jr"
PARAM$kaggle$cortes <- seq(1600, 2600, by= 100)
PARAM$kaggle$submit <- FALSE

# %% [markdown]
# ## Corrida de una configuracion
# Esta funcion hace todo el pipeline de una configuracion y libera la memoria al terminar.

# %%
correr_config <- function(cfg) {

  dir_cfg <- file.path(dir_exp, cfg$config)
  if( file.exists(file.path(dir_cfg, "config_terminada.txt")) ) {
    message("Ya terminada, la salteo: ", cfg$config)
    return(invisible("saltada"))
  }

  for( carpeta_sub in c("validacion", "prediccion", "kaggle") )
    dir.create(file.path(dir_cfg, carpeta_sub), showWarnings= FALSE, recursive= TRUE)

  t_inicio <- Sys.time()
  message("==== INICIO ", cfg$config, "   ", format(t_inicio))

  # ---- 1. dataset y catastrophe analysis
  base <- leer_dataset_base()
  dataset <- base$dataset
  campos_monetarios <- base$campos_monetarios
  rm(base)

  # ---- 2. k-means (cluster precalculado, igual para todos los metodos)
  if( cfg$variante != "intrames" ) {
    tb_km <- fread(file.path(dir_exp, "kmeans_asignacion.txt.gz"))
    dataset[tb_km, on= c("numero_de_cliente", "foto_mes"),
      `:=`(kmeans_cluster = i.kmeans_cluster, kmeans_dist = i.kmeans_dist)]
    rm(tb_km)
  }

  # ---- 3. data drifting
  set.seed(PARAM$semilla_primigenia)   # los empates del rank son al azar
  claves <- "foto_mes"
  if( cfg$variante == "kmeans_drift" ) claves <- c("foto_mes", "kmeans_cluster")

  switch( cfg$dr,
    "rank_cero_fijo"        = drift_rank_cero_fijo(dataset, campos_monetarios, claves),
    "deflacion"             = drift_por_indice(dataset, campos_monetarios, "IPC", "*"),
    "deflacion_usd_oficial" = drift_por_indice(dataset, campos_monetarios, "dolar_oficial", "/"),
    "deflacion_usd_blue"    = drift_por_indice(dataset, campos_monetarios, "dolar_blue", "/"),
    stop("metodo de drift desconocido: ", cfg$dr)
  )

  # en kmeans_drift el cluster solo sirve para rankear, no es feature
  if( cfg$variante == "kmeans_drift" )
    dataset[, c("kmeans_cluster", "kmeans_dist") := NULL ]

  # ---- 4. feature engineering
  FE_intra_mes(dataset)
  FE_historico(dataset)
  gc(verbose= FALSE)

  # ---- 5. training strategy
  dataset[, clase01 := ifelse(clase_ternaria %in% PARAM$trainingstrategy$positivos, 1L, 0L) ]

  campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01", "azar"))
  message("campos buenos: ", length(campos_buenos))

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

  # ---- 6. grid search (semilla primigenia)
  message(format(Sys.time()), "  grid search: ", nrow(PARAM$lgbm$grid), " combinaciones")
  tb_grid <- copy(PARAM$lgbm$grid)
  tb_grid[, c("AUC", "num_iterations") :=
    Estimar_AUC_lightgbm(.SD, dtrain, dvalidate), by= 1:nrow(tb_grid) ]

  setorder(tb_grid, -AUC)
  fwrite(tb_grid, file= file.path(dir_cfg, "tb_grid_search.txt"), sep= "\t")

  auc_grid <- tb_grid[1, AUC]
  mejores <- as.list(tb_grid[1])
  mejores$AUC <- NULL     # queda num_leaves, min_data_in_leaf, feature_fraction, num_iterations

  # modifyList para que valgan los mejores hiperparametros y no queden nombres repetidos
  param_base <- modifyList(PARAM$lgbm$param_fijos, mejores)
  param_base$early_stopping_rounds <- NULL

  # ---- 7. un modelo de validacion por semilla (entrena en training, predice 202107)
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

  # libero memoria antes de armar el final training
  rm(dtrain, dvalidate, mat_val, prob_val, tb_val)
  gc(full= TRUE, verbose= FALSE)

  # ---- 8. modelo final por semilla (entrena en final_train, predice 202109)
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

  # ---- 9. ensamble de las semillas y archivos de Kaggle
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

    if( PARAM$kaggle$submit ) {
      linea <- paste("kaggle competitions submit",
        "-c", PARAM$kaggle$competencia,
        "-f", archivo_kaggle,
        paste0("-m 'config=", cfg$config, "  envios=", envios, "  ensemble10'"))
      salida <- system(linea, intern= TRUE)
      Sys.sleep(30)
      cat(salida, "\n")
    }
  }

  # ---- 10. resumen y marca de configuracion terminada
  tb_resumen <- data.table(
    config= cfg$config,
    dr= cfg$dr,
    variante= cfg$variante,
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
    c(PARAM[c("experimento", "dataset", "semillas", "semilla_primigenia",
              "trainingstrategy", "kmeans", "kaggle")],
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
# ## Resumen y comparacion intra-mes vs k-means
# Compara cada variante con k-means contra `intrames` del mismo metodo de drift. Como las 10 semillas son las mismas, el Wilcoxon es **pareado por semilla**.

# %%
armar_resumen <- function() {

  carpetas <- file.path(dir_exp, tb_config$config)
  terminadas <- carpetas[ file.exists(file.path(carpetas, "config_terminada.txt")) ]
  if( length(terminadas) == 0 ) {
    message("todavia no hay configuraciones terminadas")
    return(invisible(NULL))
  }

  tb_res <- rbindlist(lapply(terminadas,
    function(d) fread(file.path(d, "resumen_config.txt"))))
  setorder(tb_res, -auc_val_media)
  fwrite(tb_res, file= file.path(dir_exp, "resumen_configuraciones.txt"), sep= "\t")

  tb_long <- rbindlist(lapply(terminadas, function(d) {
    x <- fread(file.path(d, "auc_validacion_por_semilla.txt"))
    x[, config := basename(d) ]
    x
  }))
  tb_long <- merge(tb_long, tb_config[, list(config, dr, variante)], by= "config")
  fwrite(tb_long, file= file.path(dir_exp, "auc_validacion_todas_las_semillas.txt"), sep= "\t")

  filas <- list()
  for( metodo in unique(tb_long$dr) ) {
    a <- tb_long[ dr == metodo & variante == "intrames", list(semilla, auc_a= auc_validacion) ]
    for( var_km in c("kmeans_feature", "kmeans_drift") ) {
      b <- tb_long[ dr == metodo & variante == var_km, list(semilla, auc_b= auc_validacion) ]
      if( nrow(a) == 0 || nrow(b) == 0 ) next
      ab <- merge(a, b, by= "semilla")
      p <- tryCatch(
        suppressWarnings( wilcox.test(ab$auc_b, ab$auc_a, paired= TRUE, exact= FALSE)$p.value ),
        error= function(e) NA_real_
      )
      filas[[length(filas) + 1]] <- data.table(
        dr= metodo,
        comparacion= paste0(var_km, " vs intrames"),
        auc_intrames= mean(ab$auc_a),
        auc_kmeans= mean(ab$auc_b),
        diferencia= mean(ab$auc_b - ab$auc_a),
        semillas_que_mejoran= sum(ab$auc_b > ab$auc_a),
        semillas= nrow(ab),
        p_wilcoxon_pareado= p
      )
    }
  }

  tb_comp <- rbindlist(filas)
  if( nrow(tb_comp) > 0 )
    fwrite(tb_comp, file= file.path(dir_exp, "comparacion_intrames_vs_kmeans.txt"), sep= "\t")

  list(resumen= tb_res, comparacion= tb_comp)
}

# %% [markdown]
# ## Corrida
# Tarda muchas horas: 9 configuraciones x (grid de 50 combinaciones + 10 modelos de validacion + 10 modelos finales). Si se corta, se vuelve a correr esta celda: saltea las configuraciones terminadas. Si una configuracion falla, guarda `error_<config>.txt` y sigue con la siguiente.
#
# Recomendado: antes de la corrida larga, probar con `PARAM$prueba_rapida <- TRUE` (celda de Parametros).

# %%
format(Sys.time(), "%a %b %d %X %Y")

write_yaml(PARAM[c("experimento", "dataset", "semillas", "semilla_primigenia",
                   "drift_metodos", "kmeans", "trainingstrategy", "kaggle")],
  file= file.path(dir_exp, "PARAM_global.yml"))

asignar_kmeans_global()

for( i in seq_len(nrow(tb_config)) ) {
  cfg <- tb_config[i]

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
# ## Resultados

# %%
res <- armar_resumen()
res$resumen

# %%
res$comparacion

# %%
list.files(dir_exp)
