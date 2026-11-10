# NEW201: Wilcoxon entre configuraciones y ensambles entre configuraciones
# Usa lo que ya dejo la corrida en exp/NEW201 (no reentrena nada).

require("data.table")

dir_exp <- "/content/buckets/b1/exp/NEW201"

# ---------------------------------------------------------------
# 1. AUC de validacion (202107) por configuracion y semilla
# ---------------------------------------------------------------
tb <- fread(file.path(dir_exp, "auc_validacion_todas_las_semillas.txt"))
# columnas: semilla, auc_validacion, config, dr, variante

tb_media <- tb[, list(auc_media= mean(auc_validacion),
                      auc_sd= sd(auc_validacion),
                      n= .N), by= config]
setorder(tb_media, -auc_media)
tb_media

mejor <- tb_media[1, config]

# ---------------------------------------------------------------
# 2. Wilcoxon de cada configuracion contra la mejor
#    - no pareado (rank-sum): cada configuracion tiene otros campos e
#      hiperparametros, asi que una misma semilla no es "el mismo azar"
#    - pareado por semilla: es el que usa el cuaderno, se deja como referencia
#    Holm corrige por hacer 8 comparaciones a la vez
# ---------------------------------------------------------------
a <- tb[config == mejor][order(semilla), auc_validacion]

tb_wil <- rbindlist(lapply(setdiff(tb_media$config, mejor), function(cfg) {
  b <- tb[config == cfg][order(semilla), auc_validacion]
  data.table(
    config= cfg,
    diferencia= mean(a) - mean(b),
    semillas_donde_gana_la_mejor= sum(a > b),
    p_no_pareado= wilcox.test(a, b, paired= FALSE, exact= FALSE)$p.value,
    p_pareado= suppressWarnings(wilcox.test(a, b, paired= TRUE, exact= FALSE)$p.value)
  )
}))
tb_wil[, p_no_pareado_holm := p.adjust(p_no_pareado, method= "holm")]
tb_wil[, p_pareado_holm := p.adjust(p_pareado, method= "holm")]
setorder(tb_wil, p_no_pareado)

cat("Mejor configuracion:", mejor, "\n")
tb_wil

fwrite(tb_wil, file= file.path(dir_exp, "wilcoxon_contra_la_mejor.txt"), sep= "\t")

# ---------------------------------------------------------------
# 3. Ensambles ENTRE configuraciones (con las probabilidades guardadas)
#    Cada configuracion ya es un ensamble de 10 semillas; aca se promedian
#    los ranks de varias configuraciones y se mide la AUC en 202107.
# ---------------------------------------------------------------
auc_ranks <- function(prob, y) {
  r <- frank(prob, ties.method= "average")
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

ens_config <- function(config) {
  archivos <- list.files(file.path(dir_exp, config, "validacion"), full.names= TRUE)
  x <- rbindlist(lapply(archivos, fread))
  x <- x[, list(prob= mean(prob)), by= list(numero_de_cliente, clase_ternaria)]
  setorder(x, numero_de_cliente)
  x[, clase01 := as.integer(clase_ternaria %in% c("BAJA+1", "BAJA+2"))]
  x[, rk := frank(prob) / .N]
  x[, list(numero_de_cliente, clase01, rk)]
}

lista <- lapply(tb_media$config, ens_config)
names(lista) <- tb_media$config

# todas las configuraciones tienen que tener los mismos clientes en el mismo orden
for( x in lista ) stopifnot( identical(x$numero_de_cliente, lista[[1]]$numero_de_cliente) )
y <- lista[[1]]$clase01

auc_combo <- function(cfgs) {
  auc_ranks(rowMeans(sapply(lista[cfgs], function(x) x$rk)), y)
}

nombres <- tb_media$config

tb_combo <- data.table(
  combinacion= c("la mejor sola", "top 2", "top 3", "top 5", "las 9"),
  auc_ensemble= c(
    auc_combo(nombres[1]),
    auc_combo(nombres[1:2]),
    auc_combo(nombres[1:3]),
    auc_combo(nombres[1:5]),
    auc_combo(nombres)
  )
)

# combinacion a mano: rank intra-mes + rank con k-means + deflacion con k-means
elegidas <- c("01_rank_cero_fijo_intrames",
              "05_rank_cero_fijo_kmeans_feature",
              "06_deflacion_kmeans_feature")
tb_combo <- rbind(tb_combo,
  data.table(combinacion= "01 + 05 + 06", auc_ensemble= auc_combo(elegidas)))

tb_combo
fwrite(tb_combo, file= file.path(dir_exp, "ensambles_entre_configuraciones.txt"), sep= "\t")
