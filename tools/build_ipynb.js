// Convierte un .R en formato "percent" (# %% / # %% [markdown]) a .ipynb con kernel R
// y hace un chequeo grueso de balance de (), [], {} y comillas en cada celda de codigo.
const fs = require("fs");

const [, , entrada, salida] = process.argv;
const lineas = fs.readFileSync(entrada, "utf8").split("\n");

const celdas = [];
let actual = null;
for (const l of lineas) {
  if (/^# %%/.test(l)) {
    if (actual) celdas.push(actual);
    actual = { tipo: /\[markdown\]/.test(l) ? "markdown" : "code", lineas: [] };
  } else if (actual) {
    actual.lineas.push(l);
  }
}
if (actual) celdas.push(actual);

const limpiar = (ls) => {
  let a = 0, b = ls.length;
  while (a < b && ls[a].trim() === "") a++;
  while (b > a && ls[b - 1].trim() === "") b--;
  return ls.slice(a, b);
};

let errores = 0;
const chequear = (txt, n) => {
  // saca comentarios y strings, despues cuenta balance
  let s = "", i = 0, q = null;
  while (i < txt.length) {
    const c = txt[i];
    if (q) {
      if (c === "\\") { i += 2; continue; }
      if (c === q) q = null;
      i++; continue;
    }
    if (c === '"' || c === "'" || c === "`") { q = c; i++; continue; }
    if (c === "#") { while (i < txt.length && txt[i] !== "\n") i++; continue; }
    s += c; i++;
  }
  if (q) { console.log(`celda ${n}: comilla sin cerrar (${q})`); errores++; }
  const pares = { ")": "(", "]": "[", "}": "{" };
  const pila = [];
  for (const c of s) {
    if ("([{".includes(c)) pila.push(c);
    if (")]}".includes(c)) {
      if (pila.pop() !== pares[c]) { console.log(`celda ${n}: cierre ${c} sin apertura`); errores++; return; }
    }
  }
  if (pila.length) { console.log(`celda ${n}: quedaron abiertos ${pila.join("")}`); errores++; }
};

const cells = celdas.map((c, idx) => {
  const ls = limpiar(c.lineas);
  if (c.tipo === "markdown") {
    const md = ls.map((l) => l.replace(/^# ?/, ""));
    return { cell_type: "markdown", metadata: {}, source: md.map((l, i) => (i < md.length - 1 ? l + "\n" : l)) };
  }
  chequear(ls.join("\n"), idx + 1);
  return {
    cell_type: "code", execution_count: null, metadata: {}, outputs: [],
    source: ls.map((l, i) => (i < ls.length - 1 ? l + "\n" : l)),
  };
});

const nb = {
  cells,
  metadata: {
    kernelspec: { display_name: "R", language: "R", name: "ir" },
    language_info: { name: "R", file_extension: ".r", mimetype: "text/x-r-source" },
  },
  nbformat: 4,
  nbformat_minor: 5,
};
// nbformat 4.5 exige id en cada celda
nb.cells.forEach((c, i) => { c.id = "c" + String(i + 1).padStart(3, "0"); });

fs.writeFileSync(salida, JSON.stringify(nb, null, 1));
console.log(`celdas: ${cells.length} (${cells.filter((c) => c.cell_type === "code").length} de codigo), problemas de balance: ${errores}`);
