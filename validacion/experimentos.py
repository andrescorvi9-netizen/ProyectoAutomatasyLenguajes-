"""Experimentos de validación (Etapa 4) sobre los 30 escenarios de prueba de la v1.0."""

import copy
import itertools
import json
import random
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
SALIDA = Path(sys.argv[2])
sys.path.insert(0, str(REPO))

from softcast import gnn  # noqa: E402
from softcast.automata import evaluar, preparar_cadena  # noqa: E402
from softcast.enrutador import calcular_ruta, costo_ruta  # noqa: E402
from softcast.generador import generar_escenarios, regenerar_caracteristicas  # noqa: E402
from softcast.red import ANOMALO, BLOQUEADO, ESTADOS, IDS, NINGUNA, NORMAL  # noqa: E402

modelo, _ = gnn.cargar_o_entrenar()
_, _, prueba = gnn.dividir_escenarios(generar_escenarios(gnn.CANTIDAD_ESCENARIOS))
INDICE = {i: k for k, i in enumerate(IDS)}
PARES = list(itertools.combinations(range(9), 2))


def predicciones(red):
    return gnn.predecir(modelo, red).argmax(dim=1).tolist()


def exactitud(redes, excluir=None):
    ok = tot = 0
    for red in redes:
        pred = predicciones(red)
        for nodo, p in zip(red.nodos, pred):
            if excluir is not None and nodo.indice in excluir:
                continue
            tot += 1
            ok += ESTADOS[p] == nodo.estado
    return ok / tot, ok, tot


def rutas(redes):
    sin_ruta = con = 0
    costos, saltos = [], []
    for red in redes:
        for o, d in PARES:
            r = calcular_ruta(red, o, d)
            if r is NINGUNA:
                sin_ruta += 1
            else:
                con += 1
                costos.append(costo_ruta(red, r))
                saltos.append(len(r) - 1)
    return {
        "pares": sin_ruta + con,
        "sin_ruta": sin_ruta,
        "costo_medio": sum(costos) / len(costos),
        "saltos_medio": sum(saltos) / len(saltos),
    }


def automata(redes, cadena="sadkf"):
    acept = rech = 0
    por_r = {}
    for red in redes:
        for o, d in PARES:
            r = calcular_ruta(red, o, d)
            ev = preparar_cadena(cadena, r, red)
            ok = evaluar(ev)[0]
            acept += ok
            rech += not ok
            clave = "b" if r is NINGUNA else str(ev.count("r"))
            por_r.setdefault(clave, [0, 0])
            por_r[clave][0 if ok else 1] += 1
    return {"aceptadas": acept, "rechazadas": rech, "por_numero_de_r": por_r}


def aplicar(redes, cambios, semilla):
    """Copia cada red y aplica los cambios de estado regenerando características (RF06)."""
    nuevas = []
    for k, red in enumerate(redes):
        r = copy.deepcopy(red)
        aleatorio = random.Random(semilla + k)
        for nombre, estado in cambios:
            i = INDICE[nombre]
            r.cambiar_estado(i, estado)
            regenerar_caracteristicas(r.nodos[i], estado, aleatorio)
        nuevas.append(r)
    return nuevas


def cambios_prediccion(base, nuevas, excluir):
    total = 0
    for a, b in zip(base, nuevas):
        pa, pb = predicciones(a), predicciones(b)
        total += sum(1 for i in range(9) if i not in excluir and pa[i] != pb[i])
    return total


def deteccion(redes, nombre, estado):
    i = INDICE[nombre]
    return sum(ESTADOS[predicciones(r)[i]] == estado for r in redes)


resultados = {}

base_exact = exactitud(prueba)
resultados["base"] = {
    "exactitud": base_exact,
    "estados_reales": {e: sum(n.estado == e for r in prueba for n in r.nodos) for e in ESTADOS},
    "rutas": rutas(prueba),
    "automata": automata(prueba),
    "metricas": gnn.metricas(modelo, prueba),
}

escenarios = {
    "bloqueo_srvl": [("SRV-L", BLOQUEADO)],
    "bloqueo_pc2": [("PC2", BLOQUEADO)],
    "bloqueo_srvg": [("SRV-G", BLOQUEADO)],
    "bloqueo_pc2_srvl": [("PC2", BLOQUEADO), ("SRV-L", BLOQUEADO)],
    "anomalo_srvl": [("SRV-L", ANOMALO)],
    "anomalo_pc4": [("PC4", ANOMALO)],
}
for nombre, cambios in escenarios.items():
    nuevas = aplicar(prueba, cambios, 5000)
    tocados = {INDICE[n] for n, _ in cambios}
    resultados[nombre] = {
        "cambios": cambios,
        "ya_estaban": {n: sum(r.nodos[INDICE[n]].estado == e for r in prueba) for n, e in cambios},
        "exactitud_total": exactitud(nuevas),
        "exactitud_resto": exactitud(nuevas, excluir=tocados),
        "exactitud_resto_base": exactitud(prueba, excluir=tocados),
        "deteccion": {n: deteccion(nuevas, n, e) for n, e in cambios},
        "predicciones_cambiadas_resto": cambios_prediccion(prueba, nuevas, tocados),
        "rutas": rutas(nuevas),
        "automata": automata(nuevas),
    }

# Detección de una anomalía inyectada en cada nodo (uno a la vez), sobre los 30 escenarios.
inyeccion = {}
for nombre in IDS:
    for estado in (ANOMALO, BLOQUEADO, NORMAL):
        nuevas = aplicar(prueba, [(nombre, estado)], 9000)
        inyeccion.setdefault(nombre, {})[estado] = deteccion(nuevas, nombre, estado)
resultados["inyeccion_por_nodo"] = inyeccion

# Ejemplo concreto con la semilla 1019 (la de las capturas).
from softcast.generador import generar_red  # noqa: E402

red = generar_red(1019)
ej = {"estados": {n.id: n.estado for n in red.nodos}}
for o, d in [("IoT1", "IoT3"), ("IoT2", "IoT3"), ("PC1", "IoT3"), ("PC2", "PC4")]:
    r = calcular_ruta(red, INDICE[o], INDICE[d])
    for cad in ("sadkf", "saddkf", "sadkdkf"):
        ev = preparar_cadena(cad, r, red)
        a, rec, rech = evaluar(ev)
        ej[f"{o}->{d} {cad}"] = {
            "ruta": None if r is NINGUNA else [IDS[i] for i in r],
            "costo": None if r is NINGUNA else round(costo_ruta(red, r), 1),
            "evaluada": ev,
            "aceptada": a,
            "recorrido": rec,
            "rechazo": rech,
        }
resultados["semilla_1019"] = ej

SALIDA.write_text(json.dumps(resultados, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in resultados.items() if k != "semilla_1019"}, ensure_ascii=False, default=str)[:6000])
