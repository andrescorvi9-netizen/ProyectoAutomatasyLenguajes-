"""Pruebas de la interfaz con eventos reales de Tk (clics, teclas y selecciones).

Se omiten si no hay una pantalla disponible.
"""

import tkinter as tk

import pytest

from softcast.automata import MENSAJE_CADENA_VACIA, MENSAJE_SIMBOLOS_SISTEMA
from softcast.red import ANOMALO, BLOQUEADO, IDS, NORMAL

PC2, PC4, IOT1, IOT3, SRV_L, SRV_G = 1, 3, 4, 6, 7, 8


def _hay_pantalla():
    try:
        tk.Tk().destroy()
        return True
    except tk.TclError:
        return False


pytestmark = pytest.mark.skipif(not _hay_pantalla(), reason="No hay pantalla disponible")


@pytest.fixture(scope="module")
def ventana():
    # Una sola ventana para todo el módulo: en Windows, crear y destruir muchos
    # intérpretes Tk seguidos falla de forma intermitente al leer init.tcl.
    from softcast.interfaz import AplicacionSoftCast

    aplicacion = AplicacionSoftCast(semilla=1019)
    aplicacion.update()
    yield aplicacion
    aplicacion.destroy()


@pytest.fixture
def app(ventana):
    ventana.mostrar_pesos.set(True)
    ventana.origen.set("IoT1")
    ventana._actualizar_destinos()
    ventana.destino.set("IoT3")
    ventana.entrada_cadena.delete(0, "end")
    ventana.nueva_red(1019)
    refrescar(ventana)
    return ventana


def refrescar(app):
    app.update()
    app.lienzo.draw()
    app.update()


def clic_en_nodo(app, indice):
    from softcast.interfaz import POSICIONES

    x, y = app.ejes.transData.transform(POSICIONES[indice])
    lienzo = app.lienzo.get_tk_widget()
    # Matplotlib mide y desde abajo; Tk, desde arriba.
    y_tk = int(lienzo.winfo_height() - y)
    lienzo.event_generate("<Button-1>", x=int(x), y=y_tk)
    lienzo.event_generate("<ButtonRelease-1>", x=int(x), y=y_tk)
    refrescar(app)


def pulsar(app, boton):
    """Activa un botón con la barra espaciadora, como lo haría un usuario con teclado."""
    boton.focus_force()
    boton.event_generate("<KeyPress-space>")
    refrescar(app)


def poner_estado(app, indice, estado):
    clic_en_nodo(app, indice)
    if app.red.nodos[indice].estado != estado:
        pulsar(app, app.botones_estado[estado])


def evaluar(app, origen, destino, cadena):
    app.origen.set(origen)
    app.origen.event_generate("<<ComboboxSelected>>")
    app.destino.set(destino)
    app.destino.event_generate("<<ComboboxSelected>>")
    app.entrada_cadena.delete(0, "end")
    app.entrada_cadena.insert(0, cadena)
    app.entrada_cadena.focus_force()
    app.entrada_cadena.event_generate("<Return>")
    refrescar(app)


def cadena_evaluada(app):
    return app.texto_cadena.get("1.0", "end").strip()


def test_estado_vacio_sin_seleccion(app):
    assert "Seleccione" in app.encabezado_nodo.cget("text")
    assert all(boton.instate(["disabled"]) for boton in app.botones_estado.values())


def test_clic_en_el_grafo_selecciona_y_sincroniza_la_tabla(app):
    clic_en_nodo(app, SRV_L)
    assert app.seleccionado == SRV_L
    assert app.tabla.selection() == (str(SRV_L),)


def test_flechas_en_la_tabla_cambian_la_seleccion(app):
    clic_en_nodo(app, SRV_L)
    app.tabla.focus_force()
    app.tabla.event_generate("<Down>")
    refrescar(app)
    assert app.seleccionado == SRV_G
    assert app.encabezado_nodo.cget("text").startswith("SRV-G")


def test_botones_de_estado(app):
    poner_estado(app, SRV_G, BLOQUEADO)
    nodo = app.red.nodos[SRV_G]
    assert nodo.trafico == 0 and nodo.latencia == 1000
    assert not any(a.activa for a in app.red.aristas if SRV_G in (a.origen, a.destino))
    assert app.botones_estado[BLOQUEADO].instate(["disabled"])
    assert app.mensaje_cambios.cget("text") != ""

    pulsar(app, app.botones_estado[NORMAL])
    for arista in app.red.aristas:
        if SRV_G in (arista.origen, arista.destino):
            otro = arista.destino if arista.origen == SRV_G else arista.origen
            assert arista.activa == (app.red.nodos[otro].estado != BLOQUEADO)


def test_destino_excluye_al_origen(app):
    app.origen.set("PC3")
    app.origen.event_generate("<<ComboboxSelected>>")
    refrescar(app)
    assert "PC3" not in app.destino.cget("values")

    app.destino.set("PC1")
    app.origen.set("PC1")
    app.origen.event_generate("<<ComboboxSelected>>")
    refrescar(app)
    assert app.destino.get() != "PC1"


@pytest.mark.parametrize(
    "cadena, mensaje",
    [
        ("sadrkf", MENSAJE_SIMBOLOS_SISTEMA),
        ("sax", "Símbolo no válido: x"),
        ("sad kf", "Símbolo no válido: espacio"),
        ("   ", MENSAJE_CADENA_VACIA),
    ],
)
def test_mensajes_de_validacion(app, cadena, mensaje):
    evaluar(app, "IoT1", "IoT3", cadena)
    assert app.mensaje_validacion.cget("text") == mensaje
    assert app.resultado.cget("text") == ""


def test_enter_evalua_y_muestra_costo_con_coma(app):
    evaluar(app, "IoT1", "IoT3", "  sadkf ")
    assert app.resultado.cget("text") != ""
    assert "costo" in app.texto_ruta.cget("text") and "," in app.texto_ruta.cget("text")


def test_cadena_incompleta_indica_lo_que_falta(app):
    evaluar(app, "IoT1", "IoT3", "s")
    assert app.resultado.cget("text") == (
        "Rechazada: la cadena terminó en q1 (Solicitando); falta la aceptación a."
    )


def test_casilla_mostrar_pesos(app):
    con_pesos = len(app.ejes.texts)
    app.mostrar_pesos.set(False)
    app._dibujar_grafo()
    refrescar(app)
    assert con_pesos - len(app.ejes.texts) == len(app.red.aristas)


def test_campo_de_semilla(app):
    app.entrada_semilla.delete(0, "end")
    app.entrada_semilla.insert(0, "abc")
    app.entrada_semilla.focus_force()
    app.entrada_semilla.event_generate("<Return>")
    refrescar(app)
    assert "entero" in app.error_semilla.cget("text")

    app.entrada_semilla.delete(0, "end")
    app.entrada_semilla.insert(0, "7")
    app.entrada_semilla.event_generate("<Return>")
    refrescar(app)
    assert app.red.semilla == 7 and app.error_semilla.cget("text") == ""
    assert app.seleccionado is None and app.resultado.cget("text") == ""

    app.entrada_semilla.delete(0, "end")
    app.entrada_semilla.event_generate("<Return>")
    refrescar(app)
    assert app.red.semilla >= 1000
    assert app.resumen.cget("text").startswith("Predicciones correctas:")


def test_sadkf_segun_los_anomalos_de_la_ruta(app):
    # Con SRV-L Bloqueado, toda ruta IoT1 → IoT3 pasa por IoT1, PC2, PC4 e IoT3.
    for indice in range(len(IDS)):
        poner_estado(app, indice, BLOQUEADO if indice == SRV_L else NORMAL)

    esperado = [
        ([], "sadkf", "Aceptada"),
        (["IoT1"], "sadrkf", "Aceptada"),
        (["IoT1", "IoT3"], "sadrrkf", "Aceptada"),
        (["PC2", "IoT1", "IoT3"], "sadrrrkf", "Rechazada en el símbolo 6 (r)"),
    ]
    nuevos_anomalos = [None, IOT1, IOT3, PC2]
    for nuevo, (anomalos, cadena, resultado) in zip(nuevos_anomalos, esperado):
        if nuevo is not None:
            poner_estado(app, nuevo, ANOMALO)
        evaluar(app, "IoT1", "IoT3", "sadkf")
        en_ruta = [IDS[i] for i in app.ruta if app.red.nodos[i].estado == ANOMALO]
        print(
            f"{len(en_ruta)} Anómalos | ruta: {app.texto_ruta.cget('text')} | "
            f"evaluada: {cadena_evaluada(app)} | {app.resultado.cget('text')}"
        )
        assert sorted(en_ruta) == sorted(anomalos)
        assert cadena_evaluada(app) == cadena
        assert app.resultado.cget("text") == resultado
