"""Interfaz gráfica de SoftCast: una ventana con el grafo, el nodo seleccionado,
el autómata con la ruta y la tabla de nodos."""

import platform
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

from softcast import gnn
from softcast.automata import (
    evaluar,
    mensaje_cadena_incompleta,
    preparar_cadena,
    validar_cadena,
)
from softcast.enrutador import calcular_ruta, costo_ruta
from softcast.generador import generar_red, regenerar_caracteristicas
from softcast.red import ANOMALO, BLOQUEADO, ESTADOS, IDS, NINGUNA

# Portabilidad (RNF07): fuera de Windows no existen el tema vista ni estas fuentes.
ES_WINDOWS = platform.system() == "Windows"
TEMA = "vista" if ES_WINDOWS else "clam"
FUENTE = "Segoe UI" if ES_WINDOWS else "DejaVu Sans"
FUENTE_FIJA = "Consolas" if ES_WINDOWS else "DejaVu Sans Mono"

TITULO = "SoftCast: detección de fallas en la red"

COLOR_ESTADO = {"Normal": "#81C784", ANOMALO: "#FFD54F", BLOQUEADO: "#E57373"}
COLOR_ARISTA_ACTIVA = "#6E6E6E"
COLOR_ARISTA_INACTIVA = "#949494"  # Gris claro con contraste 3:1 sobre blanco (WCAG 1.4.11).
COLOR_RUTA = "#1565C0"
COLOR_BORDE = "#212121"
COLOR_TEXTO_PESO = "#424242"
COLOR_ACEPTADA = "#1B5E20"
COLOR_RECHAZADA = "#B71C1C"
COLOR_FILA_CAMBIADA = "#E0E0E0"

# Posiciones fijas: los nodos externos van sobre una elipse de semiejes 6 y 4, con
# SRV-L en el centro; así ninguna arista atraviesa un nodo.
POSICIONES = {
    0: (-5.64, 1.37),  # PC1
    1: (-3.44, 3.28),  # PC2
    2: (3.44, 3.28),  # PC3
    3: (5.64, 1.37),  # PC4
    4: (-4.92, -2.29),  # IoT1
    5: (0.0, -4.0),  # IoT2
    6: (4.92, -2.29),  # IoT3
    7: (0.0, 0.0),  # SRV-L
    8: (0.0, 4.0),  # SRV-G
}
ANCHO_NODO = 1.6
ALTO_NODO = 0.6


def numero(valor, decimales=1):
    """Formatea con coma decimal."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def forma_nodo(tipo, x, y, ancho, alto, color, grosor=1.0):
    """PC: rectángulo redondeado; IoT: óvalo; servidores: rectángulo."""
    estilo = {"facecolor": color, "edgecolor": COLOR_BORDE, "linewidth": grosor, "zorder": 4}
    esquina = (x - ancho / 2, y - alto / 2)
    if tipo == "PC":
        return FancyBboxPatch(
            esquina, ancho, alto, boxstyle="round,pad=0,rounding_size=0.15", **estilo
        )
    if tipo == "IoT":
        return Ellipse((x, y), ancho, alto, **estilo)
    return Rectangle(esquina, ancho, alto, **estilo)


class AplicacionSoftCast(tk.Tk):
    def __init__(self, semilla=None):
        super().__init__()
        self.title(TITULO)
        self.minsize(1280, 720)
        self._configurar_estilo()

        self.modelo, _ = gnn.cargar_o_entrenar()
        self.red = None
        self.seleccionado = None
        self.ruta = NINGUNA
        self.cambiados = set()
        self.hay_resultado = False
        self.mostrar_pesos = tk.BooleanVar(value=True)

        self._construir()
        self.nueva_red(semilla)
        self._maximizar()

    # Construcción de la ventana

    def _configurar_estilo(self):
        estilo = ttk.Style(self)
        estilo.theme_use(TEMA)
        for nombre in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont"):
            tkfont.nametofont(nombre).configure(family=FUENTE, size=10)
        estilo.configure("Area.TLabelframe.Label", font=(FUENTE, 11, "bold"))
        estilo.configure("Encabezado.TLabel", font=(FUENTE, 11, "bold"))
        estilo.configure("Resultado.TLabel", font=(FUENTE, 12, "bold"))
        estilo.configure("Error.TLabel", foreground=COLOR_RECHAZADA)
        estilo.configure("Fija.TLabel", font=(FUENTE_FIJA, 11))
        estilo.configure("Treeview", rowheight=20)
        self.fondo = estilo.lookup("TFrame", "background") or "#F0F0F0"
        matplotlib.rcParams["font.family"] = FUENTE

    def _maximizar(self):
        try:
            if ES_WINDOWS:
                self.state("zoomed")
            else:
                self.attributes("-zoomed", True)
        except tk.TclError:
            self.geometry("1280x720")

    def _construir(self):
        raiz = ttk.Frame(self, padding=16)
        raiz.pack(fill="both", expand=True)
        raiz.columnconfigure(0, weight=3, uniform="columnas")
        raiz.columnconfigure(1, weight=2, uniform="columnas")
        raiz.rowconfigure(1, weight=1)

        self._construir_barra_semilla(raiz).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16)
        )

        izquierda = ttk.Frame(raiz)
        izquierda.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        izquierda.columnconfigure(0, weight=1)
        izquierda.rowconfigure(0, weight=1)
        self._construir_area_grafo(izquierda).grid(row=0, column=0, sticky="nsew")
        self._construir_area_tabla(izquierda).grid(row=1, column=0, sticky="ew", pady=(16, 0))

        derecha = ttk.Frame(raiz)
        derecha.grid(row=1, column=1, sticky="nsew")
        derecha.columnconfigure(0, weight=1)
        derecha.rowconfigure(1, weight=1)
        self._construir_area_nodo(derecha).grid(row=0, column=0, sticky="ew")
        self._construir_area_automata(derecha).grid(row=1, column=0, sticky="nsew", pady=(16, 0))

    def _construir_barra_semilla(self, padre):
        barra = ttk.Frame(padre)
        ttk.Label(barra, text="Semilla").pack(side="left")
        self.entrada_semilla = ttk.Entry(barra, width=10)
        self.entrada_semilla.pack(side="left", padx=(4, 8))
        self.entrada_semilla.bind("<Return>", lambda _evento: self._generar_desde_entrada())
        ttk.Button(barra, text="Generar nueva red", command=self._generar_desde_entrada).pack(
            side="left"
        )
        self.etiqueta_semilla = ttk.Label(barra)
        self.etiqueta_semilla.pack(side="left", padx=(16, 0))
        self.error_semilla = ttk.Label(barra, style="Error.TLabel")
        self.error_semilla.pack(side="left", padx=(16, 0))
        return barra

    def _construir_area_grafo(self, padre):
        marco = ttk.LabelFrame(padre, text="Red", padding=8, style="Area.TLabelframe")
        self.figura = Figure(figsize=(4, 2.5), dpi=100, facecolor="white")
        self.ejes = self.figura.add_axes([0, 0, 1, 1])
        self.lienzo = FigureCanvasTkAgg(self.figura, master=marco)
        self.lienzo.get_tk_widget().pack(fill="both", expand=True)
        self.lienzo.mpl_connect("button_press_event", self._clic_en_grafo)
        # Sobre el lienzo, en la esquina superior derecha, para no restarle altura al grafo.
        ttk.Checkbutton(
            marco, text="Mostrar pesos", variable=self.mostrar_pesos, command=self._dibujar_grafo
        ).place(relx=1.0, y=0, anchor="ne")
        return marco

    def _construir_area_tabla(self, padre):
        marco = ttk.LabelFrame(padre, text="Nodos", padding=8, style="Area.TLabelframe")
        columnas = ("id", "tipo", "trafico", "cpu", "latencia", "real", "predicho")
        encabezados = (
            "Id", "Tipo", "Tráfico (Mbps)", "CPU (%)", "Latencia (ms)",
            "Estado real", "Estado predicho",
        )
        self.tabla = ttk.Treeview(
            marco, columns=columnas, show="headings", height=9, selectmode="browse"
        )
        for columna, encabezado in zip(columnas, encabezados):
            numerica = columna in ("trafico", "cpu", "latencia")
            self.tabla.heading(columna, text=encabezado, anchor="e" if numerica else "w")
            self.tabla.column(columna, anchor="e" if numerica else "w", width=90, stretch=True)
        self.tabla.tag_configure(
            "cambiado", background=COLOR_FILA_CAMBIADA, font=(FUENTE, 10, "bold")
        )
        self.tabla.bind("<<TreeviewSelect>>", self._seleccion_en_tabla)
        self.tabla.pack(fill="x")
        self.resumen = ttk.Label(marco)
        self.resumen.pack(anchor="w", pady=(8, 0))
        return marco

    def _construir_area_nodo(self, padre):
        marco = ttk.LabelFrame(
            padre, text="Nodo seleccionado", padding=8, style="Area.TLabelframe"
        )
        self.encabezado_nodo = ttk.Label(marco, style="Encabezado.TLabel")
        self.encabezado_nodo.pack(anchor="w")

        cuerpo = ttk.Frame(marco)
        cuerpo.pack(fill="x", pady=(8, 0))
        datos = ttk.Frame(cuerpo)
        datos.pack(side="left", anchor="n")
        self.valores_nodo = {}
        for fila, clave in enumerate(("Tráfico", "CPU", "Latencia", "Estado real", "Estado predicho")):
            ttk.Label(datos, text=clave).grid(row=fila, column=0, sticky="w", padx=(0, 8), pady=1)
            self.valores_nodo[clave] = ttk.Label(datos, width=11)
            self.valores_nodo[clave].grid(row=fila, column=1, sticky="w")

        probabilidades = ttk.Frame(cuerpo)
        probabilidades.pack(side="left", anchor="n", padx=(16, 0))
        ttk.Label(probabilidades, text="Probabilidades de la GCN").pack(anchor="w", pady=(0, 4))
        self.barras = tk.Canvas(
            probabilidades, width=270, height=76, bg=self.fondo, highlightthickness=0
        )
        self.barras.pack(anchor="w")

        botones = ttk.Frame(marco)
        botones.pack(anchor="w", pady=(8, 0))
        ttk.Label(botones, text="Cambiar estado a").pack(side="left", padx=(0, 8))
        self.botones_estado = {}
        for estado in ESTADOS:
            boton = ttk.Button(
                botones, text=estado, command=lambda e=estado: self.cambiar_estado(e)
            )
            boton.pack(side="left", padx=(0, 4))
            self.botones_estado[estado] = boton

        self.mensaje_cambios = ttk.Label(marco, wraplength=460)
        self.mensaje_cambios.pack(anchor="w", pady=(8, 0))
        return marco

    def _construir_area_automata(self, padre):
        marco = ttk.LabelFrame(padre, text="Autómata y ruta", padding=8, style="Area.TLabelframe")
        marco.columnconfigure(1, weight=1)

        ttk.Label(marco, text="Origen").grid(row=0, column=0, sticky="w", pady=2)
        extremos = ttk.Frame(marco)
        extremos.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.origen = ttk.Combobox(extremos, values=IDS, state="readonly", width=8)
        self.origen.set("IoT1")
        self.origen.pack(side="left")
        ttk.Label(extremos, text="Destino").pack(side="left", padx=(16, 8))
        self.destino = ttk.Combobox(extremos, state="readonly", width=8)
        self.destino.pack(side="left")
        self._actualizar_destinos()
        self.destino.set("IoT3")
        self.origen.bind("<<ComboboxSelected>>", self._cambio_de_extremos)
        self.destino.bind("<<ComboboxSelected>>", self._cambio_de_extremos)

        ttk.Label(marco, text="Cadena").grid(row=2, column=0, sticky="w", pady=(6, 2))
        entrada = ttk.Frame(marco)
        entrada.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 2))
        self.entrada_cadena = ttk.Entry(entrada, font=(FUENTE_FIJA, 11), width=18)
        self.entrada_cadena.pack(side="left")
        self.entrada_cadena.bind("<Return>", lambda _evento: self.evaluar_cadena())
        ttk.Button(entrada, text="Evaluar", command=self.evaluar_cadena).pack(
            side="left", padx=(8, 0)
        )
        ttk.Label(marco, text="Escriba solo s, a, d, k y f; el sistema agrega r y b.").grid(
            row=3, column=1, sticky="w", padx=(8, 0)
        )
        self.mensaje_validacion = ttk.Label(marco, style="Error.TLabel", wraplength=420)
        self.mensaje_validacion.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

        ttk.Separator(marco).grid(row=5, column=0, columnspan=2, sticky="ew", pady=8)

        ttk.Label(marco, text="Ruta").grid(row=6, column=0, sticky="nw", pady=2)
        self.texto_ruta = ttk.Label(marco, wraplength=420)
        self.texto_ruta.grid(row=6, column=1, sticky="w", padx=(8, 0))
        ttk.Label(marco, text="Anómalos en ruta").grid(row=7, column=0, sticky="w", pady=2)
        self.texto_anomalos = ttk.Label(marco)
        self.texto_anomalos.grid(row=7, column=1, sticky="w", padx=(8, 0))
        ttk.Label(marco, text="Cadena evaluada").grid(row=8, column=0, sticky="w", pady=2)
        self.texto_cadena = tk.Text(
            marco, height=1, width=36, font=(FUENTE_FIJA, 11), bg=self.fondo,
            relief="flat", borderwidth=0, highlightthickness=0, takefocus=0,
        )
        self.texto_cadena.tag_configure("insertado", underline=True)
        self.texto_cadena.grid(row=8, column=1, sticky="w", padx=(8, 0))
        self.texto_cadena.configure(state="disabled")
        ttk.Label(marco, text="Recorrido").grid(row=9, column=0, sticky="nw", pady=2)
        self.texto_recorrido = ttk.Label(marco, style="Fija.TLabel", wraplength=420)
        self.texto_recorrido.grid(row=9, column=1, sticky="w", padx=(8, 0))

        self.resultado = ttk.Label(marco, style="Resultado.TLabel", wraplength=480)
        self.resultado.grid(row=10, column=0, columnspan=2, sticky="w", pady=(12, 0))
        return marco

    # Red y grafo

    def nueva_red(self, semilla=None):
        self.red = generar_red(semilla)
        gnn.clasificar(self.modelo, self.red)
        self.cambiados = set()
        self.seleccionado = None
        self.entrada_semilla.delete(0, "end")
        self.entrada_semilla.insert(0, str(self.red.semilla))
        self.etiqueta_semilla.configure(text=f"Red actual: semilla {self.red.semilla}")
        self.error_semilla.configure(text="")
        self.mensaje_cambios.configure(text="")
        self.tabla.selection_remove(self.tabla.selection())
        self._limpiar_resultados()
        self._refrescar()

    def _generar_desde_entrada(self):
        texto = self.entrada_semilla.get().strip()
        if not texto:
            self.nueva_red()
            return
        try:
            semilla = int(texto)
        except ValueError:
            self.error_semilla.configure(text="La semilla debe ser un número entero.")
            return
        self.nueva_red(semilla)

    def _refrescar(self):
        self._dibujar_grafo()
        self._actualizar_tabla()
        self._actualizar_panel_nodo()

    def _dibujar_grafo(self):
        ejes = self.ejes
        ejes.clear()
        ejes.set_axis_off()
        ejes.set_xlim(-6.6, 11.4)
        ejes.set_ylim(-4.5, 4.5)
        ejes.set_aspect("equal")

        pares_ruta = set()
        if self.ruta is not NINGUNA:
            pares_ruta = {frozenset(par) for par in zip(self.ruta, self.ruta[1:])}

        for arista in self.red.aristas:
            (x1, y1), (x2, y2) = POSICIONES[arista.origen], POSICIONES[arista.destino]
            if arista.activa:
                ejes.plot([x1, x2], [y1, y2], color=COLOR_ARISTA_ACTIVA, linewidth=1.5, zorder=1)
            else:
                ejes.plot(
                    [x1, x2], [y1, y2], color=COLOR_ARISTA_INACTIVA, linewidth=1,
                    linestyle=(0, (2, 3)), zorder=1,
                )
            if frozenset((arista.origen, arista.destino)) in pares_ruta:
                ejes.plot(
                    [x1, x2], [y1, y2], color=COLOR_RUTA, linewidth=4,
                    solid_capstyle="round", zorder=2,
                )
            if self.mostrar_pesos.get():
                ejes.text(
                    (x1 + x2) / 2, (y1 + y2) / 2, numero(arista.peso), fontsize=8,
                    color=COLOR_TEXTO_PESO, ha="center", va="center", zorder=3,
                    bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none"},
                )

        for nodo in self.red.nodos:
            x, y = POSICIONES[nodo.indice]
            grosor = 3 if nodo.indice == self.seleccionado else 1
            ejes.add_patch(
                forma_nodo(nodo.tipo, x, y, ANCHO_NODO, ALTO_NODO, COLOR_ESTADO[nodo.estado], grosor)
            )
            ejes.text(x, y, nodo.id, fontsize=9, ha="center", va="center", zorder=6)
            if nodo.indice in self.cambiados:
                ejes.add_patch(self._anillo(x, y, ANCHO_NODO + 0.5, ALTO_NODO + 0.5))

        self._dibujar_leyenda()
        self.lienzo.draw_idle()

    @staticmethod
    def _anillo(x, y, ancho, alto):
        return Ellipse(
            (x, y), ancho, alto, fill=False, linestyle="--", linewidth=1.5,
            edgecolor="black", zorder=5,
        )

    def _dibujar_leyenda(self):
        ejes = self.ejes
        x_figura, x_texto = 7.9, 8.5
        tamano = {"fontsize": 8.5, "va": "center"}

        def titulo(y, texto):
            ejes.text(7.5, y, texto, fontweight="bold", **tamano)

        titulo(4.2, "Estado")
        for y, estado in zip((3.6, 3.1, 2.6), ESTADOS):
            ejes.add_patch(Rectangle((x_figura - 0.25, y - 0.15), 0.5, 0.3,
                                     facecolor=COLOR_ESTADO[estado], edgecolor=COLOR_BORDE))
            ejes.text(x_texto, y, estado, **tamano)

        titulo(1.8, "Tipo")
        for y, tipo, texto in ((1.2, "PC", "PC"), (0.7, "IoT", "IoT"), (0.2, "SRV-L", "Servidor")):
            ejes.add_patch(forma_nodo(tipo, x_figura, y, 0.6, 0.32, "white"))
            ejes.text(x_texto, y, texto, **tamano)

        titulo(-0.6, "Arista")
        ejes.plot([7.6, 8.2], [-1.2, -1.2], color=COLOR_ARISTA_ACTIVA, linewidth=1.5)
        ejes.text(x_texto, -1.2, "Activa", **tamano)
        ejes.plot([7.6, 8.2], [-1.7, -1.7], color=COLOR_ARISTA_INACTIVA, linewidth=1,
                  linestyle=(0, (2, 3)))
        ejes.text(x_texto, -1.7, "Inactiva", **tamano)
        ejes.plot([7.6, 8.2], [-2.2, -2.2], color=COLOR_RUTA, linewidth=4, solid_capstyle="round")
        ejes.text(x_texto, -2.2, "Ruta calculada", **tamano)

        titulo(-3.0, "Clasificación")
        ejes.add_patch(self._anillo(x_figura, -3.6, 0.7, 0.42))
        ejes.text(x_texto, -3.6, "Predicción cambiada", **tamano)

    def _clic_en_grafo(self, evento):
        if evento.xdata is None:
            return
        for nodo in self.red.nodos:
            x, y = POSICIONES[nodo.indice]
            if abs(evento.xdata - x) <= ANCHO_NODO / 2 and abs(evento.ydata - y) <= ALTO_NODO / 2:
                self.seleccionar(nodo.indice)
                return

    # Nodo seleccionado

    def seleccionar(self, indice):
        self.seleccionado = indice
        iid = str(indice)
        if self.tabla.selection() != (iid,):
            self.tabla.selection_set(iid)
            self.tabla.focus(iid)
            self.tabla.see(iid)
        self._actualizar_panel_nodo()
        self._dibujar_grafo()

    def _seleccion_en_tabla(self, _evento):
        seleccion = self.tabla.selection()
        if seleccion and int(seleccion[0]) != self.seleccionado:
            self.seleccionar(int(seleccion[0]))

    def cambiar_estado(self, estado):
        """Cambio manual (RF06): regenera características, reclasifica y resalta (RF08)."""
        indice = self.seleccionado
        self.red.cambiar_estado(indice, estado)
        regenerar_caracteristicas(self.red.nodos[indice], estado)
        self.cambiados = set(gnn.clasificar(self.modelo, self.red))

        if self.cambiados:
            nombres = ", ".join(IDS[i] for i in sorted(self.cambiados))
            self.mensaje_cambios.configure(text=f"Cambió la predicción de: {nombres}.")
        else:
            self.mensaje_cambios.configure(text="Ninguna predicción cambió.")
        if self.hay_resultado:
            self._limpiar_resultados()
            self.resultado.configure(
                text="La red cambió; evalúe de nuevo la cadena.", foreground=""
            )
        self._refrescar()

    def _actualizar_panel_nodo(self):
        self.barras.delete("all")
        if self.seleccionado is None:
            self.encabezado_nodo.configure(text="Seleccione un nodo en el grafo o en la tabla.")
            for etiqueta in self.valores_nodo.values():
                etiqueta.configure(text="")
            for boton in self.botones_estado.values():
                boton.state(["disabled"])
            return

        nodo = self.red.nodos[self.seleccionado]
        self.encabezado_nodo.configure(text=f"{nodo.id} · {nodo.tipo}")
        self.valores_nodo["Tráfico"].configure(text=f"{numero(nodo.trafico)} Mbps")
        self.valores_nodo["CPU"].configure(text=f"{numero(nodo.cpu)} %")
        self.valores_nodo["Latencia"].configure(text=f"{numero(nodo.latencia)} ms")
        self.valores_nodo["Estado real"].configure(text=nodo.estado)
        self.valores_nodo["Estado predicho"].configure(text=nodo.estado_predicho)

        for fila, (estado, probabilidad) in enumerate(zip(ESTADOS, nodo.probabilidades)):
            y = 4 + fila * 24
            self.barras.create_text(0, y + 8, anchor="w", text=estado, font=(FUENTE, 10))
            self.barras.create_rectangle(76, y, 196, y + 16, outline="#9E9E9E")
            if probabilidad > 0:
                self.barras.create_rectangle(
                    76, y, 76 + 120 * probabilidad, y + 16,
                    fill=COLOR_ESTADO[estado], outline="",
                )
            self.barras.create_text(
                204, y + 8, anchor="w", text=f"{numero(probabilidad * 100)} %",
                font=(FUENTE, 10),
            )

        for estado, boton in self.botones_estado.items():
            boton.state(["disabled"] if estado == nodo.estado else ["!disabled"])

    def _actualizar_tabla(self):
        for nodo in self.red.nodos:
            valores = (
                nodo.id, nodo.tipo, numero(nodo.trafico), numero(nodo.cpu),
                numero(nodo.latencia), nodo.estado, nodo.estado_predicho,
            )
            etiquetas = ("cambiado",) if nodo.indice in self.cambiados else ()
            iid = str(nodo.indice)
            if self.tabla.exists(iid):
                self.tabla.item(iid, values=valores, tags=etiquetas)
            else:
                self.tabla.insert("", "end", iid=iid, values=valores, tags=etiquetas)
        correctas = sum(nodo.estado == nodo.estado_predicho for nodo in self.red.nodos)
        self.resumen.configure(text=f"Predicciones correctas: {correctas} de {len(self.red.nodos)}")

    # Autómata y ruta

    def _actualizar_destinos(self):
        opciones = [i for i in IDS if i != self.origen.get()]
        self.destino.configure(values=opciones)
        if self.destino.get() not in opciones:
            self.destino.set(opciones[0])

    def _cambio_de_extremos(self, _evento):
        self._actualizar_destinos()
        if self.hay_resultado:
            self._limpiar_resultados()
            self._dibujar_grafo()

    def _limpiar_resultados(self):
        self.ruta = NINGUNA
        self.hay_resultado = False
        self.mensaje_validacion.configure(text="")
        self.texto_ruta.configure(text="")
        self.texto_anomalos.configure(text="")
        self.texto_recorrido.configure(text="")
        self.resultado.configure(text="", foreground="")
        self._escribir_cadena("", "")

    def _escribir_cadena(self, original, preparada):
        """Muestra la cadena evaluada subrayando los símbolos que agregó el sistema."""
        self.texto_cadena.configure(state="normal")
        self.texto_cadena.delete("1.0", "end")
        # r y b nunca están en la cadena validada, así que basta recorrer ambas a la par.
        i = 0
        for simbolo in preparada:
            if i < len(original) and original[i] == simbolo:
                self.texto_cadena.insert("end", simbolo)
                i += 1
            else:
                self.texto_cadena.insert("end", simbolo, "insertado")
        self.texto_cadena.configure(state="disabled")

    def evaluar_cadena(self):
        self._limpiar_resultados()
        cadena, mensaje = validar_cadena(self.entrada_cadena.get())
        if mensaje:
            self.mensaje_validacion.configure(text=mensaje)
            self._dibujar_grafo()
            return

        origen = IDS.index(self.origen.get())
        destino = IDS.index(self.destino.get())
        self.ruta = calcular_ruta(self.red, origen, destino)
        preparada = preparar_cadena(cadena, self.ruta, self.red)
        aceptada, recorrido, rechazo = evaluar(preparada)
        self.hay_resultado = True

        if self.ruta is NINGUNA:
            self.texto_ruta.configure(text=self._motivo_sin_ruta(origen, destino))
            self.texto_anomalos.configure(text="No aplica")
        else:
            nombres = " → ".join(IDS[i] for i in self.ruta)
            costo = numero(costo_ruta(self.red, self.ruta))
            self.texto_ruta.configure(text=f"{nombres} (costo {costo} ms)")
            anomalos = [IDS[i] for i in self.ruta if self.red.nodos[i].estado == ANOMALO]
            texto = f"{len(anomalos)} ({', '.join(anomalos)})" if anomalos else "0"
            self.texto_anomalos.configure(text=texto)

        self._escribir_cadena(cadena, preparada)
        self.texto_recorrido.configure(text=" → ".join(recorrido))

        if aceptada:
            self.resultado.configure(text="Aceptada", foreground=COLOR_ACEPTADA)
        else:
            posicion, simbolo = rechazo
            if simbolo is None:
                texto = mensaje_cadena_incompleta(recorrido[-1])
            else:
                texto = f"Rechazada en el símbolo {posicion} ({simbolo})"
            self.resultado.configure(text=texto, foreground=COLOR_RECHAZADA)
        self._dibujar_grafo()

    def _motivo_sin_ruta(self, origen, destino):
        bloqueados = [IDS[i] for i in (origen, destino) if self.red.nodos[i].estado == BLOQUEADO]
        if len(bloqueados) == 2:
            motivo = f"{bloqueados[0]} y {bloqueados[1]} están Bloqueados"
        elif bloqueados:
            motivo = f"{bloqueados[0]} está Bloqueado"
        else:
            motivo = f"no hay camino entre {IDS[origen]} y {IDS[destino]}"
        return f"No existe ruta: {motivo}. El sistema insertó b."


def main(semilla=None):
    AplicacionSoftCast(semilla).mainloop()


if __name__ == "__main__":
    main()
