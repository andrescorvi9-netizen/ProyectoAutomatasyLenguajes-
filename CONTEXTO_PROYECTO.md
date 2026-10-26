# Proyecto SoftCast — Prototipo (Etapas 3 y 4)

Eres el asistente de programación de un equipo universitario (Lenguajes y Autómatas,
Ingeniería de Sistemas y Computación, Universidad de Cundinamarca). Implementa el
prototipo descrito abajo, que ya fue aprobado por el docente en la etapa de Diseño.
NO cambies decisiones de diseño por tu cuenta: si algo parece incorrecto o ambiguo,
pregunta antes.

## Ubicación y entorno
- Crea el proyecto en: C:\Users\Apharius\Desktop\octavosemestre\automatas\ProyectoSoftCast
- Todo el código, el entorno virtual y los archivos generados van dentro de esa carpeta.
- Entorno (PowerShell, dentro de la carpeta):
    py -3.14 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
- Si alguna librería no instala en 3.14, avisa antes de cambiar a Python 3.13.

## Reglas de trabajo
- Trabaja módulo por módulo en el orden indicado. Al terminar cada uno, DETENTE,
  muestra el código y un resumen, y espera aprobación antes de seguir.
- Código, nombres, comentarios, docstrings y textos de la interfaz en ESPAÑOL,
  claros y académicos.
- Aplica antislop: antislop-code para los comentarios (nada de comentarios genéricos
  que repiten el código) y antislop-ui para la interfaz. La interfaz debe ser sobria,
  funcional y respetar las convenciones visuales del Diseño (colores por estado,
  formas por tipo); antislop no puede cambiar esas convenciones.
- Código simple y explicable en una sustentación: sin abstracciones innecesarias.
- El autómata y Dijkstra se programan SIN librerías (son el núcleo de la asignatura).
- La GCN se implementa a mano con PyTorch (sin PyTorch Geometric).
- Python 3.14 (plan B: 3.13), entorno virtual .venv, requirements.txt.
- Librerías: numpy, networkx (solo apoyo/verificación), torch (CPU), matplotlib,
  tkinter (incluido con Python), pytest, pyinstaller (al final).

## Estructura
ProyectoSoftCast/
  requirements.txt
  softcast/
    red.py  generador.py  gnn.py  enrutador.py  automata.py  interfaz.py  main.py
    modelo/pesos_gcn.pt
  pruebas/
    test_automata.py  test_enrutador.py  test_gnn.py

## 1. red.py — Grafo no dirigido ponderado, 9 nodos, 21 aristas
Índices: 0 PC1, 1 PC2, 2 PC3, 3 PC4, 4 IoT1, 5 IoT2, 6 IoT3, 7 SRV-L, 8 SRV-G.
Tipos: PC, IoT, SRV-L, SRV-G.
Aristas:
- PC–IoT: PC2–IoT1, PC3–IoT2, PC4–IoT3
- PC–PC: todos entre sí (6)
- PC–SRV-L: los 4 PCs;  IoT–SRV-L: los 3 IoT
- PC–SRV-G: los 4 PCs;  SRV-L–SRV-G
Comprobación: suma de grados = 42; grado SRV-L = 8.
Clases: Nodo(indice, id, tipo, trafico, cpu, latencia, estado, estado_predicho,
probabilidades[3]); Arista(origen, destino, peso, activa);
Red(nodos, aristas, matriz_base, semilla) con cambiar_estado(), matriz_activa()
(fila y columna en 0 para los nodos bloqueados), matriz_ponderada() y
vector_caracteristicas().
Una arista queda inactiva si cualquiera de sus extremos está Bloqueado; se reactiva
al desbloquear.

## 2. generador.py — Datos aleatorios (distribución uniforme)
Probabilidad de cada estado inicial: Normal 70 %, Anómalo 20 %, Bloqueado 10 %.
Rangos por estado (Normal / Anómalo / Bloqueado):
- Tráfico PC: 10–50 / 51–100 / 0 Mbps
- Tráfico IoT: 5–30 / 31–60 / 0 Mbps
- Tráfico servidores: 40–80 / 81–150 / 0 Mbps
- CPU: 10–60 / 61–100 / 0–5 %
- Latencia: 5–50 / 150–500 / 1000 ms
Peso de cada arista: uniforme entre 1 y 10 ms.
generar_red(), generar_escenarios(200), regenerar_caracteristicas(nodo, estado):
al cambiar un estado manualmente se regeneran sus características dentro del rango
del nuevo estado (RF06). Semilla configurable (RNF04).

## 3. automata.py — AFD M = (Q, Σ, δ, q0, F)
Q = {q0..q7}; Σ = {s, a, d, k, r, f, b}; F = {q6}; q7 es sumidero.
Transiciones distintas de q7 (todas las demás van a q7):
q0-s→q1; q1-a→q2; q2-d→q3; q2-f→q6; q3-k→q2; q3-r→q4; q4-k→q2; q4-r→q5; q5-k→q2.
Lenguaje: sa(d(ε + r + rr)k)*f
evaluar(cadena) → (aceptada, recorrido de estados).

Validación de la cadena del usuario, antes de preparar_cadena():
- Solo se aceptan los símbolos s, a, d, k, f.
- Si contiene r o b, no se evalúa y se muestra: "Los símbolos r y b los agrega el
  sistema según el estado de la ruta; no deben escribirse."
- Cualquier otro símbolo fuera de Σ: "Símbolo no válido: <símbolo>".

preparar_cadena(cadena, ruta, red):
- si ruta es NINGUNA: insertar 'b' después de la primera 's'
- n = número de nodos Anómalos en la ruta (incluidos el origen y el destino)
- después de cada 'd' insertar n veces 'r'

Casos obligatorios (pytest, llamando directamente a evaluar(), sin pasar por la
validación de la caja de texto):
sadkf A, saf A, sadkdkf A, sadrrkf A, sadrrrkf R, saddkf R, sbadkf R, dkf R, sadkfd R
(A = aceptada, R = rechazada), verificando también el recorrido.
Probar además la validación: una cadena con r o b debe rechazarse con el mensaje.

## 4. enrutador.py — Dijkstra propio, O(V²)
Costo(u→v) = peso(u, v) + latencia(v). Ignora nodos Bloqueados y aristas inactivas.
Si el origen o el destino están Bloqueados, o no hay ruta: retorna NINGUNA.
Origen y destino no pueden ser el mismo nodo.
Prueba: IoT1→IoT3 con SRV-L Bloqueado debe dar IoT1→PC2→PC4→IoT3 (con pesos y
latencias controlados en la prueba).

## 5. gnn.py — GCN de 2 capas (467 parámetros)
Entrada por nodo (8 valores): tráfico, CPU y latencia normalizados min-max con cotas
fijas (tráfico 0–150, CPU 0–100, latencia 0–1000); one-hot del tipo (4); grado activo/8.
Matriz: A ponderada activa con 1/peso; Â = A + I; normalización D̂^(-1/2) Â D̂^(-1/2).
Capas: GCN 8→16 + ReLU + Dropout 0,2 → GCN 16→16 + ReLU → Lineal 16→3 + Softmax.
Verificar el conteo: 144 + 272 + 51 = 467.
Entrenamiento: 200 escenarios (140 entrenamiento / 30 validación / 30 prueba);
entropía cruzada con pesos por clase inversos a su frecuencia; Adam, lr 0,01,
weight_decay 5e-4; máximo 200 épocas, parada temprana con paciencia 20; lote de
16 escenarios; semilla fija.
Guardar los pesos en modelo/pesos_gcn.pt y cargarlos si existen (no reentrenar).
metricas(): exactitud (meta ≥ 85 %), matriz de confusión, precisión, recall y F1
por clase. Guardar la curva de pérdida como imagen para el documento.

## 6. interfaz.py — Tkinter + Matplotlib (decisión del equipo), una sola ventana con 4 áreas
- Grafo: color por estado (verde Normal, amarillo Anómalo, rojo Bloqueado);
  forma por tipo (PC rectángulo redondeado, IoT óvalo, servidor rectángulo);
  arista activa gris continua, inactiva gris claro punteada; ruta en azul grueso.
  Clic en un nodo = seleccionarlo.
- Nodo seleccionado: características, estado real, estado predicho y probabilidades;
  botones Normal / Anómalo / Bloqueado. Cada cambio regenera las características,
  reclasifica con la GNN y resalta los nodos cuya predicción cambió (RF08).
- Autómata: selectores de origen y destino, caja de texto para la cadena, botón
  Evaluar; mostrar ruta, cadena evaluada, recorrido de estados y resultado.
  Los mensajes de validación (r, b o símbolos no válidos) se muestran en esta área.
- Tabla de nodos (ttk.Treeview): id, tipo, tráfico, CPU, latencia, estado real,
  estado predicho.
- Campo de semilla y botón "Generar nueva red".

## Orden y puntos de parada
1) red.py + generador.py → 2) automata.py + pruebas → 3) enrutador.py + pruebas →
4) gnn.py + entrenamiento + métricas → 5) interfaz.py → 6) main.py y ejecutable
con PyInstaller. Detente y espera aprobación al final de cada paso.
