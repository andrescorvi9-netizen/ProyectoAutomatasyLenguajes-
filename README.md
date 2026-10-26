# SoftCast

Prototipo de las etapas 3 y 4 del proyecto de Lenguajes y Autómatas (Ingeniería de Sistemas y Computación, Universidad de Cundinamarca).

SoftCast simula una red de 9 nodos (4 PC, 3 IoT, un servidor local y uno global) unidos por 21 aristas. Cada nodo puede estar Normal, Anómalo o Bloqueado. El programa hace tres cosas sobre esa red:

- Calcula la ruta de menor costo entre dos nodos con una implementación propia de Dijkstra.
- Valida la secuencia de una transmisión con un autómata finito determinista que reconoce el lenguaje `sa(d(ε + r + rr)k)*f`. El sistema agrega `r` por cada nodo Anómalo de la ruta y `b` cuando no hay ruta.
- Clasifica el estado de cada nodo con una red convolucional de grafos (GCN) de 2 capas y 467 parámetros, escrita a mano con PyTorch. En los 30 escenarios de prueba alcanza 87,41 % de exactitud.

La interfaz muestra el grafo, el nodo seleccionado, el autómata y una tabla con todos los nodos en una sola ventana.

## Requisitos

- Python 3.14
- Las librerías de `requirements.txt`. PyTorch se instala en su versión para CPU desde el índice de PyTorch, que ya está indicado en ese archivo.

## Instalación

En PowerShell, dentro de la carpeta del proyecto:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecución

```powershell
python -m softcast.main
```

Para abrir siempre la misma red, pase una semilla entera:

```powershell
python -m softcast.main 1019
```

Los pesos entrenados de la GCN están en `softcast/modelo/pesos_gcn.pt` y se cargan al iniciar. Si se borra ese archivo, el programa vuelve a entrenar la red antes de abrir la ventana. Para entrenar y regenerar las métricas a mano:

```powershell
python -m softcast.gnn
```

## Pruebas

```powershell
python -m pytest
```

Las pruebas de `pruebas/test_interfaz.py` abren la ventana y la manejan con eventos de Tk. En un equipo sin pantalla se omiten.

## Ejecutable para Windows

El ejecutable se construye con PyInstaller a partir de `SoftCast.spec`:

```powershell
pyinstaller SoftCast.spec
```

El resultado queda en `dist/SoftCast/`. Esa carpeta no se guarda en el repositorio porque pesa cerca de 500 MB; se publica comprimida en la sección Releases.

## Estructura

```
ProyectoSoftCast/
├── softcast/
│   ├── red.py          Grafo, nodos, aristas y matrices
│   ├── generador.py    Redes y escenarios aleatorios
│   ├── automata.py     AFD, validación y preparación de la cadena
│   ├── enrutador.py    Dijkstra O(V²)
│   ├── gnn.py          GCN, entrenamiento y métricas
│   ├── interfaz.py     Ventana con Tkinter y Matplotlib
│   ├── main.py         Punto de entrada
│   └── modelo/
│       └── pesos_gcn.pt
├── pruebas/            Pruebas con pytest
├── resultados/         Métricas, curva de pérdida y capturas de la interfaz
├── requirements.txt
├── pytest.ini
├── SoftCast.spec
└── CONTEXTO_PROYECTO.md
```
