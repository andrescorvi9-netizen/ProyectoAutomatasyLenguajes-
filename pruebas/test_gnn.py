import numpy as np
import pytest
import torch

from softcast import gnn
from softcast.generador import generar_escenarios, generar_red
from softcast.red import BLOQUEADO, NORMAL


def test_conteo_de_parametros():
    parametros = gnn.contar_parametros(gnn.GCN())
    assert list(parametros.values()) == [144, 272, 51]
    assert sum(parametros.values()) == 467


def test_matriz_normalizada_coincide_con_la_formula():
    red = generar_red(7)
    pesos = red.matriz_ponderada()
    a_gorro = np.where(pesos > 0, 1.0 / np.where(pesos > 0, pesos, 1.0), 0.0) + np.eye(9)
    d = np.diag(1.0 / np.sqrt(a_gorro.sum(axis=1)))
    esperada = d @ a_gorro @ d
    obtenida = gnn.matriz_normalizada(red).numpy()
    assert np.allclose(obtenida, esperada, atol=1e-6)
    assert np.allclose(obtenida, obtenida.T)


def test_nodo_bloqueado_solo_conserva_su_lazo():
    red = generar_red(7)
    red.cambiar_estado(7, BLOQUEADO)
    fila = gnn.matriz_normalizada(red)[7]
    assert fila[7] == pytest.approx(1.0)
    assert fila.sum() == pytest.approx(1.0)


def test_probabilidades_suman_uno():
    probabilidades = gnn.predecir(gnn.GCN(), generar_red(7))
    assert probabilidades.shape == (9, 3)
    assert torch.allclose(probabilidades.sum(dim=1), torch.ones(9))


def test_metricas_calculadas_a_mano():
    # 0 = Normal, 1 = Anómalo, 2 = Bloqueado
    resultado = gnn.calcular_metricas([0, 0, 1, 1, 2], [0, 1, 1, 1, 2])
    assert resultado["exactitud"] == 0.8
    assert resultado["matriz_confusion"] == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]
    assert resultado["por_clase"]["Anómalo"]["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert resultado["por_clase"]["Normal"]["recall"] == 0.5


def test_clasificar_reporta_nodos_que_cambian():
    modelo = gnn.GCN()
    red = generar_red(7)
    assert gnn.clasificar(modelo, red) == []
    red.nodos[0].estado_predicho = "otro"
    assert gnn.clasificar(modelo, red) == [0]


def test_si_existen_pesos_se_cargan_sin_reentrenar(tmp_path, monkeypatch):
    original = gnn.GCN()
    ruta = tmp_path / "pesos_gcn.pt"
    torch.save(original.state_dict(), ruta)

    def entrenar_prohibido(*args, **kwargs):
        raise AssertionError("No debe reentrenar si el archivo de pesos existe")

    monkeypatch.setattr(gnn, "entrenar", entrenar_prohibido)
    modelo, registro = gnn.cargar_o_entrenar(ruta)
    assert registro is None
    for a, b in zip(original.parameters(), modelo.parameters()):
        assert torch.equal(a, b)


@pytest.mark.skipif(not gnn.RUTA_PESOS.exists(), reason="Ejecute primero python -m softcast.gnn")
def test_modelo_guardado_cumple_la_meta_de_exactitud():
    modelo, _ = gnn.cargar_o_entrenar()
    _, _, prueba = gnn.dividir_escenarios(generar_escenarios(gnn.CANTIDAD_ESCENARIOS))
    assert gnn.metricas(modelo, prueba)["exactitud"] >= 0.85
