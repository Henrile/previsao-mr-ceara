from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def start_app():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    assert not app.exception
    return app


def test_previsao_com_modelo_do_repositorio():
    app = start_app()
    app.button[0].click().run()
    app.button[1].click().run()

    assert not app.exception
    assert not app.error
    assert float(app.metric[0].value.split()[0]) == pytest.approx(176.91, abs=0.02)

    app.number_input[5].set_value(20.0)
    app.button[1].click().run()
    assert "granulometria" in app.error[0].value
    assert not app.metric


def test_painel_carrega_filtra_e_detecta_recorte_vazio():
    app = start_app()
    csv = (
        "Nome;Class ;Class;σ3;σd;2,0 mm (%);0,42 mm (%);0,074 mm (%);MR (MPa)\n"
        "01;A-1-B;1,2;0,021;0,041;50;25;10;100,5\n"
        "01;A-1-b;1,2;0,021;0,062;50;25;10;200,5\n"
        "02;A-2-4;2,4;0,035;0,041;80;25;10;300,5\n"
        "02;A-2-4;2,4;0,035;0,062;80;25;10;0\n"
    )
    app.file_uploader[0].upload("ensaios.csv", csv.encode(), "text/csv").run()

    assert not app.exception
    assert not app.error
    assert [metric.value for metric in app.metric] == ["4", "2", "200,5 MPa", "3"]
    assert "01" in app.selectbox[0].options
    assert app.multiselect[0].options == ["A-1-b", "A-2-4"]

    app.multiselect[0].set_value(["A-1-b"])
    app.selectbox[2].set_value(0.062).run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["1", "1", "200,5 MPa", "1"]

    app.selectbox[1].set_value(0.035).run()
    assert not app.exception
    assert not app.metric
    assert "Nenhum ensaio corresponde" in app.warning[0].value


def test_painel_orienta_quando_falta_mr():
    app = start_app()
    app.file_uploader[0].upload("ensaios.csv", b"Nome,CBR\n01,12\n", "text/csv").run()
    assert not app.exception
    assert "coluna MR" in app.error[0].value
