"""Font assets and Plotly overrides must also work on machines without Aptos."""
from pathlib import Path
import tomllib

import plotly.graph_objects as go

import dashboard_ui


def test_deployment_font_assets_cover_theme():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / ".streamlit/config.toml").read_text(encoding="utf-8"))
    assert config["server"]["enableStaticServing"]
    theme = config["theme"]
    assert theme["font"] == theme["headingFont"] == theme["codeFont"] == "Aptos, sans-serif"
    assert {face["weight"] for face in theme["fontFaces"]} == {400, 700}
    for face in theme["fontFaces"]:
        assert face["family"] == "Aptos"
        # TrueType header: catches missing assets and Git LFS pointer files.
        assert (root / face["url"].removeprefix("app/")).read_bytes()[:4] == b"\x00\x01\x00\x00"


def test_display_plot_overrides_nested_fonts_without_changing_source(monkeypatch):
    captured = []
    monkeypatch.setattr(dashboard_ui.st, "plotly_chart", lambda fig, **kwargs: captured.append(fig))
    monkeypatch.setattr(dashboard_ui, "localized_figure", lambda fig: fig)
    figure = go.Figure(go.Bar(x=["A"], y=[1], text=["1"], textfont_family="Courier New",
                             hoverlabel_font_family="Arial", insidetextfont_family="Arial"))
    figure.update_layout(font_family="Times New Roman", legend_font_family="Arial",
                         xaxis_tickfont_family="Arial", annotations=[dict(text="Label", x=0, y=0,
                                                                          font_family="Arial")])
    dashboard_ui.display_plot(figure, "font_check")
    rendered = captured[0]
    assert rendered.layout.font.family == "Aptos, sans-serif"
    assert rendered.layout.legend.font.family == "Aptos, sans-serif"
    assert rendered.layout.hoverlabel.font.family == "Aptos, sans-serif"
    assert rendered.layout.xaxis.tickfont.family == "Aptos, sans-serif"
    assert rendered.layout.annotations[0].font.family == "Aptos, sans-serif"
    assert rendered.data[0].hoverlabel.font.family == "Aptos, sans-serif"
    assert rendered.data[0].insidetextfont.family == "Aptos, sans-serif"
    assert figure.layout.font.family == "Times New Roman"


def test_map_glyph_fonts_are_preserved():
    spec = go.Figure(go.Scattermap(lat=[0], lon=[0], text=["Site"],
                                   textfont_family="Open Sans Regular",
                                   hoverlabel_font_family="Arial")).to_plotly_json()
    dashboard_ui._aptos_fonts(spec)
    assert spec["data"][0]["textfont"]["family"] == "Open Sans Regular"
    assert spec["data"][0]["hoverlabel"]["font"]["family"] == "Aptos, sans-serif"
