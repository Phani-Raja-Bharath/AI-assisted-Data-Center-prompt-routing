"""Bilingual provenance: distinguish source data from assumptions and analysis."""
from energy_model import BASE_ENERGY_WH, COOLING_SYSTEMS
from temperature_display import temperature_text


def executive_results_markdown(results, statistics=None, spanish=False):
    """Report calculated values and actual test output, never fixed significance claims."""
    if spanish:
        title = "Resultados calculados para el escenario actual"
        note = "Estos valores proceden de la simulación, no de mediciones de campo. Compare primero el número de solicitudes atendidas: una menor carga servida también reduce la energía y el calor calculados."
        header = "| Estrategia | Solicitudes atendidas / ofrecidas | Energía (Wh) | Emisiones (g CO₂) | Máximo ΔT-AR: °C [°F] |"
        names = {"Random": "Aleatoria", "Energy-Only": "Solo energía", "Heat-Only": "Solo impacto térmico", "Multi-Objective": "Multiobjetivo"}
    else:
        title = "Calculated results for the current scenario"
        note = "These values are simulation outputs, not field measurements. Compare served request counts first: serving fewer requests also reduces calculated energy and heat."
        header = "| Strategy | Served / offered requests | Energy (Wh) | Emissions (g CO₂) | Peak ΔT-AR: °C [°F] |"
        names = {}
    rows = [f"### {title}", note, "", header, "|---|---:|---:|---:|---:|"]
    for name, result in results.items():
        data = result["totals"]
        rows.append(f"| {names.get(name, name)} | {data['served_requests']:,} / {data['offered_requests']:,} | "
                    f"{data['energy_wh']:.2f} | {data['carbon_g']:.2f} | {temperature_text(data['peak_ΔT'], difference=True, precision=6)} |")
    if statistics and statistics.get("comparisons"):
        rows += ["", _comparison_markdown(statistics, names, spanish)]
    elif statistics:
        label = "Prueba sobre las realizaciones de Monte Carlo" if spanish else "Test on Monte Carlo realizations"
        scope = "describe la simulación; no constituye validación de campo" if spanish else "describes the simulation; not field validation"
        rows += ["", f"{label}: p = {statistics['p_value']:.6g}; {scope}."]
    return "\n\n".join(rows[:2]) + "\n" + "\n".join(rows[2:])


def _dz(comparison, spanish):
    """Paired effect size, or an explicit note when it is undefined (identical differences)."""
    if comparison.get("cohens_dz") is None:
        return "no definido (todas las diferencias son idénticas)" if spanish else "undefined (all differences identical)"
    return f"{comparison['cohens_dz']:+.2f}"


def _comparison_markdown(statistics, names, spanish):
    """One line per paired comparison: effect size and win rate first, corrected p-value last."""
    reference = names.get(statistics["reference"], statistics["reference"])
    if spanish:
        lines = [f"**Comparaciones pareadas de Monte Carlo (máximo ΔT-AR, {reference} frente a cada estrategia)**: "
                 "prueba de rangos con signo de Wilcoxon por ejecución, con corrección de Holm. "
                 "Los valores p simulados disminuyen al aumentar el número de ejecuciones; priorice el tamaño del efecto. "
                 "Describe la simulación; no constituye validación de campo."]
    else:
        lines = [f"**Paired Monte Carlo comparisons (peak ΔT-AR, {reference} vs each strategy)**: "
                 "Wilcoxon signed-rank test per run, Holm-corrected. Simulated p-values shrink as runs are added; "
                 "read the effect size first. Describes the simulation; not field validation."]
    for name, c in statistics["comparisons"].items():
        label = names.get(name, name)
        lower = f"{c['reference_lower_share']:.0%}"
        caveat = ""
        if c.get("unserved_runs"):
            caveat = (" Algunas ejecuciones dejaron solicitudes sin atender, lo que reduce el calor por ese motivo."
                      if spanish else " Some runs left requests unserved, which lowers heat for that reason alone.")
        if spanish:
            lines.append(f"- frente a {label}: diferencia mediana {c['median_difference']:+.3g} °C; d_z = {_dz(c, spanish)}; "
                         f"menor en {lower} de {c['n_runs']} ejecuciones; p (Holm) = {c['p_holm']:.3g}.{caveat}")
        else:
            lines.append(f"- vs {label}: median difference {c['median_difference']:+.3g} °C; d_z = {_dz(c, spanish)}; "
                         f"lower in {lower} of {c['n_runs']} runs; p (Holm) = {c['p_holm']:.3g}.{caveat}")
    return "\n".join(lines)


def recommendation_markdown(recommendation, spanish=False):
    """Describe the actual selection criterion and calculated differences neutrally."""
    if not recommendation.get("feasible", True):
        reason = recommendation.get("binding_constraint", "")
        if spanish:
            return ("### Ninguna estrategia factible\n\nNinguna estrategia atiende solicitudes en este escenario. "
                    f"Límite determinante: {reason}.")
        return f"### No feasible strategy\n\nNo strategy serves any request in this scenario. Binding limit: {reason}."
    name = recommendation["recommended_strategy"]
    values = recommendation["improvements"]
    if spanish:
        name = {"Random": "Aleatoria", "Energy-Only": "Solo energía", "Heat-Only": "Solo impacto térmico", "Multi-Objective": "Multiobjetivo"}.get(name, name)
        title = "Estrategia recomendada para este escenario"
        criterion = "La estrategia aleatoria es solo una referencia y no se recomienda. Entre las demás, se consideran las que atienden más solicitudes dentro del límite de latencia y se elige la más cercana al mejor valor en energía, emisiones, latencia y máximo ΔT-AR a la vez (brecha media respecto del mejor, con pesos iguales)."
        note = "Diferencias calculadas respecto de Solo energía; no son efectos medidos ni una demostración de superioridad general. En las reducciones, un signo positivo indica una disminución; en las variaciones de latencia y energía, un signo positivo indica un aumento."
        labels = ["Reducción del máximo ΔT-AR", "Reducción del CV de la disipación térmica", "Variación de la latencia", "Variación del consumo energético"]
    else:
        title = "Recommended strategy for this scenario"
        criterion = "Random is a reference baseline and is never recommended. Among the other strategies, those serving the most requests within the latency limit are compared, and the one closest to the best value on energy, emissions, latency and peak ΔT-AR at once is selected (average gap to the best, equal weights)."
        note = "Calculated differences relative to Energy-Only; these are not measured effects or proof of general superiority. Positive reductions indicate a decrease; positive latency and energy changes indicate an increase."
        labels = ["Peak ΔT-AR reduction", "Heat-dissipation CV reduction", "Latency change", "Energy consumption change"]
    keys = ["ΔT_reduction_pct", "cv_reduction_pct", "latency_change_pct", "energy_overhead_pct"]
    rows = [f"### {title}: {name}", criterion, note]
    rows.extend(f"- {label}: {values[key]:+.2f}%" for label, key in zip(labels, keys))
    return "\n\n".join(rows[:3]) + "\n\n" + "\n".join(rows[3:])


def provenance_rows(spanish=False):
    pue = f"{min(c['pue'] for c in COOLING_SYSTEMS.values()):.2f}–{max(c['pue'] for c in COOLING_SYSTEMS.values()):.2f}"
    if spanish:
        headings = ["Valor o magnitud", "Clasificación", "Actualización o validez", "Referencia e interpretación"]
        rows = [
            ["Meteorología actual", "Estimación obtenida de una API", "Consulta automática y caché de 5 minutos; actualización manual disponible; se indican las fechas de validez y consulta", "Open-Meteo: datos de modelos meteorológicos; no son mediciones de sensores del centro de datos."],
            ["Meteorología histórica para el entrenamiento", "Datos de modelos obtenidos de una API; alternativa sintética si falla la consulta", "Ventana 2021–2024; se reutiliza con el modelo guardado", "API de pronósticos históricos de Open-Meteo. La disponibilidad depende de la ubicación y del proveedor."],
            ["Intensidad de carbono de Argentina", "Promedio nacional anual publicado; aproximación para Neuquén", "Consulta diaria de la última anualidad; respaldo de 2025: 0.346 kg CO₂e/kWh", "OWID / Ember. La tabla de fuentes identifica el año, la consulta y el uso del valor de respaldo. No es telemetría urbana en tiempo real."],
            ["Intensidad de carbono de las demás regiones", "Valores de referencia configurados como supuestos del escenario", "Constantes estáticas; variación horaria modelada", "Los valores de cada región figuran en la tabla de fuentes. Las referencias aportan contexto; la aplicación no verifica su extracción exacta ni consulta una API eléctrica para estas regiones."],
            [f"Energía base: {BASE_ENERGY_WH} Wh/solicitud; PUE: {pue}", "Supuestos del modelo para el análisis comparativo", "Constantes del escenario; no mediciones del hardware utilizado", "Alkrush et al. (2024) proporciona contexto sobre refrigeración. Los PUE específicos y la energía base se declaran como supuestos, no como valores medidos o validados por esa publicación."],
            ["ΔT: coeficiente 0.0012; factor de viento 0.15; área 1 km²", "Indicador térmico simplificado y no calibrado", "Supuestos utilizados en la simulación", "No constituye una predicción validada de la temperatura urbana. La conversión de energía a flujo térmico presupone un intervalo de una hora."],
            ["Latencia: procesamiento 30 ms; propagación 200000 km/s; servicio 1000 solicitudes/s", "Supuestos analíticos de propagación y colas M/M/1", "Utilización = tasa de solicitudes del sitio durante el lote de una hora (solicitudes / 3600 s) dividida por la tasa de servicio; cada sitio se limita al 99 % de ella", "No son mediciones de red. Se emplean para comparar escenarios bajo las mismas hipótesis."],
            ["Predicciones energéticas y resultados estadísticos", "Resultados calculados a partir de datos y supuestos", "Dependientes de la configuración, del modelo y de las realizaciones de Monte Carlo", "Las etiquetas energéticas del entrenamiento se generan mediante fórmulas, incluso cuando la meteorología procede de una API. Los intervalos y las pruebas estadísticas describen la simulación, no una validación experimental de campo."],
        ]
    else:
        headings = ["Value or quantity", "Classification", "Refresh or validity", "Reference and interpretation"]
        rows = [
            ["Current weather", "API-fetched model estimate", "Five-minute auto-check and cache; manual refresh available; valid-at and fetched-at times shown", "Open-Meteo weather-model output; not measurements from datacenter sensors."],
            ["Historical training weather", "API-fetched model data; synthetic fallback on failure", "2021–2024 window; reused with the saved model", "Open-Meteo Historical Forecast API. Coverage depends on location and provider."],
            ["Argentina grid carbon intensity", "Published annual national average; proxy for Neuquén", "Latest annual value checked daily; 2025 fallback: 0.346 kg CO₂e/kWh", "OWID / Ember. The source table identifies year, fetched status, and fallback use. This is not live city-level grid telemetry."],
            ["Other regional carbon intensities", "Configured scenario assumptions", "Static constants with modeled hourly variation", "Actual regional values appear in the source table. References provide context; exact source-table extraction is not verified here, and these regions do not use a live electricity API."],
            [f"Base energy: {BASE_ENERGY_WH} Wh/request; PUE: {pue}", "Model assumptions for comparative analysis", "Scenario constants; not measurements of the hardware in use", "Alkrush et al. (2024) provides cooling context. The exact PUE settings and base energy are declared assumptions, not values measured or validated by that publication."],
            ["ΔT: coefficient 0.0012; wind factor 0.15; area 1 km²", "Simplified, uncalibrated thermal indicator", "Assumptions used for simulation", "Not a validated urban-temperature forecast. Converting energy to heat flux assumes a one-hour interval."],
            ["Latency: processing 30 ms; propagation 200000 km/s; service 1000 requests/s", "Analytical propagation and M/M/1 queue assumptions", "Utilization = the site's request rate over the one-hour batch (requests / 3,600 s) divided by the service rate; sites are capped at 99% of it", "Not measured network performance; used to compare scenarios under the same assumptions."],
            ["Energy predictions and statistical results", "Calculated analysis outputs", "Depend on configuration, model, and Monte Carlo realizations", "Training energy labels come from formulas even when weather is API-sourced. Confidence intervals and tests describe the simulation, not field validation."],
        ]
    if spanish:
        rows += [
            ["Emisiones, latencia y aumento de temperatura", "Cálculos deterministas condicionados a las entradas", "Reproducibles si se mantienen la asignación, la meteorología, la hora del perfil de carbono y el modelo guardado", "Las emisiones se calculan a partir de la energía y la intensidad de carbono; la latencia utiliza fórmulas de distancia y colas; ΔT utiliza un indicador térmico simplificado. La energía puede proceder de una predicción del modelo guardado. El carácter determinista no implica validación experimental."],
            ["Asignación aleatoria y análisis de Monte Carlo", "Muestreo estocástico", "Los resultados nuevos pueden variar; los resultados almacenados se reutilizan", "La estrategia Aleatoria muestrea pesos de asignación. Monte Carlo perturba la meteorología y la demanda. Los intervalos describen estas realizaciones simuladas, no la incertidumbre de mediciones de campo."],
        ]
    else:
        rows += [
            ["Emissions, latency, and temperature rise", "Deterministic calculations conditional on inputs", "Repeatable with allocation, weather, carbon-profile hour, and saved model held fixed", "Emissions use energy and grid intensity; latency uses distance and queue formulas; ΔT uses a simplified thermal proxy. Energy may be predicted by the saved model. Deterministic does not mean experimentally validated."],
            ["Random allocation and Monte Carlo analysis", "Stochastic sampling", "New results may differ; cached results are reused", "Random routing samples allocation weights. Monte Carlo perturbs weather and demand. Intervals describe these simulated realizations, not uncertainty in field measurements."],
        ]
    return [dict(zip(headings, row)) for row in rows]


def reference_markdown(spanish=False):
    if spanish:
        return """
**Fuentes de datos consultables**
- [Open-Meteo: documentación de condiciones actuales](https://open-meteo.com/en/docs). Estimaciones meteorológicas basadas en modelos.
- [Open-Meteo: Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api). Serie histórica de salidas de modelos empleada como entrada del entrenamiento.
- [OWID / Ember: intensidad de carbono de la electricidad](https://ourworldindata.org/grapher/carbon-intensity-electricity). Promedio anual nacional utilizado para Argentina; el año y el estado de la consulta figuran en la tabla de fuentes.

**Referencia contextual sobre refrigeración**

Alkrush, A. A., Salem, M. S., Abdelrehim, O., & Hegazi, A. A. (2024).
*Data centers cooling: A critical review of techniques, challenges, and energy saving solutions.*
International Journal of Refrigeration, 160, 246–262. [DOI: 10.1016/j.ijrefrig.2024.02.007](https://doi.org/10.1016/j.ijrefrig.2024.02.007).

**Alcance de las citas:** la referencia de refrigeración proporciona contexto científico; no verifica automáticamente cada constante elegida para este escenario. La energía base, los PUE específicos, los coeficientes térmicos, la latencia y la matriz cualitativa de refrigeración son supuestos declarados para el análisis. Las predicciones y los resultados estadísticos se calculan a partir de estos supuestos; no constituyen mediciones de campo.
"""
    return """
**Retrievable data sources**
- [Open-Meteo current-conditions documentation](https://open-meteo.com/en/docs): weather-model estimates.
- [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api): historical model output used as training inputs.
- [OWID / Ember electricity carbon intensity](https://ourworldindata.org/grapher/carbon-intensity-electricity): annual national average used for Argentina; the source table shows its year and fetch status.

**Contextual cooling reference**

Alkrush, A. A., Salem, M. S., Abdelrehim, O., & Hegazi, A. A. (2024).
*Data centers cooling: A critical review of techniques, challenges, and energy saving solutions.*
International Journal of Refrigeration, 160, 246–262. [DOI: 10.1016/j.ijrefrig.2024.02.007](https://doi.org/10.1016/j.ijrefrig.2024.02.007).

**Citation scope:** the cooling reference provides scientific context; it does not verify every scenario constant. Base energy, exact PUE settings, thermal and latency coefficients, and the qualitative cooling matrix are declared modeling assumptions. Predictions and statistics are calculated from these assumptions, not field measurements.
"""
