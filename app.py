import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Análisis de Riesgo de Concentración de Clientes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main {background-color: #F7F8FA;}
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .metric-card {
        background: white;
        padding: 18px 18px;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(31, 41, 55, 0.08);
        border: 1px solid #ECEFF3;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #111827;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .small-note {color:#6B7280; font-size:0.9rem;}
    .risk-low {color:#047857; font-weight:800;}
    .risk-medium {color:#B45309; font-weight:800;}
    .risk-high {color:#B91C1C; font-weight:800;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# Utilidades de datos
# ------------------------------------------------------------
def make_sample_data(n_clients: int = 24, months: int = 24, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=months, freq="M")
    clients = [f"Cliente {i:02d}" for i in range(1, n_clients + 1)]

    base = rng.pareto(1.7, n_clients) + 0.25
    base = base / base.sum()
    monthly_market = rng.normal(1.006, 0.025, months)

    rows = []
    for c, weight in zip(clients, base):
        initial = weight * 13_500_000
        trend = rng.normal(0.004, 0.012)
        margin = np.clip(rng.normal(0.31, 0.08), 0.08, 0.55)
        volatility = np.clip(rng.normal(0.085, 0.035), 0.025, 0.20)
        revenue = initial
        for idx, d in enumerate(dates):
            shock = rng.normal(0, volatility)
            revenue = max(revenue * monthly_market[idx] * (1 + trend + shock), 0)
            gross_profit = revenue * np.clip(rng.normal(margin, 0.035), 0.04, 0.65)
            rows.append({
                "Fecha": d,
                "Cliente": c,
                "Ingresos": revenue,
                "Rentabilidad": gross_profit / revenue if revenue else 0,
                "Utilidad": gross_profit,
            })
    return pd.DataFrame(rows)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    aliases = {
        "Fecha": ["fecha", "date", "periodo", "mes"],
        "Cliente": ["cliente", "client", "customer", "cuenta", "nombre cliente"],
        "Ingresos": ["ingresos", "revenue", "ventas", "facturacion", "facturación", "importe"],
        "Rentabilidad": ["rentabilidad", "margen", "margin", "profitability"],
        "Utilidad": ["utilidad", "profit", "gross profit", "beneficio", "contribucion", "contribución"],
    }
    for target, opts in aliases.items():
        for opt in opts:
            if opt in lower_cols:
                col_map[lower_cols[opt]] = target
                break
    df = df.rename(columns=col_map)

    required = {"Fecha", "Cliente", "Ingresos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(missing)}")

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Ingresos"] = pd.to_numeric(df["Ingresos"], errors="coerce").fillna(0)
    if "Rentabilidad" not in df.columns:
        df["Rentabilidad"] = np.nan
    else:
        df["Rentabilidad"] = pd.to_numeric(df["Rentabilidad"], errors="coerce")
        df.loc[df["Rentabilidad"] > 1, "Rentabilidad"] = df.loc[df["Rentabilidad"] > 1, "Rentabilidad"] / 100
    if "Utilidad" not in df.columns:
        df["Utilidad"] = df["Ingresos"] * df["Rentabilidad"].fillna(0)
    else:
        df["Utilidad"] = pd.to_numeric(df["Utilidad"], errors="coerce")
        df["Utilidad"] = df["Utilidad"].fillna(df["Ingresos"] * df["Rentabilidad"].fillna(0))

    df = df.dropna(subset=["Fecha", "Cliente"])
    df = df[df["Ingresos"] >= 0]
    return df


def client_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = df.groupby([pd.Grouper(key="Fecha", freq="M"), "Cliente"], as_index=False).agg(
        Ingresos=("Ingresos", "sum"),
        Utilidad=("Utilidad", "sum"),
    )
    pivot = monthly.pivot(index="Fecha", columns="Cliente", values="Ingresos").fillna(0)
    returns = pivot.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    total_income = pivot.sum().sum()

    agg = monthly.groupby("Cliente", as_index=False).agg(
        Ingresos=("Ingresos", "sum"),
        Utilidad=("Utilidad", "sum"),
        Ingreso_Promedio_Mensual=("Ingresos", "mean"),
    )
    first_last = pivot.apply(lambda s: pd.Series({
        "Inicial": s[s > 0].iloc[0] if (s > 0).any() else 0,
        "Final": s.iloc[-1] if len(s) else 0,
    })).T
    periods = max(len(pivot) - 1, 1)
    growth = ((first_last["Final"] / first_last["Inicial"].replace(0, np.nan)) ** (12 / periods) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
    risk = returns.std() * np.sqrt(12)

    agg["Participacion"] = agg["Ingresos"] / total_income if total_income else 0
    agg["Rentabilidad"] = np.where(agg["Ingresos"] > 0, agg["Utilidad"] / agg["Ingresos"], 0)
    agg["Riesgo_Individual"] = agg["Cliente"].map(risk).fillna(0)
    agg["Crecimiento_Historico"] = agg["Cliente"].map(growth).fillna(0)
    agg = agg.sort_values("Ingresos", ascending=False).reset_index(drop=True)
    agg["Ranking"] = np.arange(1, len(agg) + 1)
    agg["Participacion_Acumulada"] = agg["Participacion"].cumsum()
    agg["Clasificacion_Riesgo"] = pd.cut(
        agg["Participacion"],
        bins=[-0.01, 0.05, 0.10, 1],
        labels=["Bajo", "Medio", "Alto"],
    )
    return agg, pivot


def concentration_indicators(metrics: pd.DataFrame) -> dict:
    shares = metrics["Participacion"].to_numpy()
    hhi = float(np.sum(shares ** 2) * 10_000)
    top5 = float(metrics.head(5)["Participacion"].sum())
    top10 = float(metrics.head(10)["Participacion"].sum())
    dominant = float(metrics["Participacion"].max()) if len(metrics) else 0
    dominant_client = metrics.iloc[0]["Cliente"] if len(metrics) else "N/A"
    pareto_clients = int((metrics["Participacion_Acumulada"] <= 0.80).sum()) + 1 if len(metrics) else 0
    pareto_clients = min(pareto_clients, len(metrics))
    pareto_pct_clients = pareto_clients / len(metrics) if len(metrics) else 0

    if hhi < 1_000 and dominant < 0.15 and top5 < 0.45:
        risk_level = "Bajo"
    elif hhi < 1_800 and dominant < 0.25 and top5 < 0.65:
        risk_level = "Medio"
    else:
        risk_level = "Alto"

    return {
        "hhi": hhi,
        "top5": top5,
        "top10": top10,
        "dominant": dominant,
        "dominant_client": dominant_client,
        "pareto_clients": pareto_clients,
        "pareto_pct_clients": pareto_pct_clients,
        "risk_level": risk_level,
    }


def optimize_portfolio(metrics: pd.DataFrame, max_weight: float, min_profitability: float, lambda_vol: float = 0.35) -> pd.DataFrame:
    n = len(metrics)
    if n == 0:
        return metrics
    current_w = metrics["Participacion"].to_numpy()
    profitability = metrics["Rentabilidad"].fillna(0).to_numpy()
    volatility = metrics["Riesgo_Individual"].fillna(0).to_numpy()
    avg_profit = np.average(profitability, weights=current_w) if current_w.sum() else profitability.mean()
    target_profit = max(min_profitability, avg_profit * 0.92)

    def objective(w):
        concentration = np.sum(w ** 2)
        vol_penalty = np.sum(w * volatility)
        distance_penalty = 0.08 * np.sum((w - current_w) ** 2)
        return concentration + lambda_vol * vol_penalty + distance_penalty

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: np.dot(w, profitability) - target_profit},
    ]
    bounds = [(0, max_weight) for _ in range(n)]
    x0 = np.minimum(np.ones(n) / n, max_weight)
    x0 = x0 / x0.sum()

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000})
    w = result.x if result.success else x0
    out = metrics[["Cliente", "Participacion", "Rentabilidad", "Riesgo_Individual", "Ingresos"]].copy()
    out["Participacion_Optimizada"] = w
    out["Cambio_pp"] = (out["Participacion_Optimizada"] - out["Participacion"]) * 100
    out["Ingreso_Objetivo"] = out["Participacion_Optimizada"] * metrics["Ingresos"].sum()
    return out.sort_values("Participacion_Optimizada", ascending=False)


def stress_scenario(metrics: pd.DataFrame, clients_to_remove: list[str]) -> dict:
    lost_income = metrics[metrics["Cliente"].isin(clients_to_remove)]["Ingresos"].sum()
    total = metrics["Ingresos"].sum()
    remaining = metrics[~metrics["Cliente"].isin(clients_to_remove)].copy()
    remaining_total = remaining["Ingresos"].sum()
    if remaining_total > 0:
        remaining["Participacion"] = remaining["Ingresos"] / remaining_total
        new_hhi = np.sum(remaining["Participacion"] ** 2) * 10_000
        new_top5 = remaining.sort_values("Ingresos", ascending=False).head(5)["Participacion"].sum()
    else:
        new_hhi, new_top5 = 0, 0
    return {
        "Clientes Perdidos": ", ".join(clients_to_remove) if clients_to_remove else "Ninguno",
        "Ingreso Perdido": lost_income,
        "% Ingreso Perdido": lost_income / total if total else 0,
        "Ingreso Remanente": total - lost_income,
        "HHI Remanente": new_hhi,
        "Top 5 Remanente": new_top5,
    }


def format_money(x):
    return f"${x:,.0f}"


def format_pct(x):
    return f"{x:.1%}"


def export_excel(metrics, indicators, stress_df, optimized, pivot) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary = pd.DataFrame([{
            "Ingresos Totales": metrics["Ingresos"].sum(),
            "Número de Clientes": len(metrics),
            "HHI": indicators["hhi"],
            "Top 5 %": indicators["top5"],
            "Top 10 %": indicators["top10"],
            "Cliente Dominante": indicators["dominant_client"],
            "Cliente Dominante %": indicators["dominant"],
            "Semáforo de Riesgo": indicators["risk_level"],
        }])
        summary.to_excel(writer, sheet_name="Resumen Ejecutivo", index=False)
        metrics.to_excel(writer, sheet_name="Clientes", index=False)
        stress_df.to_excel(writer, sheet_name="Stress Test", index=False)
        optimized.to_excel(writer, sheet_name="Portafolio Optimizado", index=False)
        pivot.to_excel(writer, sheet_name="Histórico Ingresos")
    return output.getvalue()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")
st.sidebar.caption("Carga un Excel o usa datos demo para probar el modelo.")

uploaded = st.sidebar.file_uploader("Archivo Excel", type=["xlsx"])
use_demo = st.sidebar.toggle("Usar datos demo", value=True if uploaded is None else False)

st.sidebar.markdown("---")
max_weight = st.sidebar.slider("Límite máximo ideal por cliente", 5, 35, 12, 1) / 100
min_profitability = st.sidebar.slider("Rentabilidad mínima objetivo", 0, 60, 20, 1) / 100
horizon_note = st.sidebar.info("Formato esperado: Fecha, Cliente, Ingresos. Opcional: Rentabilidad o Utilidad.")

try:
    if uploaded is not None and not use_demo:
        raw = pd.read_excel(uploaded)
        df = normalize_columns(raw)
    else:
        df = make_sample_data()
except Exception as exc:
    st.error(f"No fue posible procesar el archivo: {exc}")
    st.stop()

metrics, pivot = client_metrics(df)
ind = concentration_indicators(metrics)
optimized = optimize_portfolio(metrics, max_weight=max_weight, min_profitability=min_profitability)

risk_class = {"Bajo": "risk-low", "Medio": "risk-medium", "Alto": "risk-high"}.get(ind["risk_level"], "")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("📊 Análisis de Riesgo de Concentración de Clientes")
st.caption("Dashboard ejecutivo para Board Meeting | Evaluación de dependencia, concentración, estrés y diversificación del portafolio.")

# KPIs
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Ingresos Totales", format_money(metrics["Ingresos"].sum()))
c2.metric("Clientes", f"{len(metrics):,}")
c3.metric("HHI", f"{ind['hhi']:,.0f}")
c4.metric("Top 5", format_pct(ind["top5"]))
c5.metric("Top 10", format_pct(ind["top10"]))
c6.metric("Dominante", format_pct(ind["dominant"]))
c7.markdown(f"<div class='metric-card'><div class='small-note'>Semáforo</div><div class='{risk_class}' style='font-size:1.7rem'>{ind['risk_level']}</div></div>", unsafe_allow_html=True)

st.markdown("---")

# Tabs
main_tabs = st.tabs([
    "Resumen Ejecutivo",
    "Concentración",
    "Histórico y Correlación",
    "Stress Test",
    "Optimización",
    "Exportación",
])

with main_tabs[0]:
    st.markdown("<div class='section-title'>Lectura ejecutiva</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.write(
            f"El portafolio registra **{format_money(metrics['Ingresos'].sum())}** de ingresos acumulados, "
            f"distribuidos en **{len(metrics)} clientes**. El índice HHI es **{ind['hhi']:,.0f}**, "
            f"con una concentración Top 5 de **{format_pct(ind['top5'])}** y un cliente dominante "
            f"(**{ind['dominant_client']}**) con **{format_pct(ind['dominant'])}** del total."
        )
        if ind["risk_level"] == "Alto":
            st.error("El portafolio presenta riesgo alto: existe dependencia relevante en pocos clientes. Se recomienda diversificación comercial y límites de exposición por cliente.")
        elif ind["risk_level"] == "Medio":
            st.warning("El portafolio presenta riesgo medio: la concentración es manejable, pero requiere monitoreo y reducción gradual de clientes críticos.")
        else:
            st.success("El portafolio presenta riesgo bajo: la distribución de ingresos es relativamente diversificada.")
    with col_b:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ind["hhi"],
            title={"text": "HHI de Concentración"},
            gauge={
                "axis": {"range": [0, 3500]},
                "bar": {"thickness": 0.28},
                "steps": [
                    {"range": [0, 1000], "color": "#D1FAE5"},
                    {"range": [1000, 1800], "color": "#FEF3C7"},
                    {"range": [1800, 3500], "color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 4}, "value": ind["hhi"]},
            },
        ))
        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge, use_container_width=True)

    st.markdown("<div class='section-title'>Top clientes por riesgo de concentración</div>", unsafe_allow_html=True)
    show_cols = ["Ranking", "Cliente", "Ingresos", "Participacion", "Participacion_Acumulada", "Riesgo_Individual", "Crecimiento_Historico", "Rentabilidad", "Clasificacion_Riesgo"]
    st.dataframe(
        metrics[show_cols].head(15).style.format({
            "Ingresos": "${:,.0f}",
            "Participacion": "{:.1%}",
            "Participacion_Acumulada": "{:.1%}",
            "Riesgo_Individual": "{:.1%}",
            "Crecimiento_Historico": "{:.1%}",
            "Rentabilidad": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with main_tabs[1]:
    st.markdown("<div class='section-title'>Pareto de clientes</div>", unsafe_allow_html=True)
    pareto = metrics.copy()
    fig_pareto = go.Figure()
    fig_pareto.add_bar(x=pareto["Cliente"], y=pareto["Participacion"], name="Participación")
    fig_pareto.add_scatter(x=pareto["Cliente"], y=pareto["Participacion_Acumulada"], name="Acumulado", yaxis="y2", mode="lines+markers")
    fig_pareto.add_hline(y=0.80, line_dash="dash", annotation_text="80%")
    fig_pareto.update_layout(
        height=460,
        yaxis=dict(title="Participación", tickformat=".0%"),
        yaxis2=dict(title="Acumulado", tickformat=".0%", overlaying="y", side="right", range=[0, 1.05]),
        xaxis=dict(tickangle=-45),
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=30, b=80),
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_tree = px.treemap(metrics, path=["Clasificacion_Riesgo", "Cliente"], values="Ingresos", color="Participacion", title="Treemap de ingresos por cliente")
        fig_tree.update_layout(height=430, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_tree, use_container_width=True)
    with col2:
        donut_df = metrics.head(10).copy()
        others = metrics.iloc[10:]["Ingresos"].sum()
        if others > 0:
            donut_df = pd.concat([donut_df, pd.DataFrame([{"Cliente": "Otros", "Ingresos": others}])], ignore_index=True)
        fig_donut = px.pie(donut_df, names="Cliente", values="Ingresos", hole=0.55, title="Dona de concentración Top 10 + Otros")
        fig_donut.update_layout(height=430, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_donut, use_container_width=True)

with main_tabs[2]:
    st.markdown("<div class='section-title'>Evolución histórica de ingresos</div>", unsafe_allow_html=True)
    top_n = st.slider("Clientes a visualizar", 3, min(15, len(metrics)), min(8, len(metrics)))
    selected_clients = metrics.head(top_n)["Cliente"].tolist()
    hist = pivot[selected_clients].reset_index().melt(id_vars="Fecha", var_name="Cliente", value_name="Ingresos")
    fig_hist = px.line(hist, x="Fecha", y="Ingresos", color="Cliente", markers=True, title="Evolución histórica - principales clientes")
    fig_hist.update_layout(height=460, margin=dict(l=20, r=20, t=45, b=20), yaxis_tickprefix="$")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<div class='section-title'>Heatmap de correlación entre clientes</div>", unsafe_allow_html=True)
    corr_clients = metrics.head(min(12, len(metrics)))["Cliente"].tolist()
    corr = pivot[corr_clients].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).corr()
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlación de variaciones mensuales de ingresos")
    fig_corr.update_layout(height=520, margin=dict(l=20, r=20, t=45, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

with main_tabs[3]:
    st.markdown("<div class='section-title'>Simulación de estrés</div>", unsafe_allow_html=True)
    top1 = [metrics.iloc[0]["Cliente"]] if len(metrics) else []
    top3 = metrics.head(3)["Cliente"].tolist()
    top5_clients = metrics.head(5)["Cliente"].tolist()
    custom_clients = st.multiselect("Escenario personalizado: clientes perdidos", metrics["Cliente"].tolist(), default=top1)

    scenarios = [
        stress_scenario(metrics, top1) | {"Escenario": "Pérdida cliente principal"},
        stress_scenario(metrics, top3) | {"Escenario": "Pérdida Top 3"},
        stress_scenario(metrics, top5_clients) | {"Escenario": "Pérdida Top 5"},
        stress_scenario(metrics, custom_clients) | {"Escenario": "Personalizado"},
    ]
    stress_df = pd.DataFrame(scenarios)[["Escenario", "Clientes Perdidos", "Ingreso Perdido", "% Ingreso Perdido", "Ingreso Remanente", "HHI Remanente", "Top 5 Remanente"]]

    st.dataframe(
        stress_df.style.format({
            "Ingreso Perdido": "${:,.0f}",
            "% Ingreso Perdido": "{:.1%}",
            "Ingreso Remanente": "${:,.0f}",
            "HHI Remanente": "{:,.0f}",
            "Top 5 Remanente": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig_stress = px.bar(stress_df, x="Escenario", y="% Ingreso Perdido", text="% Ingreso Perdido", title="Impacto por pérdida de clientes críticos")
    fig_stress.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig_stress.update_layout(height=430, yaxis_tickformat=".0%", margin=dict(l=20, r=20, t=45, b=40))
    st.plotly_chart(fig_stress, use_container_width=True)

with main_tabs[4]:
    st.markdown("<div class='section-title'>Optimización inspirada en Markowitz</div>", unsafe_allow_html=True)
    st.caption("Objetivo: reducir concentración, limitar peso máximo por cliente y conservar rentabilidad mínima objetivo.")

    current_hhi = np.sum(optimized["Participacion"] ** 2) * 10_000
    opt_hhi = np.sum(optimized["Participacion_Optimizada"] ** 2) * 10_000
    current_profit = np.average(optimized["Rentabilidad"], weights=optimized["Participacion"])
    opt_profit = np.average(optimized["Rentabilidad"], weights=optimized["Participacion_Optimizada"])
    current_vol = np.average(optimized["Riesgo_Individual"], weights=optimized["Participacion"])
    opt_vol = np.average(optimized["Riesgo_Individual"], weights=optimized["Participacion_Optimizada"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("HHI actual", f"{current_hhi:,.0f}")
    k2.metric("HHI optimizado", f"{opt_hhi:,.0f}", delta=f"{opt_hhi-current_hhi:,.0f}")
    k3.metric("Rentabilidad actual", format_pct(current_profit))
    k4.metric("Rentabilidad optimizada", format_pct(opt_profit), delta=format_pct(opt_profit-current_profit))

    frontier = []
    caps = np.linspace(max(1 / max(len(metrics), 1), 0.05), 0.35, 22)
    for cap in caps:
        tmp = optimize_portfolio(metrics, max_weight=float(cap), min_profitability=min_profitability)
        frontier.append({
            "Límite por Cliente": cap,
            "HHI": np.sum(tmp["Participacion_Optimizada"] ** 2) * 10_000,
            "Rentabilidad": np.average(tmp["Rentabilidad"], weights=tmp["Participacion_Optimizada"]),
            "Volatilidad": np.average(tmp["Riesgo_Individual"], weights=tmp["Participacion_Optimizada"]),
        })
    frontier_df = pd.DataFrame(frontier)
    fig_frontier = px.scatter(frontier_df, x="Volatilidad", y="Rentabilidad", size="Límite por Cliente", color="HHI", title="Frontera de diversificación")
    fig_frontier.add_scatter(x=[current_vol], y=[current_profit], mode="markers+text", text=["Actual"], textposition="top center", name="Actual", marker=dict(size=14, symbol="diamond"))
    fig_frontier.update_layout(height=470, xaxis_tickformat=".1%", yaxis_tickformat=".1%", margin=dict(l=20, r=20, t=45, b=20))
    st.plotly_chart(fig_frontier, use_container_width=True)

    comp = optimized.head(15).melt(id_vars="Cliente", value_vars=["Participacion", "Participacion_Optimizada"], var_name="Tipo", value_name="Participación")
    comp["Tipo"] = comp["Tipo"].replace({"Participacion": "Actual", "Participacion_Optimizada": "Optimizada"})
    fig_comp = px.bar(comp, x="Cliente", y="Participación", color="Tipo", barmode="group", title="Distribución actual vs optimizada - Top 15")
    fig_comp.update_layout(height=460, yaxis_tickformat=".0%", xaxis_tickangle=-45, margin=dict(l=20, r=20, t=45, b=80))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.dataframe(
        optimized[["Cliente", "Participacion", "Participacion_Optimizada", "Cambio_pp", "Ingreso_Objetivo", "Rentabilidad", "Riesgo_Individual"]].style.format({
            "Participacion": "{:.1%}",
            "Participacion_Optimizada": "{:.1%}",
            "Cambio_pp": "{:+.1f}",
            "Ingreso_Objetivo": "${:,.0f}",
            "Rentabilidad": "{:.1%}",
            "Riesgo_Individual": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with main_tabs[5]:
    st.markdown("<div class='section-title'>Exportación a Excel</div>", unsafe_allow_html=True)
    st.write("Descarga el análisis con resumen ejecutivo, métricas por cliente, escenarios de estrés, portafolio optimizado e histórico de ingresos.")
    default_stress = pd.DataFrame([
        stress_scenario(metrics, [metrics.iloc[0]["Cliente"]]) | {"Escenario": "Pérdida cliente principal"},
        stress_scenario(metrics, metrics.head(3)["Cliente"].tolist()) | {"Escenario": "Pérdida Top 3"},
        stress_scenario(metrics, metrics.head(5)["Cliente"].tolist()) | {"Escenario": "Pérdida Top 5"},
    ])
    excel_bytes = export_excel(metrics, ind, default_stress, optimized, pivot)
    st.download_button(
        label="📥 Descargar análisis en Excel",
        data=excel_bytes,
        file_name=f"analisis_riesgo_concentracion_clientes_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("### Plantilla de carga")
    template = pd.DataFrame({
        "Fecha": ["2026-01-31", "2026-01-31", "2026-02-28"],
        "Cliente": ["Cliente A", "Cliente B", "Cliente A"],
        "Ingresos": [1500000, 850000, 1600000],
        "Rentabilidad": [0.32, 0.25, 0.34],
        "Utilidad": [480000, 212500, 544000],
    })
    st.dataframe(template, use_container_width=True, hide_index=True)
