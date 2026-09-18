import io
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
N_SYNTHETIC_RECORDS = 1000

ACTIVITIES = [
    "Procesamiento de facturas estándar",
    "Resolución de excepciones",
    "Revisión de facturas sin PO",
    "Revisión de facturas de alto monto",
    "Seguimiento de facturas pendientes",
    "Conciliación / revisión de statements de proveedores",
]

VARIABLES = [
    "Volumen de facturas",
    "Monto promedio",
    "Tiempo de procesamiento",
    "Tasa de excepciones",
    "Cumplimiento de PO",
    "Pago a tiempo",
    "Horas de procesamiento",
]

REQUIRED_COLUMNS = {
    "Fecha",
    "Monto de factura",
    "Tiempo de procesamiento",
    "Indicador de excepción",
    "Indicador de cumplimiento de PO",
    "Indicador de pago a tiempo",
    "Horas de procesamiento",
    "Actividad",
}

BASE_ALLOCATION = np.array([0.42, 0.16, 0.10, 0.08, 0.12, 0.12])


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Optimización AP | Markowitz",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .main-title {font-size: 2.35rem; font-weight: 750; margin-bottom: 0.1rem;}
        .subtitle {font-size: 1.05rem; color: #64748b; margin-bottom: 1.3rem;}
        .section-title {font-size: 1.45rem; font-weight: 700; margin-top: 1.4rem;}
        .info-card {
            padding: 1rem 1.1rem; border-radius: 12px;
            background: #f8fafc; border: 1px solid #e2e8f0;
            margin-bottom: 0.8rem;
        }
        .privacy {
            padding: 0.9rem 1rem; border-radius: 10px;
            background: #fff7ed; border: 1px solid #fed7aa;
            color: #7c2d12; margin: 0.8rem 0 1.2rem 0;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 0.7rem 0.8rem; background: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SYNTHETIC DATA
# ============================================================

@st.cache_data
def generate_synthetic_data(n_records: int = N_SYNTHETIC_RECORDS, seed: int = SEED) -> pd.DataFrame:
    """
    Generate reproducible synthetic AP invoice records.
    No real/company/proprietary information is used.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    months = pd.date_range("2025-01-01", periods=12, freq="MS")

    activity_probs = np.array([0.43, 0.15, 0.10, 0.08, 0.13, 0.11])

    activity_params = {
        ACTIVITIES[0]: dict(time=1.7, exc=0.07, po=0.94, late=0.05, hours=0.20, amount=9500),
        ACTIVITIES[1]: dict(time=4.8, exc=0.32, po=0.76, late=0.16, hours=0.48, amount=7200),
        ACTIVITIES[2]: dict(time=5.4, exc=0.20, po=0.10, late=0.19, hours=0.55, amount=6800),
        ACTIVITIES[3]: dict(time=3.9, exc=0.11, po=0.89, late=0.10, hours=0.43, amount=42000),
        ACTIVITIES[4]: dict(time=3.0, exc=0.14, po=0.86, late=0.13, hours=0.34, amount=12500),
        ACTIVITIES[5]: dict(time=2.7, exc=0.09, po=0.91, late=0.08, hours=0.38, amount=15500),
    }

    chosen_activities = rng.choice(ACTIVITIES, size=n_records, p=activity_probs)
    chosen_dates = rng.choice(dates, size=n_records, replace=True)
    month_index = pd.DatetimeIndex(chosen_dates).month - 1

    # Common monthly pressure creates realistic cross-activity movement.
    monthly_pressure = rng.normal(0, 0.10, size=12)

    rows = []
    for activity, date, mi in zip(chosen_activities, chosen_dates, month_index):
        p = activity_params[activity]
        pressure = monthly_pressure[mi]

        amount = p["amount"] * np.exp(rng.normal(0.0, 0.48))
        amount *= np.exp(pressure * 0.8)

        time = max(
            0.35,
            p["time"] * np.exp(rng.normal(0.0, 0.22) + pressure * 0.55),
        )

        exception_probability = np.clip(p["exc"] + pressure * 0.10 + rng.normal(0, 0.015), 0.01, 0.75)
        exception = int(rng.random() < exception_probability)

        po_probability = np.clip(p["po"] - pressure * 0.06 + rng.normal(0, 0.012), 0.02, 0.99)
        po_compliant = int(rng.random() < po_probability)

        late_probability = np.clip(p["late"] + pressure * 0.08 + rng.normal(0, 0.012), 0.01, 0.65)
        paid_on_time = int(rng.random() >= late_probability)

        hours = max(
            0.03,
            p["hours"] * (0.55 + 0.45 * time / max(p["time"], 0.1))
            * np.exp(rng.normal(0.0, 0.16))
        )

        rows.append(
            {
                "Fecha": pd.Timestamp(date),
                "Mes": pd.Timestamp(date).to_period("M").strftime("%Y-%m"),
                "Actividad": activity,
                "Monto de factura": round(amount, 2),
                "Tiempo de procesamiento": round(time, 2),
                "Indicador de excepción": exception,
                "Indicador de cumplimiento de PO": po_compliant,
                "Indicador de pago a tiempo": paid_on_time,
                "Horas de procesamiento": round(hours, 3),
            }
        )

    return pd.DataFrame(rows).sort_values("Fecha").reset_index(drop=True)


# ============================================================
# DATA VALIDATION / PREPARATION
# ============================================================

def validate_input_data(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return False, "Faltan columnas requeridas: " + ", ".join(sorted(missing))

    if df.empty:
        return False, "El archivo no contiene registros."

    numeric_cols = [
        "Monto de factura",
        "Tiempo de procesamiento",
        "Indicador de excepción",
        "Indicador de cumplimiento de PO",
        "Indicador de pago a tiempo",
        "Horas de procesamiento",
    ]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"La columna '{col}' debe ser numérica."

    return True, ""


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce")
    out = out.dropna(subset=["Fecha"]).copy()
    out["Mes"] = out["Fecha"].dt.to_period("M").astype(str)

    # Binary indicators are clipped to avoid malformed values dominating the model.
    for col in [
        "Indicador de excepción",
        "Indicador de cumplimiento de PO",
        "Indicador de pago a tiempo",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(0, 1)

    for col in [
        "Monto de factura",
        "Tiempo de procesamiento",
        "Horas de procesamiento",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0)

    out["Actividad"] = out["Actividad"].astype(str)

    return out


# ============================================================
# METRICS
# ============================================================

def minmax_by_activity(series: pd.Series, inverse: bool = False) -> pd.Series:
    """Normalize values to 0-1 across activities."""
    low, high = series.min(), series.max()
    if np.isclose(high, low):
        result = pd.Series(1.0, index=series.index)
    elif inverse:
        result = (high - series) / (high - low)
    else:
        result = (series - low) / (high - low)
    return result.clip(0, 1)


def calculate_activity_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        df.groupby("Actividad")
        .agg(
            **{
                "Volumen de facturas": ("Actividad", "size"),
                "Monto promedio": ("Monto de factura", "mean"),
                "Tiempo de procesamiento": ("Tiempo de procesamiento", "mean"),
                "Tasa de excepciones": ("Indicador de excepción", "mean"),
                "Cumplimiento de PO": ("Indicador de cumplimiento de PO", "mean"),
                "Pago a tiempo": ("Indicador de pago a tiempo", "mean"),
                "Horas de procesamiento": ("Horas de procesamiento", "sum"),
            }
        )
        .reindex(ACTIVITIES)
    )

    # Efficiency: lower processing time = higher efficiency.
    grouped["Eficiencia de procesamiento"] = minmax_by_activity(
        grouped["Tiempo de procesamiento"], inverse=True
    )

    # Convert percentages to 0-1 where appropriate.
    grouped["Tasa de excepciones"] = grouped["Tasa de excepciones"].clip(0, 1)
    grouped["Cumplimiento de PO"] = grouped["Cumplimiento de PO"].clip(0, 1)
    grouped["Pago a tiempo"] = grouped["Pago a tiempo"].clip(0, 1)

    # Composite operational return.
    grouped["Rendimiento operativo"] = (
        0.40 * grouped["Pago a tiempo"]
        + 0.30 * grouped["Cumplimiento de PO"]
        + 0.30 * grouped["Eficiencia de procesamiento"]
    )

    # Monthly variability by activity.
    monthly = (
        df.groupby(["Mes", "Actividad"])
        .agg(
            tiempo_std=("Tiempo de procesamiento", "std"),
            horas_total=("Horas de procesamiento", "sum"),
            horas_mean=("Horas de procesamiento", "mean"),
            pago=("Indicador de pago a tiempo", "mean"),
            po=("Indicador de cumplimiento de PO", "mean"),
        )
        .reset_index()
    )

    # Fill sparse-month standard deviations safely.
    monthly["tiempo_std"] = monthly["tiempo_std"].fillna(0)

    variability = (
        monthly.groupby("Actividad")
        .agg(
            variabilidad_tiempo=("tiempo_std", "mean"),
            variabilidad_horas=("horas_mean", "std"),
        )
        .reindex(ACTIVITIES)
        .fillna(0)
    )

    # Normalize risk components across activities.
    risk_components = pd.DataFrame(index=ACTIVITIES)
    risk_components["Tasa de excepciones"] = grouped["Tasa de excepciones"]
    risk_components["Variabilidad del tiempo"] = minmax_by_activity(
        variability["variabilidad_tiempo"]
    )
    risk_components["Variabilidad de horas"] = minmax_by_activity(
        variability["variabilidad_horas"]
    )

    grouped["Riesgo operativo"] = (
        0.40 * risk_components["Tasa de excepciones"]
        + 0.30 * risk_components["Variabilidad del tiempo"]
        + 0.30 * risk_components["Variabilidad de horas"]
    ).clip(0, 1)

    grouped["Eficiencia ajustada por riesgo"] = (
        grouped["Rendimiento operativo"] / grouped["Riesgo operativo"].clip(lower=0.05)
    )

    return grouped, monthly


def build_monthly_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a monthly activity performance matrix used only to estimate
    correlations between activities. The seven dashboard variables remain
    descriptive inputs rather than direct terms in the Markowitz formula.
    """
    monthly = (
        df.groupby(["Mes", "Actividad"])
        .agg(
            pago=("Indicador de pago a tiempo", "mean"),
            po=("Indicador de cumplimiento de PO", "mean"),
            tiempo=("Tiempo de procesamiento", "mean"),
            volumen=("Actividad", "size"),
            horas=("Horas de procesamiento", "sum"),
        )
        .reset_index()
    )

    monthly["tiempo_eff_raw"] = -monthly["tiempo"]

    # Convert monthly processing time into a higher-is-better efficiency score
    # before building a composite operational performance series.
    monthly["tiempo_eff"] = monthly.groupby("Actividad")["tiempo_eff_raw"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if not np.isclose(s.max(), s.min()) else 1.0
    )

    monthly["performance"] = (
        0.40 * monthly["pago"]
        + 0.30 * monthly["po"]
        + 0.30 * monthly["tiempo_eff"]
    )

    pivot = monthly.pivot(index="Mes", columns="Actividad", values="performance")
    pivot = pivot.reindex(columns=ACTIVITIES)

    # Ensure complete numeric matrix.
    pivot = pivot.interpolate(limit_direction="both").ffill().bfill()

    return pivot


# ============================================================
# MARKOWITZ ADAPTATION
# ============================================================

def covariance_matrix(activity_metrics: pd.DataFrame, monthly_performance: pd.DataFrame) -> np.ndarray:
    """
    Adapt financial covariance to AP:
    individual standard deviations are the activity risk indices,
    while correlations come from co-movement of monthly operational
    performance. The resulting matrix is PSD by construction.
    """
    corr = monthly_performance.corr().reindex(index=ACTIVITIES, columns=ACTIVITIES).fillna(0.0)

    # Use an explicit writable NumPy copy. This is compatible with newer
    # NumPy/Pandas combinations where DataFrame-backed arrays may be read-only.
    corr_values = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(corr_values, 1.0)

    # Project small numerical asymmetries to a positive-semidefinite matrix.
    eigvals, eigvecs = np.linalg.eigh(corr_values)
    eigvals = np.clip(eigvals, 0.001, None)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)

    sigma = activity_metrics["Riesgo operativo"].to_numpy(dtype=float)
    sigma = np.clip(sigma, 0.03, 1.0)

    cov = np.outer(sigma, sigma) * corr_psd
    cov = (cov + cov.T) / 2
    return cov


def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    return float(weights @ expected_returns)


def portfolio_risk(weights: np.ndarray, covariance: np.ndarray) -> float:
    variance = float(weights @ covariance @ weights)
    return float(np.sqrt(max(variance, 0.0)))


def simulate_portfolios(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    n_portfolios: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Dirichlet gives non-negative weights summing exactly to 1.
    weights = rng.dirichlet(np.ones(len(ACTIVITIES)), size=n_portfolios)

    returns = weights @ expected_returns
    risks = np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance, weights))
    efficiency = returns / np.clip(risks, 1e-8, None)

    result = pd.DataFrame(weights, columns=ACTIVITIES)
    result["Rendimiento"] = returns
    result["Riesgo"] = risks
    result["Eficiencia ajustada por riesgo"] = efficiency
    return result


def optimize_min_variance(covariance: np.ndarray) -> np.ndarray:
    n = len(ACTIVITIES)
    x0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n

    result = minimize(
        lambda w: portfolio_risk(w, covariance),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    return result.x if result.success else x0


def optimize_max_efficiency(expected_returns: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    n = len(ACTIVITIES)
    x0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n

    def objective(w):
        r = portfolio_return(w, expected_returns)
        risk = portfolio_risk(w, covariance)
        return -(r / max(risk, 1e-8))

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    return result.x if result.success else x0


def efficient_frontier(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    min_return: float,
    max_return: float,
    points: int = 60,
) -> pd.DataFrame:
    """
    Approximate the efficient frontier by solving minimum-risk portfolios
    at a sequence of target returns.
    """
    targets = np.linspace(min_return, max_return, points)
    rows = []
    n = len(ACTIVITIES)
    bounds = [(0, 1)] * n

    for target in targets:
        x0 = np.ones(n) / n
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, expected_returns) - t},
        ]

        result = minimize(
            lambda w: portfolio_risk(w, covariance),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if result.success:
            rows.append(
                {
                    "Rendimiento": portfolio_return(result.x, expected_returns),
                    "Riesgo": portfolio_risk(result.x, covariance),
                }
            )

    frontier = pd.DataFrame(rows).sort_values("Rendimiento").drop_duplicates("Rendimiento")
    if frontier.empty:
        return frontier

    # Only the upper efficient branch: as return rises, retain the minimum
    # risk observed up to each point.
    frontier["min_risk_so_far"] = frontier["Riesgo"].cummin()
    frontier = frontier[frontier["Riesgo"] <= frontier["min_risk_so_far"] + 1e-8].copy()
    return frontier


def portfolio_record(name: str, weights: np.ndarray, expected_returns: np.ndarray, covariance: np.ndarray) -> Dict:
    r = portfolio_return(weights, expected_returns)
    risk = portfolio_risk(weights, covariance)
    return {
        "Portafolio": name,
        "Rendimiento": r,
        "Riesgo": risk,
        "Eficiencia ajustada por riesgo": r / max(risk, 1e-8),
        "weights": weights,
    }


# ============================================================
# INTERPRETATION
# ============================================================

def correlation_interpretation(x: str, y: str, corr: float) -> str:
    strength = abs(corr)
    if strength < 0.20:
        level = "muy débil"
    elif strength < 0.40:
        level = "débil"
    elif strength < 0.60:
        level = "moderada"
    elif strength < 0.80:
        level = "fuerte"
    else:
        level = "muy fuerte"

    direction = "positiva" if corr > 0 else "negativa" if corr < 0 else "prácticamente nula"

    if abs(corr) < 0.20:
        return (
            f"La relación entre **{x}** y **{y}** es {direction} y {level} "
            f"(r = {corr:.2f}). En estos datos sintéticos no se observa una asociación lineal importante. "
            "Correlación no implica causalidad."
        )

    return (
        f"Existe una relación lineal {direction} {level} entre **{x}** y **{y}** "
        f"(r = {corr:.2f}). Esto significa que, en estos datos sintéticos, ambas variables "
        f"tienden a moverse en esa dirección. No implica que una variable cause a la otra."
    )


def allocation_table(records: list) -> pd.DataFrame:
    data = {}
    for rec in records:
        data[rec["Portafolio"]] = rec["weights"] * 100
    table = pd.DataFrame(data, index=ACTIVITIES)
    table.index.name = "Actividad"
    return table.reset_index()


# ============================================================
# VISUALS
# ============================================================

def plot_frontier(simulated: pd.DataFrame, frontier: pd.DataFrame, records: list):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.scatter(
        simulated["Riesgo"],
        simulated["Rendimiento"],
        s=7,
        alpha=0.18,
        label="Portafolios simulados",
    )

    if not frontier.empty:
        ax.plot(
            frontier["Riesgo"],
            frontier["Rendimiento"],
            linewidth=2.6,
            label="Frontera eficiente",
        )

    markers = ["o", "D", "X"]
    for rec, marker in zip(records, markers):
        ax.scatter(
            rec["Riesgo"],
            rec["Rendimiento"],
            s=105,
            marker=marker,
            edgecolor="black",
            linewidth=0.7,
            label=rec["Portafolio"],
            zorder=5,
        )

    ax.set_xlabel("Riesgo operativo")
    ax.set_ylabel("Rendimiento operativo esperado")
    ax.set_title("Frontera eficiente — capacidad operativa de AP")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_allocations(table: pd.DataFrame):
    long = table.melt(id_vars="Actividad", var_name="Portafolio", value_name="Asignación")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=long,
        x="Actividad",
        y="Asignación",
        hue="Portafolio",
        ax=ax,
        errorbar=None,
    )
    ax.set_ylabel("Asignación de capacidad (%)")
    ax.set_xlabel("")
    ax.set_title("Comparación de asignación de capacidad")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="Actividad",
        alpha=0.55,
        s=45,
        ax=ax,
    )
    sns.regplot(
        data=df,
        x=x,
        y=y,
        scatter=False,
        ax=ax,
        ci=None,
        line_kws={"linewidth": 1.7},
    )
    ax.set_title(f"{x} vs. {y}")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


# ============================================================
# APP
# ============================================================

st.markdown('<div class="main-title">Optimización AP</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Aplicación de la lógica de Markowitz a Accounts Payable</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="privacy">
    <b>Privacidad:</b> esta aplicación utiliza datos sintéticos/simulados por defecto.
    No requiere ni utiliza información real, confidencial o propietaria de ninguna empresa.
    Si cargas un CSV, evita incluir información confidencial.
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("Parámetros del modelo")
n_portfolios = st.sidebar.slider(
    "Número de portafolios a simular",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000,
)
seed = st.sidebar.number_input("Semilla de reproducibilidad", min_value=1, max_value=999999, value=SEED)
show_activity_table = st.sidebar.checkbox("Mostrar tabla detallada de actividades", value=True)

st.sidebar.divider()
st.sidebar.subheader("Datos")
uploaded = st.sidebar.file_uploader(
    "Opcional: cargar CSV",
    type=["csv"],
    help="El CSV debe contener las columnas indicadas en README.md.",
)

if uploaded is None:
    df = generate_synthetic_data(seed=int(seed))
    data_source = "Datos sintéticos generados por la aplicación"
else:
    try:
        df_uploaded = pd.read_csv(uploaded)
        valid, message = validate_input_data(df_uploaded)
        if not valid:
            st.error(message)
            st.stop()
        df = prepare_data(df_uploaded)
        data_source = "CSV cargado por el usuario"
    except Exception as exc:
        st.error(f"No fue posible leer el CSV: {exc}")
        st.stop()

if not set(ACTIVITIES).issubset(set(df["Actividad"].unique())):
    st.warning(
        "El archivo cargado no contiene las seis actividades requeridas. "
        "La simulación de Markowitz necesita exactamente estas seis actividades."
    )
    st.stop()

df = df[df["Actividad"].isin(ACTIVITIES)].copy()

# Metrics
activity_metrics, monthly_detail = calculate_activity_metrics(df)
monthly_performance = build_monthly_performance(df)

expected_returns = activity_metrics["Rendimiento operativo"].to_numpy(dtype=float)
covariance = covariance_matrix(activity_metrics, monthly_performance)

simulated = simulate_portfolios(
    expected_returns,
    covariance,
    n_portfolios=int(n_portfolios),
    seed=int(seed),
)

min_var_weights = optimize_min_variance(covariance)
max_eff_weights = optimize_max_efficiency(expected_returns, covariance)

base_rec = portfolio_record("Portafolio base", BASE_ALLOCATION, expected_returns, covariance)
min_var_rec = portfolio_record("Mínima varianza", min_var_weights, expected_returns, covariance)
max_eff_rec = portfolio_record(
    "Eficiencia ajustada por riesgo",
    max_eff_weights,
    expected_returns,
    covariance,
)

records = [base_rec, min_var_rec, max_eff_rec]

# Header / analogy
st.markdown('<div class="section-title">¿Qué estamos optimizando?</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="info-card">
    La lógica de Markowitz se adapta aquí a <b>capacidad operativa</b>, no a dinero invertido.
    Las seis actividades de AP funcionan como los “activos” del portafolio; la capacidad mensual
    del equipo es el recurso limitado; el rendimiento es eficiencia operativa y el riesgo es
    variabilidad operacional. El modelo busca combinaciones de capacidad que permitan observar
    el intercambio entre eficiencia y predictibilidad.
    </div>
    """,
    unsafe_allow_html=True,
)

# KPIs
st.markdown('<div class="section-title">Resumen del escenario</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Facturas", f"{len(df):,}")
c2.metric("Actividades", f"{len(ACTIVITIES)}")
c3.metric("Rendimiento promedio", f"{activity_metrics['Rendimiento operativo'].mean():.1%}")
c4.metric("Riesgo promedio", f"{activity_metrics['Riesgo operativo'].mean():.1%}")
c5.metric("Horas procesadas", f"{df['Horas de procesamiento'].sum():,.0f}")

st.caption(f"Fuente del escenario: {data_source}.")

# Activity analysis
st.markdown('<div class="section-title">1. Análisis de las actividades de AP</div>', unsafe_allow_html=True)
st.write(
    "Las siete variables principales describen carga, magnitud económica, eficiencia, calidad, "
    "cumplimiento y consumo de capacidad. No todas entran directamente en la fórmula de Markowitz."
)

display_cols = [
    "Volumen de facturas",
    "Monto promedio",
    "Tiempo de procesamiento",
    "Tasa de excepciones",
    "Cumplimiento de PO",
    "Pago a tiempo",
    "Horas de procesamiento",
    "Eficiencia de procesamiento",
    "Rendimiento operativo",
    "Riesgo operativo",
]
display = activity_metrics[display_cols].copy()
for col in ["Tasa de excepciones", "Cumplimiento de PO", "Pago a tiempo", "Eficiencia de procesamiento", "Rendimiento operativo", "Riesgo operativo"]:
    display[col] = display[col].map(lambda x: f"{x:.1%}")
display["Monto promedio"] = display["Monto promedio"].map(lambda x: f"${x:,.0f}")
display["Tiempo de procesamiento"] = display["Tiempo de procesamiento"].map(lambda x: f"{x:.2f}")
display["Horas de procesamiento"] = display["Horas de procesamiento"].map(lambda x: f"{x:,.1f}")

if show_activity_table:
    st.dataframe(display, use_container_width=True)

with st.expander("¿Cómo se calcula el rendimiento y el riesgo?"):
    st.markdown(
        """
        **Rendimiento operativo**

        - 40% Pago a tiempo
        - 30% Cumplimiento de PO
        - 30% Eficiencia de procesamiento

        Para el tiempo de procesamiento se invierte la dirección: un menor tiempo produce
        una mayor eficiencia. Después se normaliza en una escala de 0 a 1 para hacerla comparable.

        **Riesgo operativo**

        - 40% Tasa de excepciones
        - 30% Variabilidad del tiempo de procesamiento
        - 30% Variabilidad de horas de procesamiento

        Los componentes se normalizan antes de combinarlos. Por eso un riesgo alto significa que
        la actividad presenta más fricción o un comportamiento menos predecible y puede requerir
        capacidad variable.

        **Importante:** estas métricas son una adaptación académica para AP. No representan
        volatilidad financiera ni una recomendación de inversión.
        """
    )

# Correlation analysis
st.markdown('<div class="section-title">2. Análisis de correlaciones</div>', unsafe_allow_html=True)

corr_source = (
    df.groupby("Mes")
    .agg(
        **{
            "Volumen de facturas": ("Actividad", "size"),
            "Monto promedio": ("Monto de factura", "mean"),
            "Tiempo de procesamiento": ("Tiempo de procesamiento", "mean"),
            "Tasa de excepciones": ("Indicador de excepción", "mean"),
            "Cumplimiento de PO": ("Indicador de cumplimiento de PO", "mean"),
            "Pago a tiempo": ("Indicador de pago a tiempo", "mean"),
            "Horas de procesamiento": ("Horas de procesamiento", "sum"),
        }
    )
)

corr = corr_source[VARIABLES].corr(method="pearson")

col_heat, col_text = st.columns([1.25, 0.75])
with col_heat:
    fig, ax = plt.subplots(figsize=(9, 6.2))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Matriz de correlación de Pearson")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_text:
    x_var = st.selectbox("Variable X", VARIABLES, index=0)
    y_default = 6 if len(VARIABLES) > 6 else 1
    y_var = st.selectbox("Variable Y", VARIABLES, index=y_default)

    pair_corr = float(corr.loc[x_var, y_var])
    st.metric("Coeficiente de Pearson", f"{pair_corr:.2f}")

    st.info(correlation_interpretation(x_var, y_var, pair_corr))

fig = plot_scatter(
    corr_source.assign(Mes=corr_source.index.astype(str)),
    x_var,
    y_var,
)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.caption(
    "La correlación resume asociación lineal en los datos observados/sintéticos. "
    "No demuestra causalidad."
)

# Markowitz simulation
st.markdown('<div class="section-title">3. Simulación de portafolios operativos</div>', unsafe_allow_html=True)
st.write(
    f"Se generaron **{len(simulated):,} combinaciones** de asignación. Cada combinación tiene "
    "pesos no negativos que suman 100%."
)

sim_table = pd.DataFrame(
    {
        "Portafolio": ["Base", "Mínima varianza", "Eficiencia ajustada por riesgo"],
        "Rendimiento": [r["Rendimiento"] for r in records],
        "Riesgo": [r["Riesgo"] for r in records],
        "Índice de eficiencia ajustada": [
            r["Eficiencia ajustada por riesgo"] for r in records
        ],
    }
)
sim_table["Rendimiento"] = sim_table["Rendimiento"].map(lambda x: f"{x:.1%}")
sim_table["Riesgo"] = sim_table["Riesgo"].map(lambda x: f"{x:.1%}")
sim_table["Índice de eficiencia ajustada"] = sim_table["Índice de eficiencia ajustada"].map(lambda x: f"{x:.2f}")
st.dataframe(sim_table, use_container_width=True, hide_index=True)

# Frontier
frontier = efficient_frontier(
    expected_returns,
    covariance,
    min_return=float(expected_returns.min()),
    max_return=float(expected_returns.max()),
    points=55,
)

fig = plot_frontier(simulated, frontier, records)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown(
    """
    **¿Qué significa la frontera eficiente aquí?**

    Es el conjunto de combinaciones que, para un determinado nivel de riesgo operativo,
    buscan ofrecer el mayor rendimiento operativo observado por el modelo. Moverse hacia
    arriba representa mayor rendimiento; moverse hacia la izquierda representa menor riesgo.
    No significa que exista una única asignación “correcta”: muestra distintos intercambios
    posibles entre eficiencia y variabilidad.
    """
)

# Allocation comparison
st.markdown('<div class="section-title">4. Comparación de asignaciones</div>', unsafe_allow_html=True)
table = allocation_table(
    [
        {"Portafolio": "Asignación Base", "weights": BASE_ALLOCATION},
        {"Portafolio": "Mínima Varianza", "weights": min_var_weights},
        {"Portafolio": "Eficiencia Ajustada por Riesgo", "weights": max_eff_weights},
    ]
)

# Friendly names in table
table.columns = [
    "Actividad",
    "Asignación Base",
    "Mínima Varianza",
    "Eficiencia Ajustada por Riesgo",
]

for col in table.columns[1:]:
    table[col] = table[col].map(lambda x: f"{x:.1f}%")

st.dataframe(table, use_container_width=True, hide_index=True)

fig = plot_allocations(
    allocation_table(
        [
            {"Portafolio": "Asignación Base", "weights": BASE_ALLOCATION},
            {"Portafolio": "Mínima Varianza", "weights": min_var_weights},
            {"Portafolio": "Eficiencia Ajustada por Riesgo", "weights": max_eff_weights},
        ]
    )
)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Natural-language interpretation
st.markdown('<div class="section-title">5. Interpretación automática</div>', unsafe_allow_html=True)

max_return_activity = activity_metrics["Rendimiento operativo"].idxmax()
min_risk_activity = activity_metrics["Riesgo operativo"].idxmin()
max_alloc_activity = ACTIVITIES[int(np.argmax(max_eff_weights))]
max_alloc_value = float(np.max(max_eff_weights))

st.info(
    f"El modelo asigna la mayor proporción del escenario de **eficiencia ajustada por riesgo** "
    f"a **{max_alloc_activity}** ({max_alloc_value:.1%}). En este conjunto sintético, "
    f"la actividad con mayor rendimiento operativo individual es **{max_return_activity}**, "
    f"mientras que la de menor riesgo operativo individual es **{min_risk_activity}**. "
    "Esto refleja las métricas y correlaciones generadas para este escenario; no implica causalidad "
    "ni constituye una recomendación financiera."
)

with st.expander("Limitaciones y lectura responsable"):
    st.markdown(
        """
        - Los datos son sintéticos y sirven para demostrar la metodología, no para medir el desempeño real de un equipo.
        - Los pesos dependen de los supuestos, de la calidad de los datos y de la ventana temporal.
        - La matriz de covarianza representa co-movimiento operacional; no es una matriz de riesgo financiero.
        - La frontera eficiente es una herramienta de análisis de trade-offs, no una instrucción operativa automática.
        - Un modelo de optimización no sustituye conocimiento del proceso, capacidad mínima por actividad,
          prioridades de negocio, SLA, controles, segregación de funciones ni restricciones regulatorias.
        - La correlación no demuestra causalidad.
        """
    )

st.markdown("---")
st.caption(
    "Modelo académico de optimización de capacidad para Accounts Payable · Datos sintéticos · "
    "Reproducible con semilla fija"
)
