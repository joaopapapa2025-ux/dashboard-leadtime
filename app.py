from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Lead Time | Papapá", page_icon="⏱️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
ALL = "Todos"
META_FATURAMENTO_DIAS = 2
META_NIVEL_SERVICO = 0.96
VENDEDORES_ESPECIAIS = {
    "PEDRO HENRIQUE KRUGER BORN",
    "RODRIGO DOS REIS E SARLO",
    "JOAO PAULO FERREIRA ALVES",
    "ANA CHRISTINA RODRIGUES",
    "THIAGO MARTINS CABRAL",
    "BERNARDO OLIVEIRA DALLEGRAVE",
    "JOAO VITOR TADRA",
}


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def parse_date(series: pd.Series) -> pd.Series:
    value = clean_text(series).replace({"00000000": "", "nan": ""})
    return pd.to_datetime(value, format="%d%m%Y", errors="coerce")


def parse_brl_number(series: pd.Series) -> pd.Series:
    value = clean_text(series).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(value, errors="coerce")


def business_days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=start.index, dtype="Int64")
    valid = start.notna() & end.notna()
    if valid.any():
        result.loc[valid] = np.busday_count(
            start.loc[valid].values.astype("datetime64[D]"),
            end.loc[valid].values.astype("datetime64[D]"),
        )
    return result


def add_business_days(start: pd.Series, days: pd.Series) -> pd.Series:
    result = pd.Series(pd.NaT, index=start.index, dtype="datetime64[ns]")
    valid = start.notna() & days.notna()
    if valid.any():
        offsets = np.ceil(days.loc[valid]).astype(int).to_numpy()
        result.loc[valid] = pd.to_datetime(
            np.busday_offset(start.loc[valid].values.astype("datetime64[D]"), offsets, roll="forward")
        )
    return result


def find_source_file(prefix: str, required: bool = False) -> Path | None:
    files = sorted(
        file for file in BASE_DIR.glob("*.xlsx")
        if file.name.upper().startswith(prefix.upper()) and not file.name.startswith("~$")
    )
    if len(files) > 1:
        st.error(f"Há mais de um arquivo começando com `{prefix}` na raiz do projeto.")
        st.stop()
    if files:
        return files[0]
    if required:
        return None
    return None


@st.cache_data(show_spinner="Calculando a referência de prazo por estado...")
def load_state_lead_time(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["Estado", "Lead time médio por estado (dias úteis)"])

    raw = pd.read_excel(path, sheet_name="tabela de lead time", header=None)
    lead = pd.DataFrame(
        {
            "Estado": clean_text(raw.iloc[3:, 2]),
            "Lead time": pd.to_numeric(raw.iloc[3:, 3], errors="coerce"),
        }
    )
    lead = lead[(lead["Estado"].ne("")) & lead["Lead time"].notna()].copy()
    return (
        lead.groupby("Estado", as_index=False)["Lead time"]
        .mean()
        .rename(columns={"Lead time": "Lead time médio por estado (dias úteis)"})
    )


@st.cache_data(show_spinner="Lendo e conciliando as bases...")
def load_data(
    pedidos_path: str,
    faturamento_path: str,
    inside_sales_path: str | None,
    state_lead_path: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pedidos_raw = pd.read_excel(pedidos_path, dtype=str)
    faturamento_raw = pd.read_excel(faturamento_path, dtype=str)

    pedidos = pd.DataFrame(
        {
            "Pedido": clean_text(pedidos_raw.iloc[:, 2]),
            "Situação pedido": clean_text(pedidos_raw.iloc[:, 4]),
            "Código cliente": clean_text(pedidos_raw.iloc[:, 5]),
            "Cliente": clean_text(pedidos_raw.iloc[:, 6]),
            "Vendedor": clean_text(pedidos_raw.iloc[:, 8]),
            "Vendedor interno": clean_text(pedidos_raw.iloc[:, 21]),  # Coluna V
            "Data pedido": parse_date(pedidos_raw.iloc[:, 9]),
            "Valor pedido": parse_brl_number(pedidos_raw.iloc[:, 13]),
        }
    )
    pedidos = pedidos[
        pedidos["Pedido"].str.endswith("00", na=False)
        & ~pedidos["Situação pedido"].isin(["CAN", "REP"])
    ].copy()

    faturamento_all = pd.DataFrame(
        {
            "Data faturamento": parse_date(faturamento_raw.iloc[:, 0]),
            "Nota fiscal": clean_text(faturamento_raw.iloc[:, 1]),
            "Código cliente NF": clean_text(faturamento_raw.iloc[:, 4]),
            "Cliente NF": clean_text(faturamento_raw.iloc[:, 5]),
            "Data prevista": parse_date(faturamento_raw.iloc[:, 11]),
            "Data entrega": parse_date(faturamento_raw.iloc[:, 12]),
            "Valor nota fiscal": parse_brl_number(faturamento_raw.iloc[:, 21]),
            "Pedido": clean_text(faturamento_raw.iloc[:, 23]),
            "Data solicitada cliente": parse_date(faturamento_raw.iloc[:, 26]),  # AA
            "Regional": clean_text(faturamento_raw.iloc[:, 31]),
            "Grupo": clean_text(faturamento_raw.iloc[:, 33]),
        }
    )

    client_dimension = (
        faturamento_all[faturamento_all["Código cliente NF"].ne("")]
        .sort_values("Data faturamento")
        .groupby("Código cliente NF", as_index=False)
        .last()
        .rename(
            columns={
                "Código cliente NF": "Código cliente",
                "Cliente NF": "Cliente cadastro",
                "Regional": "Regional cadastro",
                "Grupo": "Grupo cadastro",
            }
        )[["Código cliente", "Cliente cadastro", "Regional cadastro", "Grupo cadastro"]]
    )

    faturamento = faturamento_all[faturamento_all["Pedido"].ne("")].copy()
    faturamento_resumo = (
        faturamento.groupby("Pedido", as_index=False)
        .agg(
            **{
                "NFs": ("Nota fiscal", "nunique"),
                "Nota fiscal": ("Nota fiscal", lambda values: ", ".join(sorted(set(values.dropna())))),
                "Data faturamento": ("Data faturamento", "min"),
                "Data prevista": ("Data prevista", "max"),
                "Data entrega": ("Data entrega", "max"),
                "Data solicitada cliente": ("Data solicitada cliente", "max"),
                "Valor nota fiscal": ("Valor nota fiscal", "sum"),
                "Código cliente NF": ("Código cliente NF", "first"),
                "Cliente NF": ("Cliente NF", "first"),
                "Regional": ("Regional", "first"),
                "Grupo": ("Grupo", "first"),
            }
        )
    )

    df = pedidos.merge(faturamento_resumo, how="left", on="Pedido")
    df = df.merge(client_dimension, how="left", on="Código cliente")

    if inside_sales_path:
        inside_sales = pd.read_excel(inside_sales_path, dtype=str)
        states = pd.DataFrame(
            {
                "Código cliente": clean_text(inside_sales["CÓDIGO"]),
                "Estado": clean_text(inside_sales["UF"]),
            }
        ).drop_duplicates("Código cliente")
        df = df.merge(states, how="left", on="Código cliente")
    else:
        df["Estado"] = pd.NA

    has_nf_client = df["Cliente NF"].notna() & df["Cliente NF"].ne("")
    has_history_client = df["Cliente cadastro"].notna() & df["Cliente cadastro"].ne("")
    df["Cliente"] = df["Cliente NF"].where(
        has_nf_client, df["Cliente cadastro"].where(has_history_client, df["Cliente"])
    )
    has_nf_code = df["Código cliente NF"].notna() & df["Código cliente NF"].ne("")
    df["Código cliente"] = df["Código cliente NF"].where(has_nf_code, df["Código cliente"])
    df["Regional"] = df["Regional"].replace("", pd.NA).fillna(df["Regional cadastro"])
    df["Grupo"] = df["Grupo"].replace("", pd.NA).fillna(df["Grupo cadastro"])
    df["Regional"] = df["Regional"].replace("", pd.NA).fillna("Não informado")
    df["Grupo"] = df["Grupo"].replace("", pd.NA).fillna("Não informado")

    # Regra comercial: a equipe indicada deve ser sempre classificada como Especiais.
    # A comparação também considera a coluna I porque parte dos nomes históricos
    # aparece nela, embora a origem principal da regra seja a coluna V.
    team_normalized = {name.casefold() for name in VENDEDORES_ESPECIAIS}
    is_special_team = (
        df["Vendedor interno"].str.casefold().isin(team_normalized)
        | df["Vendedor"].str.casefold().isin(team_normalized)
    )
    df.loc[is_special_team, "Regional"] = "ESPECIAIS - BR"
    df.loc[is_special_team, "Grupo"] = "ESPECIAIS"

    df["Estado"] = df["Estado"].replace("", pd.NA).fillna("Não informado")
    df["NFs"] = df["NFs"].fillna(0).astype(int)
    df["Nota fiscal"] = df["Nota fiscal"].fillna("")
    df["Valor nota fiscal"] = df["Valor nota fiscal"].fillna(0.0)

    state_lead = load_state_lead_time(state_lead_path)
    df = df.merge(state_lead, how="left", left_on="Estado", right_on="Estado")
    df["Previsão por estado"] = add_business_days(
        df["Data faturamento"], df["Lead time médio por estado (dias úteis)"]
    )

    df["Prazo SLA"] = df["Data solicitada cliente"]
    df["Fonte prazo SLA"] = np.where(df["Prazo SLA"].notna(), "Data solicitada pelo cliente", "")
    use_state = df["Prazo SLA"].isna() & df["Previsão por estado"].notna()
    df.loc[use_state, "Prazo SLA"] = df.loc[use_state, "Previsão por estado"]
    df.loc[use_state, "Fonte prazo SLA"] = "Média de lead time por estado"
    use_carrier_forecast = df["Prazo SLA"].isna() & df["Data prevista"].notna()
    df.loc[use_carrier_forecast, "Prazo SLA"] = df.loc[use_carrier_forecast, "Data prevista"]
    df.loc[use_carrier_forecast, "Fonte prazo SLA"] = "Data prevista da transportadora"
    df["Fonte prazo SLA"] = df["Fonte prazo SLA"].replace("", "Sem prazo disponível")

    df["Pedido → faturamento (dias)"] = (df["Data faturamento"] - df["Data pedido"]).dt.days
    df["Pedido → faturamento (dias úteis)"] = business_days_between(df["Data pedido"], df["Data faturamento"])
    df["Faturamento → entrega (dias)"] = (df["Data entrega"] - df["Data faturamento"]).dt.days
    df["Lead time total (dias)"] = (df["Data entrega"] - df["Data pedido"]).dt.days
    df["Atraso entrega (dias)"] = (df["Data entrega"] - df["Prazo SLA"]).dt.days
    df["Atraso atual (dias)"] = (pd.Timestamp(date.today()) - df["Prazo SLA"]).dt.days

    df["Status logística"] = "Aguardando faturamento"
    df.loc[df["Data faturamento"].notna(), "Status logística"] = "Faturado / aguardando entrega"
    df.loc[df["Data entrega"].notna(), "Status logística"] = "Entregue"
    df.loc[df["Data entrega"].notna() & (df["Atraso entrega (dias)"] <= 0), "Status logística"] = "Entregue no prazo"
    df.loc[df["Data entrega"].notna() & (df["Atraso entrega (dias)"] > 0), "Status logística"] = "Entregue em atraso"

    return df, state_lead


def selected_values(selection: list[str]) -> list[str]:
    return [] if not selection or ALL in selection else selection


def filter_by_selection(data: pd.DataFrame, column: str, selection: list[str]) -> pd.DataFrame:
    values = selected_values(selection)
    return data if not values else data[data[column].isin(values)]


def prepare_multiselect(key: str, options: list[str], auto_single: bool = False) -> list[str]:
    current = st.session_state.get(key, [ALL])
    if isinstance(current, str):
        current = [current]
    current = [value for value in current if value == ALL or value in options]
    if ALL in current and len(current) > 1:
        current.remove(ALL)
    if not current:
        current = [ALL]
    if auto_single and len(options) == 1 and current == [ALL]:
        current = options.copy()
    st.session_state[key] = current
    return current


def clear_filters() -> None:
    for key in [
        "period_filter", "regional_filter", "grupo_filter", "vendedor_filter", "status_filter",
        "estado_filter", "client_search", "pedido_search", "nota_search",
    ]:
        st.session_state.pop(key, None)


def to_excel(data: pd.DataFrame, sheet_name: str) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl", datetime_format="DD/MM/YYYY") as writer:
        data.to_excel(writer, index=False, sheet_name=sheet_name[:31])
        worksheet = writer.sheets[sheet_name[:31]]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cells in worksheet.columns:
            width = min(max(len(str(cell.value or "")) for cell in cells) + 2, 35)
            worksheet.column_dimensions[cells[0].column_letter].width = width
    return output.getvalue()


def status_color(value: str) -> str:
    value = str(value).casefold()
    if "dentro" in value or "atingida" in value:
        return "background-color: #d1fae5; color: #065f46"
    if "limite" in value:
        return "background-color: #fef3c7; color: #92400e"
    return "background-color: #fee2e2; color: #991b1b"


def table_with_status(data: pd.DataFrame, column_config: dict | None = None) -> None:
    st.dataframe(
        data.style.map(status_color, subset=["Status"]),
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
    )


def chart_selected_regional(event: object) -> str | None:
    try:
        selection = event.get("selection", {})
        points = selection.get("points", [])
        return points[0].get("x") if points else None
    except AttributeError:
        return None


def show_detail(title: str, data: pd.DataFrame, key: str) -> None:
    if data.empty:
        st.info("Não há pedidos para este detalhamento.")
        return
    st.markdown(f"#### {title}")
    detail_cols = [
        "Pedido", "Nota fiscal", "Cliente", "Código cliente", "Vendedor", "Regional", "Grupo", "Estado",
        "Valor pedido", "Valor nota fiscal", "Data pedido", "Data faturamento", "Data solicitada cliente",
        "Previsão por estado", "Data prevista", "Prazo SLA", "Fonte prazo SLA", "Data entrega",
        "Pedido → faturamento (dias)", "Atraso entrega (dias)", "Atraso atual (dias)", "Status logística",
    ]
    detail = data[[column for column in detail_cols if column in data.columns]].copy()
    st.dataframe(
        detail,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Valor pedido": st.column_config.NumberColumn(format="R$ %.2f"),
            "Valor nota fiscal": st.column_config.NumberColumn(format="R$ %.2f"),
            "Data pedido": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Data faturamento": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Data solicitada cliente": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Previsão por estado": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Data prevista": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Prazo SLA": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Data entrega": st.column_config.DateColumn(format="DD/MM/YYYY"),
        },
    )
    st.download_button(
        "Exportar este detalhamento (Excel)",
        data=to_excel(detail, title),
        file_name=f"{key}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{key}",
    )


def billing_age_bucket(days: float) -> str:
    if days < 2:
        return "Menos de 2 dias"
    if days <= 4:
        return "De 2 a 4 dias"
    return "Mais de 4 dias"


def delivery_age_bucket(days: float) -> str:
    if pd.isna(days):
        return "Sem prazo SLA"
    if days <= 0:
        return "Dentro do prazo"
    if days <= 2:
        return "Atrasado até 2 dias"
    if days <= 5:
        return "Atrasado de 3 a 5 dias"
    if days <= 10:
        return "Atrasado de 6 a 10 dias"
    return "Atrasado há mais de 10 dias"


pedidos_file = find_source_file("SVE611", required=True)
faturamento_file = find_source_file("SVE660", required=True)
inside_sales_file = find_source_file("Base Dashboard Inside Sales")
state_lead_file = find_source_file("Tabela lead time operacao e comercial")

st.title("⏱️ Lead Time da Operação")

if not pedidos_file or not faturamento_file:
    st.error("Envie as bases SVE611 e SVE660 na mesma pasta do arquivo app.py.")
    st.stop()

try:
    base, state_lead = load_data(
        str(pedidos_file),
        str(faturamento_file),
        str(inside_sales_file) if inside_sales_file else None,
        str(state_lead_file) if state_lead_file else None,
    )
except Exception as error:
    st.exception(error)
    st.stop()

if not state_lead_file:
    st.warning("A base 'Tabela lead time operacao e comercial.xlsx' não foi encontrada. A previsão por estado não será calculada.")

st.subheader("Filtros")
min_date = base["Data pedido"].min().date()
max_date = base["Data pedido"].max().date()
regional_options = sorted(base["Regional"].dropna().unique().tolist())
regional_current = prepare_multiselect("regional_filter", regional_options)
regional_scope = filter_by_selection(base, "Regional", regional_current)
group_options = sorted(regional_scope["Grupo"].dropna().unique().tolist())
group_current = prepare_multiselect("grupo_filter", group_options, auto_single=True)
vendor_scope = filter_by_selection(regional_scope, "Grupo", group_current)
vendor_options = sorted(vendor_scope["Vendedor"].dropna().unique().tolist())
vendor_current = prepare_multiselect("vendedor_filter", vendor_options)
status_options = sorted(base["Status logística"].dropna().unique().tolist())
status_current = prepare_multiselect("status_filter", status_options)

filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)
with filter_col1:
    period = st.date_input("Período do pedido", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="period_filter")
with filter_col2:
    regional_filter = st.multiselect("Regional", [ALL, *regional_options], default=regional_current, key="regional_filter")
with filter_col3:
    grupo_filter = st.multiselect("Grupo", [ALL, *group_options], default=group_current, key="grupo_filter")
with filter_col4:
    vendedor_filter = st.multiselect("Vendedor", [ALL, *vendor_options], default=vendor_current, key="vendedor_filter")
with filter_col5:
    status_filter = st.multiselect("Status", [ALL, *status_options], default=status_current, key="status_filter")

search_col1, search_col2, search_col3, search_col4, search_col5 = st.columns(5)
with search_col1:
    client_search = st.text_input("Código do cliente", placeholder="Ex.: C62203", key="client_search")
with search_col2:
    pedido_search = st.text_input("Número do pedido", placeholder="Ex.: 14576400", key="pedido_search")
with search_col3:
    nota_search = st.text_input("Número da nota fiscal", placeholder="Ex.: 0144898", key="nota_search")
with search_col4:
    is_special = any("ESPECIAIS" in regional.upper() for regional in selected_values(regional_filter))
    state_scope = base[base["Regional"].str.contains("ESPECIAIS", case=False, na=False)] if is_special else base.iloc[0:0]
    state_options = sorted(state_scope["Estado"].dropna().unique().tolist())
    if is_special and state_options:
        state_current = prepare_multiselect("estado_filter", state_options)
        estado_filter = st.multiselect("Estado", [ALL, *state_options], default=state_current, key="estado_filter")
    else:
        estado_filter = [ALL]
with search_col5:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("Limpar filtros"):
        clear_filters()
        st.rerun()

filtered = filter_by_selection(base, "Regional", regional_filter)
filtered = filter_by_selection(filtered, "Grupo", grupo_filter)
filtered = filter_by_selection(filtered, "Vendedor", vendedor_filter)
filtered = filter_by_selection(filtered, "Status logística", status_filter)
if selected_values(estado_filter):
    special_rows = filtered["Regional"].str.contains("ESPECIAIS", case=False, na=False)
    filtered = filtered[~special_rows | filtered["Estado"].isin(selected_values(estado_filter))]
if client_search:
    code = filtered["Código cliente"].fillna("").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    search = "".join(char for char in client_search.upper() if char.isalnum())
    filtered = filtered[code.eq(search)]
if pedido_search:
    filtered = filtered[filtered["Pedido"].str.contains(pedido_search.strip(), regex=False, na=False)]
if nota_search:
    filtered = filtered[filtered["Nota fiscal"].str.contains(nota_search.strip(), regex=False, na=False)]
if isinstance(period, tuple) and len(period) == 2:
    start_date, end_date = map(pd.Timestamp, period)
    filtered = filtered[filtered["Data pedido"].between(start_date, end_date)]

if filtered.empty:
    st.warning("Nenhum pedido encontrado para os filtros selecionados.")
    st.stop()

with st.expander("Referência de lead time por estado"):
    st.caption("Média do campo 'Lead time total' da tabela operacional, em dias úteis. Ela é usada quando não há data solicitada pelo cliente.")
    st.dataframe(state_lead, hide_index=True, use_container_width=True, column_config={
        "Lead time médio por estado (dias úteis)": st.column_config.NumberColumn(format="%.2f")
    })

st.markdown("### Visão do painel")
st.caption("Escolha entre resultados concluídos e pedidos que ainda exigem acompanhamento.")
st.markdown(
    """
    <style>
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        width: 100%;
        gap: 16px;
        background: transparent;
        border: 0;
        border-radius: 0;
        padding: 0;
        margin-top: 6px;
        margin-bottom: 24px;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        width: 100%;
        height: 78px;
        min-width: 0;
        justify-content: center;
        padding: 0 28px;
        border: 1px solid #dfe5ea;
        border-radius: 12px;
        background: #ffffff;
        color: #475569;
        font-size: 18px;
        font-weight: 700;
        transition: all .15s ease;
    }
    div[data-testid="stTabs"] [aria-selected="true"] {
        background: #fff4ef;
        color: #9a4f38;
        border: 2px solid #e7a88f;
        box-shadow: 0 4px 12px rgba(154, 79, 56, .12);
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
historical, present = st.tabs([
    "Histórico - pedidos concluídos",
    "Presente - pedidos em andamento",
])

with historical:
    st.header("Histórico")
    st.caption("Metas: faturamento em até 2 dias e nível de serviço de 96%.")

    history_billing = filtered[filtered["Data faturamento"].notna()].copy()
    billing_summary = (
        history_billing.groupby("Regional", as_index=False)
        .agg(
            **{
                "Média de faturamento (dias)": ("Pedido → faturamento (dias)", "mean"),
                "Pedidos faturados": ("Pedido", "size"),
            }
        )
    )
    billing_summary["Meta (dias)"] = META_FATURAMENTO_DIAS
    billing_summary["Diferença para a meta (dias)"] = billing_summary["Média de faturamento (dias)"] - META_FATURAMENTO_DIAS
    billing_summary["Status"] = np.select(
        [billing_summary["Média de faturamento (dias)"] <= META_FATURAMENTO_DIAS,
         billing_summary["Média de faturamento (dias)"] <= META_FATURAMENTO_DIAS + 1],
        ["🟢 Dentro do prazo", "🟡 Próximo ao limite"],
        default="🔴 Prazo acima do esperado",
    )
    billing_summary = billing_summary.sort_values("Média de faturamento (dias)", ascending=False)
    billing_total_mean = history_billing["Pedido → faturamento (dias)"].mean()
    billing_total_status = (
        "🟢 Dentro do prazo" if billing_total_mean <= META_FATURAMENTO_DIAS
        else "🟡 Próximo ao limite" if billing_total_mean <= META_FATURAMENTO_DIAS + 1
        else "🔴 Prazo acima do esperado"
    )
    billing_total = pd.DataFrame([{
        "Regional": "TOTAL",
        "Média de faturamento (dias)": billing_total_mean,
        "Pedidos faturados": len(history_billing),
        "Meta (dias)": META_FATURAMENTO_DIAS,
        "Diferença para a meta (dias)": billing_total_mean - META_FATURAMENTO_DIAS,
        "Status": billing_total_status,
    }])
    billing_summary = pd.concat([billing_summary, billing_total], ignore_index=True)

    st.subheader("Tempo médio de faturamento por regional")
    table_with_status(billing_summary, column_config={
        "Média de faturamento (dias)": st.column_config.NumberColumn(format="%.2f"),
        "Meta (dias)": st.column_config.NumberColumn(format="%.2f"),
        "Diferença para a meta (dias)": st.column_config.NumberColumn(format="%.2f"),
    })
    billing_chart = px.bar(
        billing_summary[billing_summary["Regional"].ne("TOTAL")], x="Regional", y="Média de faturamento (dias)", color="Status",
        color_discrete_map={"🟢 Dentro do prazo": "#16a34a", "🟡 Próximo ao limite": "#f59e0b", "🔴 Prazo acima do esperado": "#dc2626"},
        hover_data=["Pedidos faturados", "Meta (dias)"], title="Clique em uma barra para ver os pedidos",
    )
    billing_chart.add_hline(y=META_FATURAMENTO_DIAS, line_dash="dash", line_color="#dc2626", annotation_text="Meta: 2 dias")
    billing_event = st.plotly_chart(billing_chart, use_container_width=True, key="billing_chart", on_select="rerun", selection_mode="points")
    billing_region = chart_selected_regional(billing_event)
    if billing_region:
        show_detail(f"Pedidos usados no tempo de faturamento — {billing_region}", history_billing[history_billing["Regional"].eq(billing_region)], "historico_faturamento")

    history_service = filtered[filtered["Data entrega"].notna() & filtered["Prazo SLA"].notna()].copy()
    history_service["No prazo"] = history_service["Data entrega"] <= history_service["Prazo SLA"]
    service_summary = (
        history_service.groupby("Regional", as_index=False)
        .agg(Pedidos=("Pedido", "size"), Entregues_no_prazo=("No prazo", "sum"))
    )
    service_summary["Nível de serviço"] = service_summary["Entregues_no_prazo"] / service_summary["Pedidos"]
    service_summary["Meta"] = META_NIVEL_SERVICO
    service_summary["Status"] = np.where(
        service_summary["Nível de serviço"] >= META_NIVEL_SERVICO,
        "🟢 Meta de serviço atingida", "🔴 Meta de serviço não atingida",
    )
    service_summary = service_summary.sort_values("Nível de serviço")
    total_service_orders = int(service_summary["Pedidos"].sum())
    total_on_time = int(service_summary["Entregues_no_prazo"].sum())
    total_service_level = total_on_time / total_service_orders if total_service_orders else 0
    service_total = pd.DataFrame([{
        "Regional": "TOTAL",
        "Pedidos": total_service_orders,
        "Entregues_no_prazo": total_on_time,
        "Nível de serviço": total_service_level,
        "Meta": META_NIVEL_SERVICO,
        "Status": "🟢 Meta de serviço atingida" if total_service_level >= META_NIVEL_SERVICO else "🔴 Meta de serviço não atingida",
    }])
    service_summary = pd.concat([service_summary, service_total], ignore_index=True)
    service_display = service_summary.copy()
    service_display["Nível de serviço (%)"] = service_display["Nível de serviço"] * 100
    service_display["Meta (%)"] = service_display["Meta"] * 100
    service_display = service_display.drop(columns=["Nível de serviço", "Meta"])
    service_display = service_display[[
        "Regional", "Pedidos", "Entregues_no_prazo", "Nível de serviço (%)", "Meta (%)", "Status"
    ]]

    st.subheader("Nível de serviço por regional")
    if history_service.empty:
        st.info("Não há entregas com prazo SLA disponível nos filtros atuais.")
    else:
        table_with_status(service_display, column_config={
            "Nível de serviço (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Meta (%)": st.column_config.NumberColumn(format="%.2f%%"),
        })
        service_chart = px.bar(
            service_summary[service_summary["Regional"].ne("TOTAL")], x="Regional", y="Nível de serviço", color="Status",
            color_discrete_map={"🟢 Meta de serviço atingida": "#16a34a", "🔴 Meta de serviço não atingida": "#dc2626"},
            hover_data=["Pedidos", "Entregues_no_prazo"], title="Clique em uma barra para ver os pedidos",
        )
        service_chart.update_yaxes(tickformat=".0%", range=[0, 1])
        service_chart.add_hline(y=META_NIVEL_SERVICO, line_dash="dash", line_color="#dc2626", annotation_text="Meta: 96%")
        service_event = st.plotly_chart(service_chart, use_container_width=True, key="service_chart", on_select="rerun", selection_mode="points")
        service_region = chart_selected_regional(service_event)
        if service_region:
            show_detail(f"Pedidos usados no nível de serviço — {service_region}", history_service[history_service["Regional"].eq(service_region)], "historico_servico")

with present:
    st.header("Presente")
    st.caption("Acompanhamento de pedidos que aguardam faturamento ou entrega.")

    pending_billing = filtered[filtered["Data faturamento"].isna()].copy()
    pending_billing["Dias aguardando faturamento"] = (pd.Timestamp(date.today()) - pending_billing["Data pedido"]).dt.days
    pending_billing["Faixa faturamento"] = pending_billing["Dias aguardando faturamento"].apply(billing_age_bucket)
    billing_aging = (
        pending_billing.pivot_table(index="Regional", columns="Faixa faturamento", values="Pedido", aggfunc="size", fill_value=0)
        .reindex(columns=["Menos de 2 dias", "De 2 a 4 dias", "Mais de 4 dias"], fill_value=0)
        .reset_index()
    )
    billing_aging_total = {"Regional": "TOTAL", **{
        column: int(billing_aging[column].sum()) for column in billing_aging.columns if column != "Regional"
    }}
    billing_aging = pd.concat([billing_aging, pd.DataFrame([billing_aging_total])], ignore_index=True)

    st.subheader("Pedidos aguardando faturamento")
    st.dataframe(billing_aging, hide_index=True, use_container_width=True)
    billing_aging_long = billing_aging[billing_aging["Regional"].ne("TOTAL")].melt(id_vars="Regional", var_name="Faixa", value_name="Pedidos")
    billing_aging_chart = px.bar(billing_aging_long, x="Regional", y="Pedidos", color="Faixa", barmode="stack", title="Clique em uma barra para ver os pedidos")
    pending_billing_event = st.plotly_chart(billing_aging_chart, use_container_width=True, key="pending_billing_chart", on_select="rerun", selection_mode="points")
    pending_billing_region = chart_selected_regional(pending_billing_event)
    if pending_billing_region:
        show_detail(f"Pedidos aguardando faturamento — {pending_billing_region}", pending_billing[pending_billing["Regional"].eq(pending_billing_region)], "presente_aguardando_faturamento")

    pending_delivery = filtered[filtered["Data faturamento"].notna() & filtered["Data entrega"].isna()].copy()
    pending_delivery["Faixa entrega"] = pending_delivery["Atraso atual (dias)"].apply(delivery_age_bucket)
    delivery_columns = ["Dentro do prazo", "Atrasado até 2 dias", "Atrasado de 3 a 5 dias", "Atrasado de 6 a 10 dias", "Atrasado há mais de 10 dias", "Sem prazo SLA"]
    delivery_aging = (
        pending_delivery.pivot_table(index="Regional", columns="Faixa entrega", values="Pedido", aggfunc="size", fill_value=0)
        .reindex(columns=delivery_columns, fill_value=0)
        .reset_index()
    )
    delivery_aging_total = {"Regional": "TOTAL", **{
        column: int(delivery_aging[column].sum()) for column in delivery_aging.columns if column != "Regional"
    }}
    delivery_aging = pd.concat([delivery_aging, pd.DataFrame([delivery_aging_total])], ignore_index=True)

    st.subheader("Pedidos faturados aguardando entrega")
    st.dataframe(delivery_aging, hide_index=True, use_container_width=True)
    delivery_aging_long = delivery_aging[delivery_aging["Regional"].ne("TOTAL")].melt(id_vars="Regional", var_name="Faixa", value_name="Pedidos")
    delivery_aging_chart = px.bar(delivery_aging_long, x="Regional", y="Pedidos", color="Faixa", barmode="stack", title="Clique em uma barra para ver os pedidos")
    pending_delivery_event = st.plotly_chart(delivery_aging_chart, use_container_width=True, key="pending_delivery_chart", on_select="rerun", selection_mode="points")
    pending_delivery_region = chart_selected_regional(pending_delivery_event)
    if pending_delivery_region:
        show_detail(f"Pedidos aguardando entrega — {pending_delivery_region}", pending_delivery[pending_delivery["Regional"].eq(pending_delivery_region)], "presente_aguardando_entrega")

st.divider()
show_detail("Base completa filtrada", filtered, "base_completa_filtrada")
