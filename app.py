from pathlib import Path
from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Lead Time | Papapa", page_icon="⏱️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent


def clean_text(series: pd.Series) -> pd.Series:
    """Padroniza textos sem remover zeros à esquerda de códigos."""
    return series.fillna("").astype(str).str.strip()


def parse_date(series: pd.Series) -> pd.Series:
    """Lê as datas no padrão ddmmyyyy e ignora 00000000."""
    value = clean_text(series).replace({"00000000": "", "nan": ""})
    return pd.to_datetime(value, format="%d%m%Y", errors="coerce")


def parse_brl_number(series: pd.Series) -> pd.Series:
    """Converte valores como 11.584,44 para números utilizáveis no dashboard."""
    value = clean_text(series).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(value, errors="coerce")


def business_days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Calcula dias úteis de segunda a sexta, sem considerar feriados."""
    result = pd.Series(pd.NA, index=start.index, dtype="Int64")
    valid = start.notna() & end.notna()
    if valid.any():
        result.loc[valid] = np.busday_count(
            start.loc[valid].values.astype("datetime64[D]"),
            end.loc[valid].values.astype("datetime64[D]"),
        )
    return result


def to_excel(data: pd.DataFrame) -> bytes:
    """Gera o arquivo Excel a partir da visão filtrada."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl", datetime_format="DD/MM/YYYY") as writer:
        data.to_excel(writer, index=False, sheet_name="Pedidos filtrados")
        worksheet = writer.sheets["Pedidos filtrados"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cells in worksheet.columns:
            width = min(max(len(str(cell.value or "")) for cell in cells) + 2, 35)
            worksheet.column_dimensions[cells[0].column_letter].width = width
    return output.getvalue()


def find_source_file(prefix: str, required: bool = True) -> Path | None:
    """Localiza a única planilha cujo nome começa com o prefixo informado."""
    files = sorted(
        path
        for path in BASE_DIR.glob("*.xlsx")
        if path.name.upper().startswith(prefix.upper()) and not path.name.startswith("~$")
    )
    if len(files) == 1:
        return files[0]
    if len(files) > 1:
        st.error(f"Há mais de um arquivo começando com `{prefix}` na raiz do projeto.")
        st.info("Mantenha apenas a versão mais recente de cada base antes de fazer o commit.")
        st.stop()
    if required:
        return None
    return None


@st.cache_data(show_spinner="Lendo e conciliando as bases...")
def load_data(pedidos_path: str, faturamento_path: str, inside_sales_path: str | None) -> pd.DataFrame:
    pedidos_raw = pd.read_excel(pedidos_path, dtype=str)
    faturamento_raw = pd.read_excel(faturamento_path, dtype=str)

    # Seleção por posição deixa a leitura estável mesmo se o Excel exibir acentos incorretos.
    pedidos = pd.DataFrame(
        {
            "Pedido": clean_text(pedidos_raw.iloc[:, 2]),
            "Status pedido": clean_text(pedidos_raw.iloc[:, 4]),
            "Código cliente": clean_text(pedidos_raw.iloc[:, 5]),
            "Cliente": clean_text(pedidos_raw.iloc[:, 6]),
            "Data pedido": parse_date(pedidos_raw.iloc[:, 9]),
            "Vendedor": clean_text(pedidos_raw.iloc[:, 8]),
            "Valor pedido": parse_brl_number(pedidos_raw.iloc[:, 13]),
        }
    )
    # Somente finais 00 são pedidos. Finais 50 são orçamentos; CAN e REP
    # representam pedidos cancelados e reprovados, que não entram na operação.
    pedidos = pedidos[
        pedidos["Pedido"].str.endswith("00", na=False)
        & ~pedidos["Status pedido"].isin(["CAN", "REP"])
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
            "Regional": clean_text(faturamento_raw.iloc[:, 31]),  # AF: Desc.Região
            "Grupo": clean_text(faturamento_raw.iloc[:, 33]),  # AH: Desc.Grupo Cliente
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

    # Um pedido pode ter mais de uma NF. O dashboard consolida em uma linha por pedido:
    # primeiro faturamento e última previsão/entrega, evitando duplicar pedidos nos indicadores.
    faturamento_resumo = (
        faturamento.groupby("Pedido", as_index=False)
        .agg(
            **{
                "NFs": ("Nota fiscal", "nunique"),
                "Nota fiscal": ("Nota fiscal", lambda values: ", ".join(sorted(set(values.dropna())))),
                "Data faturamento": ("Data faturamento", "min"),
                "Data prevista": ("Data prevista", "max"),
                "Data entrega": ("Data entrega", "max"),
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
        state_dimension = pd.DataFrame(
            {
                "Código cliente": clean_text(inside_sales["CÓDIGO"]),
                "Estado": clean_text(inside_sales["UF"]),
            }
        )
        state_dimension = state_dimension[state_dimension["Código cliente"].ne("")]
        state_dimension = state_dimension.drop_duplicates("Código cliente")
        df = df.merge(state_dimension, how="left", on="Código cliente")
    else:
        df["Estado"] = pd.NA

    # Para pedidos não faturados, regional e grupo vêm do último cadastro do
    # cliente visto na SVE660. Os dados do pedido nunca são apagados por NaN.
    has_nf_client = df["Cliente NF"].notna() & df["Cliente NF"].ne("")
    has_client_history = df["Cliente cadastro"].notna() & df["Cliente cadastro"].ne("")
    df["Cliente"] = df["Cliente NF"].where(
        has_nf_client, df["Cliente cadastro"].where(has_client_history, df["Cliente"])
    )
    has_nf_code = df["Código cliente NF"].notna() & df["Código cliente NF"].ne("")
    df["Código cliente"] = df["Código cliente NF"].where(has_nf_code, df["Código cliente"])
    df["Regional"] = (
        df["Regional"].replace("", pd.NA).fillna(df["Regional cadastro"])
        .replace("", pd.NA).fillna("Sem regional")
    )
    df["Grupo"] = (
        df["Grupo"].replace("", pd.NA).fillna(df["Grupo cadastro"])
        .replace("", pd.NA).fillna("Sem grupo")
    )
    df["Estado"] = df["Estado"].replace("", pd.NA).fillna("Não informado")
    df["NFs"] = df["NFs"].fillna(0).astype(int)
    df["Nota fiscal"] = df["Nota fiscal"].fillna("")
    df["Valor nota fiscal"] = df["Valor nota fiscal"].fillna(0.0)

    df["Pedido → faturamento (dias)"] = (
        df["Data faturamento"] - df["Data pedido"]
    ).dt.days
    df["Faturamento → previsão (dias)"] = (
        df["Data prevista"] - df["Data faturamento"]
    ).dt.days
    df["Faturamento → entrega (dias)"] = (
        df["Data entrega"] - df["Data faturamento"]
    ).dt.days
    df["Lead time total (dias)"] = (
        df["Data entrega"] - df["Data pedido"]
    ).dt.days
    df["Pedido → faturamento (dias úteis)"] = business_days_between(
        df["Data pedido"], df["Data faturamento"]
    )
    df["Faturamento → previsão (dias úteis)"] = business_days_between(
        df["Data faturamento"], df["Data prevista"]
    )
    df["Faturamento → entrega (dias úteis)"] = business_days_between(
        df["Data faturamento"], df["Data entrega"]
    )
    df["Lead time total (dias úteis)"] = business_days_between(
        df["Data pedido"], df["Data entrega"]
    )

    today = pd.Timestamp(date.today())
    df["Status logística"] = "Aguardando faturamento"
    df.loc[df["Data faturamento"].notna(), "Status logística"] = "Faturado / aguardando entrega"
    df.loc[df["Data entrega"].notna(), "Status logística"] = "Entregue"
    df.loc[
        df["Data entrega"].notna() & (df["Data entrega"] <= df["Data prevista"]),
        "Status logística",
    ] = "Entregue no prazo"
    df.loc[
        df["Data entrega"].notna() & (df["Data entrega"] > df["Data prevista"]),
        "Status logística",
    ] = "Entregue em atraso"
    df.loc[
        df["Data entrega"].isna()
        & df["Data prevista"].notna()
        & (df["Data prevista"] < today),
        "Status logística",
    ] = "Entrega atrasada"

    return df


def format_days(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1f} dias"


st.title("⏱️ Lead Time da Operação")

pedidos_file = find_source_file("SVE611")
faturamento_file = find_source_file("SVE660")
inside_sales_file = find_source_file("Base Dashboard Inside Sales", required=False)

if not pedidos_file or not faturamento_file:
    st.error("Arquivos de dados não encontrados.")
    st.markdown(
        "Envie uma planilha de cada tipo na mesma pasta do arquivo `app.py`:"
    )
    st.code("SVE611V (71).xlsx\nSVE660V.xlsx", language="text")
    st.info("Os nomes só precisam começar com SVE611 e SVE660. Substitua os dois arquivos diariamente no GitHub e mantenha uma versão de cada base.")
    st.stop()

try:
    base = load_data(
        str(pedidos_file),
        str(faturamento_file),
        str(inside_sales_file) if inside_sales_file else None,
    )
except Exception as error:
    st.exception(error)
    st.stop()

st.subheader("Filtros")
min_date = base["Data pedido"].min().date()
max_date = base["Data pedido"].max().date()


def groups_for_regional(regional: str) -> list[str]:
    """Retorna apenas os grupos que existem na regional selecionada."""
    scope = base if regional == "Todos" else base[base["Regional"].eq(regional)]
    return sorted(scope["Grupo"].dropna().unique().tolist())


def sync_group_with_regional() -> None:
    """Reseta ou preenche Grupo quando a Regional muda."""
    options = groups_for_regional(st.session_state.get("regional_filter", "Todos"))
    st.session_state["grupo_filter"] = options[0] if len(options) == 1 else "Todos"
    st.session_state["estado_filter"] = "Todos"


def clear_filters() -> None:
    for key in [
        "period_filter", "regional_filter", "grupo_filter", "status_filter", "estado_filter",
        "client_search", "pedido_search", "nota_search",
    ]:
        st.session_state.pop(key, None)


filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
with filter_col1:
    period = st.date_input(
        "Período do pedido", value=(min_date, max_date), min_value=min_date,
        max_value=max_date, key="period_filter"
    )
with filter_col2:
    regional_filter = st.selectbox(
        "Regional",
        ["Todos", *sorted(base["Regional"].unique())],
        key="regional_filter",
        on_change=sync_group_with_regional,
    )
with filter_col3:
    group_options = groups_for_regional(regional_filter)
    if len(group_options) == 1 and st.session_state.get("grupo_filter") != group_options[0]:
        st.session_state["grupo_filter"] = group_options[0]
    elif st.session_state.get("grupo_filter", "Todos") not in ["Todos", *group_options]:
        st.session_state["grupo_filter"] = "Todos"
    grupo_filter = st.selectbox("Grupo", ["Todos", *group_options], key="grupo_filter")
with filter_col4:
    status_filter = st.selectbox(
        "Status", ["Todos", *sorted(base["Status logística"].unique())], key="status_filter"
    )

search_col1, search_col2, search_col3, search_col4 = st.columns(4)
with search_col1:
    client_search = st.text_input(
        "Código do cliente", placeholder="Ex.: C62203", key="client_search",
        help="A busca é exata, considerando somente o código do cliente.",
    )
with search_col2:
    pedido_search = st.text_input("Número do pedido", placeholder="Ex.: 14489800", key="pedido_search")
with search_col3:
    nota_search = st.text_input("Número da nota fiscal", placeholder="Ex.: 0144898", key="nota_search")
with search_col4:
    is_special = "ESPECIAIS" in regional_filter.upper()
    state_scope = base[base["Regional"].eq(regional_filter)] if is_special else base.iloc[0:0]
    state_options = sorted(
        state_scope.loc[state_scope["Estado"].notna() & state_scope["Estado"].ne(""), "Estado"].unique()
    )
    if is_special and state_options:
        estado_filter = st.selectbox("Estado", ["Todos", *state_options], key="estado_filter")
    elif is_special:
        estado_filter = "Todos"
        st.warning("Envie a base 'Base Dashboard Inside Sales.xlsx' para habilitar Estado.")
    else:
        estado_filter = "Todos"

if st.button("Limpar filtros"):
    clear_filters()
    st.rerun()

filtered = base.copy()
if regional_filter != "Todos":
    filtered = filtered[filtered["Regional"].eq(regional_filter)]
if grupo_filter != "Todos":
    filtered = filtered[filtered["Grupo"].eq(grupo_filter)]
if status_filter != "Todos":
    filtered = filtered[filtered["Status logística"].eq(status_filter)]
if estado_filter != "Todos":
    filtered = filtered[filtered["Estado"].eq(estado_filter)]

if client_search:
    search = "".join(character for character in client_search.upper() if character.isalnum())
    code = filtered["Código cliente"].fillna("").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    filtered = filtered[code.eq(search)]
if pedido_search:
    filtered = filtered[filtered["Pedido"].str.contains(pedido_search.strip(), regex=False, na=False)]
if nota_search:
    nota = filtered["Nota fiscal"].fillna("").astype(str)
    filtered = filtered[nota.str.contains(nota_search.strip(), regex=False, na=False)]

if isinstance(period, tuple) and len(period) == 2:
    start_date, end_date = map(pd.Timestamp, period)
    filtered = filtered[filtered["Data pedido"].between(start_date, end_date)]

if filtered.empty:
    st.warning("Nenhum pedido encontrado para os filtros selecionados.")
    st.stop()

orders = len(filtered)
faturados = int(filtered["Data faturamento"].notna().sum())
entregues = int(filtered["Data entrega"].notna().sum())
no_prazo = int((filtered["Status logística"] == "Entregue no prazo").sum())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Pedidos", f"{orders:,}".replace(",", "."))
col2.metric("Faturados", f"{faturados:,}".replace(",", "."))
col3.metric("Entregues", f"{entregues:,}".replace(",", "."))
col4.metric("No prazo", f"{no_prazo / entregues:.0%}" if entregues else "—")
col5.metric("Lead time total médio", format_days(filtered["Lead time total (dias)"].mean()))

st.divider()
left, right = st.columns(2)
with left:
    by_status = (
        filtered["Status logística"].value_counts().rename_axis("Status").reset_index(name="Pedidos")
    )
    fig_status = px.bar(
        by_status,
        x="Pedidos",
        y="Status",
        orientation="h",
        color="Status",
        title="Pedidos por status",
        text="Pedidos",
    )
    fig_status.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_status, use_container_width=True)

with right:
    by_regional = (
        filtered.groupby("Regional", as_index=False)["Lead time total (dias)"]
        .median()
        .dropna()
        .sort_values("Lead time total (dias)", ascending=False)
    )
    fig_regional = px.bar(
        by_regional,
        x="Regional",
        y="Lead time total (dias)",
        title="Lead time total mediano por regional",
        text_auto=".1f",
    )
    fig_regional.update_layout(xaxis_title="", yaxis_title="Dias")
    st.plotly_chart(fig_regional, use_container_width=True)

st.subheader("Prazos médios")
lead_stages = [
    ("Pedido → faturamento", "Pedido → faturamento (dias)", "Pedido → faturamento (dias úteis)"),
    ("Faturamento → previsão", "Faturamento → previsão (dias)", "Faturamento → previsão (dias úteis)"),
    ("Faturamento → entrega", "Faturamento → entrega (dias)", "Faturamento → entrega (dias úteis)"),
    ("Lead time total", "Lead time total (dias)", "Lead time total (dias úteis)"),
]
lead_summary = pd.DataFrame(
    {
        "Etapa": [stage[0] for stage in lead_stages],
        "Média dias corridos": [filtered[stage[1]].mean() for stage in lead_stages],
        "Mediana dias corridos": [filtered[stage[1]].median() for stage in lead_stages],
        "Média dias úteis": [filtered[stage[2]].mean() for stage in lead_stages],
        "Mediana dias úteis": [filtered[stage[2]].median() for stage in lead_stages],
        "Pedidos com dado": [int(filtered[stage[1]].notna().sum()) for stage in lead_stages],
    }
)
st.caption("Dias úteis: segunda a sexta-feira, sem descontar feriados.")
st.dataframe(
    lead_summary,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Média dias corridos": st.column_config.NumberColumn(format="%.1f"),
        "Mediana dias corridos": st.column_config.NumberColumn(format="%.1f"),
        "Média dias úteis": st.column_config.NumberColumn(format="%.1f"),
        "Mediana dias úteis": st.column_config.NumberColumn(format="%.1f"),
    },
)

st.subheader("Detalhamento por pedido")
display_cols = [
    "Pedido", "Nota fiscal", "Cliente", "Código cliente", "Regional", "Grupo", "Estado", "Status logística", "NFs",
    "Valor pedido", "Valor nota fiscal",
    "Data pedido", "Data faturamento", "Data prevista", "Data entrega",
    "Pedido → faturamento (dias)", "Faturamento → previsão (dias)",
    "Faturamento → entrega (dias)", "Lead time total (dias)",
    "Pedido → faturamento (dias úteis)", "Faturamento → previsão (dias úteis)",
    "Faturamento → entrega (dias úteis)", "Lead time total (dias úteis)",
]
detail = filtered[display_cols].sort_values(["Data pedido", "Pedido"], ascending=[False, False])
st.dataframe(
    detail,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Valor pedido": st.column_config.NumberColumn("Valor pedido", format="R$ %.2f"),
        "Valor nota fiscal": st.column_config.NumberColumn("Valor nota fiscal", format="R$ %.2f"),
        "Data pedido": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Data faturamento": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Data prevista": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Data entrega": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Pedido → faturamento (dias)": st.column_config.NumberColumn(format="%.0f"),
        "Faturamento → previsão (dias)": st.column_config.NumberColumn(format="%.0f"),
        "Faturamento → entrega (dias)": st.column_config.NumberColumn(format="%.0f"),
        "Lead time total (dias)": st.column_config.NumberColumn(format="%.0f"),
        "Pedido → faturamento (dias úteis)": st.column_config.NumberColumn(format="%.0f"),
        "Faturamento → previsão (dias úteis)": st.column_config.NumberColumn(format="%.0f"),
        "Faturamento → entrega (dias úteis)": st.column_config.NumberColumn(format="%.0f"),
        "Lead time total (dias úteis)": st.column_config.NumberColumn(format="%.0f"),
    },
)

st.download_button(
    "Baixar dados filtrados (Excel)",
    data=to_excel(detail),
    file_name="leadtime_filtrado.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
