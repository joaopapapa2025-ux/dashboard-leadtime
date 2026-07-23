from pathlib import Path
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Lead Time | Papapa", page_icon="⏱️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def clean_text(series: pd.Series) -> pd.Series:
    """Padroniza textos sem remover zeros à esquerda de códigos."""
    return series.fillna("").astype(str).str.strip()


def parse_date(series: pd.Series) -> pd.Series:
    """Lê as datas no padrão ddmmyyyy e ignora 00000000."""
    value = clean_text(series).replace({"00000000": "", "nan": ""})
    return pd.to_datetime(value, format="%d%m%Y", errors="coerce")


def find_source_file(prefix: str) -> Path | None:
    """Localiza a única planilha cujo nome começa com o prefixo informado."""
    files = sorted(
        path
        for path in DATA_DIR.glob("*.xlsx")
        if path.name.upper().startswith(prefix.upper()) and not path.name.startswith("~$")
    )
    if len(files) == 1:
        return files[0]
    if len(files) > 1:
        st.error(f"Há mais de um arquivo começando com `{prefix}` na pasta `data`.")
        st.info("Mantenha apenas a versão mais recente de cada base antes de fazer o commit.")
        st.stop()
    return None


@st.cache_data(show_spinner="Lendo e conciliando as bases...")
def load_data(pedidos_path: str, faturamento_path: str) -> pd.DataFrame:
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
            "Valor pedido": clean_text(pedidos_raw.iloc[:, 13]),
        }
    )
    # Somente finais 00 são pedidos. Finais 50 são orçamentos.
    pedidos = pedidos[pedidos["Pedido"].str.endswith("00", na=False)].copy()

    faturamento = pd.DataFrame(
        {
            "Data faturamento": parse_date(faturamento_raw.iloc[:, 0]),
            "Nota fiscal": clean_text(faturamento_raw.iloc[:, 1]),
            "Código cliente NF": clean_text(faturamento_raw.iloc[:, 4]),
            "Cliente NF": clean_text(faturamento_raw.iloc[:, 5]),
            "Data prevista": parse_date(faturamento_raw.iloc[:, 11]),
            "Data entrega": parse_date(faturamento_raw.iloc[:, 12]),
            "Pedido": clean_text(faturamento_raw.iloc[:, 23]),
            "Regional": clean_text(faturamento_raw.iloc[:, 31]),  # AF: Desc.Região
            "Grupo": clean_text(faturamento_raw.iloc[:, 33]),  # AH: Desc.Grupo Cliente
        }
    )
    faturamento = faturamento[faturamento["Pedido"].ne("")].copy()

    # Um pedido pode ter mais de uma NF. O dashboard consolida em uma linha por pedido:
    # primeiro faturamento e última previsão/entrega, evitando duplicar pedidos nos indicadores.
    faturamento_resumo = (
        faturamento.groupby("Pedido", as_index=False)
        .agg(
            **{
                "NFs": ("Nota fiscal", "nunique"),
                "Data faturamento": ("Data faturamento", "min"),
                "Data prevista": ("Data prevista", "max"),
                "Data entrega": ("Data entrega", "max"),
                "Regional": ("Regional", "first"),
                "Grupo": ("Grupo", "first"),
            }
        )
    )

    df = pedidos.merge(faturamento_resumo, how="left", on="Pedido")
    df["Regional"] = df["Regional"].fillna("Sem regional")
    df["Grupo"] = df["Grupo"].fillna("Sem grupo")
    df["NFs"] = df["NFs"].fillna(0).astype(int)

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
st.caption("Pedidos finais 00 • Atualização diária pelas bases SVE611 e SVE660")

pedidos_file = find_source_file("SVE611")
faturamento_file = find_source_file("SVE660")

if not pedidos_file or not faturamento_file:
    st.error("Arquivos de dados não encontrados.")
    st.markdown(
        "Crie a pasta `data` ao lado do arquivo `app.py` e envie uma planilha de cada tipo:"
    )
    st.code("data/SVE611V (71).xlsx\ndata/SVE660V.xlsx", language="text")
    st.info("Os nomes só precisam começar com SVE611 e SVE660. Substitua os dois arquivos diariamente no GitHub e mantenha uma versão de cada base.")
    st.stop()

try:
    base = load_data(str(pedidos_file), str(faturamento_file))
except Exception as error:
    st.exception(error)
    st.stop()

with st.sidebar:
    st.header("Filtros")
    regionais = sorted(base["Regional"].dropna().unique())
    grupos = sorted(base["Grupo"].dropna().unique())
    regional_filter = st.multiselect("Regional", regionais, default=regionais)
    grupo_filter = st.multiselect("Grupo", grupos, default=grupos)
    client_search = st.text_input("Buscar cliente ou código", placeholder="Ex.: Drogaria ou C12345")
    statuses = sorted(base["Status logística"].unique())
    status_filter = st.multiselect("Status", statuses, default=statuses)

    min_date = base["Data pedido"].min().date()
    max_date = base["Data pedido"].max().date()
    period = st.date_input("Período do pedido", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = base[
    base["Regional"].isin(regional_filter)
    & base["Grupo"].isin(grupo_filter)
    & base["Status logística"].isin(status_filter)
].copy()

if client_search:
    search = client_search.strip().casefold()
    filtered = filtered[
        filtered["Cliente"].str.casefold().str.contains(search, na=False)
        | filtered["Código cliente"].str.casefold().str.contains(search, na=False)
    ]

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
lead_cols = [
    "Pedido → faturamento (dias)",
    "Faturamento → previsão (dias)",
    "Faturamento → entrega (dias)",
    "Lead time total (dias)",
]
lead_summary = pd.DataFrame(
    {
        "Etapa": lead_cols,
        "Média (dias)": [filtered[column].mean() for column in lead_cols],
        "Mediana (dias)": [filtered[column].median() for column in lead_cols],
        "Pedidos com dado": [int(filtered[column].notna().sum()) for column in lead_cols],
    }
)
st.dataframe(lead_summary, hide_index=True, use_container_width=True, column_config={"Média (dias)": st.column_config.NumberColumn(format="%.1f"), "Mediana (dias)": st.column_config.NumberColumn(format="%.1f")})

st.subheader("Detalhamento por pedido")
display_cols = [
    "Pedido", "Cliente", "Código cliente", "Regional", "Grupo", "Status logística", "NFs",
    "Data pedido", "Data faturamento", "Data prevista", "Data entrega",
    "Pedido → faturamento (dias)", "Faturamento → previsão (dias)",
    "Faturamento → entrega (dias)", "Lead time total (dias)",
]
detail = filtered[display_cols].sort_values(["Data pedido", "Pedido"], ascending=[False, False])
st.dataframe(
    detail,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Data pedido": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Data faturamento": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Data prevista": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Data entrega": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Pedido → faturamento (dias)": st.column_config.NumberColumn(format="%.0f"),
        "Faturamento → previsão (dias)": st.column_config.NumberColumn(format="%.0f"),
        "Faturamento → entrega (dias)": st.column_config.NumberColumn(format="%.0f"),
        "Lead time total (dias)": st.column_config.NumberColumn(format="%.0f"),
    },
)

csv = detail.to_csv(index=False).encode("utf-8-sig")
st.download_button("Baixar dados filtrados (CSV)", data=csv, file_name="leadtime_filtrado.csv", mime="text/csv")

