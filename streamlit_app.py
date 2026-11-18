"""Demo Streamlit: dự báo giá chip nhớ DRAM & NAND.

Chạy cục bộ:
    streamlit run streamlit_app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

import demo_core as dc
import train_models as tm

st.set_page_config(
    page_title="Dự báo giá chip DRAM & NAND",
    page_icon="💾",
    layout="wide",
)

MODEL_NAME = {
    "Naive": "Naive (giữ giá)",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
}
SERIES_COLOR = {
    "Giá thực": "#0f172a",
    "Naive (giữ giá)": "#94a3b8",
    "Random Forest": "#2563eb",
    "XGBoost": "#ea580c",
}
CHIP_ICON = {"dram": "🧠", "nand": "💽"}
MAX_HORIZON = 12

CSS = """
<style>
.block-container { max-width: 1180px; padding-top: 1.1rem; padding-bottom: 3rem; }
[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

.hero {
    background: radial-gradient(900px 320px at 85% -30%, rgba(96,165,250,.55), transparent 65%),
                linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
    color: #fff; border-radius: 20px; padding: 28px 32px;
    box-shadow: 0 18px 40px -24px rgba(15,23,42,.55);
}
.hero .badge {
    display: inline-block; background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.22); color: #dbeafe;
    font-size: .68rem; font-weight: 700; letter-spacing: .12em;
    padding: 5px 11px; border-radius: 999px;
}
.hero h1 { color: #fff; font-size: 2.05rem; font-weight: 800; letter-spacing: -.025em; margin: 12px 0 6px; }
.hero p { color: #c7d2fe; margin: 0; font-size: .95rem; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.chip {
    background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16);
    color: #e2e8f0; padding: 6px 12px; border-radius: 999px; font-size: .8rem;
}
.chip b { color: #fff; font-weight: 700; }

.section { display: flex; align-items: center; gap: 10px; margin: 6px 0 14px; }
.section .ico { font-size: 1.35rem; line-height: 1; }
.section h3 { margin: 0; font-size: 1.12rem; font-weight: 700; color: #0f172a; letter-spacing: -.01em; }
.section .price-pill {
    margin-left: auto; background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;
    padding: 4px 12px; border-radius: 999px; font-size: .78rem; font-weight: 600; white-space: nowrap;
}

div[data-testid="stMetric"] {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 13px 16px; box-shadow: 0 1px 2px rgba(15,23,42,.04);
}
div[data-testid="stMetricLabel"] p { font-size: .78rem; color: #64748b; font-weight: 600; }
div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
div[data-testid="stMetricDelta"] { font-size: .76rem; }

.stTabs [role="tablist"] {
    gap: 4px; background: #f1f5f9; padding: 5px; border-radius: 12px;
    border: none; height: auto; min-height: 0;
}
.stTabs [data-testid="stTab"] {
    height: 38px; padding: 0 16px; border-radius: 8px; background: transparent;
    color: #64748b; display: flex; align-items: center; justify-content: center;
    transition: background .15s ease, color .15s ease;
}
.stTabs [data-testid="stTab"] p { margin: 0; font-size: .9rem; font-weight: 600; color: inherit; }
.stTabs [data-testid="stTab"]:hover { background: rgba(255,255,255,.7); }
.stTabs [data-testid="stTab"][aria-selected="true"] {
    background: #fff !important; color: #0f172a !important;
    box-shadow: 0 1px 3px rgba(15,23,42,.12);
}
.stTabs .react-aria-SelectionIndicator { display: none; }

table.cmp { width: 100%; border-collapse: separate; border-spacing: 0; background: #fff;
    border: 1px solid #e2e8f0; border-radius: 14px; overflow: hidden; }
table.cmp th {
    text-align: right; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
    color: #64748b; font-weight: 700; padding: 12px 16px; background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
}
table.cmp th:first-child, table.cmp td:first-child { text-align: left; }
table.cmp td { padding: 13px 16px; text-align: right; border-bottom: 1px solid #f1f5f9;
    font-variant-numeric: tabular-nums; color: #0f172a; font-weight: 600; font-size: .95rem; }
table.cmp tr:last-child td { border-bottom: none; }
table.cmp td.chip-name { font-weight: 700; white-space: nowrap; }
table.cmp td.base { color: #334155; }
table.cmp .chg { font-size: .73rem; font-weight: 600; margin-top: 2px; }
table.cmp .chg.up { color: #dc2626; }
table.cmp .chg.down { color: #16a34a; }
table.cmp .chg.flat { color: #94a3b8; }

.note {
    background: #fff7ed; border: 1px solid #fed7aa; color: #9a3412;
    border-radius: 12px; padding: 12px 15px; font-size: .86rem; line-height: 1.55;
}
.note.ok { background: #f0fdf4; border-color: #bbf7d0; color: #166534; }
.note.info { background: #eff6ff; border-color: #bfdbfe; color: #1e40af; }

.card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 17px 19px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(15,23,42,.03);
}
.card h4 { margin: 0 0 7px; font-size: .95rem; font-weight: 700; color: #0f172a; }
.card p, .card li { margin: 0; color: #475569; font-size: .87rem; line-height: 1.6; }
.card ol, .card ul { margin: 0; padding-left: 18px; }

.small { color: #64748b; font-size: .78rem; line-height: 1.6; }
.legend { display: flex; flex-direction: column; gap: 7px; font-size: .85rem; color: #334155; }
.legend .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------- DỮ LIỆU / CACHE -----------------------------
@st.cache_data
def data_range():
    """(tháng đầu, tháng cuối) của chuỗi giá từng loại chip."""
    spans = {}
    for name, fname in tm.TARGETS.items():
        s = tm.load_price(fname, f"{name}_price")
        spans[name] = (s.index.min(), s.index.max())
    return spans


@st.cache_data(show_spinner="Đang huấn luyện mô hình…")
def get_results(horizon: int, stamp: int):
    return dc.get_results(horizon)


@st.cache_data
def load_metrics(horizon: int, stamp: int):
    path = results_dir(horizon) / "metrics.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner="Đang chạy backtest…")
def run_backtest(horizon: int, n_last: int, stamp: int):
    res = get_results(horizon, stamp)
    return {name: dc.backtest_table(r, name, n_last=n_last) for name, r in res.items()}


def results_dir(horizon: int):
    return tm.base_dir / ("results" if horizon == 1 else f"results_h{horizon}")


def results_stamp(horizon: int) -> int:
    """mtime mới nhất trong results*/ — đổi file kết quả là cache tự làm mới."""
    folder = results_dir(horizon)
    if not folder.exists():
        return 0
    return max((p.stat().st_mtime_ns for p in folder.glob("*")), default=0)


def fmt(value: float) -> str:
    """Định dạng giá gọn cho cả DRAM (~4.85) và NAND (~0.079)."""
    return f"{value:,.4g}"


def pct(value: float, base: float) -> str:
    return f"{(value / base - 1) * 100:+.1f}%"


# ----------------------------- BIỂU ĐỒ -----------------------------
def color_scale():
    return alt.Scale(domain=list(SERIES_COLOR), range=list(SERIES_COLOR.values()))


def forecast_chart(res, months: int = 36):
    """Giá lịch sử (vùng tô) + đường dự báo của 3 mô hình."""
    df = res["df"]
    hist = df["price"].tail(months)
    hist_df = pd.DataFrame({"date": hist.index, "value": hist.values})
    frames = [pd.DataFrame({"date": hist.index, "value": hist.values, "series": "Giá thực"})]
    last_date, last_price = df.index[-1], float(df["price"].iloc[-1])
    for model, value in res["preds"].items():
        label = MODEL_NAME[model]
        frames.append(pd.DataFrame({
            "date": [last_date, res["next_month"]],
            "value": [last_price, value],
            "series": [label, label],
        }))
    data = pd.concat(frames, ignore_index=True)

    base = alt.Chart(data).encode(
        x=alt.X("date:T", title=None, scale=alt.Scale(padding=14), axis=alt.Axis(
            format="%m/%y", labelFontSize=11, tickCount=8, grid=False, domainOpacity=.3)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False, nice=True),
                axis=alt.Axis(labelFontSize=11, grid=True, gridColor="#eef2f7", domainOpacity=.3)),
        color=alt.Color(
            "series:N", scale=color_scale(), title=None,
            sort=["Giá thực", "Naive (giữ giá)", "Random Forest", "XGBoost"],
            legend=alt.Legend(symbolOpacity=1, symbolSize=110, symbolStrokeWidth=3,
                              labelFontSize=11, labelColor="#334155"),
        ),
        tooltip=[
            alt.Tooltip("date:T", format="%m/%Y", title="Tháng"),
            alt.Tooltip("series:N", title="Chuỗi"),
            alt.Tooltip("value:Q", format=".4f", title="Giá"),
        ],
    )
    area = alt.Chart(hist_df).mark_area(color="#93c5fd", opacity=0.38).encode(
        x=alt.X("date:T"), y=alt.Y("value:Q"))
    line = base.mark_line(strokeWidth=2.4)
    points = base.mark_point(filled=True, size=85).transform_filter(
        alt.datum.series != "Giá thực")
    now_rule = alt.Chart(pd.DataFrame({"date": [last_date]})).mark_rule(
        color="#cbd5e1", strokeDash=[3, 4]).encode(x="date:T")
    return alt.layer(area, line, points, now_rule).properties(height=340)


def mase_chart(metrics: pd.DataFrame):
    d = metrics[(metrics["split"] == "test") & (metrics["model"] != "Naive")].copy()
    if d.empty:
        return None
    d["chip"] = d["target"].map(dc.LABEL)
    d["model"] = d["model"].map(MODEL_NAME)
    ymax = max(1.35, float(d["MASE"].max()) * 1.12)
    order = ["Random Forest", "XGBoost"]
    scale_y = alt.Scale(domain=[0, ymax])

    bars = alt.Chart(d).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=34).encode(
        x=alt.X("model:N", title=None, sort=order, axis=alt.Axis(labelAngle=0, labelFontSize=12)),
        xOffset=alt.XOffset("chip:N", sort=["DRAM", "NAND"]),
        y=alt.Y("MASE:Q", title="MASE (test)", scale=scale_y,
                axis=alt.Axis(grid=True, gridColor="#eef2f7", domainOpacity=.3)),
        color=alt.Color("chip:N", title=None,
                        scale=alt.Scale(domain=["DRAM", "NAND"], range=["#2563eb", "#10b981"])),
        tooltip=[alt.Tooltip("chip:N", title="Chip"),
                 alt.Tooltip("model:N", title="Mô hình"),
                 alt.Tooltip("MASE:Q", format=".3f", title="MASE")],
    )
    labels = alt.Chart(d).mark_text(dy=-8, fontSize=11, color="#475569").encode(
        x=alt.X("model:N", sort=order),
        xOffset=alt.XOffset("chip:N", sort=["DRAM", "NAND"]),
        y=alt.Y("MASE:Q", scale=scale_y),
        text=alt.Text("MASE:Q", format=".2f"),
    )
    rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
        color="#ef4444", strokeDash=[5, 4], strokeWidth=1.5).encode(y="y:Q")
    return (bars + labels + rule).properties(height=300)


def backtest_chart(bt: pd.DataFrame):
    long = bt.melt(
        id_vars=["month"],
        value_vars=["actual", "Naive", "RandomForest", "XGBoost"],
        var_name="series", value_name="value",
    )
    long["series"] = long["series"].map(lambda s: "Giá thực" if s == "actual" else MODEL_NAME[s])
    base = alt.Chart(long).encode(
        x=alt.X("month:T", title=None, scale=alt.Scale(padding=14),
                axis=alt.Axis(format="%m/%y", labelFontSize=11, grid=False)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False, nice=True),
                axis=alt.Axis(labelFontSize=11, grid=True, gridColor="#eef2f7")),
        color=alt.Color(
            "series:N", scale=color_scale(), title=None,
            sort=["Giá thực", "Naive (giữ giá)", "Random Forest", "XGBoost"],
            legend=alt.Legend(symbolOpacity=1, symbolSize=110, symbolStrokeWidth=3,
                              labelFontSize=11, labelColor="#334155"),
        ),
        tooltip=[
            alt.Tooltip("month:T", format="%m/%Y", title="Tháng"),
            alt.Tooltip("series:N", title="Chuỗi"),
            alt.Tooltip("value:Q", format=".4f", title="Giá"),
        ],
    )
    actual = alt.Chart(bt[["month", "actual"]].rename(columns={"actual": "value"})).mark_area(
        color="#93c5fd", opacity=0.35).encode(x=alt.X("month:T"), y=alt.Y("value:Q"))
    line = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45, filled=True))
    return alt.layer(actual, line).properties(height=320)


# ----------------------------- HTML HELPERS -----------------------------
def hero_html(first_month, last_month, target_month, horizon, tuned_count):
    return f"""
    <div class="hero">
      <span class="badge">DEMO HỌC THUẬT</span>
      <h1>💾 Dự báo giá chip nhớ DRAM &amp; NAND</h1>
      <p>Random Forest · XGBoost · dự báo log-return · dữ liệu {first_month:%m/%Y} – {last_month:%m/%Y}</p>
      <div class="chips">
        <span class="chip">Mốc dự báo <b>{target_month:%m/%Y}</b></span>
        <span class="chip">h = <b>{horizon}</b> tháng</span>
        <span class="chip">Đã tinh chỉnh <b>{tuned_count}/{MAX_HORIZON}</b> mốc</span>
      </div>
    </div>
    """


def overview_html(results):
    head = "".join(f"<th>{h}</th>" for h in
                   ["Loại chip", "Giá hiện tại", "Naive", "Random Forest", "XGBoost"])
    rows = []
    for name, res in results.items():
        cells = [
            f"<td class='chip-name'>{CHIP_ICON[name]} {dc.dc_label(name)}</td>",
            f"<td class='base'>{fmt(res['price_now'])}</td>",
        ]
        for model in ["Naive", "RandomForest", "XGBoost"]:
            value = res["preds"][model]
            delta = (value / res["price_now"] - 1) * 100
            cls = "up" if delta > 0.05 else ("down" if delta < -0.05 else "flat")
            cells.append(f"<td>{fmt(value)}<div class='chg {cls}'>{delta:+.1f}%</div></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(rows)
    return f"<table class='cmp'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def section_html(name, res):
    return (f"<div class='section'><span class='ico'>{CHIP_ICON[name]}</span>"
            f"<h3>{dc.dc_label(name)}</h3>"
            f"<span class='price-pill'>Tháng {res['month_now']:%m/%Y} · {fmt(res['price_now'])}</span></div>")


def legend_html():
    items = [
        ("#94a3b8", "Naive — giá không đổi"),
        ("#2563eb", "Random Forest"),
        ("#ea580c", "XGBoost"),
    ]
    rows = "".join(
        f"<div><span class='dot' style='background:{c}'></span>{t}</div>" for c, t in items
    )
    return f"<div class='legend'>{rows}</div>"


# ----------------------------- SIDEBAR -----------------------------
spans = data_range()
first_month = min(start for start, _ in spans.values())
last_month = min(end for _, end in spans.values())

with st.sidebar:
    st.markdown("### ⚙️ Thiết lập")
    periods = [pd.Period(last_month, "M") + i for i in range(1, MAX_HORIZON + 1)]
    options = {f"{p:%m/%Y}  ·  {i} tháng": i for i, p in enumerate(periods, 1)}
    choice = st.selectbox("Tháng cần dự báo", list(options), index=0)
    horizon = options[choice]
    target_month = pd.Period(last_month, "M") + horizon

    if not (results_dir(horizon) / "best_params.csv").exists():
        st.warning(f"Chưa có tham số tinh chỉnh cho h={horizon}; "
                   "mô hình dùng tham số mặc định.", icon="ℹ️")

    st.caption(f"→ Dự báo giá tháng **{target_month:%m/%Y}**")
    st.divider()

    st.markdown("#### 🎛️ Biểu đồ")
    window = st.select_slider(
        "Số tháng lịch sử hiển thị",
        options=[24, 36, 60, 120],
        value=36,
        format_func=lambda v: f"{v} tháng",
    )

    st.divider()
    st.markdown("#### 🤖 Mô hình so sánh")
    st.markdown(legend_html(), unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<div class='small'>Nguồn giá: <code>data/dram_price.csv</code>, "
        "<code>data/nand_price.csv</code><br>Biến thị trường: MU, SK Hynix, Samsung, WDC, "
        "SOX, USD/KRW, FEDFUNDS</div>",
        unsafe_allow_html=True,
    )

# ----------------------------- HEADER -----------------------------
tuned_count = sum((results_dir(h) / "best_params.csv").exists()
                  for h in range(1, MAX_HORIZON + 1))
st.markdown(hero_html(first_month, last_month, target_month, horizon, tuned_count), unsafe_allow_html=True)
st.write("")

try:
    results = get_results(horizon, results_stamp(horizon))
except Exception as exc:  # noqa: BLE001 - hiển thị lỗi thân thiện trên UI
    st.error(f"Không chạy được mô hình: {exc}")
    st.stop()

tab_forecast, tab_metrics, tab_backtest, tab_about = st.tabs(
    ["📈 Dự báo", "🎯 Hiệu năng", "🧪 Backtest", "ℹ️ Giới thiệu"]
)

# ----------------------------- TAB 1: DỰ BÁO -----------------------------
with tab_forecast:
    st.markdown("#### Tổng quan")
    st.markdown(overview_html(results), unsafe_allow_html=True)
    st.write("")
    st.markdown(
        "<div class='note'>⚠️ Kết quả chỉ mang tính minh hoạ học thuật. Trên tập test, đa số mô hình "
        "có MASE ≥ 1 (chưa tốt hơn naive) — xem tab <b>Hiệu năng</b> để đánh giá đầy đủ.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    for i, (name, res) in enumerate(results.items()):
        if i:
            st.divider()
        st.markdown(section_html(name, res), unsafe_allow_html=True)

        prev_price = float(res["df"]["price"].iloc[-2])
        cols = st.columns(4)
        cols[0].metric(
            "Giá hiện tại",
            fmt(res["price_now"]),
            f"{pct(res['price_now'], prev_price)} so với tháng trước",
            delta_color="off",
        )
        for col, (model, value) in zip(cols[1:], res["preds"].items()):
            col.metric(MODEL_NAME[model], fmt(value), pct(value, res["price_now"]),
                       delta_color="off")

        st.write("")
        st.altair_chart(forecast_chart(res, window), width="stretch")
        st.caption(
            f"Vùng tô: giá thực {window} tháng gần nhất · nét đứt: thời điểm hiện tại · "
            f"điểm cuối: dự báo {res['next_month']:%m/%Y}"
        )

        with st.expander("Bảng dự báo chi tiết"):
            detail = pd.DataFrame([
                {"Mô hình": MODEL_NAME[m], "Giá dự báo": v,
                 "Thay đổi (%)": (v / res["price_now"] - 1) * 100}
                for m, v in res["preds"].items()
            ])
            st.dataframe(
                detail, hide_index=True,
                column_config={
                    "Giá dự báo": st.column_config.NumberColumn(format="%.4f"),
                    "Thay đổi (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )

# ----------------------------- TAB 2: HIỆU NĂNG -----------------------------
with tab_metrics:
    metrics = load_metrics(horizon, results_stamp(horizon))
    if metrics is None:
        st.info(
            f"Chưa có `metrics.csv` cho h={horizon}. "
            f"Chạy `HORIZON={horizon} python train_models.py` để tạo kết quả đánh giá.",
            icon="📄",
        )
    else:
        st.markdown("#### MASE trên tập Test")
        st.caption("Cột thấp hơn đường đỏ (naive = 1) là tốt hơn mô hình “giá không đổi”.")
        chart = mase_chart(metrics)
        if chart is not None:
            st.altair_chart(chart, width="stretch")

        test = metrics[metrics["split"] == "test"].copy()
        test["target"] = test["target"].map(dc.LABEL)
        test["model"] = test["model"].map(MODEL_NAME)
        test["DirAcc"] = (pd.to_numeric(test["DirAcc"], errors="coerce") * 100).map(
            lambda v: f"{v:.0f}" if pd.notna(v) else "—")
        test = test.drop(columns=["split"])

        show = test.rename(columns={
            "target": "Loại chip", "model": "Mô hình", "MAPE%": "MAPE (%)",
            "sMAPE%": "sMAPE (%)", "DirAcc": "Đúng hướng (%)",
        })
        st.dataframe(
            show.style.format({
                "MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}",
                "sMAPE (%)": "{:.2f}", "MASE": "{:.3f}",
                "R2": "{:.3f}",
            }, na_rep="—").apply(
                lambda s: ["background-color: #dcfce7" if v < 1 else ""
                           for v in s], subset=["MASE"],
            ),
            hide_index=True,
        )
        st.caption("MASE < 1 (nền xanh) nghĩa là tốt hơn naive · DirAcc để trống với naive "
                   "vì naive không dự đoán hướng.")

        best = show.loc[show["MASE"].idxmin()]
        if best["MASE"] < 1:
            st.markdown(
                f"<div class='note ok'>✅ Tốt nhất trên test: <b>{best['Mô hình']}</b> cho "
                f"<b>{best['Loại chip']}</b> (MASE = {best['MASE']:.3f}), tốt hơn naive.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='note'>Trên tập test, <b>không mô hình nào tốt hơn naive</b> "
                "(MASE ≥ 1). Nên xem kết quả như minh hoạ phương pháp, không phải dự báo đầu tư.</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Chỉ số trên tập Validation"):
            val = metrics[metrics["split"] == "val"].copy()
            val["target"] = val["target"].map(dc.LABEL)
            val["model"] = val["model"].map(MODEL_NAME)
            val["DirAcc"] = (pd.to_numeric(val["DirAcc"], errors="coerce") * 100).map(
                lambda v: f"{v:.0f}" if pd.notna(v) else "—")
            val = val.drop(columns=["split"]).rename(columns={
                "target": "Loại chip", "model": "Mô hình", "MAPE%": "MAPE (%)",
                "sMAPE%": "sMAPE (%)", "DirAcc": "Đúng hướng (%)",
            })
            st.dataframe(
                val.style.format({
                    "MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}",
                    "sMAPE (%)": "{:.2f}", "MASE": "{:.3f}",
                    "R2": "{:.3f}",
                }, na_rep="—"),
                hide_index=True,
            )

# ----------------------------- TAB 3: BACKTEST -----------------------------
with tab_backtest:
    st.markdown("#### Backtest cuốn chiếu (walk-forward)")
    st.caption(
        "Tại mỗi tháng, mô hình chỉ được huấn luyện bằng các mẫu mà nhãn đã biết tại thời điểm đó, "
        "rồi dự đoán giá tháng kế tiếp. Đây là kiểm tra trung thực hơn so với tập test cố định."
    )
    n_last = st.slider("Số tháng backtest gần nhất", min_value=6, max_value=36,
                       value=18, step=1)

    if st.button("▶️ Chạy backtest", type="primary"):
        st.session_state["backtest_done"] = True
    if st.session_state.get("backtest_done"):
        backtests = run_backtest(horizon, n_last, results_stamp(horizon))
        for name, res in results.items():
            bt = backtests.get(name)
            st.markdown(section_html(name, res), unsafe_allow_html=True)
            if bt is None or bt.empty:
                st.info("Không đủ dữ liệu để backtest.")
                continue

            errors = []
            for model in ["Naive", "RandomForest", "XGBoost"]:
                err = (bt[model] - bt["actual"]).abs()
                errors.append({
                    "model": model,
                    "Mô hình": MODEL_NAME[model],
                    "MAE": err.mean(),
                    "MAPE (%)": (err / bt["actual"]).mean() * 100,
                })
            errors = pd.DataFrame(errors)

            cols = st.columns(3)
            for col, (_, row) in zip(cols, errors.iterrows()):
                col.metric(row["Mô hình"], f"{row['MAE']:.4f}")
                col.caption(f"MAPE {row['MAPE (%)']:.1f}%")

            st.write("")
            st.altair_chart(backtest_chart(bt), width="stretch")
            st.caption("Sai số MAE càng thấp càng tốt · naive là mốc so sánh.")

            with st.expander("Bảng số liệu backtest"):
                table = bt.copy()
                table["month"] = table["month"].dt.strftime("%m/%Y")
                table["next_month"] = table["next_month"].dt.strftime("%m/%Y")
                table = table.rename(columns={
                    "month": "Tháng", "next_month": "Dự báo cho", "price_now": "Giá hiện tại",
                    "actual": "Giá thực tế", **{m: MODEL_NAME[m] for m in MODEL_NAME},
                })
                st.dataframe(table, hide_index=True,
                             column_config={c: st.column_config.NumberColumn(format="%.4f")
                                            for c in table.columns if c not in ("Tháng", "Dự báo cho")})
    else:
        st.markdown(
            "<div class='note info'>⏱️ Bấm <b>Chạy backtest</b> để bắt đầu — mất khoảng vài chục "
            "giây vì phải huấn luyện lại nhiều lần.</div>",
            unsafe_allow_html=True,
        )

# ----------------------------- TAB 4: GIỚI THIỆU -----------------------------
with tab_about:
    st.markdown("#### Pipeline")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "<div class='card'><h4>🗂️ Dữ liệu</h4><p>Giá DRAM/NAND theo tháng (2018 → nay) và "
            "biến thị trường hằng ngày: MU, SK Hynix, Samsung, WDC, SOX, USD/KRW, FEDFUNDS. "
            "Dữ liệu ngày được gộp về tháng, chỉ <code>ffill</code> từ quá khứ.</p></div>"
            "<div class='card'><h4>🧮 Đặc trưng</h4><ul>"
            "<li>Log-return giá tháng hiện tại + 2 lag</li>"
            "<li>Độ lệch so với MA3, độ biến động 3 tháng</li>"
            "<li>Log-return tháng của cổ phiếu/chỉ số thị trường</li>"
            "<li>Lãi suất Fed (FEDFUNDS)</li></ul></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            "<div class='card'><h4>🤖 Mô hình</h4><p>Random Forest và XGBoost trong Pipeline có "
            "<code>SimpleImputer</code> (fit riêng từng fold để chống rò rỉ). Siêu tham số tinh chỉnh "
            "bằng <code>RandomizedSearchCV</code> + <code>TimeSeriesSplit</code>, lưu ở "
            "<code>results*/best_params.csv</code>.</p></div>"
            "<div class='card'><h4>🎯 Dự báo</h4><p>Nhãn là log-return sau <i>h</i> tháng: "
            "<code>log(P[t+h]) − log(P[t])</code>. Fit lại trên toàn bộ dữ liệu có nhãn, dự đoán "
            "rồi quy đổi <code>giá = giá hiện tại × exp(log-return)</code>.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='note'>⚠️ <b>Hạn chế:</b> chuỗi giá chỉ ~105 tháng nên kết quả nhạy với cách chia "
        "train/val/test; giá chip chịu ảnh hưởng chu kỳ cung cầu và tin tức nên mô hình không nắm được "
        "cú sốc. Đây là demo học thuật, <b>không phải khuyến nghị đầu tư</b>.</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("#### Chạy cục bộ")
    st.code("pip install -r requirements.txt\nstreamlit run streamlit_app.py", language="bash")

st.write("")
st.markdown(
    "<div class='small' style='text-align:center'>Demo Streamlit · dữ liệu trong <code>data/</code> · "
    "scikit-learn + XGBoost</div>",
    unsafe_allow_html=True,
)
