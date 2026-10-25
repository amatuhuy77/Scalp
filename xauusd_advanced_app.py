"""
XAUUSD Scalper Pro v3
Jalankan:  streamlit run xauusd_advanced_app.py
"""
import io
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import engine as E

st.set_page_config(page_title="XAUUSD Scalper Pro", page_icon="🪙", layout="wide",
                   initial_sidebar_state="collapsed")

BUY, SELL, GOLD, MUTED = "#2FB59A", "#E4574F", "#D8A93A", "#9A9AA3"

# =============================================================================
# CSS - satu sistem desain untuk HP dan desktop
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input { font-family: 'Manrope', system-ui, sans-serif; }
.block-container { max-width: 1180px; padding: 1.1rem 1rem 3rem; }
#MainMenu, footer { visibility: hidden; }
h1 { font-size: clamp(1.35rem, 4.5vw, 1.9rem) !important; font-weight: 800 !important; letter-spacing: -.01em; margin-bottom: .2rem; }
.sub { color: #9A9AA3; font-size: .86rem; margin: 0 0 .9rem; }
.num { font-variant-numeric: tabular-nums; }

/* Kartu keputusan */
.hero { background: #1C1F25; border: 1px solid #2A2E36; border-left: 6px solid #4A4F5A;
        border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: .8rem; }
.hero.buy { border-left-color: #2FB59A; } .hero.sell { border-left-color: #E4574F; }
.hero.ready { border-left-color: #D8A93A; }
.hero .st { font-size: clamp(1.6rem, 6.2vw, 2.3rem); font-weight: 800; line-height: 1.1; letter-spacing: -.02em; }
.hero.buy .st { color: #2FB59A; } .hero.sell .st { color: #E4574F; } .hero.ready .st { color: #D8A93A; }
.hero .why { color: #D4D0C6; font-size: .95rem; line-height: 1.5; margin-top: .45rem; }
.bar { height: 6px; background: #2A2E36; border-radius: 4px; margin-top: .8rem; overflow: hidden; }
.bar > i { display: block; height: 100%; background: #D8A93A; border-radius: 4px; }
.barlbl { display: flex; justify-content: space-between; color: #9A9AA3; font-size: .78rem; margin-top: .35rem; }

/* Rantai keputusan M15 -> M5 -> M1 */
.chain { background: #1C1F25; border: 1px solid #2A2E36; border-radius: 14px; padding: .7rem .9rem; margin-bottom: .8rem; }
.chain .ttl { color: #9A9AA3; font-size: .8rem; margin-bottom: .35rem; }
.node { display: grid; grid-template-columns: 3.2rem 1fr auto; align-items: center; gap: .6rem; padding: .5rem 0; position: relative; }
.node + .node { border-top: 1px dashed #2A2E36; }
.node .tf { font-weight: 800; font-size: 1.05rem; }
.node .role { color: #9A9AA3; font-size: .76rem; display: block; font-weight: 500; }
.node .dir { font-weight: 700; } .node .dir.buy { color: #2FB59A; } .node .dir.sell { color: #E4574F; }
.chip { font-size: .74rem; font-weight: 700; padding: .18rem .55rem; border-radius: 99px; white-space: nowrap; }
.chip.pass { background: rgba(47,181,154,.15); color: #2FB59A; }
.chip.fail { background: rgba(216,169,58,.16); color: #D8A93A; }
.chip.idle { background: #262A31; color: #7C7F88; }
.node.idle .tf, .node.idle .dir { opacity: .45; }

/* Grid metrik: otomatis 2 kolom di HP, 5 di desktop */
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .6rem; margin-bottom: .8rem; }
.cell { background: #1C1F25; border: 1px solid #2A2E36; border-radius: 12px; padding: .65rem .8rem; }
.cell .k { color: #9A9AA3; font-size: .76rem; }
.cell .v { font-size: 1.28rem; font-weight: 800; font-variant-numeric: tabular-nums; margin-top: .1rem; }
.cell .s { color: #7C7F88; font-size: .74rem; margin-top: .05rem; }

/* Rencana trade */
.plan { background: #1C1F25; border: 1px solid #2A2E36; border-radius: 14px; padding: .85rem 1rem; margin-bottom: .8rem; }
.plan h4 { margin: 0 0 .5rem; font-size: 1rem; }
.plan .row { display: flex; justify-content: space-between; padding: .28rem 0; font-size: .93rem; font-variant-numeric: tabular-nums; }
.plan .row + .row { border-top: 1px solid #262A31; }
.plan .row span:first-child { color: #9A9AA3; }
.note { border-radius: 10px; padding: .6rem .8rem; font-size: .85rem; line-height: 1.45; margin-top: .6rem; }
.note.warn { background: rgba(216,169,58,.12); color: #E8C777; border: 1px solid rgba(216,169,58,.3); }
.note.bad  { background: rgba(228,87,79,.12); color: #F08A84; border: 1px solid rgba(228,87,79,.3); }
.note.good { background: rgba(47,181,154,.12); color: #6FD3BC; border: 1px solid rgba(47,181,154,.3); }
.note.info { background: #1C1F25; color: #B9B6AD; border: 1px solid #2A2E36; }

/* Tab & kontrol nyaman disentuh */
button[data-baseweb="tab"] { font-weight: 700; padding: .6rem .9rem; }
.stButton > button, .stDownloadButton > button { width: 100%; min-height: 2.8rem; font-weight: 700; border-radius: 10px; }
[data-testid="stExpander"] { border-radius: 12px; }
@media (max-width: 640px) {
  .block-container { padding: .7rem .65rem 3rem; }
  .hero { padding: .85rem .9rem; } .cell .v { font-size: 1.12rem; }
  .node { grid-template-columns: 2.8rem 1fr auto; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def h(s: str) -> str:
    """Rapatkan HTML supaya Markdown tidak menganggapnya blok kode."""
    return re.sub(r"\n\s*", "", s).strip()


def money(v, d=2):
    return f"${v:,.{d}f}"


# =============================================================================
# DATA
# =============================================================================
@st.cache_data(ttl=20, show_spinner=False)
def fetch(ticker: str, tf: str, period: str):
    try:
        df = yf.download(ticker, period=period, interval=tf, progress=False,
                         auto_adjust=False, threads=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df = df[~df.index.duplicated()].sort_index()
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return E.add_indicators(df)


def load_set(periods: dict):
    """Semua TF HARUS dari ticker yang sama supaya harga konsisten."""
    for tk in ("GC=F", "XAUUSD=X"):
        out = {tf: fetch(tk, tf, per) for tf, per in periods.items()}
        if all(v is not None and len(v) > E.WARMUP + 5 for v in out.values()):
            return out, tk
    return None, None


# =============================================================================
# HEADER + PENGATURAN
# =============================================================================
st.title("🪙 XAUUSD Scalper Pro")
st.markdown('<p class="sub">Satu keputusan dari tiga timeframe: M15 menentukan arah, M5 menyiapkan setup, '
            'M1 hanya untuk timing.</p>', unsafe_allow_html=True)

TZ_OPT = {"WIB (UTC+7)": "Asia/Jakarta", "WITA (UTC+8)": "Asia/Makassar", "WIT (UTC+9)": "Asia/Jayapura"}

with st.expander("⚙️ Modal & pengaturan", expanded=False):
    c1, c2, c3 = st.columns(3)
    tipe = c1.selectbox("Tipe akun", ["Standard / Raw", "Cent"])
    modal = c2.number_input("Modal ($)", min_value=5.0, value=50.0, step=5.0)
    risiko = c3.slider("Risiko per trade (%)", 0.5, 3.0, 1.0, 0.5)
    c4, c5, c6 = st.columns(3)
    min_skor = c4.slider("Skor setup minimum (%)", 50, 90, 70, 5,
                         help="Makin tinggi = makin sedikit tapi makin selektif.")
    pull = c5.slider("Toleransi pullback (x ATR)", 0.3, 1.5, 0.8, 0.1)
    rr = c6.slider("Rasio TP : SL", 1.0, 3.0, 1.5, 0.25)
    c7, c8, c9 = st.columns(3)
    tz_name = c7.selectbox("Zona waktu", list(TZ_OPT), index=1)
    pakai_sesi = c8.toggle("Hanya sesi London-NY", value=True)
    auto = c9.toggle("Auto-refresh 15 detik", value=True)
    adv = st.checkbox("Pengaturan lanjutan")
    if adv:
        a1, a2, a3, a4 = st.columns(4)
        contract = a1.number_input("Oz per 1.0 lot", value=100.0 if tipe.startswith("Standard") else 1.0, min_value=0.1)
        min_lot = a2.number_input("Lot minimum", value=0.01, min_value=0.001, format="%.3f")
        lot_step = a3.number_input("Kelipatan lot", value=0.01, min_value=0.001, format="%.3f")
        spread = a4.number_input("Spread+slippage ($/oz)", value=0.30, min_value=0.0, step=0.05)
        a5, a6, a7 = st.columns(3)
        sl_atr = a5.slider("SL (x ATR M5)", 1.0, 3.0, 1.5, 0.25)
        adx_min = a6.slider("ADX minimum", 10, 30, 18)
        max_loss = a7.slider("Stop hari ini setelah N loss", 1, 5, 2)
        a8, a9, a10 = st.columns(3)
        slip_sl = a8.number_input("Slippage saat SL ($/oz)", value=0.15, min_value=0.0, step=0.05)
        spike_mult = a9.slider("Lonjakan = candle ≥ N× ATR", 2.0, 6.0, 3.0, 0.5)
        use_spike = a10.toggle("Tahan sinyal setelah lonjakan", value=True)
    else:
        contract = 100.0 if tipe.startswith("Standard") else 1.0
        min_lot, lot_step, spread, sl_atr, adx_min, max_loss = 0.01, 0.01, 0.30, 1.5, 18, 2
        slip_sl, spike_mult, use_spike = 0.15, 3.0, True
    st.caption("Ukuran kontrak dan lot minimum berbeda tiap broker. Cocokkan dengan spesifikasi akun Anda.")

P = E.Params(min_score=min_skor, adx_min=float(adx_min), pullback_atr=pull, sl_atr=sl_atr, rr=rr,
             spread=spread, use_session=pakai_sesi, max_losses_day=max_loss, capital=modal,
             risk_pct=risiko, contract=contract, min_lot=min_lot, lot_step=lot_step,
             slip_sl=slip_sl, use_spike=use_spike, spike_mult=spike_mult)
TZ = TZ_OPT[tz_name]
MAX_RISK = modal * risiko / 100.0

tab_live, tab_bt, tab_journal, tab_news, tab_guide = st.tabs(
    ["📡 Live", "🧪 Backtest", "📒 Jurnal", "📰 Berita", "📖 Panduan"])


# =============================================================================
# KOMPONEN TAMPILAN
# =============================================================================
def dir_txt(v):
    return {1: ("Naik ▲", "buy"), -1: ("Turun ▼", "sell"), 0: ("Netral", "")}[v]


def chain_html(s):
    order = ["m15", "m5", "m1"]
    br = s["broken"]

    def state(k):
        if br is None:
            return "pass"
        if br in ("sesi", "guard"):
            return "idle" if k == "m1" else "pass"
        b, i = order.index(br), order.index(k)
        return "pass" if i < b else ("fail" if i == b else "idle")

    chips = {"pass": "✓ lolos", "fail": "✕ berhenti di sini", "idle": "belum dicek"}
    rows = []
    for k, tf, role, v, extra in (
        ("m15", "M15", "Arah besar", s["b15"], ""),
        ("m5", "M5", "Setup", s["b5"], f" · skor {s['score']}%" if s["side"] and s["b5"] == s["side"] else ""),
        ("m1", "M1", "Pemicu", s["m1"], ""),
    ):
        txt, cls = dir_txt(v)
        stt = state(k)
        rows.append(f'<div class="node {stt}"><div class="tf">{tf}<span class="role">{role}</span></div>'
                    f'<div class="dir {cls}">{txt}{extra}</div><span class="chip {stt}">{chips[stt]}</span></div>')
    return h('<div class="chain"><div class="ttl">Rantai keputusan - timeframe tinggi selalu menang</div>'
             + "".join(rows) + "</div>")


def hero_html(s):
    bar = ""
    if s["side"] != 0:
        bar = (f'<div class="bar"><i style="width:{s["score"]}%"></i></div>'
               f'<div class="barlbl"><span>Skor setup M5: <b>{s["score"]}%</b></span>'
               f'<span>minimum {P.min_score}%</span></div>')
    return h(f'<div class="hero {s["tone"]}"><div class="st">{s["status"]}</div>'
             f'<div class="why">{s["why"]}</div>{bar}</div>')


def grid_html(s):
    lim = P.pullback_atr * s["atr5"]
    cells = [
        ("Harga terakhir", money(s["last_close"]), "dari candle M1"),
        ("Jarak ke EMA 50 (M5)", money(s["dist50"]), f"batas {money(lim)}"),
        ("ATR M5", money(s["atr5"]), f"SL ≈ {money(P.sl_atr * s['atr5'])}"),
        ("ADX M5", f"{s['adx5']:.1f}", f"minimum {P.adx_min:.0f}"),
        ("RSI M5", f"{s['rsi5']:.1f}", "zona sehat 45-68 / 32-55"),
    ]
    return h('<div class="grid">' + "".join(
        f'<div class="cell"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{sub}</div></div>'
        for k, v, sub in cells) + "</div>")


def plan_html(pl, ready):
    col = BUY if pl["side"] == "BUY" else SELL
    ttl = "Rencana trade" if ready else "Rencana jika entry terpicu"
    risk_pct = pl["real_risk"] / modal * 100
    notes = ""
    if pl["too_small"]:
        lvl = "bad" if pl["real_risk"] > 3 * MAX_RISK else "warn"
        notes += (f'<div class="note {lvl}">Modal terlalu kecil untuk risiko {risiko:g}%. Lot ideal '
                  f'{pl["raw"]:.4f}, tetapi lot minimum {P.min_lot:g}. Risiko nyata jadi '
                  f'<b>{money(pl["real_risk"])} ({risk_pct:.1f}% modal)</b>, bukan {money(MAX_RISK)}. '
                  f'Pertimbangkan akun Cent atau lewati sinyal saat ATR sedang besar.</div>')
    notes += (f'<div class="note info">Setelah {P.max_losses_day} loss dalam sehari, berhenti. '
              f'Jangan menambah lot untuk balas kerugian.</div>')
    return h(f'<div class="plan"><h4 style="color:{col}">{pl["side"]} · {ttl}</h4>'
             f'<div class="row"><span>Entry (kira-kira)</span><span>{money(pl["entry"])}</span></div>'
             f'<div class="row"><span>Stop loss</span><span>{money(pl["sl"])} · jarak {money(pl["sl_dist"])}</span></div>'
             f'<div class="row"><span>Take profit</span><span>{money(pl["tp"])}</span></div>'
             f'<div class="row"><span>Lot</span><span><b>{pl["lot"]:g}</b></span></div>'
             f'<div class="row"><span>Risiko nyata</span><span>{money(pl["real_risk"])} ({risk_pct:.1f}%)</span></div>'
             f'{notes}</div>')


def make_chart(frames, tf, plan):
    n = {"1m": 120, "5m": 90, "15m": 80}[tf]
    d = frames[tf].tail(n)
    lab = d.index.tz_convert(TZ).strftime("%d/%m %H:%M")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=lab, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                 increasing_line_color=BUY, decreasing_line_color=SELL,
                                 increasing_fillcolor=BUY, decreasing_fillcolor=SELL,
                                 name="XAUUSD", showlegend=False), row=1, col=1)
    for col, color, nm in (("EMA20", "#8E94A3", "EMA 20"), ("EMA50", GOLD, "EMA 50"), ("EMA200", "#6FA8DC", "EMA 200")):
        fig.add_trace(go.Scatter(x=lab, y=d[col], mode="lines", name=nm, line=dict(color=color, width=1.4)),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=lab, y=d["RSI"], mode="lines", name="RSI", showlegend=False,
                             line=dict(color="#B7A6E8", width=1.4)), row=2, col=1)
    for y in (30, 70):
        fig.add_hline(y=y, line=dict(color="rgba(255,255,255,.18)", width=1, dash="dot"), row=2, col=1)
    lo, hi = float(d["Low"].min()), float(d["High"].max())
    if plan:
        for key, color, nm in (("sl", SELL, "SL"), ("tp", BUY, "TP")):
            fig.add_hline(y=plan[key], line=dict(color=color, width=1.2, dash="dash"), row=1, col=1,
                          annotation_text=nm, annotation_position="right", annotation_font_color=color)
            lo, hi = min(lo, plan[key]), max(hi, plan[key])
    pad = (hi - lo) * 0.06
    fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1, tickvals=[30, 70])
    fig.update_xaxes(type="category", nticks=5, rangeslider_visible=False, showgrid=False, fixedrange=True)
    fig.update_yaxes(fixedrange=True, gridcolor="rgba(255,255,255,.06)")
    fig.update_layout(template="plotly_dark", height=470, margin=dict(l=2, r=2, t=6, b=2),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", dragmode=False,
                      legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)),
                      font=dict(family="Manrope, system-ui, sans-serif"))
    return fig


# =============================================================================
# TAB BERITA (jadwal manual) - dieksekusi sebelum Live supaya jadwal terbaru terbaca
# =============================================================================
EV_COLS = ["Berita", "Waktu", "Sebelum (mnt)", "Sesudah (mnt)"]


def empty_events():
    return pd.DataFrame({"Berita": pd.Series(dtype="object"), "Waktu": pd.Series(dtype="datetime64[ns]"),
                         "Sebelum (mnt)": pd.Series(dtype="int64"), "Sesudah (mnt)": pd.Series(dtype="int64")})


for k_, v_ in (("events_base", None), ("events", None), ("ev_gen", 0), ("ev_up_id", None)):
    if k_ not in st.session_state:
        st.session_state[k_] = empty_events() if k_ in ("events_base", "events") else v_


def events_utc():
    out = []
    ev = st.session_state.get("events")
    if ev is None or len(ev) == 0:
        return out
    for _, r in ev.iterrows():
        if pd.isna(r["Waktu"]) or not str(r["Berita"]).strip() or str(r["Berita"]) == "nan":
            continue
        t = pd.Timestamp(r["Waktu"]).tz_localize(TZ).tz_convert("UTC")
        b = 30 if pd.isna(r["Sebelum (mnt)"]) else int(r["Sebelum (mnt)"])
        a = 30 if pd.isna(r["Sesudah (mnt)"]) else int(r["Sesudah (mnt)"])
        out.append({"name": str(r["Berita"]), "time": t, "before": b, "after": a})
    return out


def fmt_menit(m):
    return f"{m:.0f} menit" if m < 60 else f"{int(m // 60)} jam {int(m % 60)} menit"


with tab_news:
    st.markdown('<p class="sub">Isi jadwal berita besar (NFP, CPI, FOMC, pidato The Fed). Selama jendela aktif, '
                'sinyal ditahan. Jadwalnya cek di kalender ekonomi, filter dampak tinggi untuk USD.</p>',
                unsafe_allow_html=True)
    ed = st.data_editor(
        st.session_state["events_base"], num_rows="dynamic", hide_index=True,
        key=f"ev_editor_{st.session_state['ev_gen']}",
        column_config={
            "Berita": st.column_config.TextColumn("Berita", width="medium"),
            "Waktu": st.column_config.DatetimeColumn(f"Waktu ({tz_name.split()[0]})", format="DD/MM/YYYY HH:mm", step=60),
            "Sebelum (mnt)": st.column_config.NumberColumn("Tahan sebelum", min_value=0, max_value=240, default=30),
            "Sesudah (mnt)": st.column_config.NumberColumn("Tahan sesudah", min_value=0, max_value=240, default=30),
        })
    st.session_state["events"] = ed
    n_ev = len(events_utc())
    st.markdown(f'<div class="note info">{n_ev} jadwal aktif. Waktu dibaca dalam {tz_name}. '
                'Jadwal hilang jika aplikasi restart, jadi unduh file untuk menyimpannya.</div>',
                unsafe_allow_html=True)
    n1, n2 = st.columns(2)
    n1.download_button("⬇ Simpan jadwal (CSV)", ed.to_csv(index=False).encode(), "jadwal_berita.csv", "text/csv")
    up = n2.file_uploader("Muat jadwal (CSV)", type="csv", key="ev_upload", label_visibility="collapsed")
    if up is not None and st.session_state["ev_up_id"] != (up.name, up.size):
        loaded = False
        try:
            df_up = pd.read_csv(up, parse_dates=["Waktu"])
            if set(EV_COLS) <= set(df_up.columns):
                st.session_state["events_base"] = df_up[EV_COLS]
                st.session_state["events"] = df_up[EV_COLS]
                st.session_state["ev_up_id"] = (up.name, up.size)
                st.session_state["ev_gen"] += 1
                loaded = True
        except Exception:
            pass
        if loaded:
            st.rerun()
        else:
            st.error("File tidak cocok. Pakai file yang diunduh dari tombol Simpan jadwal.")
    st.markdown('<div class="note warn">Penjaga ini mencegah sinyal baru, tetapi tidak melindungi posisi yang '
                'sudah terbuka. Saat berita besar, SL bisa terisi jauh dari harga yang Anda pasang. '
                'Jangan memegang posisi menjelang rilis.</div>', unsafe_allow_html=True)


# =============================================================================
# TAB LIVE
# =============================================================================
LIVE_PERIODS = {"1m": "5d", "5m": "5d", "15m": "15d"}


def _live():
    frames, src = load_set(LIVE_PERIODS)
    if frames is None:
        st.error("Data pasar belum bisa diambil. Periksa koneksi internet, lalu muat ulang halaman.")
        return
    now = pd.Timestamp.now(tz="UTC")
    s = E.live_snapshot(frames, P, now, MAX_RISK)
    if "error" in s:
        st.warning(s["error"])
        return

    age = (now - s["last_bar_end"]).total_seconds() / 60
    local_t = s["last_bar_end"].tz_convert(TZ).strftime("%H:%M")
    if age > 12:
        s = {**s, "status": "DATA TERTUNDA", "tone": "wait", "broken": "m15", "plan": None,
             "why": f"Candle terakhir {age:.0f} menit lalu ({local_t}). Pasar kemungkinan tutup. "
                    "Jangan entry berdasarkan data lama."}

    guard = E.news_guard(now, events_utc(), frames["1m"], P)
    if guard["blocked"] and age <= 12:
        until = guard["until"].tz_convert(TZ).strftime("%H:%M")
        s = {**s, "status": "TRADING DITAHAN" if guard["kind"] == "news" else "VOLATILITAS EKSTREM",
             "tone": "wait", "broken": "guard", "plan": None,
             "why": f"{guard['reason']}. Sinyal baru diizinkan lagi sekitar {until} {tz_name.split()[0]}. "
                    "Jika ada posisi terbuka, periksa SL Anda."}

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(hero_html(s), unsafe_allow_html=True)
    with right:
        st.markdown(chain_html(s), unsafe_allow_html=True)

    nx = guard["next"]
    if nx and not guard["blocked"] and nx["minutes"] < 1440:
        st.markdown(f'<div class="note info">Berita berikutnya: <b>{nx["name"]}</b> dalam {fmt_menit(nx["minutes"])}.</div>',
                    unsafe_allow_html=True)
    st.markdown(grid_html(s), unsafe_allow_html=True)

    if s["plan"]:
        st.markdown(plan_html(s["plan"], s["status"].startswith("ENTRY")), unsafe_allow_html=True)

    tf_lbl = st.radio("Grafik", ["M1", "M5", "M15"], index=1, horizontal=True, key="chart_tf",
                      label_visibility="collapsed")
    plan_for_chart = s["plan"] if s["status"].startswith("ENTRY") else None
    st.plotly_chart(make_chart(frames, {"M1": "1m", "M5": "5m", "M15": "15m"}[tf_lbl], plan_for_chart),
                    config={"displayModeBar": False})
    st.caption(f"Grafik hanya tampilan. Sinyal selalu dihitung dari M15→M5→M1, apa pun grafik yang dipilih. "
               f"Candle terakhir tutup {local_t} {tz_name.split()[0]} · sumber {src} · "
               f"{'refresh otomatis' if auto else 'refresh manual'}")
    if src == "GC=F":
        st.caption("GC=F adalah futures emas. Harganya bisa beda beberapa dolar dari XAUUSD spot di broker. "
                   "Pakai untuk arah dan struktur, bukan untuk angka entry persis.")


with tab_live:
    (st.fragment(run_every=15)(_live) if auto else st.fragment(_live))()


# =============================================================================
# TAB BACKTEST
# =============================================================================
def verdict(hier, oos):
    n = hier.get("trades", 0)
    if n < 30:
        return "warn", (f"Sampel terlalu kecil ({n} trade). Angka di bawah belum bisa dipercaya; "
                        "perpanjang periode data atau longgarkan skor minimum.")
    if hier["exp_R"] <= 0 or hier["pf"] < 1:
        return "bad", ("Belum menguntungkan setelah spread. Jangan pakai uang asli dengan pengaturan ini.")
    if oos.get("trades", 0) >= 8 and oos["exp_R"] <= 0:
        return "warn", ("Untung di periode awal tetapi tidak bertahan di periode akhir. "
                        "Tanda khas overfitting; strategi belum terbukti.")
    return "good", ("Lolos uji awal pada data ini. Ini bukan jaminan: uji lagi di akun demo "
                    "minimal 2 minggu sebelum memakai uang asli.")


def fmt_stats(st_, label):
    if st_.get("trades", 0) == 0:
        return {"Mode": label, "Trade": 0}
    pf = st_["pf"]
    return {"Mode": label, "Trade": st_["trades"], "Win %": round(st_["winrate"], 1),
            "Profit factor": "∞" if pf == float("inf") else round(pf, 2),
            "Expectancy (R)": round(st_["exp_R"], 2), "Hasil %": round(st_["net_pct"], 1),
            "Max drawdown %": round(st_["max_dd"], 1), "Rugi beruntun": st_["max_losing_streak"],
            "Risiko rata-rata %": round(st_["avg_risk_pct"], 2)}


def run_backtest(entry_label, days):
    entry = "5m" if entry_label.startswith("M5") else "1m"
    per = {"15m": f"{days}d", "5m": f"{days}d"}
    if entry == "1m":
        per = {"15m": "59d", "5m": "30d", "1m": "7d"}
    frames, src = load_set(per)
    if frames is None:
        return None
    out = {"entry": entry, "src": src, "P": P, "days": days if entry == "5m" else 7}
    for key, hier in (("hier", True), ("naive", False)):
        tr, ruined = E.run_full(frames, entry, P, hier)
        out[key] = {"tr": tr, "stats": E.stats(tr, P.capital, ruined)}
    e_idx = frames[entry].index
    out["is"], out["oos"], out["cut"] = E.split_in_out(out["hier"]["tr"], e_idx)
    out["frames"] = frames
    return out


with tab_bt:
    st.markdown('<p class="sub">Menguji aturan yang sama persis dengan sinyal live pada data historis. '
                'Entry di open candle berikutnya, SL dianggap kena lebih dulu bila SL dan TP tersentuh '
                'dalam satu candle, dan spread dihitung.</p>', unsafe_allow_html=True)
    b1, b2 = st.columns([1, 1])
    entry_label = b1.radio("Timeframe entry", ["M5 (disarankan)", "M1 (hanya 7 hari)"], horizontal=False)
    days = b2.select_slider("Panjang data (M5)", options=[7, 14, 30, 45, 59], value=30,
                            disabled=entry_label.startswith("M1"))
    if entry_label.startswith("M1"):
        st.markdown('<div class="note warn">Yahoo Finance hanya menyimpan 7 hari data M1. Hasilnya cepat '
                    'berubah dan sampelnya kecil, jadi jangan dijadikan satu-satunya dasar.</div>',
                    unsafe_allow_html=True)
    if st.button("▶ Jalankan backtest", type="primary"):
        with st.spinner("Mengambil data dan menguji aturan..."):
            res = run_backtest(entry_label, days)
        if res is None:
            st.error("Data tidak cukup/gagal diambil. Coba periode lebih pendek atau ulangi beberapa saat lagi.")
        else:
            st.session_state["bt"] = res

    res = st.session_state.get("bt")
    if res:
        hs, ns = res["hier"]["stats"], res["naive"]["stats"]
        stI, stO = E.stats_R_only(res["is"]), E.stats_R_only(res["oos"])
        kind, msg = verdict(hs, stO)
        st.markdown(f'<div class="note {kind}"><b>Kesimpulan:</b> {msg}</div>', unsafe_allow_html=True)
        if hs.get("ruined"):
            st.markdown('<div class="note bad">Modal habis di tengah pengujian.</div>', unsafe_allow_html=True)
        if hs.get("trades", 0) and hs["avg_risk_pct"] > 2 * P.risk_pct:
            st.markdown(f'<div class="note warn">Karena lot minimum, risiko nyata rata-rata '
                        f'<b>{hs["avg_risk_pct"]:.1f}%</b> per trade, bukan {P.risk_pct:g}%. '
                        'Inilah yang membuat modal kecil cepat tergerus.</div>', unsafe_allow_html=True)

        st.subheader("M1 vs M5 yang bertentangan: seberapa berpengaruh?")
        st.caption("Naif = hanya melihat satu timeframe (kondisi yang membingungkan). "
                   "Hierarki = M15 menentukan arah, M5 setup, M1 pemicu.")
        st.dataframe(pd.DataFrame([fmt_stats(ns, "Naif (1 timeframe)"), fmt_stats(hs, "Hierarki M15→M5→M1")]),
                     hide_index=True)

        tr = res["hier"]["tr"]
        if len(tr):
            fig = go.Figure()
            n_tr = res["naive"]["tr"]
            if len(n_tr):
                fig.add_trace(go.Scatter(x=n_tr["exit_time"].dt.tz_convert(TZ).dt.tz_localize(None),
                                         y=n_tr["equity"], name="Naif", mode="lines",
                                         line=dict(color=MUTED, width=1.2, dash="dot")))
            fig.add_trace(go.Scatter(x=tr["exit_time"].dt.tz_convert(TZ).dt.tz_localize(None), y=tr["equity"],
                                     name="Hierarki", mode="lines", line=dict(color=GOLD, width=2.2)))
            fig.add_hline(y=P.capital, line=dict(color="rgba(255,255,255,.2)", dash="dot"))
            fig.update_layout(template="plotly_dark", height=330, margin=dict(l=2, r=2, t=8, b=2),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", y=1.1, x=0), dragmode=False,
                              yaxis=dict(title="Ekuitas ($)", fixedrange=True, gridcolor="rgba(255,255,255,.06)"),
                              xaxis=dict(fixedrange=True, showgrid=False))
            st.plotly_chart(fig, config={"displayModeBar": False})

            st.subheader("Apakah bertahan di data yang tidak dipakai 'belajar'?")
            cut = res["cut"].tz_convert(TZ).strftime("%d/%m %H:%M")
            st.caption(f"Periode awal (70%) vs akhir (30%, mulai {cut}). Strategi sehat terlihat mirip di keduanya.")
            def row(lbl, x):
                if x.get("trades", 0) == 0:
                    return {"Periode": lbl, "Trade": 0}
                return {"Periode": lbl, "Trade": x["trades"], "Win %": round(x["winrate"], 1),
                        "Profit factor": "∞" if x["pf"] == float("inf") else round(x["pf"], 2),
                        "Expectancy (R)": round(x["exp_R"], 2)}
            st.dataframe(pd.DataFrame([row("Awal (in-sample)", stI), row("Akhir (out-of-sample)", stO)]),
                         hide_index=True)

            st.subheader("Seberapa aman modal Anda? (Monte Carlo)")
            n_mc = st.select_slider("Panjang simulasi (jumlah trade ke depan)", options=[50, 100, 200], value=100,
                                    key="mc_n")
            mc = E.monte_carlo(tr, res["P"], n_mc, 1000)
            if mc is None:
                st.info("Butuh minimal 10 trade untuk simulasi. Perpanjang data atau turunkan skor minimum.")
            else:
                if len(tr) < 30:
                    st.markdown(f'<div class="note warn">Baru {len(tr)} trade sebagai dasar simulasi. '
                                'Hasil di bawah masih kasar.</div>', unsafe_allow_html=True)
                bad = "color:#E4574F" if mc["p_near_mc"] >= 10 else ""
                cells = [
                    ("Peluang modal turun 30%", f"{mc['p_loss30']:.0f}%", "pada titik mana pun"),
                    ("Peluang modal tinggal 20%", f"<span style='{bad}'>{mc['p_near_mc']:.0f}%</span>", "mendekati margin call"),
                    ("Modal akhir (median)", money(mc["median_final"], 0), f"dari {money(res['P'].capital, 0)}"),
                    ("Skenario buruk (5%)", money(mc["p5_final"], 0), f"skenario baik {money(mc['p95_final'], 0)}"),
                    ("Drawdown terburuk", f"{mc['dd95']:.0f}%", "batas 95% simulasi"),
                    ("Peluang akhir untung", f"{mc['p_profit']:.0f}%", f"setelah {n_mc} trade"),
                ]
                st.markdown(h('<div class="grid">' + "".join(
                    f'<div class="cell"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{sub}</div></div>'
                    for k, v, sub in cells) + "</div>"), unsafe_allow_html=True)
                xs = list(range(n_mc + 1))
                fm = go.Figure()
                fm.add_trace(go.Scatter(x=xs, y=mc["bands"][2], mode="lines", line=dict(width=0),
                                        name="Skenario baik (95%)", hoverinfo="skip"))
                fm.add_trace(go.Scatter(x=xs, y=mc["bands"][0], mode="lines", line=dict(width=0), fill="tonexty",
                                        fillcolor="rgba(216,169,58,.18)", name="Rentang 90%", hoverinfo="skip"))
                fm.add_trace(go.Scatter(x=xs, y=mc["bands"][1], mode="lines", name="Median",
                                        line=dict(color=GOLD, width=2.2)))
                fm.add_hline(y=res["P"].capital, line=dict(color="rgba(255,255,255,.25)", dash="dot"))
                fm.update_layout(template="plotly_dark", height=300, margin=dict(l=2, r=2, t=8, b=2),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", dragmode=False,
                                 legend=dict(orientation="h", y=1.12, x=0),
                                 xaxis=dict(title="Trade ke-", fixedrange=True, showgrid=False),
                                 yaxis=dict(title="Ekuitas ($)", fixedrange=True, gridcolor="rgba(255,255,255,.06)"))
                st.plotly_chart(fm, config={"displayModeBar": False})
                if mc["p_near_mc"] >= 10 or mc["p_loss30"] >= 40:
                    st.markdown('<div class="note bad">Risiko tinggi: ada peluang berarti modal Anda tergerus '
                                'dalam. Turunkan risiko per trade, naikkan skor minimum, atau pakai akun Cent '
                                'sebelum trading nyata.</div>', unsafe_allow_html=True)
                st.caption(f"Dari {len(tr)} trade backtest, diambil acak (bootstrap) 1.000 kali. Simulasi "
                           "mengasumsikan masa depan mirip masa lalu, yang tidak selalu benar.")

            with st.expander("Uji ketahanan parameter (3×3)"):
                st.caption("Jika hanya satu sel yang untung, hasilnya kebetulan. Cari area luas yang konsisten positif.")
                if st.button("Jalankan uji ketahanan"):
                    with st.spinner("Menguji 9 kombinasi..."):
                        sens = E.sensitivity(res["frames"], res["entry"], P).round(2)
                    st.dataframe(sens, hide_index=True)

            with st.expander("Daftar trade"):
                show = tr.copy()
                for c in ("entry_time", "exit_time"):
                    show[c] = show[c].dt.tz_convert(TZ).dt.strftime("%d/%m %H:%M")
                show = show[["entry_time", "side", "entry", "sl", "tp", "lot", "exit", "reason", "R", "pnl", "equity"]]
                st.dataframe(show.round(2).iloc[::-1], hide_index=True, height=320)
                st.download_button("⬇ Unduh CSV", show.round(4).to_csv(index=False).encode(),
                                   "backtest_xauusd.csv", "text/csv")
        else:
            st.info("Tidak ada trade yang memenuhi aturan pada periode ini. Turunkan skor minimum atau perpanjang data.")

        st.caption(f"Sumber {res['src']} · entry {res['entry'].upper()} · {res['days']} hari · "
                   f"spread {money(P.spread)}/oz · data futures bisa berbeda dari spot broker.")


# =============================================================================
# TAB JURNAL
# =============================================================================
def empty_journal():
    return pd.DataFrame({"waktu": pd.Series(dtype="object"), "arah": pd.Series(dtype="object"),
                         "entry": pd.Series(dtype="float64"), "sl": pd.Series(dtype="float64"),
                         "exit": pd.Series(dtype="float64"), "lot": pd.Series(dtype="float64"),
                         "catatan": pd.Series(dtype="object")})


for k_ in ("journal_base", "journal"):
    if k_ not in st.session_state:
        st.session_state[k_] = empty_journal()
for k_, v_ in (("jr_gen", 0), ("jr_up_id", None)):
    if k_ not in st.session_state:
        st.session_state[k_] = v_

with tab_journal:
    st.markdown('<p class="sub">Catat trade dari akun demo (atau nyata) dengan harga entry, SL, dan harga keluar '
                'sesungguhnya. Aplikasi menghitung R dan membandingkannya dengan backtest.</p>',
                unsafe_allow_html=True)
    jed = st.data_editor(
        st.session_state["journal_base"], num_rows="dynamic", hide_index=True,
        key=f"jr_editor_{st.session_state['jr_gen']}",
        column_config={
            "waktu": st.column_config.TextColumn("Waktu"),
            "arah": st.column_config.SelectboxColumn("Arah", options=["BUY", "SELL"]),
            "entry": st.column_config.NumberColumn("Entry", format="%.2f"),
            "sl": st.column_config.NumberColumn("SL", format="%.2f"),
            "exit": st.column_config.NumberColumn("Keluar", format="%.2f"),
            "lot": st.column_config.NumberColumn("Lot", format="%.2f"),
            "catatan": st.column_config.TextColumn("Catatan"),
        })
    st.session_state["journal"] = jed
    j1, j2 = st.columns(2)
    j1.download_button("⬇ Simpan jurnal (CSV)", jed.to_csv(index=False).encode(), "jurnal_trade.csv", "text/csv")
    jup = j2.file_uploader("Muat jurnal (CSV)", type="csv", key="jr_upload", label_visibility="collapsed")
    if jup is not None and st.session_state["jr_up_id"] != (jup.name, jup.size):
        loaded = False
        try:
            dj = pd.read_csv(jup)
            if set(E.JOURNAL_COLS) <= set(dj.columns):
                st.session_state["journal_base"] = dj[E.JOURNAL_COLS]
                st.session_state["journal"] = dj[E.JOURNAL_COLS]
                st.session_state["jr_up_id"] = (jup.name, jup.size)
                st.session_state["jr_gen"] += 1
                loaded = True
        except Exception:
            pass
        if loaded:
            st.rerun()
        else:
            st.error("File tidak cocok. Pakai file yang diunduh dari tombol Simpan jurnal.")

    js = E.journal_stats(jed)
    if js["trades"] == 0:
        st.markdown('<div class="note info">Belum ada trade lengkap. Isi arah, entry, SL, dan harga keluar.</div>',
                    unsafe_allow_html=True)
    else:
        bt_ = st.session_state.get("bt")
        hs_ = bt_["hier"]["stats"] if bt_ and bt_["hier"]["stats"].get("trades") else None
        rows = [("Jumlah trade", "-" if not hs_ else str(hs_["trades"]), str(js["trades"])),
                ("Win rate", "-" if not hs_ else f"{hs_['winrate']:.0f}%", f"{js['winrate']:.0f}%"),
                ("Expectancy (R)", "-" if not hs_ else f"{hs_['exp_R']:+.2f}", f"{js['exp_R']:+.2f}"),
                ("Rata-rata R saat rugi", "-" if not hs_ else f"{hs_['avg_loss_R']:.2f}", f"{js['avg_loss_R']:.2f}")]
        st.dataframe(pd.DataFrame(rows, columns=["", "Backtest", "Jurnal Anda"]), hide_index=True)
        if js["trades"] < 20:
            st.markdown(f'<div class="note info">Baru {js["trades"]} trade. Butuh minimal 20 agar perbandingan '
                        'berarti.</div>', unsafe_allow_html=True)
        elif hs_ is None:
            st.markdown('<div class="note info">Jalankan backtest dulu agar bisa dibandingkan.</div>',
                        unsafe_allow_html=True)
        else:
            if js["avg_loss_R"] < hs_["avg_loss_R"] - 0.2:
                st.markdown('<div class="note warn">Rugi Anda lebih dalam dari backtest. Biasanya karena slippage, '
                            'spread broker lebih lebar, atau SL digeser. Periksa jam trade dan broker.</div>',
                            unsafe_allow_html=True)
            if js["exp_R"] < hs_["exp_R"] - 0.15:
                st.markdown('<div class="note bad">Hasil nyata jauh di bawah backtest. Jangan naikkan lot. '
                            'Cari penyebabnya dulu: eksekusi, jam trade, atau backtest terlalu optimis.</div>',
                            unsafe_allow_html=True)
            elif js["exp_R"] > 0:
                st.markdown('<div class="note good">Hasil nyata sejalan dengan backtest dan positif. Teruskan di '
                            'demo sampai sampel cukup besar.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="note warn">Hasil nyata sejalan dengan backtest, tetapi expectancy-nya '
                            'masih negatif. Strategi belum menguntungkan; jangan pakai uang asli.</div>',
                            unsafe_allow_html=True)
    st.caption("R = (harga keluar - entry) / jarak entry ke SL, dengan tanda sesuai arah. "
               "Jurnal hilang jika aplikasi restart, jadi simpan CSV secara rutin.")


# =============================================================================
# TAB PANDUAN
# =============================================================================
def build_pdf() -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    ss = getSampleStyleSheet()
    t1 = ParagraphStyle("t1", parent=ss["Heading1"], fontSize=18, textColor=colors.HexColor("#1E293B"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, textColor=colors.HexColor("#0F172A"))
    bd = ParagraphStyle("bd", parent=ss["Normal"], fontSize=9.5, leading=14, textColor=colors.HexColor("#334155"))
    story = [Paragraph("XAUUSD Scalper Pro v3 - Panduan", t1), HRFlowable(width="100%", color=colors.HexColor("#CBD5E1")),
             Paragraph("1. Cara kerja: hierarki timeframe", h2),
             Paragraph("M15 menentukan arah (hanya BUY bila M15 naik, hanya SELL bila M15 turun). M5 mencari setup: "
                       "harga pullback ke EMA 50, ADX cukup, MACD dan RSI searah. M1 hanya pemicu timing. "
                       "Bila M1 berlawanan dengan M5/M15, M1 diabaikan.", bd),
             Paragraph("2. Bobot skor setup (M5)", h2)]
    rows = [["Komponen", "Syarat BUY", "Syarat SELL", "Bobot"],
            ["Tren (wajib)", "Harga > EMA50 > EMA200", "Harga < EMA50 < EMA200", "30"],
            ["Pullback (wajib)", "Low 3 candle dekat EMA50", "High 3 candle dekat EMA50", "20"],
            ["MACD", "Histogram > 0 dan naik", "Histogram < 0 dan turun", "20"],
            ["ADX", "ADX >= minimum", "ADX >= minimum", "15"],
            ["RSI", "45 - 68", "32 - 55", "15"]]
    t = Table(rows, colWidths=[90, 170, 170, 50])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                           ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#CBD5E1"))]))
    story += [t, Spacer(1, 6), Paragraph("3. Manajemen risiko", h2),
              Paragraph("Maks rugi = Modal x Risiko%. Jarak SL = 1.5 x ATR M5. Lot = Maks rugi / (Jarak SL x oz per lot). "
                        "Jika hasil lot di bawah lot minimum broker, risiko nyata lebih besar dari yang direncanakan. "
                        "Berhenti trading setelah 2 loss dalam sehari.", bd),
              Paragraph("4. Batasan", h2),
              Paragraph("Data dari Yahoo Finance (futures GC=F) bisa berbeda dari spot broker. Backtest bukan jaminan. "
                        "Uji di akun demo sebelum memakai uang asli.", bd)]
    doc.build(story)
    return buf.getvalue()


with tab_guide:
    st.markdown("### Kalau M1 Buy tetapi M5 Sell?")
    st.markdown(h('<div class="note good"><b>Ikuti timeframe yang lebih tinggi.</b> M1 yang berlawanan dengan M5 '
                  'hampir selalu hanya koreksi kecil. Aplikasi ini sekarang mengunci keputusan: '
                  'M15 → M5 → M1, dan M1 tidak bisa membatalkan M5.</div>'), unsafe_allow_html=True)
    st.table(pd.DataFrame({
        "M15": ["Turun", "Turun", "Turun", "Naik", "Netral"],
        "M5": ["Turun", "Turun", "Naik", "Naik", "apa saja"],
        "M1": ["Turun", "Naik", "apa saja", "Naik", "apa saja"],
        "Keputusan": ["ENTRY SELL", "Tunggu M1 kembali turun (jangan ikut M1)", "Tidak trade",
                      "ENTRY BUY (bila pullback & skor terpenuhi)", "Tidak trade"],
    }))
    st.markdown("### Kenapa sinyal live bisa berbeda dari versi lama?")
    st.markdown(
        "- **Hanya candle yang sudah tutup** yang dipakai; candle berjalan sering berubah dan memicu sinyal palsu.\n"
        "- **RSI, ATR, ADX memakai metode Wilder** seperti di MT4/MT5/TradingView.\n"
        "- **Pullback ke EMA 50 dan tren searah bersifat wajib**, bukan sekadar poin skor.\n"
        "- **Filter ADX** menahan entry saat pasar sideways.\n"
        "- **Filter sesi** menghindari jam sepi dengan spread lebar.\n"
        "- **Lot minimum diperhitungkan**: aplikasi memberi tahu risiko nyata, bukan angka ideal.\n"
        "- **Penjaga berita**: sinyal ditahan di jendela berita yang Anda jadwalkan dan setelah candle M1 "
        "melonjak ≥ 3× ATR.\n"
        "- **Backtest memasukkan slippage SL**, jadi hasilnya tidak seoptimis versi sebelumnya.\n"
        "- **Monte Carlo** menunjukkan peluang modal tergerus; **Jurnal** membandingkan hasil nyata dengan backtest.")
    st.markdown("### Rumus manajemen risiko")
    st.code("Maks rugi ($) = Modal x Risiko %\nJarak SL ($)   = SL(x ATR) x ATR M5\n"
            "Lot            = Maks rugi / (Jarak SL x oz per lot)\n"
            "Risiko nyata   = max(Lot, Lot minimum) x oz per lot x Jarak SL", language="text")
    st.markdown("### Batasan yang perlu diketahui")
    st.markdown(
        "- Data berasal dari Yahoo Finance (futures **GC=F**), bukan feed broker Anda.\n"
        "- Backtest M1 hanya 7 hari; M5/M15 maksimal sekitar 60 hari.\n"
        "- Backtest bukan jaminan. Selalu uji di akun demo dahulu.\n"
        "- Trading emas dengan leverage dapat menghabiskan seluruh modal.")
    try:
        st.download_button("⬇ Unduh panduan PDF", build_pdf(), "XAUUSD_Scalper_Pro_Panduan.pdf",
                           "application/pdf")
    except Exception:
        st.caption("Instal reportlab untuk mengunduh PDF.")
