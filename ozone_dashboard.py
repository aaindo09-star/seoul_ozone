# -*- coding: utf-8 -*-
"""
서울 광화학 오존의 생성 조건과 생물 노출 위험 분석  (팀 오존 랩)
- 데이터: ozone_hourly.csv (에어코리아 최종확정 측정자료 + 기상청 ASOS 서울(108)을 시각 기준으로 합친 시간별 자료)
- 구조: 1) 데이터 불러오기  2) 사이드바  3) 분석 항목별 화면
"""
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="서울 오존 노출 분석 | 오존 랩", layout="wide")

# ───────── 기준값 (출처는 '처음 화면'의 표에 정리) ─────────
DATA_FILE = Path(__file__).parent / "ozone_hourly.csv"
ALERT_PPM, ENV_1H_PPM, ENV_8H_PPM = 0.12, 0.10, 0.06   # ppm
AOT_PPB = 40                                            # AOT40 기준: 40 ppb = 0.04 ppm
DAY_START, DAY_END = 8, 20      # 낮 시간: 08:00~20:00 (시작 시각 8~19시)
HUMAN_START, HUMAN_END = 9, 16  # 사람 활동 시간: 09:00~16:00 (시작 시각 9~15시)
CROP_LEVEL, FOREST_LEVEL = 3.0, 5.0   # UNECE 임계값(ppm·h): 작물 3개월, 산림 6개월
MIN_VALID = 0.75                # 월 유효 측정 비율 기준
# 측정소 유형은 직접 조사해서 채우세요. 예: STATION_GROUP = {"중구": "도심", "강서구": "외곽"}
STATION_GROUP = {}

CRITERIA = {
    "알림 기준 · 오존주의보 (1시간 0.12 ppm 이상)": ("O3", ALERT_PPM, True),
    "환경기준 · 1시간 평균 (0.1 ppm 초과)": ("O3", ENV_1H_PPM, False),
    "환경기준 · 8시간 평균 (0.06 ppm 초과)": ("O3_8h", ENV_8H_PPM, False),
}
PAGES = ["처음 화면", "① 일사·기온과 오존", "② 사람 기준 초과", "③ 식물 누적 노출 (AOT40)", "데이터 점검과 한계"]
WEEK = ["월", "화", "수", "목", "금", "토", "일"]


# ───────── 1) 데이터 불러오기 ─────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["날짜"])
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    return df


raw = load_data()


# ───────── 계산 함수 ─────────
def exceed(df, crit):
    """기준 초과 여부와 유효(값이 있는) 여부"""
    col, limit, inclusive = CRITERIA[crit]
    v = df[col]
    flag = (v >= limit) if inclusive else (v > limit)
    return flag & v.notna(), v.notna()


def human_monthly(h, crit):
    """사람 활동 시간(9~16시) 중 기준 초과: 측정소·월별 초과 시간과 일수"""
    x = h[h["시작시"].between(HUMAN_START, HUMAN_END - 1)].copy()
    x["초과"], x["유효"] = exceed(x, crit)
    x = x[x["유효"]]
    keys = ["측정소명", "연도", "월"]
    ps = x.groupby(keys).agg(초과시간=("초과", "sum"), 유효시간=("초과", "size")).reset_index()
    days = x[x["초과"]].groupby(keys)["날짜"].nunique().rename("초과일수").reset_index()
    ps = ps.merge(days, on=keys, how="left").fillna({"초과일수": 0})
    return x, ps


def aot_monthly(h):
    """낮 시간(8~20시) 40 ppb 초과분의 월별 누적(AOT40, ppm·h)"""
    x = h[h["시작시"].between(DAY_START, DAY_END - 1)].copy()
    x["초과분"] = (x["O3"] * 1000 - AOT_PPB).clip(lower=0)   # ppb 단위, 40 이하는 0
    a = x.groupby(["측정소명", "연도", "월"]).agg(초과합=("초과분", "sum"), 유효시간=("초과분", "count")).reset_index()
    a["AOT40"] = a["초과합"] / 1000                            # ppb·h → ppm·h
    a["가능시간"] = [calendar.monthrange(y, m)[1] * (DAY_END - DAY_START) for y, m in zip(a["연도"], a["월"])]
    a["유효비율"] = a["유효시간"] / a["가능시간"]
    return a


def weather_daily(df):
    w = df.drop_duplicates(["날짜", "시작시"])
    g = w.groupby("날짜")
    d = pd.DataFrame({"일최고기온": g["기온"].max(), "일최고일사": g["일사"].max(), "강수합": g["강수량"].sum(),
                      "평균풍속": g["풍속"].mean(), "평균습도": g["습도"].mean()})
    for col, name in (("기온", "기온피크시"), ("일사", "일사피크시")):
        d[name] = w.dropna(subset=[col]).sort_values(col, ascending=False).drop_duplicates("날짜").set_index("날짜")["시작시"]
    return d.reset_index()


def ozone_daily(h):
    x = h.dropna(subset=["O3"]).sort_values("O3", ascending=False).drop_duplicates(["측정소명", "날짜"])
    return x[["측정소명", "날짜", "O3", "시작시"]].rename(columns={"O3": "일최고O3", "시작시": "오존피크시"})


# ───────── 2) 사이드바 ─────────
st.sidebar.title("오존 랩 🌫️")
page = st.sidebar.radio("분석 항목 선택", PAGES)
years_all = sorted(raw["연도"].unique())
if len(years_all) > 1:
    years = st.sidebar.slider("연도 범위", int(years_all[0]), int(years_all[-1]), (int(years_all[0]), int(years_all[-1])))
else:
    years = (int(years_all[0]), int(years_all[0]))
    st.sidebar.caption(f"분석 연도: {years_all[0]}년")
months = st.sidebar.multiselect("월 선택", [5, 6, 7, 8, 9], default=[5, 6, 7, 8, 9], format_func=lambda m: f"{m}월")
st_all = sorted(raw["측정소명"].unique())
stations = st.sidebar.multiselect("측정소 (도시대기)", st_all, default=st_all)
crit = st.sidebar.selectbox("사람 기준값", list(CRITERIA))
step = st.sidebar.slider("기온 구간 간격 (℃)", 1, 5, 2)
drop_out = st.sidebar.checkbox("이상치 후보 값 제외하고 계산", value=False)
st.sidebar.caption("이상치 후보: 월·시간대별 IQR 기준. 주의보 수준의 고농도는 실제 현상일 수 있어 기본은 포함합니다.")

if not months or not stations:
    st.warning("월과 측정소를 하나 이상 선택해 주세요.")
    st.stop()

h = raw[raw["월"].isin(months) & raw["측정소명"].isin(stations) & raw["연도"].between(*years)].copy()
if drop_out:
    h.loc[h["이상치후보"] == 1, ["O3", "O3_8h"]] = np.nan

st.title("서울 오존 노출 분석 대시보드")
YEAR_TXT = f"{years_all[0]}~{years_all[-1]}년" if len(years_all) > 1 else f"{years_all[0]}년"
st.caption(f"서울 광화학 오존의 생성 조건과 생물 노출 위험 · 팀 오존 랩 · 서울 도시대기 측정소(에어코리아)와 서울 기상관측소(108번, 기상청) · {YEAR_TXT} 5~9월")

# ───────── 3) 화면 ─────────
if page == "처음 화면":
    st.subheader("큰 질문")
    st.info("서울 지역의 오존 농도는 어떤 조건에서 높아지고, 그때 생물은 얼마나 노출될까?")
    c = st.columns(3)
    c[0].markdown("**① 일사·기온**  \n일사량과 기온이 오존에 미치는 영향, 피크 시차")
    c[1].markdown("**② 사람**  \n활동 시간(9~16시)에 기준을 넘는 시간")
    c[2].markdown("**③ 식물**  \n낮 시간 40 ppb 초과 누적량(AOT40)")

    valid = h["O3"].dropna()
    top = h.loc[h["O3"].idxmax()] if len(valid) else None
    m = st.columns(4)
    m[0].metric("측정소 수", f"{h['측정소명'].nunique()}개")
    m[1].metric("분석 기간", f"{h['날짜'].min():%Y.%m.%d} ~ {h['날짜'].max():%Y.%m.%d}")
    m[2].metric("주의보 수준(0.12 ppm↑) 측정소·시간", f"{int((h['O3'] >= ALERT_PPM).sum()):,}")
    if top is not None:
        m[3].metric("최고 1시간 오존", f"{top['O3']:.3f} ppm", f"{top['측정소명']} · {top['날짜']:%m.%d} {int(top['시작시'])}시")

    st.subheader("기준값과 출처")
    st.markdown("""
| 기준 | 값 | 출처 |
|---|---|---|
| 오존주의보 / 경보 / 중대경보 | 1시간 평균 **0.12 / 0.3 / 0.5 ppm 이상** | 서울시 대기환경정보 [cleanair.seoul.go.kr/information/info13](https://cleanair.seoul.go.kr/information/info13) |
| 오존 환경기준 | 1시간 평균 **0.1 ppm 이하**, 8시간 평균 **0.06 ppm 이하** | 「환경정책기본법 시행령」 별표 1 (국가법령정보센터 law.go.kr) |
| AOT40 정의 | 낮 시간 동안 **40 ppb 초과분의 누적** (ppb·h). 유럽은 08~20시(CET)를 낮 시간으로 씀 | EEA, [AOT40 설명](https://www.eea.europa.eu/en/analysis/publications/impacts-of-air-pollution-on-ecosystems-in-europe/table-1) |
| AOT40 임계값 | 작물 **3 ppm·h**(3개월), 산림 **5 ppm·h**(6개월) | UNECE 2010 (APIS 인용: [산림](https://www.apis.ac.uk/node/863), [작물](https://www.apis.ac.uk/node/865)). 신뢰도는 '전문가 판단' 수준 |
""")
    st.caption("사람 기준은 사이드바에서 '알림 기준(주의보)'과 '환경기준'을 고를 수 있습니다. AOT40은 유럽 기준을 참고한 지표라 한국 식물의 실제 피해를 뜻하지 않습니다.")
    st.markdown(f"""
**데이터 출처**
- 한국환경공단 에어코리아, 최종확정 측정자료 (서울 도시대기 측정소, {YEAR_TXT} 5~9월, 공공누리 제3유형: 출처표시·변경금지). 이 앱의 값은 원자료를 가공한 결과입니다.
- 기상청 기상자료개방포털, 종관기상관측(ASOS) 서울(지점 108) 시간 자료 (기온, 일사, 강수량, 풍속, 습도).
""")
    st.warning("측정소 일부와 관측소 한 곳의 자료이므로 서울 전체를 대표하지 않고, 측정소 농도는 개인이 실제 마신 양이 아닙니다.")

elif page == "① 일사·기온과 오존":
    st.header("① 일사량과 기온이 오존 농도에 미치는 영향")
    wd = weather_daily(h)
    dd = ozone_daily(h).merge(wd, on="날짜", how="left")
    dd["시차"] = dd["오존피크시"] - dd["일사피크시"]

    st.subheader("하루 중 오존과 일사의 변화 (선택한 기간 평균)")
    prof_o3 = h.groupby("시작시")["O3"].mean()
    prof_sun = h.drop_duplicates(["날짜", "시작시"]).groupby("시작시")["일사"].mean()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=prof_o3.index, y=prof_o3.values, name="오존 (ppm)", mode="lines+markers"), secondary_y=False)
    fig.add_trace(go.Scatter(x=prof_sun.index, y=prof_sun.values, name="일사 (MJ/m²)", mode="lines+markers"), secondary_y=True)
    fig.update_xaxes(title_text="시각 (시작 시, 0~23시)", dtick=2)
    fig.update_yaxes(title_text="오존 (ppm)", secondary_y=False)
    fig.update_yaxes(title_text="일사 (MJ/m²)", secondary_y=True)
    st.plotly_chart(fig)
    if prof_o3.notna().any() and prof_sun.notna().any():
        t_o, t_s = int(prof_o3.idxmax()), int(prof_sun.idxmax())
        st.markdown(f"**평균 곡선에서** 일사는 **{t_s}시**, 오존은 **{t_o}시**에 가장 높습니다. 오존 피크가 일사 피크보다 **{t_o - t_s}시간** 늦습니다.")

    st.subheader("날마다의 시차 (일 최고 오존 시각 − 일 최고 일사 시각)")
    lag = dd.dropna(subset=["시차"])
    if len(lag):
        st.plotly_chart(px.histogram(lag, x="시차", nbins=int(lag["시차"].max() - lag["시차"].min() + 1), labels={"시차": "시차 (시간)"}))
        st.markdown(f"중앙값 **{lag['시차'].median():.0f}시간**, 시차가 0~3시간인 날 **{lag['시차'].between(0, 3).mean() * 100:.0f}%** (측정소·날짜 {len(lag):,}건). 극단적인 값은 비 온 날 등 예외일 수 있어 아래 '반대 사례'에서 확인합니다.")

    st.subheader("일 최고기온 구간별 일 최고 오존")
    t = dd.dropna(subset=["일최고기온", "일최고O3"]).copy()
    if len(t):
        lo = np.floor(t["일최고기온"].min() / step) * step
        hi = np.ceil((t["일최고기온"].max() + 0.01) / step) * step
        bins = np.arange(lo, hi + step, step)
        labels = [f"{b:.0f}~{b + step:.0f}℃" for b in bins[:-1]]
        t["구간"] = pd.cut(t["일최고기온"], bins, right=False, labels=labels)
        st.plotly_chart(px.box(t, x="구간", y="일최고O3", category_orders={"구간": labels}, points="outliers", labels={"일최고O3": "일 최고 오존 (ppm)"}))
        tab = t.groupby("구간", observed=True)["일최고O3"].agg(건수="count", 중앙값="median").reset_index()
        st.dataframe(tab.round(4))
        ok = tab[tab["건수"] >= 30].reset_index(drop=True)
        if len(ok) > 1:
            jump = ok["중앙값"].diff()
            k = int(jump.idxmax())
            st.markdown(f"건수가 30건 이상인 구간 중 중앙값이 가장 크게 오르는 곳은 **{ok.loc[k - 1, '구간']} → {ok.loc[k, '구간']}** (+{jump.iloc[k]:.3f} ppm)입니다. 건수가 적은 구간은 결과가 쉽게 흔들립니다.")

        st.subheader("일 최고기온과 일 최고 오존의 관계")
        r = np.corrcoef(t["일최고기온"], t["일최고O3"])[0, 1]
        st.plotly_chart(px.scatter(t, x="일최고기온", y="일최고O3", color=t["날짜"].dt.month.astype(str), opacity=0.5,
                                   labels={"일최고기온": "일 최고기온 (℃)", "일최고O3": "일 최고 오존 (ppm)", "color": "월"}))
        st.markdown(f"상관계수 r = **{r:.2f}** (n = {len(t):,}). 기온이 오존의 원인이라는 뜻은 아닙니다. 일사량도 함께 영향을 주어 광화학 반응 때문일 가능성이 있습니다.")

        st.subheader("반대 사례: 기온으로 예상한 값과 차이가 큰 날")
        slope, icpt = np.polyfit(t["일최고기온"], t["일최고O3"], 1)
        t["차이"] = t["일최고O3"] - (slope * t["일최고기온"] + icpt)
        day = t.groupby("날짜").agg(차이=("차이", "mean"), 일최고기온=("일최고기온", "first"), 일최고오존_평균=("일최고O3", "mean"),
                                 강수합=("강수합", "first"), 평균풍속=("평균풍속", "first")).reset_index()
        pick = pd.concat([day.nlargest(5, "차이"), day.nsmallest(5, "차이")])
        pick["요일"] = pick["날짜"].dt.weekday.map(lambda i: WEEK[i])
        pick["날짜"] = pick["날짜"].dt.strftime("%Y-%m-%d")
        st.dataframe(pick[["날짜", "요일", "일최고기온", "일최고오존_평균", "차이", "강수합", "평균풍속"]].round(3))
        st.caption("차이가 +이면 기온으로 예상한 것보다 오존이 높았던 날, −이면 낮았던 날입니다. 강수와 풍속을 보고 한계로 기록하세요.")
    st.caption("기준값 출처는 '처음 화면'의 표를 보세요.")

elif page == "② 사람 기준 초과":
    st.header("② 사람이 야외활동을 많이 하는 9~16시에 기준을 넘는 시간")
    st.markdown(f"선택한 기준: **{crit}**  ·  시간대: 시작 시각 {HUMAN_START}~{HUMAN_END - 1}시 (09:00~16:00)")
    x, ps = human_monthly(h, crit)
    if len(x) == 0:
        st.warning("선택한 조건에 유효한 자료가 없습니다.")
    else:
        mon = ps.groupby(["연도", "월"]).agg(측정소당_초과시간=("초과시간", "mean"), 측정소당_초과일수=("초과일수", "mean"),
                                         측정소_시간합=("초과시간", "sum")).reset_index()
        m = st.columns(3)
        m[0].metric("기준 초과 측정소·시간 합계", f"{int(ps['초과시간'].sum()):,}")
        m[1].metric("초과가 가장 많은 달", f"{int(mon.loc[mon['측정소당_초과시간'].idxmax(), '월'])}월")
        m[2].metric("초과가 한 번이라도 있던 측정소", f"{int((ps.groupby('측정소명')['초과시간'].sum() > 0).sum())} / {ps['측정소명'].nunique()}")

        st.subheader("월별 초과 시간 (측정소 한 곳당 평균)")
        st.plotly_chart(px.bar(mon, x="월", y="측정소당_초과시간", color=mon["연도"].astype(str), barmode="group",
                               labels={"측정소당_초과시간": "측정소당 초과 시간", "color": "연도"}))
        st.subheader("월별 초과 일수 (측정소 한 곳당 평균)")
        st.plotly_chart(px.bar(mon, x="월", y="측정소당_초과일수", color=mon["연도"].astype(str), barmode="group",
                               labels={"측정소당_초과일수": "측정소당 초과 일수", "color": "연도"}))
        st.subheader("월 × 시각 히트맵 (유효 측정 중 기준을 넘은 비율, %)")
        piv = (x.groupby(["월", "시작시"])["초과"].mean() * 100).unstack("시작시")
        st.plotly_chart(px.imshow(piv, aspect="auto", color_continuous_scale="YlOrRd", labels={"x": "시각 (시작 시)", "y": "월", "color": "%"}))
        st.subheader("측정소별 초과 시간 합계")
        by_st = ps.groupby("측정소명")["초과시간"].sum().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(by_st, x="측정소명", y="초과시간"))
        st.markdown("평균뿐 아니라 **기준을 넘은 시간 수**를 함께 봅니다. 측정소마다 결측 시간이 달라서 합계보다 평균으로 월을 비교했습니다.")
    st.caption("기준값 출처는 '처음 화면'의 표를 보세요. 8시간 평균은 이동평균(6개 이상 유효 시 계산)으로 구한 값입니다.")

elif page == "③ 식물 누적 노출 (AOT40)":
    st.header("③ 식물이 받는 40 ppb 초과 누적 오존(AOT40)")
    st.markdown(f"낮 시간 {DAY_START}~{DAY_END}시, 40 ppb(=0.04 ppm)를 넘은 만큼만 시간별로 더한 값입니다. 단위는 ppm·h.")
    a = aot_monthly(h)
    low = a["유효비율"] < MIN_VALID
    excl = st.checkbox(f"낮 시간 유효 측정 비율이 {MIN_VALID * 100:.0f}% 미만인 측정소·달 제외", value=True)
    b = a[~low] if excl else a
    st.caption(f"제외 대상 {int(low.sum())}건 / 전체 {len(a)}건")
    if len(b) == 0:
        st.warning("선택한 조건에 유효한 자료가 없습니다.")
    else:
        mon = b.groupby(["연도", "월"])["AOT40"].mean().reset_index()
        st.subheader("월별 AOT40 (측정소 평균)")
        st.plotly_chart(px.bar(mon, x="월", y="AOT40", color=mon["연도"].astype(str), barmode="group", labels={"AOT40": "AOT40 (ppm·h)", "color": "연도"}))
        st.subheader("측정소 × 월 AOT40")
        pv = b.pivot_table(index="측정소명", columns="월", values="AOT40")
        pv = pv.loc[pv.sum(axis=1).sort_values(ascending=False).index]
        st.plotly_chart(px.imshow(pv, aspect="auto", color_continuous_scale="YlGn", labels={"x": "월", "y": "측정소", "color": "ppm·h"}))
        if STATION_GROUP:
            g = b.assign(유형=b["측정소명"].map(STATION_GROUP).fillna("미분류")).groupby(["유형", "월"])["AOT40"].mean().reset_index()
            st.subheader("측정소 유형별 비교")
            st.plotly_chart(px.bar(g, x="월", y="AOT40", color="유형", barmode="group"))
        else:
            st.info("도심/외곽 비교는 코드 위쪽 STATION_GROUP에 측정소 유형을 직접 채우면 이 자리에 나타납니다.")

        st.subheader("유럽(UNECE) 임계값과 비교")
        piv2 = a.pivot_table(index=["측정소명", "연도"], columns="월", values="AOT40")
        res = pd.DataFrame(index=piv2.index)
        if all(mm in piv2.columns for mm in (5, 6, 7)):
            res["작물 구간(5~7월) AOT40"] = piv2[[5, 6, 7]].sum(axis=1, min_count=3)
            res["작물 3 ppm·h 초과"] = np.where(res["작물 구간(5~7월) AOT40"] > CROP_LEVEL, "O", "X")
        if all(mm in piv2.columns for mm in range(5, 10)):
            res["5~9월 합계"] = piv2[list(range(5, 10))].sum(axis=1, min_count=5)
            res["산림 5 ppm·h 초과(4월 제외한 최소값)"] = np.where(res["5~9월 합계"] > FOREST_LEVEL, "O (최소 이만큼)", "판정 불가")
        if len(res.columns):
            st.dataframe(res.round(2).reset_index())
            st.caption("산림 기준은 4~9월(6개월)인데 이 자료는 5~9월이라 4월이 빠집니다. 5~9월 합이 이미 기준을 넘으면 '넘는다'고 말할 수 있지만, 못 넘으면 판정할 수 없습니다. 월 선택을 일부만 했다면 표가 비어 있을 수 있습니다.")
        else:
            st.info("임계값 비교는 사이드바에서 5~9월을 모두 선택했을 때 나타납니다.")

        st.subheader("사람이 위험한 달과 식물이 위험한 달은 같을까?")
        _, ps = human_monthly(h, crit)
        hm = ps.groupby("월")["초과시간"].mean().rename("사람: 측정소당 초과 시간")
        am = b.groupby("월")["AOT40"].mean().rename("식물: AOT40 (ppm·h)")
        cmp = pd.concat([hm, am], axis=1)
        cmp["사람 순위"] = cmp.iloc[:, 0].rank(ascending=False)
        cmp["식물 순위"] = cmp.iloc[:, 1].rank(ascending=False)
        st.dataframe(cmp.round(2).reset_index())
        if cmp.iloc[:, :2].notna().all(axis=None) and len(cmp):
            ms, mp = int(cmp.iloc[:, 0].idxmax()), int(cmp.iloc[:, 1].idxmax())
            st.markdown(f"사람 기준 초과가 가장 많은 달은 **{ms}월**, AOT40이 가장 큰 달은 **{mp}월**입니다." + (" 두 달이 같습니다." if ms == mp else " 두 달이 다릅니다."))
    st.caption("기준값 출처는 '처음 화면'의 표를 보세요. AOT40은 유럽 기준을 참고한 지표이고 식물의 실제 피해는 수종·수분 상태에 따라 다릅니다.")

else:  # 데이터 점검과 한계
    st.header("데이터 점검과 한계")
    st.subheader("1. 분석에 쓰인 유효 시간 비율 (측정소 × 월)")
    cnt = raw.groupby(["측정소명", "연도", "월"])["O3"].count().reset_index()
    cnt["가능"] = [calendar.monthrange(y, m)[1] * 24 for y, m in zip(cnt["연도"], cnt["월"])]
    cnt["유효비율"] = cnt["O3"] / cnt["가능"]
    pv = cnt.pivot_table(index="측정소명", columns="월", values="유효비율")
    st.plotly_chart(px.imshow(pv, aspect="auto", color_continuous_scale="Blues", zmin=0.5, zmax=1, labels={"x": "월", "y": "측정소", "color": "유효 비율"}))
    st.caption("하루 유효 시간이 75% 미만인 날은 분석에서 뺐고, 연속 2시간 이하의 빈 구간만 앞뒤 값으로 보정했습니다. 그래서 1보다 낮은 칸은 제외된 날이 있다는 뜻입니다.")

    st.subheader("2. 박스플롯: 낮 시간 오존 분포")
    day = raw[raw["시작시"].between(DAY_START, DAY_END - 1)].dropna(subset=["O3"])
    st.plotly_chart(px.box(day, x="월", y="O3", points="outliers", labels={"O3": "오존 (ppm)"}))
    st.plotly_chart(px.box(day, x="시작시", y="O3", color=day["월"].astype(str), labels={"O3": "오존 (ppm)", "color": "월"}))
    n_out, n_all = int(raw["이상치후보"].sum()), int(raw["O3"].notna().sum())
    st.markdown(f"이상치 후보 **{n_out:,}개 / 유효 {n_all:,}개** ({n_out / n_all * 100:.1f}%). 여름 낮의 고농도가 이상치로 몰리지 않도록 월·시간대별로 나눠 IQR(Q1−1.5×IQR, Q3+1.5×IQR)을 적용했습니다.")

    st.subheader("3. 이상치 후보를 빼도 결론이 같은가?")
    a1 = day.groupby("월")["O3"].mean()
    a2 = day[day["이상치후보"] == 0].groupby("월")["O3"].mean()
    c2 = pd.DataFrame({"유지(월 평균, ppm)": a1, "제거(월 평균, ppm)": a2})
    c2["순위(유지)"] = c2.iloc[:, 0].rank(ascending=False)
    c2["순위(제거)"] = c2.iloc[:, 1].rank(ascending=False)
    st.dataframe(c2.round(4).reset_index())
    st.markdown("순위가 같으면 이상치 후보를 빼도 달 사이의 순서는 바뀌지 않는다는 뜻입니다." if (c2["순위(유지)"] == c2["순위(제거)"]).all() else "순위가 바뀐 달이 있습니다. 한계에 기록하세요.")

    st.subheader("4. 원본과 대조하기 (AI 계산 검증)")
    cc = st.columns(2)
    stn = cc[0].selectbox("측정소", st_all)
    dte = cc[1].date_input("날짜", value=raw["날짜"].min().date(), min_value=raw["날짜"].min().date(), max_value=raw["날짜"].max().date())
    one = raw[(raw["측정소명"] == stn) & (raw["날짜"] == pd.Timestamp(dte))].sort_values("시작시")
    if len(one) and one["O3"].notna().any():
        mx = one.loc[one["O3"].idxmax()]
        st.markdown(f"**{stn} {dte}**: 일 최고 오존 **{mx['O3']:.4f} ppm** ({int(mx['시작시'])}시, 에어코리아 표기로는 {int(mx['시작시']) + 1}시), 일 최고기온 **{one['기온'].max():.1f}℃**")
        st.dataframe(one[["시작시", "O3", "기온", "일사", "강수량", "풍속", "습도"]])
    else:
        st.info("이 날은 유효한 자료가 없습니다(결측이거나 하루 유효 비율 75% 미만으로 제외).")
    st.caption("에어코리아 원본 파일에서 같은 측정소·날짜의 O3 최댓값과, 기상청 파일에서 같은 날의 기온 최댓값을 찾아 위 숫자와 비교하세요.")

    st.subheader("5. 한계")
    st.markdown(f"""
- 일부 측정소(서울 도시대기)와 관측소 한 곳(108번)의 자료이므로 서울 전체를 대표하지 않습니다. 오존 측정소 25곳에 기상 값은 같은 관측소 것을 연결했습니다.
- 측정소 농도는 개인이 실제 마신 양이 아닙니다(실내 활동, 이동).
- 이 분석은 **{YEAR_TXT} 5~9월** 자료입니다. 연도에 따라 결과가 달라질 수 있어, 연도 범위를 바꿔 같은 경향인지 확인하세요.
- AOT40은 유럽 기준을 참고한 지표이고, 식물의 실제 피해는 수종·수분 상태에 따라 달라 이 자료로는 알 수 없습니다. 이 자료는 5~9월이라 유럽의 작물(5~7월)·산림(4~9월) 기간과 다릅니다.
- 기상청 자료는 9월 30일이 빠져 있어, 그날의 일 최고기온 등은 계산되지 않았습니다.
- 상관이 있다고 해서 원인이라고 단정할 수 없습니다.
""")
