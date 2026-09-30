# -*- coding: utf-8 -*-
"""
Borusan Otomotiv — Karar Destek Sistemi (KDS)
Satış, müşteri davranışı, mevsimsellik ve stok erken uyarı dashboard'u.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
APP_TITLE = "Borusan Otomotiv | Karar Destek Sistemi"
CSV_NAME = "borusan_otomotiv_gelismis_stok_analizi_veriseti.csv"
DEAD_STOCK_GUN = 45

COLORWAY = [
    "#0B3A5B",
    "#C8102E",
    "#2E8B8B",
    "#D4A017",
    "#4A6FA5",
    "#6B4C7A",
    "#3D7A4A",
    "#8B5A2B",
]
TEMPLATE = "plotly_white"

COL_MUSTERI = "Müşteri No"
COL_MUSTERI_TIPI = "Müşteri Tipi"
COL_YAS = "Yaş"
COL_CINSIYET = "Cinsiyet"
COL_MARKA = "Satın Alınan Marka"
COL_MODEL = "Satın Alınan Araç Modeli"
COL_GOVDE = "Gövde Tipi"
COL_YAKIT = "Motor / Yakıt Tipi"
COL_DONANIM = "Donanım Paketi"
COL_FIYAT = "Satış Fiyatı (TL)"
COL_ODEME = "Ödeme Yöntemi"
COL_KAMPANYA = "Kampanya Uygulandı mı?"
COL_TARIH = "Satın Alma Tarihi"
COL_AY = "Satın Alma Ayı"
COL_SAAT = "Satın Alma Saati"
COL_IL = "Satın Alındığı Bayi İli"
COL_TEDARIK = "Tedarik Süresi (Gün)"
COL_BEKLEME = "Stokta Bekleme Süresi (Gün)"
COL_STOK = "Satış Anındaki Bayi Stoğu"

WEEKDAY_TR = {
    "Monday": "Pazartesi",
    "Tuesday": "Salı",
    "Wednesday": "Çarşamba",
    "Thursday": "Perşembe",
    "Friday": "Cuma",
    "Saturday": "Cumartesi",
    "Sunday": "Pazar",
}
WEEKDAY_ORDER = [
    "Pazartesi",
    "Salı",
    "Çarşamba",
    "Perşembe",
    "Cuma",
    "Cumartesi",
    "Pazar",
]


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def fmt_int(value: float) -> str:
    try:
        return f"{int(round(float(value))):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "—"


def fmt_money(value: float) -> str:
    try:
        return f"{float(value):,.0f} ₺".replace(",", ".")
    except (TypeError, ValueError):
        return "—"


def fmt_gun(value: float) -> str:
    try:
        return f"{float(value):.1f} gün"
    except (TypeError, ValueError):
        return "—"


def empty_guard(df: pd.DataFrame, message: str = "Seçilen filtrelere uygun kayıt bulunamadı.") -> bool:
    if df is None or df.empty:
        st.warning(message)
        return True
    return False


def apply_chart_layout(fig, title: str, height: int = 420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#0B3A5B")),
        template=TEMPLATE,
        colorway=COLORWAY,
        height=height,
        margin=dict(l=40, r=30, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="white", font_size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F7F9FC",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#1F2933"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E5EAF0", zeroline=False)
    return fig


def yas_grubu(raw) -> str:
    if pd.isna(raw):
        return "Kurumsal / Yaş Yok"
    text = str(raw).strip()
    if text in {"", "-", "—", "nan", "None"}:
        return "Kurumsal / Yaş Yok"
    try:
        yas = int(float(text.replace(",", ".")))
    except ValueError:
        return "Kurumsal / Yaş Yok"
    if yas <= 30:
        return "18-30"
    if yas <= 45:
        return "31-45"
    if yas <= 60:
        return "46-60"
    return "60+"


def odeme_grubu(raw) -> str:
    text = str(raw).lower()
    if "takas" in text:
        return "Takas"
    if "kredi" in text:
        return "Kredi"
    if "nakit" in text:
        return "Nakit"
    return str(raw)


def kampanya_evet(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["evet", "e", "yes", "1", "true"])


def stok_devir_hizi(ortalama_bekleme: float) -> float:
    if pd.isna(ortalama_bekleme) or ortalama_bekleme <= 0:
        return np.nan
    return 365.0 / float(ortalama_bekleme)


# ---------------------------------------------------------------------------
# Veri yükleme ve ön işleme
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    candidates = [
        Path(__file__).resolve().parent / CSV_NAME,
        Path.cwd() / CSV_NAME,
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {CSV_NAME}. Lütfen dosyayı uygulama klasörüne koyun."
        )

    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "cp1254"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"CSV okunamadı: {last_error}")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    missing = [c for c in [COL_FIYAT, COL_TARIH, COL_IL, COL_MARKA] if c not in data.columns]
    if missing:
        raise KeyError(f"Beklenen kolonlar eksik: {', '.join(missing)}")

    data[COL_TARIH] = pd.to_datetime(data[COL_TARIH], errors="coerce")
    data[COL_FIYAT] = pd.to_numeric(data[COL_FIYAT], errors="coerce")
    data[COL_TEDARIK] = pd.to_numeric(data[COL_TEDARIK], errors="coerce")
    data[COL_BEKLEME] = pd.to_numeric(data[COL_BEKLEME], errors="coerce")
    data[COL_STOK] = pd.to_numeric(data[COL_STOK], errors="coerce")

    saat = pd.to_datetime(data[COL_SAAT].astype(str), format="%H:%M", errors="coerce")
    data["Saat"] = saat.dt.hour
    data["Hafta Günü"] = data[COL_TARIH].dt.day_name().map(WEEKDAY_TR)
    data["Yıl-Ay"] = data[COL_TARIH].dt.to_period("M").astype(str)
    data["Ay Sıra"] = data[COL_TARIH].dt.to_period("M")
    data["Yaş Grubu"] = data[COL_YAS].map(yas_grubu)
    data["Ödeme Grubu"] = data[COL_ODEME].map(odeme_grubu)
    data["Kampanya"] = np.where(kampanya_evet(data[COL_KAMPANYA]), "Kampanyalı", "Kampanyasız")
    data["Satış Adedi"] = 1
    return data.dropna(subset=[COL_TARIH, COL_FIYAT])


def apply_filters(
    df: pd.DataFrame,
    iller: list[str],
    markalar: list[str],
    musteri_tipleri: list[str],
    tarih_araligi,
) -> pd.DataFrame:
    filtered = df.copy()
    if iller:
        filtered = filtered[filtered[COL_IL].isin(iller)]
    if markalar:
        filtered = filtered[filtered[COL_MARKA].isin(markalar)]
    if musteri_tipleri:
        filtered = filtered[filtered[COL_MUSTERI_TIPI].isin(musteri_tipleri)]
    if tarih_araligi and len(tarih_araligi) == 2:
        start, end = pd.to_datetime(tarih_araligi[0]), pd.to_datetime(tarih_araligi[1])
        filtered = filtered[(filtered[COL_TARIH] >= start) & (filtered[COL_TARIH] <= end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
    return filtered


# ---------------------------------------------------------------------------
# Dinamik yorumlar
# ---------------------------------------------------------------------------
def yorum_kpi(df: pd.DataFrame) -> str:
    adet = len(df)
    ciro = df[COL_FIYAT].sum()
    tedarik = df[COL_TEDARIK].mean()
    bekleme = df[COL_BEKLEME].mean()
    en_marka = df.groupby(COL_MARKA)[COL_FIYAT].sum().sort_values(ascending=False)
    pay = 100 * en_marka.iloc[0] / ciro if ciro else 0
    aksiyon = (
        "Tedarik süresi 25 günün üzerinde; sipariş planını öne çekin ve kritik modeller için güvenlik stoğu artırın."
        if tedarik >= 25
        else "Tedarik süresi kontrollü; mevcut sipariş ritmi korunabilir, kapasiteyi kampanya dönemlerine kaydırın."
    )
    return (
        f"**Analitik Değerlendirme & Yorum:** Seçili dönemde **{fmt_int(adet)}** adet satışla **{fmt_money(ciro)}** ciro "
        f"üretilmiştir. Ciro lideri **{en_marka.index[0]}** (%{pay:.1f} pay). "
        f"Ortalama tedarik **{tedarik:.1f} gün**, stokta bekleme **{bekleme:.1f} gün**. {aksiyon}"
    )


def yorum_marka_model(df: pd.DataFrame) -> str:
    marka = df.groupby(COL_MARKA).agg(adet=("Satış Adedi", "sum"), ciro=(COL_FIYAT, "sum"))
    marka["ortalama"] = marka["ciro"] / marka["adet"]
    ciro_lid = marka["ciro"].idxmax()
    adet_lid = marka["adet"].idxmax()
    ort_lid = marka["ortalama"].idxmax()
    model = df.groupby(COL_MODEL)[COL_FIYAT].sum().sort_values(ascending=False)
    return (
        f"**Analitik Değerlendirme & Yorum:** Adet lideri **{adet_lid}**, ciro lideri **{ciro_lid}**, "
        f"ortalama işlem tutarı en yüksek marka **{ort_lid}** ({fmt_money(marka.loc[ort_lid, 'ortalama'])}). "
        f"Tek model bazında en yüksek ciro **{model.index[0]}**. "
        f"Aksiyon: Adet/ciro ayrışması varsa düşük ortalama tutarlı modelleri paket/aksesuar ve donanım upsell ile destekleyin; "
        f"premium modellerde teslimat vaadini kısaltın."
    )


def yorum_bayi(tablo: pd.DataFrame) -> str:
    if tablo.empty:
        return "Bayi karşılaştırması için yeterli kayıt yok."
    ciro_lid = tablo.sort_values("Toplam Ciro (TL)", ascending=False).iloc[0]
    hiz_lid = tablo.sort_values("Ort. Stok Devir Hızı", ascending=False).iloc[0]
    yavas = tablo.sort_values("Ort. Stok Devir Hızı", ascending=True).iloc[0]
    return (
        f"**Analitik Değerlendirme & Yorum:** Ciroda öne çıkan bayi ili **{ciro_lid[COL_IL]}** "
        f"({fmt_money(ciro_lid['Toplam Ciro (TL)'])}). En hızlı stok devri **{hiz_lid[COL_IL]}** "
        f"({hiz_lid['Ort. Stok Devir Hızı']:.1f} tur/yıl). En yavaş devir **{yavas[COL_IL]}**. "
        f"Aksiyon: Yavaş illerde stok kompozisyonunu gözden geçirin; hızlı illere transfer ve ortak stok havuzu değerlendirin."
    )


def yorum_yas_govde(df: pd.DataFrame) -> str:
    alt = df[df["Yaş Grubu"] != "Kurumsal / Yaş Yok"]
    if alt.empty:
        return "**Analitik Değerlendirme & Yorum:** Yaş bilgisi ağırlıklı olarak kurumsal kayıtlarda boş. Bireysel segmentte yaş bazlı hedefleme için veri kalitesini artırın."
    ct = pd.crosstab(alt["Yaş Grubu"], alt[COL_GOVDE])
    zirve = ct.stack().idxmax()
    yas_g, govde = zirve
    suv_pay = 100 * alt[alt[COL_GOVDE].str.contains("SUV", case=False, na=False)].shape[0] / len(alt)
    return (
        f"**Analitik Değerlendirme & Yorum:** En güçlü yaş × gövde kesişimi **{yas_g} / {govde}**. "
        f"Bireysel satışlarda SUV payı %{suv_pay:.0f}. "
        f"Aksiyon: {yas_g} grubuna {govde} odaklı dijital ve showroom senaryoları kurun; "
        f"diğer yaş gruplarında zayıf gövde tiplerini test sürüşü ve kampanya ile dengeleyin."
    )


def yorum_cinsiyet(df: pd.DataFrame) -> str:
    ort = df.groupby(COL_CINSIYET)[COL_FIYAT].mean().sort_values(ascending=False)
    if ort.empty:
        return "Cinsiyet kırılımı üretilemedi."
    marka_mod = (
        df.groupby([COL_CINSIYET, COL_MARKA]).size().reset_index(name="n").sort_values("n", ascending=False).iloc[0]
    )
    donanim = (
        df.groupby([COL_CINSIYET, COL_DONANIM]).size().reset_index(name="n").sort_values("n", ascending=False).iloc[0]
    )
    fark = 100 * (ort.iloc[0] - ort.iloc[-1]) / ort.iloc[-1] if len(ort) > 1 and ort.iloc[-1] else 0
    return (
        f"**Analitik Değerlendirme & Yorum:** Ortalama ödenen tutar **{ort.index[0]}** segmentinde daha yüksek "
        f"({fmt_money(ort.iloc[0])}, diğerine göre %{fark:.1f} fark). "
        f"Hacimde öne çıkan kombinasyon: **{marka_mod[COL_CINSIYET]} × {marka_mod[COL_MARKA]}**, "
        f"donanımda **{donanim[COL_CINSIYET]} × {donanim[COL_DONANIM]}**. "
        f"Aksiyon: Yüksek ortalama tutarlı segmente üst donanım ve finansman paketleri; "
        f"diğer segmente giriş modelleri ve kişiselleştirilmiş teklif kurgulayın."
    )


def yorum_odeme_kampanya(df: pd.DataFrame) -> str:
    kamp_orani = 100 * (df["Kampanya"] == "Kampanyalı").mean()
    tipi = (
        df.assign(_kamp=(df["Kampanya"] == "Kampanyalı").astype(float))
        .groupby(COL_MUSTERI_TIPI)["_kamp"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    odeme = df.groupby([COL_MUSTERI_TIPI, "Ödeme Grubu"]).size().reset_index(name="n")
    zirve = odeme.sort_values("n", ascending=False).iloc[0]
    kurumsal = df[df[COL_MUSTERI_TIPI].astype(str).str.contains("Kurumsal", case=False, na=False)]
    bireysel = df[df[COL_MUSTERI_TIPI].astype(str).str.contains("Bireysel", case=False, na=False)]
    k_odeme = kurumsal["Ödeme Grubu"].mode().iloc[0] if not kurumsal.empty else "—"
    b_odeme = bireysel["Ödeme Grubu"].mode().iloc[0] if not bireysel.empty else "—"
    return (
        f"**Analitik Değerlendirme & Yorum:** Genel kampanya kullanım oranı **%{kamp_orani:.1f}**. "
        f"Kampanyayı en yoğun kullanan müşteri tipi **{tipi.index[0]}** (%{tipi.iloc[0]:.1f}). "
        f"En sık kesişim **{zirve[COL_MUSTERI_TIPI]} × {zirve['Ödeme Grubu']}**. "
        f"Kurumsal ödeme eğilimi: **{k_odeme}**, bireysel: **{b_odeme}**. "
        f"Aksiyon: Kurumsal filoda toplu alım ve nakit/kredi mix'ini yeniden fiyatlayın; "
        f"bireyselde takas + kredi paketini vitrin teklifi yapın."
    )


def yorum_aylik_trend(df: pd.DataFrame) -> str:
    aylik = df.groupby("Yıl-Ay").size().sort_index()
    if aylik.empty:
        return "Aylık trend için yeterli tarih yok."
    zirve_ay = aylik.idxmax()
    dip_ay = aylik.idxmin()
    yakit = df.groupby([COL_YAKIT]).size().sort_values(ascending=False)
    elektrikli = df[df[COL_YAKIT].astype(str).str.contains("Elektrik", case=False, na=False)]
    e_pay = 100 * len(elektrikli) / len(df)
    return (
        f"**Analitik Değerlendirme & Yorum:** Satış zirvesi **{zirve_ay}** ({int(aylik.max())} adet), "
        f"en zayıf ay **{dip_ay}**. Yakıt karmasında lider **{yakit.index[0]}**; elektrikli payı %{e_pay:.1f}. "
        f"Aksiyon: Zayıf aylara gövde/yakıt bazlı erken sipariş ve teslim kampanyası koyun; "
        f"elektrikli/hybrid ivmesini şarj ve filoya özel tekliflerle destekleyin."
    )


def yorum_heatmap(df: pd.DataFrame) -> str:
    alt = df.dropna(subset=["Saat", "Hafta Günü", COL_IL])
    if alt.empty:
        return "Yoğunluk haritası için saat/gün bilgisi yetersiz."
    saat_mod = int(alt["Saat"].mode().iloc[0])
    gun_mod = alt["Hafta Günü"].mode().iloc[0]
    il_mod = alt[COL_IL].mode().iloc[0]
    return (
        f"**Analitik Değerlendirme & Yorum:** Satışlar **{gun_mod}** günü ve saat **{saat_mod:02d}:00** civarında yoğunlaşıyor. "
        f"Hacimde öne çıkan il **{il_mod}**. "
        f"Aksiyon: Yoğun saatlerde test sürüşü ve finansman danışmanı kapasitesini artırın; "
        f"düşük yoğunluklu dilimlere dijital randevu ve outbound kampanya yerleştirin."
    )


def yorum_kampanya_etki(df: pd.DataFrame) -> str:
    g = df.groupby("Kampanya").agg(
        adet=("Satış Adedi", "sum"),
        ciro=(COL_FIYAT, "sum"),
        gun=(COL_TARIH, "nunique"),
        bekleme=(COL_BEKLEME, "mean"),
    )
    g["hiz"] = g["adet"] / g["gun"].replace(0, np.nan)
    if "Kampanyalı" not in g.index or "Kampanyasız" not in g.index:
        return "**Analitik Değerlendirme & Yorum:** Kampanyalı/kampanyasız kıyas için her iki grup da gerekli. Filtreleri genişletin."
    lift = 100 * (g.loc["Kampanyalı", "hiz"] / g.loc["Kampanyasız", "hiz"] - 1)
    yon = "hızlandırıyor" if lift > 0 else "yavaşlatıyor veya nötr"
    return (
        f"**Analitik Değerlendirme & Yorum:** Kampanyalı dönemde günlük satış hızı **{g.loc['Kampanyalı', 'hiz']:.2f} adet/gün**, "
        f"kampanyasız **{g.loc['Kampanyasız', 'hiz']:.2f}**. Etki: **%{lift:.1f}** ({yon}). "
        f"Kampanyalı ortalama stokta bekleme {g.loc['Kampanyalı', 'bekleme']:.1f} gün. "
        f"Aksiyon: Lift pozitifse bütçeyi hızlı dönen modellere kaydırın; negatif/zayıfsa kampanyayı fiyat yerine teslimat ve donanım avantajına çevirin."
    )


def yorum_risk_matrisi(agg: pd.DataFrame) -> str:
    if agg.empty:
        return "Risk matrisi için model kırılımı oluşmadı."
    kritik = agg[(agg[COL_TEDARIK] >= agg[COL_TEDARIK].median()) & (agg[COL_BEKLEME] <= agg[COL_BEKLEME].median())]
    olu = agg[agg[COL_BEKLEME] >= DEAD_STOCK_GUN]
    k_txt = ", ".join(kritik[COL_MODEL].head(3).tolist()) or "yok"
    o_txt = ", ".join(olu[COL_MODEL].head(3).tolist()) or "yok"
    return (
        f"**Analitik Değerlendirme & Yorum:** Sağ üst bölge (uzun tedarik + uzun bekleme) sermaye bağlar; "
        f"sol alt hızlı döner. Tedarik uzun / bekleme kısa adaylar: **{k_txt}**. "
        f"{DEAD_STOCK_GUN}+ gün bekleyen modeller: **{o_txt}**. "
        f"Aksiyon: Kritik modellerde siparişi öne çekin; ölü stokta iller arası transfer veya hedefli indirim uygulayın."
    )


def yorum_acil_siparis(tablo: pd.DataFrame) -> str:
    if tablo.empty:
        return (
            "**Analitik Değerlendirme & Yorum:** Mevcut filtrede 'uzun tedarik + hızlı tükenen stok' eşiğini aşan model yok. "
            "Güvenlik stoklarını izlemeye devam edin."
        )
    ilk = tablo.iloc[0]
    return (
        f"**Analitik Değerlendirme & Yorum:** **{len(tablo)}** model ACİL SİPARİŞ bandında. "
        f"Öncelik: **{ilk[COL_MODEL]}** (tedarik {ilk[COL_TEDARIK]:.0f} gün, stokta kalma {ilk[COL_BEKLEME]:.0f} gün). "
        f"Aksiyon: Tedarikçiye acil tahsis isteyin, alternatif renk/donanım açın ve bu modellerde satış taahhüdünü stoğa bağlayın."
    )


def yorum_olu_stok(tablo: pd.DataFrame, kilitli_ciro: float) -> str:
    if tablo.empty:
        return (
            f"**Analitik Değerlendirme & Yorum:** {DEAD_STOCK_GUN} günü aşan yavaş stok kaydı yok. "
            "Mevcut kompozisyon sağlıklı; kampanya bütçesini hızlı modellere kaydırabilirsiniz."
        )
    return (
        f"**Analitik Değerlendirme & Yorum:** **{len(tablo)}** satış kaydı 45+ gün bekleyerek el değiştirmiş; "
        f"bağlı ciro yaklaşık **{fmt_money(kilitli_ciro)}**. "
        f"Aksiyon: Bu model-il çiftlerine 30-45 günlük hedefli kampanya, takas prim ve filoya özel teklif uygulayın; "
        f"90 güne yaklaşanlarda daha agresif iskonto açın."
    )


def yorum_transfer(tablo: pd.DataFrame) -> str:
    if tablo.empty:
        return (
            "**Analitik Değerlendirme & Yorum:** Aynı model için 'yavaş il → hızlı il' eşleşmesi bulunamadı. "
            "Filtreleri genişletin veya merkezi stok havuzu senaryosunu değerlendirin."
        )
    ilk = tablo.iloc[0]
    return (
        f"**Analitik Değerlendirme & Yorum:** **{len(tablo)}** transfer fırsatı üretildi. Öncelikli hat: "
        f"**{ilk['Kaynak İl']} → {ilk['Hedef İl']} / {ilk['Model']}**. "
        f"Aksiyon: Lojistik maliyeti, teslimat SLA ve renk/donanım uyumunu kontrol ederek 48 saat içinde kaydırma emri verin; "
        f"kaynak ilde yeni siparişi durdurun."
    )


# ---------------------------------------------------------------------------
# Sekmeler
# ---------------------------------------------------------------------------
def render_kpi_cards(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Satış Adedi", fmt_int(len(df)))
    c2.metric("Toplam Ciro", fmt_money(df[COL_FIYAT].sum()))
    c3.metric("Ort. Tedarik Süresi", fmt_gun(df[COL_TEDARIK].mean()))
    c4.metric("Ort. Stokta Bekleme", fmt_gun(df[COL_BEKLEME].mean()))


def tab_executive(df: pd.DataFrame) -> None:
    st.subheader("Yönetici Özeti")
    render_kpi_cards(df)
    st.success(yorum_kpi(df))

    st.markdown("##### Marka bazında ciro ve adet katkısı")
    marka = (
        df.groupby(COL_MARKA, as_index=False)
        .agg(Adet=("Satış Adedi", "sum"), Ciro=(COL_FIYAT, "sum"))
        .sort_values("Ciro", ascending=False)
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=marka[COL_MARKA], y=marka["Ciro"], name="Ciro (TL)", marker_color=COLORWAY[0], opacity=0.9),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=marka[COL_MARKA],
            y=marka["Adet"],
            name="Satış Adedi",
            mode="lines+markers",
            marker=dict(size=10, color=COLORWAY[1]),
            line=dict(width=3, color=COLORWAY[1]),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Ciro (TL)", secondary_y=False, tickformat=",")
    fig.update_yaxes(title_text="Satış Adedi", secondary_y=True)
    st.plotly_chart(apply_chart_layout(fig, "Marka: Ciro (bar) × Adet (çizgi)", 430), use_container_width=True)

    st.markdown("##### Model bazında ciro ve adet")
    model = (
        df.groupby([COL_MARKA, COL_MODEL], as_index=False)
        .agg(Adet=("Satış Adedi", "sum"), Ciro=(COL_FIYAT, "sum"))
        .sort_values("Ciro", ascending=False)
    )
    fig_m = px.bar(
        model,
        x=COL_MODEL,
        y="Ciro",
        color=COL_MARKA,
        hover_data=["Adet"],
        labels={COL_MODEL: "Model", "Ciro": "Ciro (TL)"},
    )
    st.plotly_chart(apply_chart_layout(fig_m, "Modellerin ciro katkısı (adet tooltip'te)", 460), use_container_width=True)
    st.info(yorum_marka_model(df))

    st.markdown("##### Bayi ili performans tablosu")
    bayi = (
        df.groupby(COL_IL, as_index=False)
        .agg(
            **{
                "Satış Adedi": ("Satış Adedi", "sum"),
                "Toplam Ciro (TL)": (COL_FIYAT, "sum"),
                "Ort. Tedarik (Gün)": (COL_TEDARIK, "mean"),
                "Ort. Bekleme (Gün)": (COL_BEKLEME, "mean"),
                "Ort. Bayi Stoğu": (COL_STOK, "mean"),
            }
        )
    )
    bayi["Ort. Stok Devir Hızı"] = bayi["Ort. Bekleme (Gün)"].map(stok_devir_hizi)
    bayi = bayi.sort_values("Toplam Ciro (TL)", ascending=False)
    goster = bayi.copy()
    goster["Toplam Ciro (TL)"] = goster["Toplam Ciro (TL)"].map(lambda x: f"{x:,.0f}".replace(",", "."))
    for col in ["Ort. Tedarik (Gün)", "Ort. Bekleme (Gün)", "Ort. Bayi Stoğu", "Ort. Stok Devir Hızı"]:
        goster[col] = goster[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    st.dataframe(goster, use_container_width=True, hide_index=True)
    st.info(yorum_bayi(bayi))


def tab_demografi(df: pd.DataFrame) -> None:
    st.subheader("Demografi ve Müşteri Davranışı")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Yaş grubu × gövde tipi")
        yas_df = df.copy()
        stacked = yas_df.groupby(["Yaş Grubu", COL_GOVDE], as_index=False)["Satış Adedi"].sum()
        fig = px.bar(
            stacked,
            x="Yaş Grubu",
            y="Satış Adedi",
            color=COL_GOVDE,
            barmode="stack",
            category_orders={"Yaş Grubu": ["18-30", "31-45", "46-60", "60+", "Kurumsal / Yaş Yok"]},
        )
        st.plotly_chart(apply_chart_layout(fig, "Yaş grubuna göre gövde tipi tercihi", 420), use_container_width=True)
    with c2:
        st.markdown("##### Yaş grubu × model (treemap)")
        tree = df.groupby(["Yaş Grubu", COL_MARKA, COL_MODEL], as_index=False).agg(
            Adet=("Satış Adedi", "sum"), Ciro=(COL_FIYAT, "sum")
        )
        fig_t = px.treemap(
            tree,
            path=["Yaş Grubu", COL_MARKA, COL_MODEL],
            values="Adet",
            color="Ciro",
            color_continuous_scale="Tealrose",
        )
        st.plotly_chart(apply_chart_layout(fig_t, "Yaş × marka × model hacmi", 420), use_container_width=True)
    st.info(yorum_yas_govde(df))

    st.markdown("##### Cinsiyet bazlı segmentasyon")
    g1, g2, g3 = st.columns(3)
    with g1:
        marka_c = df.groupby([COL_CINSIYET, COL_MARKA], as_index=False)["Satış Adedi"].sum()
        fig = px.bar(marka_c, x=COL_MARKA, y="Satış Adedi", color=COL_CINSIYET, barmode="group")
        st.plotly_chart(apply_chart_layout(fig, "Cinsiyet × marka", 380), use_container_width=True)
    with g2:
        don = df.groupby([COL_CINSIYET, COL_DONANIM], as_index=False)["Satış Adedi"].sum()
        fig = px.bar(don, x=COL_DONANIM, y="Satış Adedi", color=COL_CINSIYET, barmode="group")
        st.plotly_chart(apply_chart_layout(fig, "Cinsiyet × donanım", 380), use_container_width=True)
    with g3:
        ort = df.groupby(COL_CINSIYET, as_index=False)[COL_FIYAT].mean().rename(columns={COL_FIYAT: "Ort. Ödenen"})
        fig = px.bar(ort, x=COL_CINSIYET, y="Ort. Ödenen", color=COL_CINSIYET)
        st.plotly_chart(apply_chart_layout(fig, "Cinsiyete göre ortalama tutar", 380), use_container_width=True)
    st.info(yorum_cinsiyet(df))

    st.markdown("##### Müşteri tipi × ödeme yöntemi ve kampanya")
    p1, p2 = st.columns(2)
    with p1:
        cross = df.groupby([COL_MUSTERI_TIPI, "Ödeme Grubu"], as_index=False)["Satış Adedi"].sum()
        fig = px.bar(cross, x=COL_MUSTERI_TIPI, y="Satış Adedi", color="Ödeme Grubu", barmode="stack")
        st.plotly_chart(apply_chart_layout(fig, "Müşteri tipi × ödeme grubu", 400), use_container_width=True)
    with p2:
        kam = (
            df.groupby(COL_MUSTERI_TIPI)["Kampanya"]
            .apply(lambda s: 100 * (s == "Kampanyalı").mean())
            .reset_index(name="Kampanya Oranı (%)")
        )
        fig = px.bar(kam, x=COL_MUSTERI_TIPI, y="Kampanya Oranı (%)", color=COL_MUSTERI_TIPI)
        st.plotly_chart(apply_chart_layout(fig, "Müşteri tipine göre kampanya kullanım oranı", 400), use_container_width=True)

    heat = pd.crosstab(df[COL_MUSTERI_TIPI], df[COL_ODEME], normalize="index") * 100
    fig_h = px.imshow(
        heat,
        text_auto=".1f",
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(color="Satır %"),
    )
    st.plotly_chart(apply_chart_layout(fig_h, "Ödeme yöntemi dağılımı (müşteri tipi satır %)", 360), use_container_width=True)
    st.success(yorum_odeme_kampanya(df))


def tab_zaman(df: pd.DataFrame) -> None:
    st.subheader("Mevsimsellik ve Zaman Trendleri")

    aylik_yakit = df.groupby(["Ay Sıra", "Yıl-Ay", COL_YAKIT], as_index=False)["Satış Adedi"].sum().sort_values("Ay Sıra")
    fig = px.line(aylik_yakit, x="Yıl-Ay", y="Satış Adedi", color=COL_YAKIT, markers=True)
    st.plotly_chart(apply_chart_layout(fig, "Aylık satış — yakıt tipi", 420), use_container_width=True)

    aylik_govde = df.groupby(["Ay Sıra", "Yıl-Ay", COL_GOVDE], as_index=False)["Satış Adedi"].sum().sort_values("Ay Sıra")
    fig_g = px.line(aylik_govde, x="Yıl-Ay", y="Satış Adedi", color=COL_GOVDE, markers=True)
    st.plotly_chart(apply_chart_layout(fig_g, "Aylık satış — gövde tipi talebi", 420), use_container_width=True)
    st.info(yorum_aylik_trend(df))

    st.markdown("##### Saatlik satış yoğunluğu")
    h1, h2 = st.columns(2)
    heat_df = df.dropna(subset=["Saat", "Hafta Günü"])
    with h1:
        pivot = (
            heat_df.groupby(["Hafta Günü", "Saat"])["Satış Adedi"]
            .sum()
            .unstack(fill_value=0)
            .reindex(WEEKDAY_ORDER)
        )
        fig = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            labels=dict(x="Saat", y="Haftanın günü", color="Adet"),
        )
        st.plotly_chart(apply_chart_layout(fig, "Haftanın günü × saat ısı haritası", 420), use_container_width=True)
    with h2:
        pivot_il = heat_df.groupby([COL_IL, "Saat"])["Satış Adedi"].sum().unstack(fill_value=0)
        fig = px.imshow(
            pivot_il,
            aspect="auto",
            color_continuous_scale="YlGnBu",
            labels=dict(x="Saat", y="Bayi ili", color="Adet"),
        )
        st.plotly_chart(apply_chart_layout(fig, "İl × saat ısı haritası", 420), use_container_width=True)
    st.info(yorum_heatmap(df))

    st.markdown("##### Kampanya etki analizi")
    kamp = df.groupby("Kampanya", as_index=False).agg(
        Adet=("Satış Adedi", "sum"),
        Ciro=(COL_FIYAT, "sum"),
        Distinct_Gün=(COL_TARIH, "nunique"),
        Ort_Bekleme=(COL_BEKLEME, "mean"),
        Ort_Fiyat=(COL_FIYAT, "mean"),
    )
    kamp["Satış Hızı (adet/gün)"] = kamp["Adet"] / kamp["Distinct_Gün"].replace(0, np.nan)

    k1, k2 = st.columns(2)
    with k1:
        fig = px.bar(kamp, x="Kampanya", y="Satış Hızı (adet/gün)", color="Kampanya")
        st.plotly_chart(apply_chart_layout(fig, "Kampanyalı vs kampanyasız satış hızı", 380), use_container_width=True)
    with k2:
        fig = px.bar(kamp, x="Kampanya", y="Adet", color="Kampanya")
        st.plotly_chart(apply_chart_layout(fig, "Kampanya kırılımında satış adedi", 380), use_container_width=True)

    goster = kamp.copy()
    goster["Ciro"] = goster["Ciro"].map(lambda x: f"{x:,.0f}".replace(",", "."))
    goster["Ort_Fiyat"] = goster["Ort_Fiyat"].map(lambda x: f"{x:,.0f}".replace(",", "."))
    goster["Ort_Bekleme"] = goster["Ort_Bekleme"].map(lambda x: f"{x:.1f}")
    goster["Satış Hızı (adet/gün)"] = goster["Satış Hızı (adet/gün)"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    st.dataframe(goster, use_container_width=True, hide_index=True)
    st.success(yorum_kampanya_etki(df))


def model_stok_ozet(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby([COL_MARKA, COL_MODEL], as_index=False)
        .agg(
            **{
                COL_TEDARIK: (COL_TEDARIK, "mean"),
                COL_BEKLEME: (COL_BEKLEME, "mean"),
                "Satış Adedi": ("Satış Adedi", "sum"),
                "Ciro": (COL_FIYAT, "sum"),
                COL_STOK: (COL_STOK, "mean"),
            }
        )
    )


def acil_siparis_tablosu(df: pd.DataFrame) -> pd.DataFrame:
    agg = model_stok_ozet(df)
    if agg.empty:
        return agg
    tedarik_esik = max(float(agg[COL_TEDARIK].median()), 20.0)
    bekleme_esik = min(float(agg[COL_BEKLEME].median()), 20.0)
    stok_esik = 3.0
    mask = (agg[COL_TEDARIK] >= tedarik_esik) & (agg[COL_BEKLEME] <= bekleme_esik)
    yumusak = mask | ((agg[COL_TEDARIK] >= tedarik_esik) & (agg[COL_STOK] <= stok_esik) & (agg[COL_BEKLEME] <= agg[COL_BEKLEME].median()))
    out = agg[yumusak].copy()
    out["Risk Skoru"] = (out[COL_TEDARIK] / (out[COL_BEKLEME] + 1.0)) * out["Satış Adedi"]
    out = out.sort_values("Risk Skoru", ascending=False)
    out["Uyarı"] = "ACİL SİPARİŞ VERİLMELİ"
    out["Gerekçe"] = (
        "Tedarik uzun, stok hızlı tükeniyor / satış anı stoğu düşük. Güvenlik stoğu ve sipariş öne çekilmeli."
    )
    return out


def olu_stok_tablosu(df: pd.DataFrame) -> pd.DataFrame:
    alt = df[df[COL_BEKLEME] >= DEAD_STOCK_GUN].copy()
    if alt.empty:
        return alt
    alt["Öneri"] = np.where(
        alt[COL_BEKLEME] >= 55,
        "Agresif indirim + takas primi + filo teklifi",
        "Hedefli kampanya, vade avantajı ve showroom görünürlüğü",
    )
    kolonlar = [
        COL_IL,
        COL_MARKA,
        COL_MODEL,
        COL_DONANIM,
        COL_BEKLEME,
        COL_TEDARIK,
        COL_STOK,
        COL_FIYAT,
        "Kampanya",
        "Öneri",
    ]
    return alt[kolonlar].sort_values(COL_BEKLEME, ascending=False)


def transfer_onerileri(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby([COL_IL, COL_MARKA, COL_MODEL], as_index=False)
        .agg(
            Adet=("Satış Adedi", "sum"),
            Ort_Bekleme=(COL_BEKLEME, "mean"),
            Ort_Tedarik=(COL_TEDARIK, "mean"),
            Ort_Stok=(COL_STOK, "mean"),
            Ciro=(COL_FIYAT, "sum"),
        )
    )
    if agg.empty:
        return agg

    median_bekleme = float(agg["Ort_Bekleme"].median())
    yavas = agg[agg["Ort_Bekleme"] >= max(median_bekleme, 30)]
    hizli = agg.sort_values(["Adet", "Ort_Bekleme"], ascending=[False, True])

    recs = []
    for _, src in yavas.iterrows():
        dest = hizli[
            (hizli[COL_MODEL] == src[COL_MODEL])
            & (hizli[COL_IL] != src[COL_IL])
            & (hizli["Ort_Bekleme"] < src["Ort_Bekleme"])
            & (hizli["Adet"] >= src["Adet"])
        ]
        if dest.empty:
            dest = hizli[(hizli[COL_MODEL] == src[COL_MODEL]) & (hizli[COL_IL] != src[COL_IL])]
            dest = dest[dest["Ort_Bekleme"] < src["Ort_Bekleme"]]
        if dest.empty:
            continue
        d = dest.iloc[0]
        recs.append(
            {
                "Kaynak İl": src[COL_IL],
                "Hedef İl": d[COL_IL],
                "Marka": src[COL_MARKA],
                "Model": src[COL_MODEL],
                "Kaynak Ort. Bekleme (Gün)": round(src["Ort_Bekleme"], 1),
                "Hedef Ort. Bekleme (Gün)": round(d["Ort_Bekleme"], 1),
                "Kaynak Adet": int(src["Adet"]),
                "Hedef Adet": int(d["Adet"]),
                "Öneri": (
                    f"{src[COL_IL]} bayisinde yavaş dönen {src[COL_MODEL]} stoğu, "
                    f"yüksek talepli {d[COL_IL]} bayisine kaydırılmalı."
                ),
            }
        )
    if not recs:
        return pd.DataFrame()
    out = pd.DataFrame(recs).drop_duplicates(subset=["Kaynak İl", "Hedef İl", "Model"])
    out["Bekleme Farkı (Gün)"] = out["Kaynak Ort. Bekleme (Gün)"] - out["Hedef Ort. Bekleme (Gün)"]
    return out.sort_values("Bekleme Farkı (Gün)", ascending=False)


def tab_stok_kds(df: pd.DataFrame) -> None:
    st.subheader("Akıllı Stok Yönetimi ve Erken Uyarı Merkezi")
    st.caption("KDS motoru: tedarik süresi, stokta kalma ve satış hızını birlikte okuyarak sipariş, kampanya ve transfer önerir.")

    agg = model_stok_ozet(df)
    fig = px.scatter(
        agg,
        x=COL_TEDARIK,
        y=COL_BEKLEME,
        size="Satış Adedi",
        color=COL_MARKA,
        hover_name=COL_MODEL,
        size_max=48,
        labels={COL_TEDARIK: "Tedarik Süresi (Gün)", COL_BEKLEME: "Stokta Bekleme Süresi (Gün)"},
    )
    fig.add_hline(y=DEAD_STOCK_GUN, line_dash="dot", line_color="#C8102E", annotation_text="45 gün ölü stok eşiği")
    if not agg.empty:
        fig.add_vline(
            x=float(agg[COL_TEDARIK].median()),
            line_dash="dot",
            line_color="#0B3A5B",
            annotation_text="Medyan tedarik",
        )
    st.plotly_chart(apply_chart_layout(fig, "Stok risk matrisi (nokta büyüklüğü = satış adedi)", 480), use_container_width=True)
    st.info(yorum_risk_matrisi(agg))

    st.markdown("##### Kritik stok uyarısı — Out-of-Stock riski")
    acil = acil_siparis_tablosu(df)
    if acil.empty:
        st.success("Kritik eşikte model yok.")
    else:
        goster = acil.copy()
        goster[COL_TEDARIK] = goster[COL_TEDARIK].map(lambda x: f"{x:.1f}")
        goster[COL_BEKLEME] = goster[COL_BEKLEME].map(lambda x: f"{x:.1f}")
        goster[COL_STOK] = goster[COL_STOK].map(lambda x: f"{x:.1f}")
        goster["Ciro"] = goster["Ciro"].map(lambda x: f"{x:,.0f}".replace(",", "."))
        goster["Risk Skoru"] = goster["Risk Skoru"].map(lambda x: f"{x:.1f}")
        st.dataframe(
            goster[[COL_MARKA, COL_MODEL, COL_TEDARIK, COL_BEKLEME, COL_STOK, "Satış Adedi", "Ciro", "Risk Skoru", "Uyarı"]],
            use_container_width=True,
            hide_index=True,
        )
    st.info(yorum_acil_siparis(acil))

    st.markdown(f"##### Yavaş dönen stok alarmı (Dead Stock, {DEAD_STOCK_GUN}+ gün)")
    olu = olu_stok_tablosu(df)
    kilitli = float(df.loc[df[COL_BEKLEME] >= DEAD_STOCK_GUN, COL_FIYAT].sum()) if not df.empty else 0.0
    if olu.empty:
        st.success(f"{DEAD_STOCK_GUN} günü aşan kayıt yok.")
    else:
        goster = olu.copy()
        goster[COL_FIYAT] = goster[COL_FIYAT].map(lambda x: f"{x:,.0f}".replace(",", "."))
        st.dataframe(goster, use_container_width=True, hide_index=True)
    st.info(yorum_olu_stok(olu, kilitli))

    st.markdown("##### İller arası transfer öneri sistemi")
    tr = transfer_onerileri(df)
    if tr.empty:
        st.success("Eşleşen transfer fırsatı yok.")
    else:
        st.dataframe(tr, use_container_width=True, hide_index=True)
    st.success(yorum_transfer(tr))


# ---------------------------------------------------------------------------
# Uygulama
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
            [data-testid="stMetric"] {
                background: linear-gradient(180deg, #F4F7FB 0%, #FFFFFF 100%);
                border: 1px solid #D9E2EC;
                border-radius: 14px;
                padding: 12px 14px;
                box-shadow: 0 4px 14px rgba(11, 58, 91, 0.06);
            }
            [data-testid="stMetricLabel"] {color: #486581; font-weight: 600;}
            [data-testid="stMetricValue"] {color: #0B3A5B; font-weight: 700;}
            h1, h2, h3 {color: #0B3A5B;}
            div[data-testid="stSidebar"] {background-color: #F7F9FC;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🚗", layout="wide")
    inject_css()
    st.title("Borusan Otomotiv Karar Destek Sistemi")
    st.caption("Satış performansı, müşteri davranışı, mevsimsellik ve stok erken uyarı — üst yönetim dashboard'u")

    try:
        raw = load_raw_data()
        data = preprocess(raw)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Veri yüklenemedi: {exc}")
        st.stop()
        return

    with st.sidebar:
        st.header("Global Filtreler")
        st.caption("Filtreler tüm sekmeleri etkiler.")
        iller = sorted(data[COL_IL].dropna().unique().tolist())
        markalar = sorted(data[COL_MARKA].dropna().unique().tolist())
        tipler = sorted(data[COL_MUSTERI_TIPI].dropna().unique().tolist())

        sec_il = st.multiselect("Bayi İli", iller, default=iller)
        sec_marka = st.multiselect("Marka", markalar, default=markalar)
        sec_tip = st.multiselect("Müşteri Tipi", tipler, default=tipler)

        min_d = data[COL_TARIH].min().date()
        max_d = data[COL_TARIH].max().date()
        tarih = st.date_input("Tarih / ay aralığı", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        st.markdown("---")
        st.caption("Kaynak: borusan_otomotiv_gelismis_stok_analizi_veriseti.csv")

    if isinstance(tarih, (list, tuple)) and len(tarih) == 2:
        tarih_araligi = tarih
    else:
        tarih_araligi = (tarih, tarih)

    try:
        filtered = apply_filters(data, sec_il, sec_marka, sec_tip, tarih_araligi)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Filtre uygulanamadı: {exc}")
        st.stop()
        return

    st.markdown(
        f"Aktif görünüm: **{len(filtered)}** kayıt / **{len(data)}** toplam · "
        f"{filtered[COL_TARIH].min().date() if not filtered.empty else '—'} → "
        f"{filtered[COL_TARIH].max().date() if not filtered.empty else '—'}"
    )

    t1, t2, t3, t4 = st.tabs(
        [
            "Genel Performans & Yönetici Özeti",
            "Demografi & Müşteri Davranışı",
            "Mevsimsellik & Zaman Trendleri",
            "Akıllı Stok & Erken Uyarı (KDS)",
        ]
    )

    with t1:
        try:
            if not empty_guard(filtered):
                tab_executive(filtered)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Yönetici özeti hesaplanamadı: {exc}")

    with t2:
        try:
            if not empty_guard(filtered):
                tab_demografi(filtered)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Demografi analizi hesaplanamadı: {exc}")

    with t3:
        try:
            if not empty_guard(filtered):
                tab_zaman(filtered)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Zaman trendi hesaplanamadı: {exc}")

    with t4:
        try:
            if not empty_guard(filtered):
                tab_stok_kds(filtered)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Stok KDS motoru çalıştırılamadı: {exc}")


if __name__ == "__main__":
    main()
