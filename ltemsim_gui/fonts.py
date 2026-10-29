"""matplotlib font set-up: keep DejaVu Sans (the scripts' look) and fall back to a CJK font for Chinese text."""
from __future__ import annotations

CJK_CANDIDATES = ("PingFang SC", "Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "Microsoft YaHei",
                  "Noto Sans CJK SC", "WenQuanYi Zen Hei")


def install_cjk_fallback():
    """Only an explicit font.family list falls back per glyph; the generic 'sans-serif' family does not."""
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((n for n in CJK_CANDIDATES if n in names), None)
    fam = matplotlib.rcParams["font.family"]
    fam = [fam] if isinstance(fam, str) else list(fam)
    if fam == ["sans-serif"]:
        fam = ["DejaVu Sans"]
    if cjk and cjk not in fam:
        fam.append(cjk)
    matplotlib.rcParams["font.family"] = fam
    matplotlib.rcParams["axes.unicode_minus"] = False
