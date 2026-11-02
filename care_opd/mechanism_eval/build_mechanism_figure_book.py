#!/usr/bin/env python3
"""Build a figures-only Word report for the CARE-OPD mechanism study."""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
CARE = "D55E00"
TEAL = "2A9D8F"
BLUE = "0072B2"
LIGHT = "EAF2F1"


def set_east_asia_font(run, name: str = "微软雅黑") -> None:
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), name)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_heading(text, level=level)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(5)
    for r in p.runs:
        set_east_asia_font(r)
        r.font.color.rgb = RGBColor.from_string("264653")


def add_body(doc: Document, text: str, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(4)
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        r.bold = True
        rest = text[len(bold_lead):]
        p.add_run(rest)
    else:
        p.add_run(text)
    for r in p.runs:
        r.font.size = Pt(9.5)
        set_east_asia_font(r)


def add_figure(doc: Document, file: str, caption: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIGURES / file), width=Cm(23.7))
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cp.add_run(caption)
    r.italic = True
    r.font.size = Pt(8.5)
    set_east_asia_font(r)


def build(output: Path) -> None:
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Cm(1.35)
    section.bottom_margin = Cm(1.35)
    section.left_margin = Cm(1.45)
    section.right_margin = Cm(1.45)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("CARE-OPD 机制实验图解报告")
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor.from_string("264653")
    set_east_asia_font(r)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("行为恢复 · 局部监督 · 参数内化 · 在线失败迁移")
    sr.font.size = Pt(11)
    sr.font.color.rgb = RGBColor.from_string("5B6570")
    set_east_asia_font(sr)
    add_body(doc, "核心结论：CARE-OPD 是比较方法中唯一获得正向匹配净收益的方法；其冻结参数首次平均偏好 verifier 验证通过的修复回答，并且相对 GKD 的额外改善在 verifier 选中的修复字段上更大。三轮失败集合的高周转进一步说明学生错误分布随策略变化，支持反复进行 on-policy 重采样。")

    add_heading(doc, "1　行为层：失败救回与已有能力保持")
    add_figure(doc, "fig1_behavior_rescue_regression.png",
               "图1　以 Clean-SFT 为固定参照的恢复—保持权衡。CARE-OPD 救回 91.61% 的参照失败，保留 80.44% 的参照成功，得到 +509 条净成功；条件切片由 Clean-SFT 冻结定义。")
    add_body(doc, "可用结论：CARE 的优势来自高救回率与可控回退的联合，而不是简单提高回答激进程度。Recovery-SFT 的回退率略低于 CARE，但只救回 36.10% 的失败；CARE 在综合权衡上明显最好。", "可用结论：")

    add_heading(doc, "2　监督层与参数层：修复字段定位和偏好反转")
    add_figure(doc, "fig2_localization_and_margin.png",
               "图2　左：verifier 选中的修复字段覆盖 63.67% 的响应 token。右：长度归一化 oracle−rejected teacher-forced margin；正值表示冻结模型更偏好验证通过的修复回答。")
    add_body(doc, "机制对比：GKD 相对 Clean-SFT 已明显削弱错误偏好，但均值仍为负；CARE 在 GKD 基础上将 full/localized margin 进一步提高 +1.4205/+1.6331 并完成符号翻转。CARE 相对 GKD 的额外收益在 selected fields 上比 outside fields 多 +0.6451，但该差异目前没有置信区间，只作描述性机制证据。", "机制对比：")

    add_heading(doc, "3　过程层：在线轮次与非平稳失败分布")
    add_figure(doc, "fig3_online_round_dynamics.png",
               "图3　三轮总体通过率上升，但后续失败中仍有大量新增状态。failure-set turnover 支持每轮模型更新后重新收集 on-policy 失败。")
    add_body(doc, "解释边界：通过率从 72.08% 提高到 77.24%，总失败相对减少 18.48%；但语义失败在第二至第三轮增加 9 条，不能描述为逐轮单调下降。由于 checkpoint 与采样随机性同时变化，集合转移只能证明失败分布非平稳，不能作为样本级因果修复证明。", "解释边界：")

    add_heading(doc, "4　总体证据链与可主张范围")
    for text in [
        "行为层：CARE-OPD 救回 1048 条 Clean-SFT 失败，同时回退 539 条原成功，净增加 509 条成功。",
        "监督层：局部掩码只监督 63.67% 的响应 token，避免 36.33% 的外围 token 被无差别蒸馏；投影缺失、零选中和截断均为 0%。",
        "参数层：只有 CARE 的平均 full、localized 和 outside margin 全部由负转正，且相对 GKD 的额外收益在 selected fields 上更大。",
        "过程层：更新后的模型持续产生新失败状态，为多轮 on-policy 采样提供直接动机。",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(text)
        for rr in p.runs:
            rr.font.size = Pt(9.5)
            set_east_asia_font(rr)
    add_body(doc, "可主张：四层结果共同支持“on-policy失败状态 + verifier局部修复 + 风险/信任/保持约束”的联合 CARE-OPD 机制。不可主张：单个组件的独立因果贡献、margin等于生成准确率、或该QZBC条件诊断已经证明跨领域通用泛化。", "可主张：")

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "CARE-OPD_机制实验图解报告.docx")
    args = parser.parse_args()
    build(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
