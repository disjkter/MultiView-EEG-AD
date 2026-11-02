from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUTPUT = Path("output/docx/REVE论文学习报告与下一步研究设想.docx")


def set_run_font(run, latin: str, east_asia: str, size: float, bold: bool = False):
    run.font.name = latin
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_bottom_border(paragraph, color="D9D9D9", size="6", space="4"):
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), space)
    bottom.set(qn("w:color"), color)
    p_bdr.append(bottom)


def remove_paragraph_borders(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is not None:
        p_pr.remove(p_bdr)


def configure_styles(doc: Document):
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    normal.paragraph_format.line_spacing = 1.18
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    title = doc.styles["Title"]
    title.font.name = "Times New Roman"
    title.font.size = Pt(17)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(7)
    title_p_pr = title._element.find(qn("w:pPr"))
    if title_p_pr is not None:
        title_border = title_p_pr.find(qn("w:pBdr"))
        if title_border is not None:
            title_p_pr.remove(title_border)

    for style_name in ("Heading 1",):
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(12.5)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.paragraph_format.space_before = Pt(7)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True


def add_body_paragraph(doc: Document, text: str, first_line=True):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(22) if first_line else Pt(0)
    p.paragraph_format.keep_together = True
    r = p.add_run(text)
    set_run_font(r, "Times New Roman", "宋体", 11)
    return p


def add_numbered_idea(doc: Document, number: str, label: str, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(0)
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_together = True
    r = p.add_run(f"{number}  {label}：")
    set_run_font(r, "Times New Roman", "黑体", 11, bold=True)
    r = p.add_run(text)
    set_run_font(r, "Times New Roman", "宋体", 11)


def build_report():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.62)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)
    section.header_distance = Inches(0.25)
    section.footer_distance = Inches(0.30)
    configure_styles(doc)

    title = doc.add_paragraph(style="Title")
    title.add_run("REVE论文学习报告与下一步研究设想")
    remove_paragraph_borders(title)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_after = Pt(9)
    r = meta.add_run("学习论文  REVE A Foundation Model for EEG  NeurIPS 2025")
    set_run_font(r, "Times New Roman", "宋体", 9.5)
    r.font.color.rgb = RGBColor(89, 89, 89)

    h1 = doc.add_paragraph("一 论文概况", style="Heading 1")
    add_bottom_border(h1)
    overview = (
        "REVE是NeurIPS 2025发表的脑电基础模型，目标是让同一个模型适应不同的导联布局和信号长度。"
        "该方法把每个通道的脑电信号切分为时间片段，利用电极三维坐标和时间位置构成四维位置编码，并通过时空块掩码重建进行自监督预训练。"
        "作者汇集了92个数据集、24,274名参与者和61,415小时脑电，在运动想象、癫痫检测、睡眠分期、认知负荷和情绪识别等10项下游任务上取得了较强结果。"
        "这篇论文最值得我们借鉴的地方，是把不同数据集之间通道数、导联位置和记录长度不统一的问题直接纳入模型设计，从而降低模型对固定输入设置和大量任务标注的依赖。"
    )
    add_body_paragraph(doc, overview)

    h2 = doc.add_paragraph("二 下一步研究想法", style="Heading 1")
    add_bottom_border(h2)
    relation = (
        "我们当前的工作重点解决阿尔茨海默病脑电分类中标注参与者较少的问题，但仍依赖统一的19通道、采样率和窗口长度。"
        "下一项工作可以在此基础上进一步研究不同设备、导联配置和数据集之间的差异，形成从固定设置下的参与者高效学习到异构设置下跨数据集泛化的递进关系。"
    )
    add_body_paragraph(doc, relation)

    add_numbered_idea(
        doc,
        "1",
        "研究问题",
        "在不要求各数据集使用完全相同导联配置的条件下，如何学习稳定的脑电表示，并用少量标注参与者完成跨数据集的阿尔茨海默病分类。",
    )
    add_numbered_idea(
        doc,
        "2",
        "方法设想",
        "先依据电极位置和通道关系构建脑电图结构，采用图掩码自编码器学习通用表示；再通过疾病分类分支与域分类分支进行因果解耦，使模型保留疾病相关信息并减弱设备和数据集差异的影响。",
    )
    add_numbered_idea(
        doc,
        "3",
        "验证方案",
        "在CAUEEG和ds004504等队列上开展留一数据集验证，严格按参与者划分训练集与测试集，并采用参与者级概率投票。首先完成冻结编码器和轻量微调基线，再验证图预训练与因果解耦是否带来稳定提升。",
    )

    direction = doc.add_paragraph()
    direction.paragraph_format.space_before = Pt(3)
    direction.paragraph_format.space_after = Pt(5)
    direction.paragraph_format.keep_together = True
    r = direction.add_run("拟定方向  ")
    set_run_font(r, "Times New Roman", "黑体", 10.5, bold=True)
    r = direction.add_run("Topology-Aware EEG Foundation Model with Causal Decoupling for Cross-Dataset Alzheimer’s Disease Classification")
    set_run_font(r, "Times New Roman", "宋体", 10.5)

    ref = doc.add_paragraph()
    ref.paragraph_format.space_before = Pt(4)
    ref.paragraph_format.space_after = Pt(0)
    add_bottom_border(ref, color="BFBFBF", size="4", space="5")
    r = ref.add_run("参考文献  El Ouahidi et al. REVE: A Foundation Model for EEG Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects. NeurIPS 2025.")
    set_run_font(r, "Times New Roman", "宋体", 8.8)
    r.font.color.rgb = RGBColor(89, 89, 89)

    doc.core_properties.title = "REVE论文学习报告与下一步研究设想"
    doc.core_properties.subject = "EEG基础模型学习报告"
    doc.core_properties.author = "梁正扬"
    doc.save(OUTPUT)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    build_report()
