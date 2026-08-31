"""Build the doctor-facing integrated ultrasound AI research summary."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs/project/膝关节超声AI识别研究工作总结与阶段结果_2026-08-30.docx"

BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
NAVY = "203748"
MUTED = "667085"
LIGHT_GRAY = "F2F4F7"
PALE_BLUE = "EAF2F8"
PALE_GOLD = "FFF7E0"
WHITE = "FFFFFF"
BLACK = "202124"
TABLE_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120


def set_run_font(
    run,
    *,
    size: float | None = None,
    color: str | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
    east_asia: str = "Microsoft YaHei",
) -> None:
    run.font.name = "Calibri"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Calibri")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Calibri")
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_margins(cell, top: int = 80, bottom: int = 80, start: int = 120, end: int = 120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("bottom", bottom), ("start", start), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_fill(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_table_geometry(table, widths_dxa: list[int]) -> None:
    if sum(widths_dxa) != TABLE_WIDTH_DXA:
        raise ValueError("Table widths must total 9360 DXA")
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(TABLE_WIDTH_DXA))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(TABLE_INDENT_DXA))
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            cell.width = Inches(widths_dxa[index] / 1440)
            tc_w = cell._tc.get_or_add_tcPr().first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                cell._tc.get_or_add_tcPr().append(tc_w)
            tc_w.set(qn("w:w"), str(widths_dxa[index]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_paragraph_border_and_fill(paragraph, *, fill: str, border: str = "B8C7D9") -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    p_pr.append(shading)
    borders = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "6")
        node.set(qn("w:space"), "6")
        node.set(qn("w:color"), border)
        borders.append(node)
    p_pr.append(borders)


def add_page_field(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    set_run_font(run, size=9, color=MUTED)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    field_run = OxmlElement("w:r")
    field_run.append(begin)
    field_run.append(instr)
    field_run.append(separate)
    field_run.append(text)
    field_run.append(end)
    paragraph._p.append(field_run)
    tail = paragraph.add_run(" 页")
    set_run_font(tail, size=9, color=MUTED)


def style_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    section.different_first_page_header_footer = True

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.color.rgb = RGBColor.from_string(BLACK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10
    normal.paragraph_format.widow_control = True

    for style_name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ):
        style = styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True

    for list_name in ("List Bullet", "List Number"):
        style = styles[list_name]
        style.font.name = "Calibri"
        style.font.size = Pt(11)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.paragraph_format.left_indent = Inches(0.5)
        style.paragraph_format.first_line_indent = Inches(-0.25)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.line_spacing = 1.167

    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hr = hp.add_run("膝关节超声AI识别研究 | 阶段工作总结")
    set_run_font(hr, size=9, color=MUTED)
    add_page_field(section.footer.paragraphs[0])


def add_cover(doc: Document) -> None:
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(82)
    kicker = doc.add_paragraph()
    kicker.alignment = WD_ALIGN_PARAGRAPH.CENTER
    kicker.paragraph_format.space_after = Pt(16)
    run = kicker.add_run("阶段研究报告 · 医院沟通稿")
    set_run_font(run, size=11, color=BLUE, bold=True)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(12)
    run = title.add_run("膝关节超声AI识别\n研究工作总结与阶段结果")
    set_run_font(run, size=28, color=NAVY, bold=True)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(42)
    run = subtitle.add_run("覆盖患者多图分类、自动有效视野、临床信息融合与语义病灶轮廓研究")
    set_run_font(run, size=13, color=MUTED)

    for label, value in (
        ("报告日期", "2026年8月30日"),
        ("研究阶段", "开发研究完成；“膝关节2026未标注”数据的预测结果已固定，医院参考诊断待回收"),
        ("报告用途", "向医院说明已完成工作、阶段证据、当前边界及后续协作事项"),
    ):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(f"{label}：")
        set_run_font(r, size=10.5, color=DARK_BLUE, bold=True)
        r = p.add_run(value)
        set_run_font(r, size=10.5, color=BLACK)

    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(46)
    note.paragraph_format.space_after = Pt(0)
    note.paragraph_format.left_indent = Inches(0.35)
    note.paragraph_format.right_indent = Inches(0.35)
    set_paragraph_border_and_fill(note, fill=PALE_GOLD, border="D9B65D")
    r = note.add_run("使用边界：")
    set_run_font(r, size=10.5, color="7A5A00", bold=True)
    r = note.add_run("本报告为合作研究阶段总结，不授权AI独立诊断，也不替代医生形成的临床诊断。")
    set_run_font(r, size=10.5, color=BLACK)
    doc.add_page_break()


def add_heading(doc: Document, text: str, level: int = 1):
    return doc.add_paragraph(text, style=f"Heading {level}")


def add_body(doc: Document, text: str, *, bold_lead: str | None = None):
    p = doc.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        first = p.add_run(bold_lead)
        set_run_font(first, bold=True)
        rest = p.add_run(text[len(bold_lead) :])
        set_run_font(rest)
    else:
        run = p.add_run(text)
        set_run_font(run)
    return p


def add_bullet(doc: Document, text: str):
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    set_run_font(run)
    return p


def create_numbering_instance(doc: Document) -> int:
    """Create a fresh decimal-list instance that starts at one."""

    numbering = doc.part.numbering_part.element
    style = doc.styles["List Number"]
    base_num_id = int(style._element.pPr.numPr.numId.val)
    base_num = numbering.find(f"./w:num[@w:numId='{base_num_id}']", numbering.nsmap)
    if base_num is None:
        raise ValueError("List Number numbering definition is unavailable")
    abstract_num_id = base_num.find("w:abstractNumId", numbering.nsmap).get(qn("w:val"))
    existing = [int(node.get(qn("w:numId"))) for node in numbering.findall("w:num", numbering.nsmap)]
    new_num_id = max(existing) + 1
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(new_num_id))
    abstract = OxmlElement("w:abstractNumId")
    abstract.set(qn("w:val"), abstract_num_id)
    num.append(abstract)
    override = OxmlElement("w:lvlOverride")
    override.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:startOverride")
    start.set(qn("w:val"), "1")
    override.append(start)
    num.append(override)
    numbering.append(num)
    return new_num_id


def add_number(doc: Document, text: str, *, num_id: int):
    p = doc.add_paragraph(style="List Number")
    p_pr = p._p.get_or_add_pPr()
    num_pr = p_pr.get_or_add_numPr()
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num_id_element = OxmlElement("w:numId")
    num_id_element.set(qn("w:val"), str(num_id))
    num_pr.append(ilvl)
    num_pr.append(num_id_element)
    run = p.add_run(text)
    set_run_font(run)
    return p


def add_callout(doc: Document, label: str, text: str, *, fill: str = PALE_BLUE):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.right_indent = Inches(0.18)
    set_paragraph_border_and_fill(p, fill=fill)
    run = p.add_run(f"{label}：")
    set_run_font(run, color=DARK_BLUE, bold=True)
    run = p.add_run(text)
    set_run_font(run)
    return p


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths_dxa: list[int]):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    set_repeat_header(table.rows[0])
    for index, value in enumerate(headers):
        cell = table.rows[0].cells[index]
        set_cell_fill(cell, LIGHT_GRAY)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(value)
        set_run_font(run, size=9.5, color=NAVY, bold=True)
    for row_values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row_values):
            p = cells[index].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            if index > 0 and len(value) < 24:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(value)
            set_run_font(run, size=9.25)
    set_table_geometry(table, widths_dxa)
    after = doc.add_paragraph()
    after.paragraph_format.space_before = Pt(0)
    after.paragraph_format.space_after = Pt(2)
    return table


def build_document() -> Document:
    doc = Document()
    style_document(doc)
    add_cover(doc)

    add_heading(doc, "一、执行摘要")
    add_body(
        doc,
        "本项目围绕膝关节超声图像的有效视野定位、患者多图六分类、临床Excel信息融合及人工病灶轮廓辅助诊断开展研究。开发队列共967名患者、4,543张图像；另对“膝关节2026未标注”数据中的225名患者、1,819张图像完成了一次固定预测。",
    )
    add_body(
        doc,
        "现阶段已经完成本机可开展的主要开发研究、阴性路线审计、自动有效视野验证、人工病灶轮廓研究以及“膝关节2026未标注”数据的预测交付。尚未完成的是医院为该数据补充独立参考诊断后的最终效果评价。",
    )
    add_callout(
        doc,
        "核心结论",
        "在现有历史内部数据中，医生提供的分类别病灶结构化信息（M1）取得最高macro-F1 0.6351；但这一结果包含真实医学征象与历史标注流程的共同影响，不能直接表述为AI独立看图准确率。现有全自动图像六分类能力仍有限。",
    )
    add_heading(doc, "二、研究目标与评价框架")
    add_body(
        doc,
        "研究目标是在医院项目可接受的复杂度下，形成一条可复核的膝关节超声患者级疾病识别路线，并明确全自动与医生参与的半自动方案各自能达到的阶段水平。疾病分为正常、类风湿性关节炎、痛风性关节炎、脊柱关节炎、骨性关节炎和损伤六类。",
    )
    add_table(
        doc,
        ["项目", "设计"],
        [
            ["开发数据", "967名患者、4,543张图像；固定患者级五折，患者不跨折"],
            ["膝关节2026未标注", "225名患者、1,819张图像；已完成一次固定预测"],
            ["主要指标", "患者级macro-F1，并同时观察逐类灵敏度、F1、Top-2和校准"],
            ["数据隔离", "外层测试不参与训练、早停或选模；“膝关节2026未标注”数据的参考诊断未用于路线选择"],
            ["研究原则", "保存阴性结果；控制复杂度；不在同一OOF上反复搜索参数"],
        ],
        [2300, 7060],
    )
    add_body(
        doc,
        "说明：macro-F1是六类F1的平均值，每类权重相同，适合观察少数类别是否被总体准确率掩盖；它不是单纯的“正确人数比例”。",
    )

    add_heading(doc, "三、研究过程与路线演进")
    add_heading(doc, "3.1 图像基线与患者多图学习", level=2)
    add_body(
        doc,
        "研究先建立整图及人工有效视野ROI图像基线，再比较患者级多图训练。患者级学习能够提高内部macro-F1，但复杂注意力模型出现注意力过度集中、折间不稳和过度自信，因此没有仅凭最高内部得分直接选为交付模型。",
    )
    add_bullet(doc, "E1：完整有效视野ROI内每张图分别预测，再做患者级概率平均，macro-F1 0.2155。")
    add_bullet(doc, "E2：端到端患者多图注意力模型macro-F1 0.3663，但注意力塌缩率0.6225、ECE 0.4519。")
    add_bullet(doc, "E3：无注意力患者级训练macro-F1 0.3010，仍未达到预设稳定性和安全门。")

    add_heading(doc, "3.2 自动有效视野定位", level=2)
    add_body(
        doc,
        "有效视野ROI用于去除设备界面和无关边框，不是病灶区域。由于有效视野边界不存在唯一答案，本研究没有使用僵硬的IoU门槛，而是比较自动ROI对最终患者分类的影响并进行代表病例审阅。",
    )
    add_bullet(doc, "4,543张开发图像均成功得到自动有效视野；代表病例未见系统性坐标故障。")
    add_bullet(doc, "自动/人工ROI条件下macro-F1为0.2189/0.2155，差+0.0034，置信区间跨0。")
    add_bullet(doc, "结论：医院无需为有效视野另画矩形，自动有效视野可以进入研究流程。")

    add_heading(doc, "3.3 Excel临床信息", level=2)
    add_body(
        doc,
        "C3仅使用冻结的八项允许数值，在767名异常患者内部完成五分类，macro-F1为0.7185。该结果说明临床信息很强，但开发正常组缺少同标准Excel，且检查项目选择本身与疾病有关，因此不能把C3单独解释为完整六分类临床性能。",
    )

    add_heading(doc, "3.4 人工病灶轮廓：从无语义到分类别语义", level=2)
    add_body(
        doc,
        "早期L0保留完整ROI图像，并把所有病灶合并为一个二值mask通道。它较纯图像略有提高，但仅凭“是否被画、画了多少、面积多大”就能得到更高分，说明标注行为本身与疾病高度相关。因此研究进一步开展M0-M3分类别语义轮廓实验，避免把不同病灶全部合并。",
    )
    m_series_numbering = create_numbering_instance(doc)
    add_number(doc, "M0：完整有效视野ROI RGB图像基线。", num_id=m_series_numbering)
    add_number(doc, "M1：13类病灶的存在性、对象数量和面积比例，共39项患者级结构化征象。", num_id=m_series_numbering)
    add_number(doc, "M2：每张完整ROI同时读取RGB和13类可重叠轮廓通道，保留病灶外信息、相对位置和患者多图。", num_id=m_series_numbering)
    add_number(doc, "M3：在M2基础上固定加入Excel C3，不搜索融合权重。", num_id=m_series_numbering)

    add_heading(doc, "四、主要阶段结果")
    add_table(
        doc,
        ["方案", "患者级输入", "macro-F1（95% CI）", "阶段解释"],
        [
            ["M0 / E1", "ROI图像", "0.2155（0.1912-0.2389）", "纯图像基线，绝对性能不足"],
            ["M1", "病灶类别、数量、面积", "0.6351（0.6039-0.6647）", "内部最高；医生结构化征象路线"],
            ["M2", "ROI图像+13类轮廓", "0.3327（0.2991-0.3648）", "高于M0，但明显低于M1"],
            ["M0+C3", "图像正常门控+Excel异常分类", "0.6018（0.5715-0.6318）", "更简单的自动分层对照"],
            ["M3", "M2+Excel C3固定融合", "0.5676（0.5360-0.5977）", "高于M2，但低于M0+C3"],
        ],
        [1350, 2450, 2260, 3300],
    )
    add_table(
        doc,
        ["关键比较", "macro-F1差", "配对95% CI", "正向fold", "结论"],
        [
            ["M1-M0", "+0.4196", "+0.3811至+0.4589", "5/5", "结构化征象明显强于图像基线"],
            ["M2-M0", "+0.1172", "+0.0799至+0.1557", "5/5", "语义轮廓通道对纯图像有帮助"],
            ["M2-M1", "-0.3024", "-0.3469至-0.2562", "0/5", "未证明形状、纹理超过结构化征象"],
            ["M3-M2", "+0.2348", "+0.1958至+0.2735", "5/5", "主要增益来自Excel临床信息"],
            ["M3-(M0+C3)", "-0.0342", "-0.0578至-0.0096", "2/5", "固定M3未超过更简单对照"],
        ],
        [1850, 1450, 2050, 1200, 2810],
    )
    add_callout(
        doc,
        "阶段判断",
        "M1是现有内部研究中数值最好的方式；M2证明分类别轮廓可以帮助图像模型，但没有复现M1的结构化信息优势；M3加入Excel后仍未超过更简单的M0+C3。因此当前不继续在同一历史数据上搜索语义网络或融合权重。",
    )

    add_heading(doc, "五、如何正确理解M1")
    add_heading(doc, "5.1 M1实际使用了什么", level=2)
    add_body(
        doc,
        "M1不读取RGB图像，也不直接分析轮廓形状。医生在图像上标明病灶类别及轮廓后，系统按患者汇总13类病灶是否出现、对象数量和面积比例，再进行六分类。也就是说，它评估的是“医生结构化病灶征象+AI患者级分类”的半自动路线。",
    )
    add_heading(doc, "5.2 M1为什么得分较高", level=2)
    add_body(
        doc,
        "痛风石、结晶、骨赘等病灶类别本身具有明确医学信息，M1能够直接利用这些类别。同时，历史标注并非逐图穷尽，医生的切面选择、是否标注及可能已知诊断也会进入模型。因此0.6351同时包含医学信号与标注流程信号。",
    )
    add_heading(doc, "5.3 当前能够和不能够得出的结论", level=2)
    add_bullet(doc, "可以说：M1是本轮历史内部五折中表现最好的候选，标准化结构化病灶征象值得进一步验证。")
    add_bullet(doc, "不能说：AI仅凭超声图像已经达到0.6351，或M1已经具备临床独立诊断能力。")
    add_bullet(doc, "不能直接把M1结果用于“膝关节2026未标注”数据重选模型；已经生成的预测结果必须保持不变。")
    add_heading(doc, "5.4 若医院继续验证M1", level=2)
    add_body(
        doc,
        "建议采用统一的新标注协议：逐图记录病灶存在、不存在、当前切面不可见或不确定；轮廓必须带明确类别；标注者尽量不知道最终诊断和AI结果。新患者数据应作为独立验证，不回填现有五折。可先验证M1的低复杂度结构化路线，再决定是否投入更复杂的M2图像轮廓模型。",
    )

    add_heading(doc, "六、“膝关节2026未标注”数据与当前交付状态")
    add_body(
        doc,
        "“膝关节2026未标注”数据共225名患者、1,819张图像及225行同结构临床表。研究已经完成一次固定推理：五个自动有效视野模型均成功定位，0张图使用整图兜底；图像单模态和分层多模态两个结果均已保留。推理时没有读取参考诊断。",
    )
    add_callout(
        doc,
        "尚不能报告的结果",
        "目前不能给出“膝关节2026未标注”数据的真实macro-F1、准确率或逐类灵敏度，因为医院参考诊断尚未回收。模型预测分布不能代替真实诊断分布，也不能据此判断模型好坏。",
        fill=PALE_GOLD,
    )
    add_body(
        doc,
        "医院完成结构化参考诊断回表后，将按预先确定的评价方案一次性计算图像单模态与分层多模态的总体、逐类、校准、置信亚组和配对结果；不得根据“膝关节2026未标注”数据的结果回头调参。",
    )

    add_heading(doc, "七、研究局限与风险")
    for item in (
        "现有开发数据为同源单项目队列，设备、操作者、采集数量、界面和检查流程可能形成非医学捷径。",
        "图像六分类绝对性能仍不足，部分模型存在折间不稳、类别不安全或校准问题。",
        "历史人工病灶轮廓不是逐图穷尽标注；没有轮廓不等于医学阴性。",
        "M1的高分不能区分真实医学特异性与标注者已知诊断后的标注选择。",
        "C3开发数据的可用性与异常诊断相关，不属于随机缺失，内部高分可能包含检查流程偏差。",
        "自动有效视野只负责去除界面和无关区域，不等于自动病灶识别。",
        "本研究尚未完成跨中心、前瞻性临床决策影响或医疗器械级安全评价。",
    ):
        add_bullet(doc, item)

    add_heading(doc, "八、阶段结论与建议")
    conclusion_numbering = create_numbering_instance(doc)
    add_number(doc, "本机可开展的主要开发研究已经完成，包含成功结果、阴性结果和停止决定；不建议继续在现有历史数据上追加同源调参。", num_id=conclusion_numbering)
    add_number(doc, "当前全自动研究路线仍采用自动有效视野、E1图像模型和C3分层公式；该路线已对“膝关节2026未标注”数据完成固定预测，但最终效果必须等待医院参考诊断。", num_id=conclusion_numbering)
    add_number(doc, "M1是内部数据表现最好的半自动候选，适合成为未来标准化新标注的优先验证对象；在独立验证前不作为临床交付模型。", num_id=conclusion_numbering)
    add_number(doc, "若医院暂不提供“膝关节2026未标注”数据的参考诊断回表或新轮廓标注，当前阶段可以结束并作为基础研究成果归档。", num_id=conclusion_numbering)

    add_heading(doc, "九、建议医院协作事项")
    add_table(
        doc,
        ["优先级", "协作事项", "目的", "当前状态"],
        [
            ["第一", "完成“膝关节2026未标注”数据的结构化六分类参考诊断回表", "形成该数据的最终效果与验收结论", "待医院反馈"],
            ["第二（可选）", "讨论是否开展统一协议的新病灶类别/轮廓标注", "独立验证M1半自动结构化征象路线", "医院决定后另行预登记"],
            ["第三（可选）", "确认参考诊断来源及医生是否见过AI结果", "评估参考标准可能受到的AI可见性影响", "随该数据的回表记录"],
        ],
        [1200, 3300, 3260, 1600],
    )

    add_heading(doc, "十、研究过程留痕与可复核性")
    add_body(
        doc,
        "研究已保存预登记方案、配置文件、患者级五折OOF结果、逐类指标、bootstrap区间、模型权重或融合输入、运行日志、MLflow记录、Git版本、文件哈希、阴性实验和架构决策记录。“膝关节2026未标注”数据的预测结果与医院回表评价方案均已独立固定。以上技术材料作为内部审计附件保留，本报告仅呈现医生判断所需的主要证据。",
    )
    add_body(
        doc,
        "最终工作边界：本报告总结截至2026年8月30日已经完成的开发工作，以及对“膝关节2026未标注”数据完成的固定预测；医院参考诊断返回后对该数据开展的R3独立评价及最终验收结论，不在本报告已完成范围内。",
    )

    return doc


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = build_document()
    doc.core_properties.title = "膝关节超声AI识别研究工作总结与阶段结果"
    doc.core_properties.subject = "医院合作项目阶段研究总结"
    doc.core_properties.author = "膝关节超声AI识别研究项目组"
    doc.core_properties.keywords = "膝关节超声, AI识别, 患者级分类, 阶段研究"
    doc.core_properties.comments = "由项目研究记录整合生成；研究用途，不授权AI独立诊断。"
    doc.save(OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
