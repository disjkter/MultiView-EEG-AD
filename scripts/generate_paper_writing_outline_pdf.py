"""Generate a Chinese PDF for the current paper writing outline."""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


TITLE = "论文写作大纲（ESWA 风格）"
SUBTITLE = "面向跨数据集 EEG 阿尔茨海默症检测的多阶段迁移学习框架"

SECTIONS = [
    (
        "题目方向",
        [
            "Staged transfer learning with multi-source EEG priors for cross-dataset Alzheimer's disease detection",
            "A multi-stage transfer framework with source-task enhanced initialization for cross-dataset EEG-based Alzheimer's disease detection",
            "Learning transferable EEG priors through multi-source pretraining for cross-dataset Alzheimer's disease detection",
        ],
    ),
    (
        "摘要",
        [
            "背景：基于 EEG 的阿尔茨海默症检测具有应用潜力，但跨数据集泛化仍然受强 domain gap 限制。",
            "问题：源域上学到的 decision boundary 无法直接迁移到新的目标数据集。",
            "方法：采用多源时间-频率一致性预训练、source-task 增强初始化以及目标域学习的三阶段框架。",
            "实验：在 CAUEEG 和 ds004504 上评估单数据集微调、direct transfer、source init + target fine-tuning 和初始化消融。",
            "ESWA 风格摘要重点：方法动机、任务场景、系统对照实验、应用价值、核心量化结论。",
            "预期摘要结论：multi-source pretraining 有效且更稳；source-task knowledge 可在目标域重新利用；direct transfer 整体失败。",
        ],
    ),
    (
        "1. 引言",
        [
            "介绍 AD 早筛的重要性，以及 EEG 作为低成本、非侵入式信号的优势。",
            "指出现有方法大多停留在单数据集高性能，但跨数据集泛化能力不足。",
            "提出本文核心思想：不是直接迁移 source decision boundary，而是迁移通用表征先验与 source-task knowledge。",
            "ESWA 风格写法：引言要从真实应用问题切入，再自然过渡到方法缺口和本文方案。",
            "本节结尾给出三点式贡献总结。",
        ],
    ),
    (
        "1.4 创新点（建议单独列出）",
        [
            "创新点 1：问题层面，将研究重点从单数据集 AD/HC 分类提升到跨数据集 EEG 阿尔茨海默症泛化，明确把 cross-dataset generalization 作为核心问题。",
            "创新点 2：方法层面，提出一个分阶段迁移框架：Stage 1 学习 multi-source representation prior，Stage 2 构建 source-task enhanced prior，Stage 3 在目标域中重新利用这些先验。",
            "创新点 3：机制层面，显式区分并比较通用表征先验与源域任务先验，系统分析不同层次先验知识对目标域学习的贡献。",
            "创新点 4：实验发现层面，揭示了 direct transfer failure 与 source knowledge reuse 的关系：source-domain decision boundary 不能直接跨域复用，但 source-domain knowledge 经过目标域 fine-tuning 后仍具有迁移价值。",
            "创新点 5：通过 single-source 与 multi-source 初始化消融，证明 multi-source pretraining 相比 single-source 更稳定、更平衡，是更合理的默认初始化策略。",
        ],
    ),
    (
        "2. 相关工作",
        [
            "基于 EEG 的阿尔茨海默症检测。",
            "EEG 自监督学习与预训练表示学习。",
            "EEG 跨数据集迁移与域适应。",
            "说明现有工作与本文分阶段迁移框架之间的差异。",
        ],
    ),
    (
        "3. 方法",
        [
            "3.1 问题定义：source datasets、target dataset、AD/HC 二分类与 subject-level evaluation。",
            "3.2 Stage 1：基于时间-频率一致性学习的 multi-source pretraining。",
            "3.3 Stage 2：在 pretrained encoder 基础上构建 source-task 增强初始化。",
            "3.4 Stage 3：面向目标域的 fine-tuning / learning；若保留 adaptation，则作为探索性结果描述。",
            "3.5 初始化策略比较：random init、single-source pretrained init、multi-source pretrained init、source-task init。",
            "方法撰写时要强调：本文的方法创新不是全新 backbone，而是对迁移先验进行分层建模，并将其统一到 staged transfer framework 中。",
            "ESWA 风格方法部分要突出：模块清楚、训练流程清楚、变量定义清楚、每个设计对应的问题清楚。",
        ],
    ),
    (
        "4. 实验设置",
        [
            "数据集：选择 CAUEEG 与 ds004504 作为两个互补的 AD EEG 目标数据集。",
            "预处理：统一为 19 channels、128 Hz、1-second windows，并采用 subject-level split。",
            "训练协议：subject-level train/val/test = 0.7 / 0.1 / 0.2，3 个 seeds，阈值调优与 checkpoint selection。",
            "评估指标：Accuracy、Balanced Accuracy、F1、AUC、Sensitivity、Specificity。",
        ],
    ),
    (
        "5. 实验结果",
        [
            "5.1 单数据集微调：建立单数据集基线，并说明 multi-source pretraining 有效。",
            "CAUEEG：random init -> Acc 0.8744 +/- 0.0347，BalAcc 0.8605 +/- 0.0398，AUC 0.9249 +/- 0.0319；multi-source -> Acc 0.8841 +/- 0.0258，BalAcc 0.8786 +/- 0.0271，AUC 0.9364 +/- 0.0206。",
            "ds004504：random init -> Acc 0.7692 +/- 0.0628，BalAcc 0.7619 +/- 0.0680，AUC 0.8095 +/- 0.0514；multi-source -> Acc 0.7949 +/- 0.0363，BalAcc 0.8016 +/- 0.0312，AUC 0.8492 +/- 0.0224。",
            "5.2 Direct transfer：展示双向失败，并揭示明显的跨数据集 domain gap。",
            "CAUEEG -> ds004504 direct transfer：Acc 0.5500 +/- 0.0758，BalAcc 0.4977 +/- 0.0551，AUC 0.4456 +/- 0.1020，预测明显偏向 AD。",
            "ds004504 -> CAUEEG direct transfer：Acc 0.6947 +/- 0.0313，BalAcc 0.5493 +/- 0.0545，AUC 0.7469 +/- 0.0375，预测明显偏向 HC。",
            "5.3 Source initialization + target fine-tuning：说明 source knowledge 可在目标域重新利用。",
            "CAUEEG init -> ds004504 fine-tune：Acc 0.7778 +/- 0.1416，BalAcc 0.7524 +/- 0.1831，F1 0.8390 +/- 0.0771，AUC 0.9429 +/- 0.0467。",
            "ds004504 init -> CAUEEG fine-tune：Acc 0.8696 +/- 0.0414，BalAcc 0.8696 +/- 0.0363，F1 0.8184 +/- 0.0518，AUC 0.9299 +/- 0.0212。",
            "5.4 初始化消融：random、BrainLat19 single-source、auditory40hz single-source、multi-source、source-task init。",
            "在 ds004504 上，BrainLat19 single-source 在 Acc / F1 / AUC 上最强（0.8056 / 0.8434 / 0.8667），而 multi-source 在 BalAcc / Spec 上最好（0.8016 / 0.8889）。",
            "在 CAUEEG 上，multi-source 综合最优：Acc 0.8841 +/- 0.0258，BalAcc 0.8786 +/- 0.0271，F1 0.8325 +/- 0.0347，AUC 0.9364 +/- 0.0206。",
            "BrainLat19 single-source 在 CAUEEG 上同样有效：Acc 0.8816 +/- 0.0326，BalAcc 0.8659 +/- 0.0377，AUC 0.9336 +/- 0.0202。",
            "auditory40hz single-source 有效但更弱，也更体现 source dependence：ds004504 上 BalAcc 0.7222 +/- 0.0647；CAUEEG 上 BalAcc 0.8533 +/- 0.0277。",
            "ESWA 风格结果呈现重点：主表先给最关键结论，再用消融和对照解释机制，不要把读者淹没在过多次级实验里。",
        ],
    ),
    (
        "6. 讨论",
        [
            "为什么 direct transfer 会失败：domain gap、decision-boundary mismatch 与 dataset-specific cue。",
            "为什么 source-task initialization 仍然有帮助：source knowledge 不是无效，而是需要在目标域中重新学习与适配。",
            "为什么 multi-source pretraining 更可靠：先验更丰富、对单一 source 偏置依赖更小；当前消融已显示 BrainLat19 与 auditory40hz 之间存在明显 source dependence。",
            "本工作创新的落点应明确写成：框架创新、先验建模创新和迁移机制发现，而不是单一新网络模块创新。",
            "局限性：当前 source diversity 仍有限，target 数据规模有限，多数设置仅有 3 个 seeds。",
            "未来工作：更大规模的 multi-source benchmark，以及更细粒度的 disease-relevant adaptation 设计。",
            "ESWA 风格讨论要强调：方法为什么有效、哪里还不够、对真实医疗 AI 场景有什么意义。",
        ],
    ),
    (
        "7. 结论",
        [
            "再次强调本文目标是跨数据集 AD EEG 泛化，而不仅仅是单数据集准确率。",
            "总结：multi-source pretraining 提供了最可靠的初始化先验；direct transfer 整体失败；source-task knowledge 可通过目标域 fine-tuning 被重新利用。",
            "核心结论：single-source pretraining 是有效的，但具有明显 source dependence；相比之下，multi-source pretraining 在两个 target 数据集上都表现出更稳定、更平衡的下游性能。",
        ],
    ),
    (
        "ESWA 投稿取向提示",
        [
            "题目要突出方法框架和应用场景，不必追求过度花哨的算法命名。",
            "摘要里要给出 2 到 4 个关键结果数字，体现系统实验支撑。",
            "结果章节优先放最强主线：in-dataset、direct transfer、source init + target fine-tuning、初始化消融。",
            "把 adaptation 相关内容放到次要位置，除非后续有明显正向结果。",
            "全文语气要偏问题驱动和应用价值导向，而不是只强调模型结构创新。",
        ],
    ),
    (
        "建议配图与表格",
        [
            "三阶段迁移框架图。",
            "初始化消融总表。",
            "包含 BalAcc、AUC、Sensitivity、Specificity 的初始化消融总图。",
            "direct transfer 与 source initialization 对照表。",
        ],
    ),
    (
        "建议写作顺序",
        [
            "1. 引言",
            "2. 方法",
            "3. 实验设置",
            "4. 实验结果",
            "5. 讨论",
            "6. 摘要",
            "7. 结论",
        ],
    ),
]


def add_wrapped_line(lines: list[str], text: str, width: int = 48, prefix: str = "- ") -> None:
    """Append wrapped bullet text to a line buffer."""

    wrapped = wrap(text, width=width)
    if not wrapped:
        lines.append(prefix)
        return
    lines.append(prefix + wrapped[0])
    indent = " " * len(prefix)
    for extra in wrapped[1:]:
        lines.append(indent + extra)


def build_pages() -> list[list[str]]:
    """Build paginated line buffers for the PDF."""

    pages: list[list[str]] = [[]]
    max_lines = 39

    def ensure_space(required: int = 1) -> None:
        if len(pages[-1]) + required > max_lines:
            pages.append([])

    for title, bullets in SECTIONS:
        ensure_space(3)
        pages[-1].append(title)
        pages[-1].append("")
        for bullet in bullets:
            required = max(1, len(wrap(bullet, width=48)))
            ensure_space(required + 1)
            add_wrapped_line(pages[-1], bullet)
        pages[-1].append("")

    return pages


def render_page(ax: plt.Axes, lines: list[str], page_idx: int, num_pages: int) -> None:
    """Render one text page."""

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if page_idx == 0:
        ax.text(0.5, 0.965, TITLE, ha="center", va="top", fontsize=20, fontweight="bold", color="#0D47A1")
        ax.text(0.5, 0.925, SUBTITLE, ha="center", va="top", fontsize=11.5, color="#455A64")
        start_y = 0.88
    else:
        ax.text(0.5, 0.96, TITLE, ha="center", va="top", fontsize=15, fontweight="bold", color="#0D47A1")
        start_y = 0.92

    section_titles = {section[0] for section in SECTIONS}
    y = start_y
    line_height = 0.0205
    for line in lines:
        if line in section_titles:
            ax.text(0.05, y, line, ha="left", va="top", fontsize=12, fontweight="bold", color="#1B5E20")
        else:
            ax.text(0.06, y, line, ha="left", va="top", fontsize=10.2, color="#263238")
        y -= line_height

    ax.text(0.95, 0.025, f"第 {page_idx + 1}/{num_pages} 页", ha="right", va="bottom", fontsize=9, color="#607D8B")


def main() -> None:
    """Generate the paper outline PDF."""

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paper_writing_outline.pdf"

    pages = build_pages()
    with PdfPages(out_path) as pdf:
        for idx, lines in enumerate(pages):
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            render_page(ax, lines, idx, len(pages))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(out_path)


if __name__ == "__main__":
    main()
