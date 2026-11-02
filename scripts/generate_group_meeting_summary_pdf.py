from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "group_meeting_experiment_summary_2026-05-26.pdf"


def get_font() -> FontProperties:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return FontProperties(fname=str(path))
    return FontProperties()


FONT = get_font()


PAGES = [
    {
        "title": "BioFoundation 实验汇报总结",
        "lines": [
            "1. 研究目标",
            "   我们当前关注的问题不是单数据集内 AD vs HC 分类本身，而是如何利用多源自监督预训练",
            "   学习对阿尔茨海默症检测有用、并且能够跨 EEG 数据集泛化的疾病相关表征。",
            "",
            "2. 当前实验设置",
            "   - 主目标数据集：CAUEEG, ds004504",
            "   - 统一输入：19 channels, 128 Hz, 1 s windows",
            "   - 统一评估：subject-level holdout, 3 seeds, threshold tuning, subject-level aggregation",
            "   - 模型结构：dual-stream encoder",
            "       * spatial-temporal branch",
            "       * frequency-domain branch",
            "       * feature fusion + classification head",
            "   - 预训练方式：multi-source self-supervised pretraining",
            "       * time/frequency reconstruction",
            "       * branch consistency",
            "       * crop-view consistency",
            "   - 下游训练：full fine-tuning，不是 frozen encoder，也不是 linear probe",
            "",
            "3. 已完成的单数据集结论",
            "   - CAUEEG：Ours 在当前共同模型集合中表现最好。",
            "   - ds004504：Ours 处于第一梯队，虽然个别 baseline 如 TCN 更强，但预训练模型整体有竞争力。",
            "",
            "4. 预训练是否有效",
            "   CAUEEG:",
            "   - Random init     Acc 0.8744, BalAcc 0.8605, F1 0.8128, AUC 0.9249",
            "   - Pretrained init Acc 0.8841, BalAcc 0.8786, F1 0.8325, AUC 0.9364",
            "   ds004504:",
            "   - Random init     Acc 0.7692, BalAcc 0.7619, F1 0.8024, AUC 0.8095",
            "   - Pretrained init Acc 0.7949, BalAcc 0.8016, F1 0.7845, AUC 0.8492",
            "",
            "结论：预训练在两个 target 上都有效；在更困难的 ds004504 上，预训练对 BalAcc、AUC 和",
            "Specificity 的提升更明显。",
        ],
    },
    {
        "title": "跨数据集实验与下一步计划",
        "lines": [
            "5. 已完成的跨数据集 direct transfer",
            "   A. CAUEEG -> ds004504（pretrained init）",
            "      Acc 0.5500, BalAcc 0.4977, F1 0.6621, AUC 0.4456, Sens 0.7593, Spec 0.2361",
            "      现象：模型明显偏向预测 AD，Specificity 很低，表现接近随机。",
            "",
            "   B. ds004504 -> CAUEEG（pretrained init）",
            "      Acc 0.6947, BalAcc 0.5493, F1 0.1719, AUC 0.7469, Sens 0.1116, Spec 0.9869",
            "      现象：模型明显偏向预测 HC，Sensitivity 很低。",
            "",
            "6. 当前最重要的阶段性结论",
            "   - 单数据集内：多源预训练有效。",
            "   - 双向 direct transfer：都不理想，且类别偏置方向不同。",
            "   - 说明主要问题不是单数据集分类能力，而是明显的 cross-dataset domain gap。",
            "",
            "7. 当前创新点的定位",
            "   - 不是单纯提出一个新层，而是把 AD EEG 检测从单数据集分类推进到",
            "     多源预训练 + 跨数据集迁移 + 疾病相关表征泛化的问题框架中。",
            "   - 方法上强调 multi-source self-supervised dual-stream representation learning。",
            "   - 实验上建立了 random init / pretrained init / direct transfer 的分层证据链。",
            "",
            "8. 下一阶段实验",
            "   第一优先：source initialization + target fine-tuning",
            "   - CAUEEG init -> ds004504 fine-tune -> ds004504 test",
            "   - ds004504 init -> CAUEEG fine-tune -> CAUEEG test",
            "",
            "   第二优先：比较三种初始化",
            "   - random init",
            "   - single-source init",
            "   - multi-source pretrained init",
            "",
            "   如果 fine-tuning 后仍然不稳，再考虑显式 domain adaptation 或 disease-relevant alignment。",
            "",
            "9. 汇报一句话总结",
            "   当前实验表明，多源自监督预训练能够稳定提升阿尔茨海默症 EEG 检测的单数据集内表现，",
            "   但双向 direct cross-dataset transfer 都存在明显 domain gap，因此下一阶段将重点研究",
            "   source initialization 与 target fine-tuning 对跨数据集泛化的作用。",
        ],
    },
]


def draw_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.05,
        0.965,
        title,
        fontproperties=FONT,
        fontsize=18,
        fontweight="bold",
        va="top",
    )

    y = 0.93
    for line in lines:
        if line == "":
            y -= 0.02
            continue
        ax.text(0.05, y, line, fontproperties=FONT, fontsize=11.5, va="top")
        y -= 0.028

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with PdfPages(OUTPUT_PATH) as pdf:
        for page in PAGES:
            draw_page(pdf, page["title"], page["lines"])
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
