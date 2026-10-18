#!/usr/bin/env python3
"""Điểm vào dòng lệnh (CLI Runner) phân hệ hợp nhất động học hải quân.

Cung cấp công cụ thực thi và đánh giá độc lập:
    python run_naval_fusion.py --scenario scenario2 --eval
    python run_naval_fusion.py --scenario scenario2 --ablation
    python run_naval_fusion.py --all

Kết quả báo cáo định dạng Markdown tự động xuất ra màn hình và lưu vào
tệp chỉ định (mặc định output/naval_fusion_report.md).
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

# Đảm bảo gói src được tự động thêm vào sys.path
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from naval_fusion.models import EvaluationMetrics
from naval_fusion.pipeline import NavalFusionPipeline

logger = logging.getLogger("run_naval_fusion")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Thiết lập và phân tích các đối số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="CLI Runner phân hệ hợp nhất động học hải quân.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="scenario2",
        help="Tên kịch bản (ví dụ: scenario2) hoặc đường dẫn thư mục kịch bản.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Chạy đánh giá định lượng đối chứng với dữ liệu chuẩn Ground Truth.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Chạy thử nghiệm đối chứng (Radar-Only vs Radar+IR vs Toàn diện).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Chạy kiểm chứng tự động toàn bộ các kịch bản trong thư mục tập dữ liệu.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/Autoferry-sensor_fusion_dataset",
        help="Thư mục gốc chứa tập dữ liệu Autoferry Benchmark Dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/naval_fusion_report.md",
        help="Đường dẫn tệp Markdown lưu kết quả báo cáo đánh giá.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Mức độ hiển thị nhật ký thực thi (log level).",
    )

    return parser.parse_args(argv)


def resolve_scenario_dir(scenario_arg: str, dataset_dir: str | Path) -> Path:
    """Xác định đường dẫn thư mục kịch bản từ tên hoặc đường dẫn truyền vào.

    Args:
        scenario_arg: Tên kịch bản (như scenario2) hoặc đường dẫn đầy đủ.
        dataset_dir: Thư mục gốc chứa các kịch bản.

    Returns:
        Path: Đường dẫn thư mục kịch bản hợp lệ.

    Raises:
        FileNotFoundError: Nếu không tìm thấy thư mục kịch bản.
    """
    path_direct = Path(scenario_arg)
    if path_direct.is_dir():
        return path_direct

    candidates = [
        Path(dataset_dir) / scenario_arg,
        Path(__file__).resolve().parent / dataset_dir / scenario_arg,
        Path(__file__).resolve().parent.parent / dataset_dir / scenario_arg,
    ]
    for c in candidates:
        if c.is_dir():
            return c

    msg = (
        f"Không tìm thấy thư mục kịch bản: '{scenario_arg}'. "
        f"Đã thử kiểm tra: '{path_direct}' và {[str(c) for c in candidates]}."
    )
    logger.error(msg)
    raise FileNotFoundError(msg)


def discover_all_scenarios(dataset_dir: str | Path) -> list[Path]:
    """Tìm kiếm tất cả các thư mục kịch bản hợp lệ trong thư mục dataset."""
    candidates = [
        Path(dataset_dir),
        Path(__file__).resolve().parent / dataset_dir,
        Path(__file__).resolve().parent.parent / dataset_dir,
    ]
    root: Path | None = None
    for c in candidates:
        if c.is_dir():
            root = c
            break

    if root is None:
        msg = f"Thư mục tập dữ liệu không tồn tại: {dataset_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    scenario_dirs: list[Path] = []
    for item in sorted(root.iterdir()):
        if item.is_dir() and item.name.startswith("scenario"):
            # Kiểm tra xem có chứa tệp detections JSON không
            has_det = any(item.glob("*_detections.json"))
            if has_det:
                scenario_dirs.append(item)

    # Sắp xếp tự nhiên theo số thứ tự kịch bản
    def _sort_key(p: Path) -> int:
        digits = "".join(ch for ch in p.name if ch.isdigit())
        return int(digits) if digits else 9999

    scenario_dirs.sort(key=_sort_key)
    return scenario_dirs


def generate_all_scenarios_report(
    results: dict[str, EvaluationMetrics],
) -> str:
    """Tạo báo cáo tổng hợp Markdown cho toàn bộ các kịch bản thử nghiệm."""
    table_header = (
        "| Kịch bản | RMSE Vị trí (m) | RMSE Vận tốc (m/s) | "
        "Số vết xác nhận | ID Switches | Tỷ lệ bám vết (MT) |"
    )
    lines: list[str] = [
        "# Báo Cáo Tổng Hợp Đánh Giá Toàn Bộ Kịch Bản Autoferry Benchmark",
        "**Phân hệ Hợp nhất Động học Hải quân (Naval Kinematic Fusion Engine)**",
        "",
        table_header,
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for scen_name, metrics in results.items():
        mt_str = f"{metrics.mostly_tracked_ratio * 100:.1f}%"
        row = (
            f"| `{scen_name}` | {metrics.rmse_position:.2f} | "
            f"{metrics.rmse_velocity:.2f} | {metrics.num_confirmed_tracks} | "
            f"{metrics.id_switches} | {mt_str} |"
        )
        lines.append(row)

    lines.extend(
        [
            "",
            "### Tổng kết toàn diện:",
            f"- **Tổng số kịch bản đã kiểm chứng:** {len(results)} kịch bản.",
        ]
    )

    all_rmse_pos = [m.rmse_position for m in results.values() if m.rmse_position > 0]
    if all_rmse_pos:
        mean_pos = sum(all_rmse_pos) / len(all_rmse_pos)
        lines.append(f"- **RMSE Vị trí trung bình:** `{mean_pos:.2f}m`.")

    all_mt = [m.mostly_tracked_ratio * 100 for m in results.values()]
    if all_mt:
        mean_mt = sum(all_mt) / len(all_mt)
        lines.append(
            f"- **Tỷ lệ duy trì bám vết (MT) trung bình:** `{mean_mt:.1f}%`."
        )

    total_id_switches = sum(m.id_switches for m in results.values())
    lines.append(f"- **Tổng số lần nhảy ID (ID Switches):** {total_id_switches}.")

    return "\n".join(lines) + "\n"



def write_and_print_report(report_md: str, output_path: Path) -> None:
    """In báo cáo ra terminal và lưu vào tệp Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_md, encoding="utf-8")
    print("\n" + "=" * 80)
    print(report_md)
    print("=" * 80)
    print(f"Báo cáo kết quả đã được lưu tại: {output_path.resolve()}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào chính của CLI Runner."""
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = NavalFusionPipeline()
    output_path = Path(args.output)

    try:
        if args.all:
            # Chạy toàn bộ các kịch bản tìm thấy
            scenario_dirs = discover_all_scenarios(args.dataset_dir)
            if not scenario_dirs:
                logger.error("Không tìm thấy kịch bản nào trong %s", args.dataset_dir)
                return 1

            print(f"Bắt đầu chạy kiểm chứng toàn bộ {len(scenario_dirs)} kịch bản...")
            all_results: dict[str, EvaluationMetrics] = {}
            for s_dir in scenario_dirs:
                print(f" -> Đang xử lý: {s_dir.name}...")
                _, metrics = pipeline.run(s_dir, enabled_sensors=None)
                all_results[s_dir.name] = metrics

            report_md = generate_all_scenarios_report(all_results)
            write_and_print_report(report_md, output_path)
            return 0

        # Xử lý một kịch bản cụ thể
        scenario_path = resolve_scenario_dir(args.scenario, args.dataset_dir)
        scen_name = scenario_path.name

        if args.ablation:
            print(f"Bắt đầu thử nghiệm đối chứng (Ablation Study) trên {scen_name}...")
            ablation_results = pipeline.run_ablation(scenario_path)
            report_md = pipeline.evaluator.generate_markdown_report(
                ablation_results, scenario_name=scen_name
            )
            write_and_print_report(report_md, output_path)
            return 0

        # Mặc định hoặc có cờ --eval: chạy hợp nhất toàn diện và đánh giá
        print(f"Bắt đầu thực thi hợp nhất động học trên {scen_name}...")
        _, metrics = pipeline.run(scenario_path, enabled_sensors=None)
        single_result = {
            "Hợp nhất Toàn diện (Radar + IR + EO + LiDAR)": metrics,
        }
        report_md = pipeline.evaluator.generate_markdown_report(
            single_result, scenario_name=scen_name
        )
        write_and_print_report(report_md, output_path)
        return 0

    except Exception as exc:
        logger.exception("Đã xảy ra lỗi trong quá trình thực thi: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
