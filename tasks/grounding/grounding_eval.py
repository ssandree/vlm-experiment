"""
Grounding evaluation metrics for RefCOCO-style datasets.
"""

from typing import Dict, Any, List, Tuple


def bbox_xywh_to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return (x1, y1, x2, y2)


def compute_iou_xyxy(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    intersection = inter_w * inter_h

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def compute_iou_xywh(
    box1_xywh: List[float],
    box2_xywh: List[float],
) -> float:
    xyxy1 = bbox_xywh_to_xyxy(box1_xywh)
    xyxy2 = bbox_xywh_to_xyxy(box2_xywh)
    return compute_iou_xyxy(xyxy1, xyxy2)


def _normalize_pred_bboxes(pred: Any) -> List[List[float]]:
    """예측이 단일 bbox / bbox 리스트 / 이미지별 bbox 리스트일 때 평탄한 bbox 리스트 반환."""
    if pred is None:
        return []
    if isinstance(pred, dict):
        pred = pred.get("bbox", pred)
    if isinstance(pred, (list, tuple)):
        if not pred:
            return []
        first = pred[0]
        if isinstance(first, (int, float)) and len(pred) == 4:
            return [list(pred)]
        if isinstance(first, (list, tuple)):
            if len(first) == 4 and all(isinstance(x, (int, float)) for x in first):
                return [list(p) for p in pred if isinstance(p, (list, tuple)) and len(p) == 4]
            out = []
            for item in pred:
                out.extend(_normalize_pred_bboxes(item))
            return out
    return []


def _count_detections(pred: Any) -> int:
    """clip당 검출 개수 (단일 이미지: bbox 개수, 다중 이미지: 모든 이미지 bbox 합)."""
    if pred is None:
        return 0
    if isinstance(pred, dict):
        pred = pred.get("bbox", pred)
    if isinstance(pred, (list, tuple)):
        if not pred:
            return 0
        first = pred[0]
        if isinstance(first, (int, float)) and len(pred) == 4:
            return 1
        if isinstance(first, (list, tuple)):
            if len(first) == 4 and all(isinstance(x, (int, float)) for x in first):
                return len(pred)
            return sum(_count_detections(item) for item in pred)
    return 0


def evaluate_grounding(
    predictions: Dict[str, Any],
    references: Dict[str, Any],
    image_ids: List[str],
) -> Dict[str, Any]:
    """
    Grounding 평가. 이 프로젝트는 문장→단일 bbox 벤치마크가 아니라
    프레임 내 모든 객체 bbox를 반환하므로, 참조가 없을 때는 검출 개수 요약만 반환.
    참조 bbox가 있으면 기존처럼 mean IoU / Acc@0.5 도 계산.
    """
    details: Dict[str, Dict[str, Any]] = {}
    iou_sum = 0.0
    num_correct = 0
    total_detections = 0
    has_ref_bbox = False

    for image_id in image_ids:
        pred = predictions.get(image_id)
        pred_bboxes = _normalize_pred_bboxes(pred)
        ref = references.get(image_id)
        ref_bbox = ref.get("bbox", None) if isinstance(ref, dict) else ref

        num_det = _count_detections(pred)
        total_detections += num_det
        details[image_id] = {"num_detections": num_det}

        if ref_bbox is not None and isinstance(ref_bbox, (list, tuple)) and len(ref_bbox) == 4:
            has_ref_bbox = True
            ref_list = [float(x) for x in ref_bbox]
            if num_det == 0:
                iou, correct = 0.0, False
            else:
                best_iou = 0.0
                for pb in pred_bboxes:
                    if len(pb) != 4:
                        continue
                    iou = compute_iou_xywh([float(x) for x in pb], ref_list)
                    best_iou = max(best_iou, iou)
                iou = best_iou
                correct = iou >= 0.5
            details[image_id]["iou"] = round(iou, 6)
            details[image_id]["correct"] = correct
            iou_sum += iou
            if correct:
                num_correct += 1

    n = len(image_ids)
    result: Dict[str, Any] = {
        "num_samples": n,
        "total_detections": total_detections,
        "avg_detections_per_image": round(total_detections / n, 4) if n > 0 else 0.0,
        "details": details,
    }
    if has_ref_bbox and n > 0:
        result["mean_iou"] = round(iou_sum / n, 6)
        result["acc@0.5"] = round(num_correct / n, 6)
    return result

