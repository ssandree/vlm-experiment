from typing import Dict, Any, List, Optional, Union
from PIL import Image

from tasks.base_task import BaseTask
from tasks.grounding.grounding_eval import evaluate_grounding


class GroundingTask(BaseTask):
    """
    Grounding = 프레임에 보이는 모든 객체의 bbox를 표시.
    문장(phrase)에 맞는 단일 bbox 벤치마크가 아닌, prompt에 맞게 전체 객체 검출.
    """

    @property
    def task_name(self) -> str:
        return "grounding"

    def build_inputs(self, sample: Dict[str, Any], prompt_cfg: Dict[str, Any]) -> Dict[str, Any]:
        # prompt 설정만 사용. phrase/ref_sentence는 사용하지 않음 (벤치마크 태스크 아님).
        user_prompt = prompt_cfg.get("user_prompt", "")
        return {
            "image": sample["image"],
            "system_prompt": prompt_cfg.get("system_prompt", ""),
            "user_prompt": user_prompt.strip() or "List all objects visible in the image with bbox [x, y, w, h] in JSON.",
        }

    def run_inference(
        self,
        model: Any,
        inputs: Dict[str, Any],
        generation_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        image_or_list = inputs["image"]
        if isinstance(image_or_list, (list, tuple)):
            all_bboxes: List[List[List[float]]] = []
            for img in image_or_list:
                raw_output = model.run(
                    task="grounding",
                    image=img,
                    system_prompt=inputs["system_prompt"],
                    user_prompt=inputs["user_prompt"],
                    gen_cfg=generation_cfg,
                )
                bboxes_xywh = self._parse_bboxes_from_text(raw_output)
                w, h = img.size
                scaled = [
                    self._scale_bbox_to_pixels(b, w, h) for b in bboxes_xywh
                ]
                all_bboxes.append(scaled)
            return {"bbox": all_bboxes, "raw_output": None}

        raw_output = model.run(
            task="grounding",
            image=image_or_list,
            system_prompt=inputs["system_prompt"],
            user_prompt=inputs["user_prompt"],
            gen_cfg=generation_cfg,
        )
        bboxes_xywh = self._parse_bboxes_from_text(raw_output)
        img_width, img_height = image_or_list.size
        scaled = [
            self._scale_bbox_to_pixels(b, img_width, img_height)
            for b in bboxes_xywh
        ]
        return {"bbox": scaled, "raw_output": raw_output}

    def _scale_bbox_to_pixels(
        self,
        bbox: List[float],
        img_width: int,
        img_height: int,
        scale: int = 1000,
    ) -> List[float]:
        x, y, w, h = bbox
        return [
            x * img_width / scale,
            y * img_height / scale,
            w * img_width / scale,
            h * img_height / scale,
        ]

    def _parse_bboxes_from_text(self, text: str) -> List[List[float]]:
        """Parse all bboxes from model output. Returns list of [x, y, w, h] in 0–1000 scale."""
        import json
        import re

        if not text:
            return []

        cleaned = text.strip()
        if "```" in cleaned:
            code_block_match = re.search(
                r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL
            )
            if code_block_match:
                cleaned = code_block_match.group(1).strip()

        def try_parse(s: str) -> List[List[float]]:
            try:
                parsed = json.loads(s)
                return self._extract_all_bboxes_from_parsed(parsed)
            except json.JSONDecodeError:
                return []

        bboxes = try_parse(cleaned)
        if bboxes:
            return self._normalize_bbox_scale(bboxes)

        json_patterns = [
            r"\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\]",
            r"\[\s*\[[\d.,\s]+\][\s,]*(?:\[[\d.,\s]+\][\s,]*)*\]",
            r"\{[^{}]*\"(?:objects|detections|items|boxes|results)\"[^{}]*\[[\s\S]*?\]",
        ]
        for pattern in json_patterns:
            for m in re.finditer(pattern, cleaned):
                try:
                    bboxes = try_parse(m.group(0))
                    if bboxes:
                        return self._normalize_bbox_scale(bboxes)
                except Exception:
                    continue

        array_match = re.search(r"\[\s*\{[^\[\]]*\"(?:bbox_2d|bbox|box)\"[\s\S]*?\}\s*\]", cleaned)
        if array_match:
            bboxes = try_parse(array_match.group(0))
            if bboxes:
                return self._normalize_bbox_scale(bboxes)

        # Truncated JSON fallback: 완전한 {"bbox_2d": [x1,y1,x2,y2], ...} 블록을 모두 추출
        bbox_obj_pattern = re.compile(
            r'"(?:bbox_2d|bbox|box)"\s*:\s*\[([\d.,\s]+)\]',
            re.IGNORECASE
        )
        partial_bboxes: List[List[float]] = []
        for m in bbox_obj_pattern.finditer(cleaned):
            coords = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
            if len(coords) == 4:
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    partial_bboxes.append([coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]])
                else:
                    partial_bboxes.append(coords)
        if partial_bboxes:
            return self._normalize_bbox_scale(partial_bboxes)

        numbers = re.findall(r"[\d.]+", cleaned)
        if len(numbers) >= 4:
            coords = [float(x) for x in numbers[:4]]
            if all(0 <= c <= 1 for c in coords):
                coords = [c * 1000 for c in coords]
            if coords[2] > coords[0] and coords[3] > coords[1]:
                return self._normalize_bbox_scale([[coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]]])
            return self._normalize_bbox_scale([coords])

        return []

    def _normalize_bbox_scale(self, bboxes: List[List[float]]) -> List[List[float]]:
        """0-1 좌표를 0-1000으로 변환. 이미 0-1000이면 그대로 반환."""
        out = []
        for b in bboxes:
            if len(b) != 4:
                continue
            if all(0 <= c <= 1.001 for c in b):
                b = [x * 1000 for x in b]
            out.append([float(x) for x in b])
        return out

    def _extract_all_bboxes_from_parsed(self, parsed: Any) -> List[List[float]]:
        """Extract every bbox from parsed JSON (array of objects or single object)."""
        out: List[List[float]] = []
        if isinstance(parsed, list):
            for item in parsed:
                bbox = self._extract_bbox_from_parsed(item)
                if bbox is not None:
                    out.append(bbox)
            return out
        if isinstance(parsed, dict):
            for key in ("objects", "detections", "items", "boxes", "results", "bboxes"):
                if key in parsed and isinstance(parsed[key], list):
                    for item in parsed[key]:
                        bbox = self._extract_bbox_from_parsed(item)
                        if bbox is not None:
                            out.append(bbox)
                    if out:
                        return out
        bbox = self._extract_bbox_from_parsed(parsed)
        if bbox is not None:
            out.append(bbox)
        return out

    def _extract_bbox_from_parsed(self, parsed: Any) -> Optional[List[float]]:
        """Single bbox from one JSON object. xywh, scale 0–1000."""
        if isinstance(parsed, list) and len(parsed) == 4:
            coords = [float(x) for x in parsed]
            if coords[2] > coords[0] and coords[3] > coords[1]:
                return [coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]]
            return coords

        if isinstance(parsed, dict):
            for key in ("bbox_2d", "bbox", "box", "coordinates", "location"):
                val = parsed.get(key)
                if val is None or not isinstance(val, list) or len(val) != 4:
                    continue
                coords = [float(x) for x in val]
                if key == "bbox_2d" and coords[2] > coords[0] and coords[3] > coords[1]:
                    return [coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]]
                return coords
            if all(k in parsed for k in ["x", "y", "w", "h"]):
                return [
                    float(parsed["x"]),
                    float(parsed["y"]),
                    float(parsed["w"]),
                    float(parsed["h"]),
                ]
        return None

    def evaluate(
        self,
        predictions: Dict[str, Any],
        references: Dict[str, Any],
        image_ids: List[str],
        image_paths: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        return evaluate_grounding(
            predictions=predictions,
            references=references,
            image_ids=image_ids,
        )

