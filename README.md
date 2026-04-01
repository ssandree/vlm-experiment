# vlm_experiment

벤치마크와 무관한 **실험용** 설정·로더·입력전략·비디오샘플링을 모아둔 폴더입니다.  
메인 코드베이스는 `vlm_system`이고, 여기 있는 내용은 실험(anomaly CCTV, UCFCrime subset, segment_aware 샘플링) 전용입니다.

## 포함 내용

| 구분 | 내용 |
|------|------|
| **Dataset** | `anomaly_cctv`, `ucfcrime_subset`, `image_only`, `image_multi`, `video_only`, `video_image_multi` (config + loader/pipeline) |
| **입력 전략** | `identity`, `multi_image`, `top_right_crop` (image) / `multi`, `grid_no_resize`, `image_strip` (video aggregation) |
| **Video sampling** | `uniform`, `fps_sampling`, `segment_aware`, `manual`, `middle` |

## 디렉터리 구조

```
vlm_experiment/
├── configs/test_dataset/
│   ├── anomaly_cctv.yaml
│   ├── ucfcrime_subset.yaml
│   ├── image_only.yaml
│   ├── image_multi.yaml
│   ├── video_only.yaml
│   └── video_image_multi.yaml
├── data/
│   ├── loader/experiment_loaders.py   # UCFCrimeSubsetEvalLoader, anomaly_cctv_loader, image_only_loader, video_only_loader
│   ├── input_strategies/              # base, multi_image, image_merge, top_right_crop
│   └── video_sampling/                # sampling_strategy, build_sampling_strategy, uniform_sampling, fps_sampling, manual_sampling
└── README.md
```

## vlm_system과 연동

- **실험 실행**: `vlm_system` 루트에서 실행할 때, `vlm_system` 쪽 코드가 `anomaly_cctv` / `ucfcrime_subset` / `segment_aware` 사용 시 **vlm_experiment** 패키지에서 해당 구현을 import하도록 되어 있으면 됩니다.
- **PYTHONPATH**: vlm_experiment를 패키지로 쓰려면 상위 디렉터리가 필요합니다.  
  예: `PYTHONPATH=/home/vailab02 python -m run_inferences.run_video_captioning ...` (vlm_system 루트에서 실행 시 `/home/vailab02`에 vlm_experiment와 vlm_system이 있다고 가정).

## config 사용

실험용 dataset config만 쓰고 싶을 때는 `vlm_system`의 `configs/test_dataset/` 대신 이쪽 config를 참조하도록 경로만 맞추면 됩니다.  
(예: experiment.yaml의 dataset 설정에서 `vlm_experiment/configs/test_dataset/ucfcrime_subset.yaml` 경로 사용)
