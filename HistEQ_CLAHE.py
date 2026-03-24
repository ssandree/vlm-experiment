import cv2 as cv
import numpy as np
import sys


# ============================================================
# 1. 사용자 설정 파라미터
# ============================================================

#IMAGE_PATH = "lena_gray_b.jpg"
IMAGE_PATH = "insu_OCR1.jpg"

# Stretching 방식 선택
STRETCH_MODE = "percentile"   # "minmax" or "percentile"

# Percentile stretching 파라미터
LOW_PERCENTILE = 0.1
HIGH_PERCENTILE = 99.9

# Histogram Equalization 구현 방식 선택
USE_MANUAL_EQUALIZATION = True

# ------------------------------------------------------------
# CLAHE 구현 방식 선택
# ------------------------------------------------------------
# False -> OpenCV CLAHE 사용 (기본값)
# True  -> 직접 구현한 manual CLAHE 사용
USE_MANUAL_CLAHE = False

# ------------------------------------------------------------
# CLAHE 파라미터
# ------------------------------------------------------------
# tile(grid) 개수
CLAHE_GRID_ROWS = 8
CLAHE_GRID_COLS = 8

# clip limit 기본 제안값
CLAHE_CLIP_LIMIT = 2.0

# Figure 크기
FIGSIZE_EQ = (20, 18)
FIGSIZE_COMPARE = (20, 22)

# 폰트 크기
TITLE_FONTSIZE = 10
SUPTITLE_FONTSIZE = 16
X_LABEL_FONTSIZE = 6
Y_LABEL_FONTSIZE = 9
TICK_FONTSIZE = 7
LEGEND_FONTSIZE = 7


# ============================================================
# 2. BT.709 기준 BGR <-> YCbCr 변환 함수
# ============================================================

def bgr_to_ycbcr_bt709_full(img_bgr):
    """
    BGR 8비트 컬러 영상을 BT.709 full-range YCbCr로 변환합니다.
    출력 채널 순서는 [Y, Cb, Cr] 입니다.
    """
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB).astype(np.float32)

    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Cb = (B - Y) / (2.0 * (1.0 - 0.0722)) + 128.0
    Cr = (R - Y) / (2.0 * (1.0 - 0.2126)) + 128.0

    ycbcr = np.stack([Y, Cb, Cr], axis=2)
    ycbcr = np.clip(ycbcr, 0, 255).astype(np.uint8)
    return ycbcr


def ycbcr_bt709_full_to_bgr(img_ycbcr):
    """
    BT.709 full-range YCbCr 영상을 다시 BGR로 변환합니다.
    """
    img_ycbcr = img_ycbcr.astype(np.float32)

    Y = img_ycbcr[:, :, 0]
    Cb = img_ycbcr[:, :, 1] - 128.0
    Cr = img_ycbcr[:, :, 2] - 128.0

    R = Y + Cr * (2.0 * (1.0 - 0.2126))
    B = Y + Cb * (2.0 * (1.0 - 0.0722))
    G = (Y - 0.2126 * R - 0.0722 * B) / 0.7152

    img_rgb = np.stack([R, G, B], axis=2)
    img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
    return img_bgr


# ============================================================
# 3. 입력 영상 로드 및 Y 채널 추출
# ============================================================

def load_image_and_extract_y(path):
    """
    입력 영상을 읽고 처리 대상 Y 채널을 추출합니다.

    지원 입력:
    1) 8비트 grayscale
    2) 16비트 grayscale (예: 16bit tif)
    3) 8비트 color

    반환:
        original_display_rgb : matplotlib 표시용 RGB 영상
        y_original           : 처리 대상 Y 채널 (uint8 또는 uint16)
        is_color             : 컬러 영상 여부
        ycbcr_if_color       : 컬러 영상일 경우 원본 YCbCr
        input_bit_depth      : 8 또는 16
    """
    img = cv.imread(path, cv.IMREAD_UNCHANGED)

    if img is None:
        sys.exit(f"입력 영상을 읽을 수 없습니다: {path}")

    if len(img.shape) == 2:
        is_color = False
        y_original = img.copy()
        ycbcr_if_color = None

        if img.dtype == np.uint16:
            input_bit_depth = 16
            img_disp_8 = convert_16bit_to_8bit_linear(img)
            original_display_rgb = cv.cvtColor(img_disp_8, cv.COLOR_GRAY2RGB)
        elif img.dtype == np.uint8:
            input_bit_depth = 8
            original_display_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        else:
            sys.exit("지원하지 않는 grayscale dtype입니다. uint8 또는 uint16만 지원합니다.")

    elif len(img.shape) == 3 and img.shape[2] >= 3:
        if img.dtype != np.uint8:
            sys.exit("컬러 영상은 현재 8비트 입력만 지원합니다.")

        is_color = True
        input_bit_depth = 8

        img_bgr = img[:, :, :3].copy()
        original_display_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        img_ycbcr = bgr_to_ycbcr_bt709_full(img_bgr)
        y_original = img_ycbcr[:, :, 0].copy()
        ycbcr_if_color = img_ycbcr

    else:
        sys.exit("지원하지 않는 영상 형식입니다.")

    return original_display_rgb, y_original, is_color, ycbcr_if_color, input_bit_depth


def reconstruct_display_image(processed_y_8bit, is_color, original_ycbcr):
    """
    처리된 8비트 Y 채널을 다시 화면 표시용 RGB 영상으로 복원합니다.
    """
    if not is_color:
        return cv.cvtColor(processed_y_8bit, cv.COLOR_GRAY2RGB)

    ycbcr_new = original_ycbcr.copy()
    ycbcr_new[:, :, 0] = processed_y_8bit

    bgr_new = ycbcr_bt709_full_to_bgr(ycbcr_new)
    rgb_new = cv.cvtColor(bgr_new, cv.COLOR_BGR2RGB)
    return rgb_new


# ============================================================
# 4. bit-depth 변환 관련 함수
# ============================================================

def convert_16bit_to_8bit_linear(img16):
    """
    16비트 grayscale 영상을 8비트로 선형 변환합니다.
    """
    img16_f = img16.astype(np.float64)
    img8 = np.round(img16_f * 255.0 / 65535.0)
    img8 = np.clip(img8, 0, 255).astype(np.uint8)
    return img8


def convert_to_8bit_after_stretch(img):
    """
    Stretching 결과를 8비트로 바꿉니다.
    """
    if img.dtype == np.uint8:
        return img.copy()
    elif img.dtype == np.uint16:
        return convert_16bit_to_8bit_linear(img)
    else:
        sys.exit("지원하지 않는 dtype입니다. uint8 또는 uint16만 지원합니다.")


# ============================================================
# 5. Histogram / CDF / LUT 직접 구현 함수
# ============================================================

def compute_histogram_manual(img_gray_8):
    """
    8비트 grayscale 영상의 histogram을 직접 계산합니다.
    """
    if img_gray_8.dtype != np.uint8:
        raise ValueError("compute_histogram_manual()은 uint8 입력만 받습니다.")

    hist = np.zeros(256, dtype=np.float64)

    h, w = img_gray_8.shape
    for y in range(h):
        for x in range(w):
            gray_value = int(img_gray_8[y, x])
            hist[gray_value] += 1

    return hist


def compute_cdf_from_hist_manual(hist):
    """
    histogram으로부터 cumulative distribution function(CDF)를 직접 계산합니다.
    """
    cdf = np.zeros_like(hist, dtype=np.float64)

    running_sum = 0.0
    for i in range(len(hist)):
        running_sum += hist[i]
        cdf[i] = running_sum

    return cdf


def make_identity_lut_8():
    """
    8비트용 y=x identity LUT
    """
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = i
    return lut


def make_identity_lut_same_dtype(dtype):
    """
    dtype에 맞는 identity LUT 생성
    """
    if dtype == np.uint8:
        return np.arange(256, dtype=np.uint8)
    elif dtype == np.uint16:
        return np.arange(65536, dtype=np.uint16)
    else:
        sys.exit("지원하지 않는 dtype입니다.")


def apply_lut_manual_8bit(img_gray_8, lut):
    """
    8비트 grayscale 영상에 LUT를 직접 적용합니다.
    """
    if img_gray_8.dtype != np.uint8:
        raise ValueError("apply_lut_manual_8bit()은 uint8 입력만 받습니다.")

    h, w = img_gray_8.shape
    out = np.zeros_like(img_gray_8, dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            pixel_value = int(img_gray_8[y, x])
            out[y, x] = lut[pixel_value]

    return out


def apply_lut_same_dtype(img, lut):
    """
    입력 영상과 같은 bit-depth를 유지하면서 LUT를 적용합니다.
    """
    h, w = img.shape
    out = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            pixel_value = int(img[y, x])
            out[y, x] = lut[pixel_value]

    return out


# ============================================================
# 6. Histogram Stretching LUT 생성 함수
# ============================================================

def build_stretch_lut_from_percentiles(img_gray, low_percentile=0.0, high_percentile=100.0):
    """
    percentile 구간을 full range로 stretch 하는 LUT를 생성합니다.

    8비트 입력이면 0~255 LUT
    16비트 입력이면 0~65535 LUT
    """
    in_low = np.percentile(img_gray, low_percentile)
    in_high = np.percentile(img_gray, high_percentile)

    if in_high <= in_low:
        lut = make_identity_lut_same_dtype(img_gray.dtype)
        return lut, in_low, in_high

    if img_gray.dtype == np.uint8:
        max_value = 255
        lut = np.zeros(256, dtype=np.uint8)

        for i in range(256):
            stretched_value = (i - in_low) * max_value / (in_high - in_low)
            if stretched_value < 0:
                stretched_value = 0
            elif stretched_value > max_value:
                stretched_value = max_value
            lut[i] = int(round(stretched_value))

    elif img_gray.dtype == np.uint16:
        max_value = 65535
        lut = np.zeros(65536, dtype=np.uint16)

        for i in range(65536):
            stretched_value = (i - in_low) * max_value / (in_high - in_low)
            if stretched_value < 0:
                stretched_value = 0
            elif stretched_value > max_value:
                stretched_value = max_value
            lut[i] = int(round(stretched_value))
    else:
        sys.exit("지원하지 않는 dtype입니다.")

    return lut, in_low, in_high


# ============================================================
# 7. Histogram Equalization 직접 구현 함수 (8비트 전용)
# ============================================================

def build_equalization_lut_manual(img_gray_8):
    """
    Histogram Equalization LUT를 직접 구현합니다.
    입력은 반드시 8비트 grayscale이어야 합니다.
    """
    if img_gray_8.dtype != np.uint8:
        raise ValueError("build_equalization_lut_manual()은 uint8 입력만 받습니다.")

    hist = compute_histogram_manual(img_gray_8)
    cdf = compute_cdf_from_hist_manual(hist)

    total_pixels = img_gray_8.shape[0] * img_gray_8.shape[1]

    first_nonzero_index = -1
    for i in range(256):
        if hist[i] > 0:
            first_nonzero_index = i
            break

    if first_nonzero_index == -1:
        return make_identity_lut_8(), hist, cdf

    cdf_min = cdf[first_nonzero_index]

    if total_pixels == cdf_min:
        return make_identity_lut_8(), hist, cdf

    lut = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        mapped_value = (cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255.0

        if mapped_value < 0:
            mapped_value = 0
        elif mapped_value > 255:
            mapped_value = 255

        lut[i] = int(round(mapped_value))

    return lut, hist, cdf


def equalize_manual(img_gray_8):
    """
    직접 구현한 Histogram Equalization 수행
    """
    lut, _, _ = build_equalization_lut_manual(img_gray_8)
    out_img = apply_lut_manual_8bit(img_gray_8, lut)
    return out_img, lut


def equalize_opencv(img_gray_8):
    """
    비교용 OpenCV equalizeHist() 버전
    """
    out_img = cv.equalizeHist(img_gray_8)

    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    lut = cv.equalizeHist(ramp).flatten()

    return out_img, lut


def equalize_selected(img_gray_8, use_manual=True):
    if use_manual:
        return equalize_manual(img_gray_8)
    else:
        return equalize_opencv(img_gray_8)


# ============================================================
# 8. CLAHE 직접 구현 관련 함수 (8비트 전용)
# ============================================================

def make_tile_boundaries(length, num_tiles):
    """
    한 축을 num_tiles 개의 블록으로 나눌 때 경계 index 계산

    주의:
    여기서 num_tiles는 "tile 개수"입니다.
    즉, tile의 크기가 8x8 픽셀로 고정되는 것이 아니라,
    CLAHE_GRID_ROWS, CLAHE_GRID_COLS 값에 따라
    전체 영상을 몇 개의 블록으로 나눌지가 결정됩니다.
    """
    boundaries = np.linspace(0, length, num_tiles + 1, dtype=int)
    return boundaries


def clip_histogram_manual(hist, clip_limit_count):
    """
    히스토그램의 각 bin이 clip_limit_count를 넘지 않도록 자른 뒤,
    excess를 전체 bin에 재분배합니다.
    """
    clipped_hist = hist.copy()
    excess = 0.0

    for i in range(256):
        if clipped_hist[i] > clip_limit_count:
            excess += (clipped_hist[i] - clip_limit_count)
            clipped_hist[i] = clip_limit_count

    add_all = int(excess // 256)
    remainder = int(excess % 256)

    if add_all > 0:
        for i in range(256):
            clipped_hist[i] += add_all

    for i in range(remainder):
        clipped_hist[i] += 1

    return clipped_hist


def build_clahe_lut_for_tile_manual(tile_8, clip_limit=2.0, disable_clip=False):
    """
    하나의 tile에 대해 CLAHE용 LUT를 직접 만듭니다.
    """
    if tile_8.dtype != np.uint8:
        raise ValueError("build_clahe_lut_for_tile_manual()은 uint8 입력만 받습니다.")

    hist = compute_histogram_manual(tile_8)
    tile_pixels = tile_8.shape[0] * tile_8.shape[1]

    # grid=1x1이면 CLAHE가 HistEq와 같아지도록 clipping을 끄는 옵션
    if disable_clip:
        cdf = compute_cdf_from_hist_manual(hist)

        first_nonzero_index = -1
        for i in range(256):
            if hist[i] > 0:
                first_nonzero_index = i
                break

        if first_nonzero_index == -1:
            return make_identity_lut_8(), hist, hist.copy(), cdf

        cdf_min = cdf[first_nonzero_index]

        if tile_pixels == cdf_min:
            return make_identity_lut_8(), hist, hist.copy(), cdf

        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            mapped_value = (cdf[i] - cdf_min) / (tile_pixels - cdf_min) * 255.0
            if mapped_value < 0:
                mapped_value = 0
            elif mapped_value > 255:
                mapped_value = 255
            lut[i] = int(round(mapped_value))

        return lut, hist, hist.copy(), cdf

    # 일반 CLAHE
    average_bin_count = tile_pixels / 256.0
    clip_limit_count = max(1.0, clip_limit * average_bin_count)

    clipped_hist = clip_histogram_manual(hist, clip_limit_count)
    cdf = compute_cdf_from_hist_manual(clipped_hist)

    first_nonzero_index = -1
    for i in range(256):
        if clipped_hist[i] > 0:
            first_nonzero_index = i
            break

    if first_nonzero_index == -1:
        return make_identity_lut_8(), hist, clipped_hist, cdf

    cdf_min = cdf[first_nonzero_index]

    if tile_pixels == cdf_min:
        return make_identity_lut_8(), hist, clipped_hist, cdf

    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapped_value = (cdf[i] - cdf_min) / (tile_pixels - cdf_min) * 255.0
        if mapped_value < 0:
            mapped_value = 0
        elif mapped_value > 255:
            mapped_value = 255
        lut[i] = int(round(mapped_value))

    return lut, hist, clipped_hist, cdf


def build_all_tile_luts_manual(img_gray_8, grid_rows=8, grid_cols=8, clip_limit=2.0):
    """
    전체 영상을 grid_rows x grid_cols 개의 tile로 나누고,
    각 tile에 대해 CLAHE LUT를 계산합니다.

    주의:
    여기서 grid_rows, grid_cols는 tile 개수입니다.
    즉, tile 크기가 8x8 픽셀로 고정된 것이 아니라,
    전체 영상을 몇 개의 행/열 블록으로 나눌 것인지를 의미합니다.
    """
    if img_gray_8.dtype != np.uint8:
        raise ValueError("build_all_tile_luts_manual()은 uint8 입력만 받습니다.")

    h, w = img_gray_8.shape

    row_bounds = make_tile_boundaries(h, grid_rows)
    col_bounds = make_tile_boundaries(w, grid_cols)

    tile_luts = np.zeros((grid_rows, grid_cols, 256), dtype=np.uint8)

    disable_clip = (grid_rows == 1 and grid_cols == 1)

    for r in range(grid_rows):
        for c in range(grid_cols):
            y0, y1 = row_bounds[r], row_bounds[r + 1]
            x0, x1 = col_bounds[c], col_bounds[c + 1]

            tile = img_gray_8[y0:y1, x0:x1]

            lut, _, _, _ = build_clahe_lut_for_tile_manual(
                tile,
                clip_limit=clip_limit,
                disable_clip=disable_clip
            )
            tile_luts[r, c, :] = lut

    return tile_luts, row_bounds, col_bounds


def apply_clahe_manual(img_gray_8, grid_rows=8, grid_cols=8, clip_limit=2.0):
    """
    CLAHE를 직접 구현하여 8비트 영상에 적용합니다.

    주의:
    grid_rows, grid_cols는 tile 개수입니다.
    """
    if img_gray_8.dtype != np.uint8:
        raise ValueError("apply_clahe_manual()은 uint8 입력만 받습니다.")

    h, w = img_gray_8.shape

    tile_luts, row_bounds, col_bounds = build_all_tile_luts_manual(
        img_gray_8,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        clip_limit=clip_limit
    )

    row_centers = np.zeros(grid_rows, dtype=np.float64)
    col_centers = np.zeros(grid_cols, dtype=np.float64)

    for r in range(grid_rows):
        row_centers[r] = 0.5 * (row_bounds[r] + row_bounds[r + 1] - 1)

    for c in range(grid_cols):
        col_centers[c] = 0.5 * (col_bounds[c] + col_bounds[c + 1] - 1)

    out = np.zeros_like(img_gray_8, dtype=np.uint8)

    for y in range(h):
        if y <= row_centers[0]:
            r0 = r1 = 0
            wy = 0.0
        elif y >= row_centers[-1]:
            r0 = r1 = grid_rows - 1
            wy = 0.0
        else:
            r0 = 0
            while not (row_centers[r0] <= y <= row_centers[r0 + 1]):
                r0 += 1
            r1 = r0 + 1
            wy = (y - row_centers[r0]) / (row_centers[r1] - row_centers[r0])

        for x in range(w):
            if x <= col_centers[0]:
                c0 = c1 = 0
                wx = 0.0
            elif x >= col_centers[-1]:
                c0 = c1 = grid_cols - 1
                wx = 0.0
            else:
                c0 = 0
                while not (col_centers[c0] <= x <= col_centers[c0 + 1]):
                    c0 += 1
                c1 = c0 + 1
                wx = (x - col_centers[c0]) / (col_centers[c1] - col_centers[c0])

            pixel_value = int(img_gray_8[y, x])

            v00 = tile_luts[r0, c0, pixel_value]
            v01 = tile_luts[r0, c1, pixel_value]
            v10 = tile_luts[r1, c0, pixel_value]
            v11 = tile_luts[r1, c1, pixel_value]

            top = (1.0 - wx) * v00 + wx * v01
            bottom = (1.0 - wx) * v10 + wx * v11
            value = (1.0 - wy) * top + wy * bottom

            out[y, x] = int(round(value))

    return out, tile_luts, row_bounds, col_bounds


def apply_clahe_opencv(img_gray_8, grid_rows=8, grid_cols=8, clip_limit=2.0):
    """
    OpenCV에서 제공하는 CLAHE 함수를 사용합니다.

    OpenCV는 tileGridSize를 (cols, rows) 순서로 받으므로 주의합니다.
    """
    if img_gray_8.dtype != np.uint8:
        raise ValueError("apply_clahe_opencv()은 uint8 입력만 받습니다.")

    clahe = cv.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_cols, grid_rows)
    )
    out = clahe.apply(img_gray_8)

    # OpenCV CLAHE는 내부 LUT를 직접 제공하지 않으므로,
    # 시각화용 local mapping은 수동 구현으로 다시 계산합니다.
    tile_luts, row_bounds, col_bounds = build_all_tile_luts_manual(
        img_gray_8,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        clip_limit=clip_limit
    )

    return out, tile_luts, row_bounds, col_bounds


def apply_clahe_selected(img_gray_8, use_manual=False, grid_rows=8, grid_cols=8, clip_limit=2.0):
    """
    사용자 설정에 따라 manual CLAHE 또는 OpenCV CLAHE를 선택합니다.
    """
    if use_manual:
        return apply_clahe_manual(
            img_gray_8,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            clip_limit=clip_limit
        )
    else:
        return apply_clahe_opencv(
            img_gray_8,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            clip_limit=clip_limit
        )


# ============================================================
# 9. 시각화 함수
# ============================================================

def style_axis_labels(ax):
    ax.xaxis.label.set_size(X_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(Y_LABEL_FONTSIZE)
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)


def plot_image(ax, img_rgb, title):
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.axis("off")


def plot_histogram(ax, hist, title):
    ax.bar(np.arange(256), hist, width=1.0, color="gray")
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Gray Level")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    style_axis_labels(ax)


def plot_cdf(ax, cdf, title):
    ax.plot(np.arange(256), cdf, color="black", linewidth=2)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Gray Level")
    ax.set_ylabel("Cumulative Count")
    ax.grid(True, alpha=0.3)
    style_axis_labels(ax)


def plot_mapping_8bit(ax, lut, title, mapping_color="red", show_identity=True):
    x = np.arange(256)

    if show_identity:
        ax.plot(x, x, linestyle="--", color="black", linewidth=1.5, label="y = x")

    ax.plot(x, lut, color=mapping_color, linewidth=3, label="mapping")

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    style_axis_labels(ax)


def plot_mapping_16bit(ax, lut16, title, mapping_color="red", show_identity=True):
    """
    16비트 stretching LUT를 0~65535 축 기준으로 직접 표시합니다.
    """
    step = max(1, len(lut16) // 2048)
    x = np.arange(0, len(lut16), step, dtype=np.int64)
    y = lut16[x].astype(np.float64)

    if show_identity:
        ax.plot([0, 65535], [0, 65535], linestyle="--", color="black", linewidth=1.5, label="y = x")

    ax.plot(x, y, color=mapping_color, linewidth=2.5, label="mapping")

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlim(0, 65535)
    ax.set_ylim(0, 65535)
    ax.set_xlabel("Input (16-bit)")
    ax.set_ylabel("Output (16-bit)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    style_axis_labels(ax)


def plot_tile_mappings_grid(ax, tile_luts, title):
    """
    하나의 subplot 안에 Grid Box를 그리고,
    각 tile 안에 local mapping 함수를 그립니다.
    """
    grid_rows, grid_cols, _ = tile_luts.shape

    ax.set_xlim(0, grid_cols)
    ax.set_ylim(grid_rows, 0)
    ax.set_aspect('equal')

    for c in range(grid_cols + 1):
        ax.plot([c, c], [0, grid_rows], color='black', linewidth=0.7)

    for r in range(grid_rows + 1):
        ax.plot([0, grid_cols], [r, r], color='black', linewidth=0.7)

    x = np.arange(256, dtype=np.float64)

    for r in range(grid_rows):
        for c in range(grid_cols):
            lut = tile_luts[r, c, :].astype(np.float64)
            x_local = c + x / 255.0
            y_local = r + 1.0 - lut / 255.0
            ax.plot(x_local, y_local, color='blue', linewidth=0.6)

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])


# ============================================================
# 10. 메인 처리
# ============================================================

def main():
    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # 10-1. 입력 영상 읽기 및 Y 채널 추출
    # --------------------------------------------------------
    original_display_rgb, y_original, is_color, original_ycbcr, input_bit_depth = load_image_and_extract_y(IMAGE_PATH)

    # --------------------------------------------------------
    # 10-2. 원본을 8비트로 변환한 버전 준비
    # --------------------------------------------------------
    if input_bit_depth == 16:
        y_original_8 = convert_16bit_to_8bit_linear(y_original)
    else:
        y_original_8 = y_original.copy()

    hist_original = compute_histogram_manual(y_original_8)
    cdf_original = compute_cdf_from_hist_manual(hist_original)
    lut_identity = make_identity_lut_8()

    # --------------------------------------------------------
    # 10-3. Stretching 수행
    # --------------------------------------------------------
    if STRETCH_MODE.lower() == "minmax":
        stretch_lut, in_low, in_high = build_stretch_lut_from_percentiles(
            y_original, 0.0, 100.0
        )
        stretch_title = f"Original + Stretch (Min-Max)\nlow={in_low:.2f}, high={in_high:.2f}"

    elif STRETCH_MODE.lower() == "percentile":
        stretch_lut, in_low, in_high = build_stretch_lut_from_percentiles(
            y_original, LOW_PERCENTILE, HIGH_PERCENTILE
        )
        stretch_title = (
            f"Original + Stretch (Percentile)\n"
            f"low={LOW_PERCENTILE}%, high={HIGH_PERCENTILE}%\n"
            f"value={in_low:.2f}~{in_high:.2f}"
        )
    else:
        sys.exit("STRETCH_MODE는 'minmax' 또는 'percentile' 이어야 합니다.")

    y_stretched_same_depth = apply_lut_same_dtype(y_original, stretch_lut)
    y_stretched_8 = convert_to_8bit_after_stretch(y_stretched_same_depth)

    img_stretched_rgb = reconstruct_display_image(y_stretched_8, is_color, original_ycbcr)

    hist_stretched = compute_histogram_manual(y_stretched_8)
    cdf_stretched = compute_cdf_from_hist_manual(hist_stretched)

    # --------------------------------------------------------
    # 10-4. Histogram Equalization 계산
    # --------------------------------------------------------
    y_stretch_eq, lut_eq_after_stretch = equalize_selected(
        y_stretched_8, use_manual=USE_MANUAL_EQUALIZATION
    )
    img_stretch_eq_rgb = reconstruct_display_image(y_stretch_eq, is_color, original_ycbcr)
    hist_stretch_eq = compute_histogram_manual(y_stretch_eq)
    cdf_stretch_eq = compute_cdf_from_hist_manual(hist_stretch_eq)

    y_eq_direct, lut_eq_direct = equalize_selected(
        y_original_8, use_manual=USE_MANUAL_EQUALIZATION
    )
    img_eq_direct_rgb = reconstruct_display_image(y_eq_direct, is_color, original_ycbcr)
    hist_eq_direct = compute_histogram_manual(y_eq_direct)
    cdf_eq_direct = compute_cdf_from_hist_manual(hist_eq_direct)

    eq_name = "Manual HistEq" if USE_MANUAL_EQUALIZATION else "OpenCV HistEq"
    clahe_name = "Manual CLAHE" if USE_MANUAL_CLAHE else "OpenCV CLAHE"

    # --------------------------------------------------------
    # 10-5. CLAHE 계산
    # --------------------------------------------------------
    # (A) 원본 8비트 버전에 CLAHE
    y_clahe_direct, tile_luts_direct, _, _ = apply_clahe_selected(
        y_original_8,
        use_manual=USE_MANUAL_CLAHE,
        grid_rows=CLAHE_GRID_ROWS,
        grid_cols=CLAHE_GRID_COLS,
        clip_limit=CLAHE_CLIP_LIMIT
    )
    img_clahe_direct_rgb = reconstruct_display_image(y_clahe_direct, is_color, original_ycbcr)
    hist_clahe_direct = compute_histogram_manual(y_clahe_direct)

    # 비교 기준용 global HistEq LUT
    lut_global_original, _, _ = build_equalization_lut_manual(y_original_8)

    # (B) Stretch 후 8비트 버전에 CLAHE
    y_clahe_after_stretch, tile_luts_stretch, _, _ = apply_clahe_selected(
        y_stretched_8,
        use_manual=USE_MANUAL_CLAHE,
        grid_rows=CLAHE_GRID_ROWS,
        grid_cols=CLAHE_GRID_COLS,
        clip_limit=CLAHE_CLIP_LIMIT
    )
    img_clahe_after_stretch_rgb = reconstruct_display_image(
        y_clahe_after_stretch, is_color, original_ycbcr
    )
    hist_clahe_after_stretch = compute_histogram_manual(y_clahe_after_stretch)

    lut_global_stretched, _, _ = build_equalization_lut_manual(y_stretched_8)

    # --------------------------------------------------------
    # 10-6. 첫 번째 큰 Plot:
    #      Histogram Equalization 결과 4x4
    # --------------------------------------------------------
    fig_eq, axes_eq = plt.subplots(4, 4, figsize=FIGSIZE_EQ)

    # Row 1 : Original
    plot_image(axes_eq[0, 0], original_display_rgb, f"Original Image ({input_bit_depth}-bit input)")
    plot_histogram(axes_eq[0, 1], hist_original, "Histogram of Original (8-bit version)")
    plot_cdf(axes_eq[0, 2], cdf_original, "CDF of Original (8-bit version)")
    plot_mapping_8bit(
        axes_eq[0, 3],
        lut_identity,
        "Identity Mapping",
        mapping_color="black",
        show_identity=False
    )

    # Row 2 : Stretch
    plot_image(axes_eq[1, 0], img_stretched_rgb, stretch_title + "\n(then converted to 8-bit)")
    plot_histogram(axes_eq[1, 1], hist_stretched, "Histogram of Stretched (8-bit)")
    plot_cdf(axes_eq[1, 2], cdf_stretched, "CDF of Stretched (8-bit)")

    if input_bit_depth == 16:
        plot_mapping_16bit(
            axes_eq[1, 3],
            stretch_lut,
            "Stretch Mapping (16-bit)",
            mapping_color="red",
            show_identity=True
        )
    else:
        plot_mapping_8bit(
            axes_eq[1, 3],
            stretch_lut,
            "Stretch Mapping (8-bit)",
            mapping_color="red",
            show_identity=True
        )

    # Row 3 : Stretch 입력에 대한 HistEq
    plot_image(
        axes_eq[2, 0],
        img_stretch_eq_rgb,
        f"Stretched (8-bit) + {eq_name}"
    )
    plot_histogram(axes_eq[2, 1], hist_stretch_eq, "Histogram of Stretch + HistEq")
    plot_cdf(axes_eq[2, 2], cdf_stretch_eq, "CDF of Stretch + HistEq")
    plot_mapping_8bit(
        axes_eq[2, 3],
        lut_eq_after_stretch,
        "HistEq Mapping (Input = Stretched 8-bit Image)",
        mapping_color="blue",
        show_identity=True
    )

    # Row 4 : Original 입력에 대한 HistEq
    plot_image(
        axes_eq[3, 0],
        img_eq_direct_rgb,
        f"Original 8-bit Version + {eq_name}"
    )
    plot_histogram(axes_eq[3, 1], hist_eq_direct, "Histogram of Direct HistEq")
    plot_cdf(axes_eq[3, 2], cdf_eq_direct, "CDF of Direct HistEq")
    plot_mapping_8bit(
        axes_eq[3, 3],
        lut_eq_direct,
        "HistEq Mapping (Input = Original 8-bit Version)",
        mapping_color="green",
        show_identity=True
    )

    plt.figure(fig_eq.number)
    plt.suptitle(
        "Histogram Stretching and Histogram Equalization on BT.709 Y Channel",
        fontsize=SUPTITLE_FONTSIZE
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # --------------------------------------------------------
    # 10-7. 두 번째 큰 Plot:
    #      Hist Eq와 CLAHE 비교 결과 5x4
    # --------------------------------------------------------
    fig_compare, axes_cmp = plt.subplots(5, 4, figsize=FIGSIZE_COMPARE)

    # Row 1 : Original
    plot_image(axes_cmp[0, 0], original_display_rgb, f"Original Image ({input_bit_depth}-bit input)")
    plot_histogram(axes_cmp[0, 1], hist_original, "Histogram of Original (8-bit version)")
    plot_cdf(axes_cmp[0, 2], cdf_original, "CDF of Original (8-bit version)")
    plot_mapping_8bit(
        axes_cmp[0, 3],
        lut_identity,
        "Identity Mapping",
        mapping_color="black",
        show_identity=False
    )

    # Row 2 : Stretch
    plot_image(axes_cmp[1, 0], img_stretched_rgb, stretch_title + "\n(then converted to 8-bit)")
    plot_histogram(axes_cmp[1, 1], hist_stretched, "Histogram of Stretched (8-bit)")
    plot_cdf(axes_cmp[1, 2], cdf_stretched, "CDF of Stretched (8-bit)")

    if input_bit_depth == 16:
        plot_mapping_16bit(
            axes_cmp[1, 3],
            stretch_lut,
            "Stretch Mapping (16-bit)",
            mapping_color="red",
            show_identity=True
        )
    else:
        plot_mapping_8bit(
            axes_cmp[1, 3],
            stretch_lut,
            "Stretch Mapping (8-bit)",
            mapping_color="red",
            show_identity=True
        )

    # Row 3 : Original + HistEq
    plot_image(
        axes_cmp[2, 0],
        img_eq_direct_rgb,
        f"Original 8-bit Version + {eq_name}"
    )
    plot_histogram(axes_cmp[2, 1], hist_eq_direct, "Histogram of Direct HistEq")
    plot_cdf(axes_cmp[2, 2], cdf_eq_direct, "CDF of Direct HistEq")
    plot_mapping_8bit(
        axes_cmp[2, 3],
        lut_eq_direct,
        "HistEq Mapping (Input = Original 8-bit Version)",
        mapping_color="green",
        show_identity=True
    )

    # Row 4 : Original + CLAHE
    plot_image(
        axes_cmp[3, 0],
        img_clahe_direct_rgb,
        f"Original 8-bit Version + {clahe_name}\n"
        f"grid={CLAHE_GRID_ROWS}x{CLAHE_GRID_COLS}, clip_limit={CLAHE_CLIP_LIMIT}"
    )
    plot_histogram(axes_cmp[3, 1], hist_clahe_direct, "Histogram of Original + CLAHE")
    plot_tile_mappings_grid(
        axes_cmp[3, 2],
        tile_luts_direct,
        f"Local Tile Mappings ({CLAHE_GRID_ROWS}x{CLAHE_GRID_COLS})"
    )
    plot_mapping_8bit(
        axes_cmp[3, 3],
        lut_global_original,
        "Global HistEq Mapping (Input = Original 8-bit Version)",
        mapping_color="purple",
        show_identity=True
    )

    # Row 5 : Stretch + CLAHE
    plot_image(
        axes_cmp[4, 0],
        img_clahe_after_stretch_rgb,
        f"Stretched (8-bit) + {clahe_name}\n"
        f"grid={CLAHE_GRID_ROWS}x{CLAHE_GRID_COLS}, clip_limit={CLAHE_CLIP_LIMIT}"
    )
    plot_histogram(axes_cmp[4, 1], hist_clahe_after_stretch, "Histogram of Stretch + CLAHE")
    plot_tile_mappings_grid(
        axes_cmp[4, 2],
        tile_luts_stretch,
        f"Local Tile Mappings on Stretched Image ({CLAHE_GRID_ROWS}x{CLAHE_GRID_COLS})"
    )
    plot_mapping_8bit(
        axes_cmp[4, 3],
        lut_global_stretched,
        "Global HistEq Mapping (Input = Stretched 8-bit Image)",
        mapping_color="purple",
        show_identity=True
    )

    plt.figure(fig_compare.number)
    plt.suptitle(
        "Comparison of Histogram Equalization and CLAHE on BT.709 Y Channel",
        fontsize=SUPTITLE_FONTSIZE
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # --------------------------------------------------------
    # 10-8. 화면 출력
    # --------------------------------------------------------
    plt.show()


# ============================================================
# 11. 프로그램 시작점
# ============================================================
if __name__ == "__main__":
    main()