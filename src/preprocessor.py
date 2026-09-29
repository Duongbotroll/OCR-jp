"""Tầng 1 - Tiền xử lý ảnh & phân tích bố cục cho tài liệu viết tay tiếng Nhật.

Chuỗi xử lý:
    Resize -> Grayscale -> Denoise -> CLAHE -> Binarize -> Deskew -> Crop borders
    -> Detect text direction (横書き / 縦書き) -> Layout analysis -> Reading order

Mục tiêu KHÔNG phải là "làm đẹp" ảnh, mà là giảm các yếu tố gây nhiễu cho VLM
(bóng đổ khi chụp điện thoại, giấy ố vàng, ảnh bị nghiêng) và tạo ra các khối văn bản
có thứ tự đọc đúng chuẩn tiếng Nhật.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from config import PreprocessConfig
from src.utils import ensure_bgr, resize_max_side, setup_logger

logger = setup_logger(__name__)

Orientation = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class TextBlock:
    """Một khối văn bản được phát hiện bởi layout analysis (tọa độ theo pixel)."""

    x: int
    y: int
    w: int
    h: int
    order: int = 0  # Thứ tự đọc (0 là khối được đọc đầu tiên)

    @property
    def area(self) -> int:
        """Diện tích khối (pixel^2)."""
        return self.w * self.h

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Tọa độ dạng (x1, y1, x2, y2)."""
        return self.x, self.y, self.x + self.w, self.y + self.h

    def crop(self, image: np.ndarray, padding: int = 8) -> np.ndarray:
        """Cắt vùng ảnh tương ứng với khối, có thêm lề để không mất nét chữ ở biên."""
        height, width = image.shape[:2]
        x1 = max(0, self.x - padding)
        y1 = max(0, self.y - padding)
        x2 = min(width, self.x + self.w + padding)
        y2 = min(height, self.y + self.h + padding)
        return image[y1:y2, x1:x2].copy()


@dataclass
class PreprocessResult:
    """Kết quả của tầng tiền xử lý."""

    image: np.ndarray  # Ảnh BGR đã xử lý - đầu vào cho OCR engine
    binary: np.ndarray  # Ảnh nhị phân (chữ trắng / nền đen) dùng cho phân tích layout
    orientation: Orientation  # Hướng viết đã xác định
    skew_angle: float  # Góc đã xoay để chỉnh nghiêng (độ)
    blocks: list[TextBlock] = field(default_factory=list)
    steps: dict[str, np.ndarray] = field(default_factory=dict)  # Ảnh trung gian để debug/UI


class ImagePreprocessor:
    """Bộ tiền xử lý ảnh tài liệu viết tay tiếng Nhật dựa trên OpenCV."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # API chính
    # ------------------------------------------------------------------
    def process(self, image: np.ndarray) -> PreprocessResult:
        """Chạy toàn bộ chuỗi tiền xử lý trên một trang tài liệu.

        Args:
            image: Ảnh đầu vào (BGR, BGRA hoặc xám).

        Returns:
            PreprocessResult chứa ảnh đã xử lý, các khối văn bản và ảnh trung gian.
        """
        steps: dict[str, np.ndarray] = {}
        bgr = resize_max_side(ensure_bgr(image), self.config.max_side)
        steps["01_input"] = bgr.copy()

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.config.denoise:
            gray = self.denoise(gray)
            steps["02_denoised"] = gray.copy()
        if self.config.clahe:
            gray = self.enhance_contrast(gray)
            steps["03_contrast"] = gray.copy()

        binary = self.binarize(gray)

        skew_angle = 0.0
        if self.config.deskew:
            skew_angle = self.estimate_skew(binary)
            # Bỏ qua góc quá nhỏ: xoay ảnh luôn gây nội suy làm mờ nét chữ
            if abs(skew_angle) >= 0.1:
                gray = self.rotate(gray, skew_angle, border_value=255)
                binary = self.binarize(gray)
                steps["04_deskewed"] = gray.copy()
            else:
                skew_angle = 0.0

        if self.config.crop_borders:
            x, y, w, h = self.content_bounding_box(binary)
            gray = gray[y : y + h, x : x + w]
            binary = binary[y : y + h, x : x + w]
            steps["05_cropped"] = gray.copy()

        steps["06_binary"] = binary.copy()

        if self.config.text_direction == "auto":
            orientation = self.detect_orientation(binary)
        else:
            orientation = self.config.text_direction  # type: ignore[assignment]

        blocks: list[TextBlock] = []
        if self.config.layout_analysis:
            blocks = self.analyze_layout(binary, orientation)

        # VLM hiện đại được huấn luyện trên ảnh tự nhiên => mặc định dùng ảnh xám đã tăng
        # tương phản. Ảnh nhị phân chỉ nên dùng cho giấy rất bẩn hoặc nền có hoa văn.
        ocr_source = cv2.bitwise_not(binary) if self.config.binarize_for_ocr else gray
        ocr_image = cv2.cvtColor(ocr_source, cv2.COLOR_GRAY2BGR)

        logger.debug(
            "Preprocess done: size=%s, skew=%.2f, orientation=%s, blocks=%d",
            ocr_image.shape[:2], skew_angle, orientation, len(blocks),
        )
        return PreprocessResult(
            image=ocr_image,
            binary=binary,
            orientation=orientation,
            skew_angle=skew_angle,
            blocks=blocks,
            steps=steps,
        )

    # ------------------------------------------------------------------
    # Tăng chất lượng ảnh
    # ------------------------------------------------------------------
    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Khử nhiễu Non-Local Means: giữ nét chữ mảnh tốt hơn Gaussian/Median blur."""
        return cv2.fastNlMeansDenoising(
            gray, None, h=float(self.config.denoise_strength),
            templateWindowSize=7, searchWindowSize=21,
        )

    @staticmethod
    def enhance_contrast(gray: np.ndarray) -> np.ndarray:
        """CLAHE: cân bằng histogram cục bộ, xử lý tốt ảnh có bóng đổ không đều."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def binarize(gray: np.ndarray) -> np.ndarray:
        """Nhị phân hóa thích nghi (adaptive threshold).

        Trả về ảnh chữ TRẮNG trên nền ĐEN (thuận tiện cho findContours/morphology).
        Kích thước cửa sổ tỉ lệ với kích thước ảnh để ổn định trên nhiều độ phân giải.
        """
        height, width = gray.shape[:2]
        block_size = max(15, (min(height, width) // 40) | 1)  # luôn là số lẻ
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            block_size, 15,
        )
        # Mở hình thái học để xóa các chấm nhiễu li ti (bụi scan, hạt giấy)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # ------------------------------------------------------------------
    # Chỉnh nghiêng
    # ------------------------------------------------------------------
    @staticmethod
    def rotate(
        image: np.ndarray, angle: float, border_value: int = 255, expand: bool = True
    ) -> np.ndarray:
        """Xoay ảnh quanh tâm một góc ``angle`` độ (dương = ngược chiều kim đồng hồ).

        Args:
            image: Ảnh cần xoay.
            angle: Góc xoay (độ).
            border_value: Giá trị pixel lấp vào vùng trống (255 = trắng cho ảnh xám).
            expand: Mở rộng canvas để không bị cắt mất góc ảnh.
        """
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        size = (width, height)
        if expand:
            cos_a, sin_a = abs(matrix[0, 0]), abs(matrix[0, 1])
            new_width = int(height * sin_a + width * cos_a)
            new_height = int(height * cos_a + width * sin_a)
            matrix[0, 2] += new_width / 2.0 - center[0]
            matrix[1, 2] += new_height / 2.0 - center[1]
            size = (new_width, new_height)
        return cv2.warpAffine(
            image, matrix, size, flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
        )

    def estimate_skew(self, binary: np.ndarray) -> float:
        """Ước lượng góc nghiêng bằng phương pháp Projection Profile.

        Ý tưởng: khi dòng (横書き) hoặc cột (縦書き) chữ thẳng hàng, tổng pixel theo hàng
        hoặc theo cột sẽ dao động mạnh nhất (variance lớn nhất). Ta thử các góc trong
        [-max_skew_angle, +max_skew_angle], tìm thô bước 0.5° rồi tinh chỉnh bước 0.1°.
        Cách này ổn định hơn minAreaRect với chữ viết tay và hoạt động cho cả hai hướng viết.
        """
        max_angle = float(self.config.max_skew_angle)
        if max_angle <= 0 or cv2.countNonZero(binary) == 0:
            return 0.0

        small = resize_max_side(binary, 1000)  # Giảm kích thước để tìm nhanh

        def score(angle: float) -> float:
            rotated = self.rotate(small, angle, border_value=0, expand=False)
            row_profile = rotated.sum(axis=1, dtype=np.float64)
            col_profile = rotated.sum(axis=0, dtype=np.float64)
            return max(float(np.var(row_profile)), float(np.var(col_profile)))

        coarse_angles = np.arange(-max_angle, max_angle + 1e-6, 0.5)
        best = float(max(coarse_angles, key=score))
        fine_angles = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1)
        best = float(max(fine_angles, key=score))
        return round(best, 2)

    @staticmethod
    def content_bounding_box(binary: np.ndarray, margin: int = 20) -> tuple[int, int, int, int]:
        """Tìm hình chữ nhật bao toàn bộ nội dung chữ để cắt bỏ viền giấy trống."""
        height, width = binary.shape[:2]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        coords = cv2.findNonZero(cleaned)
        if coords is None:
            return 0, 0, width, height
        x, y, w, h = cv2.boundingRect(coords)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(width, x + w + margin), min(height, y + h + margin)
        return x1, y1, x2 - x1, y2 - y1

    # ------------------------------------------------------------------
    # Hướng viết & bố cục
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_char_size(binary: np.ndarray) -> int:
        """Ước lượng kích thước ký tự (pixel) bằng trung vị kích thước thành phần liên thông."""
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return 20
        sizes = stats[1:, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
        valid = sizes[(sizes[:, 0] > 3) & (sizes[:, 1] > 3)]
        if len(valid) == 0:
            return 20
        return int(max(8, np.median(valid.max(axis=1))))

    def detect_orientation(self, binary: np.ndarray) -> Orientation:
        """Xác định văn bản viết ngang (横書き) hay dọc (縦書き).

        Heuristic: giãn nở (dilate) ảnh theo phương ngang và phương dọc với cùng độ dài.
        Phương nào nối được nhiều ký tự thành ít thành phần liên thông hơn chính là hướng
        viết. Đây là heuristic nhanh, không cần model; có thể ghi đè qua text_direction.
        """
        if cv2.countNonZero(binary) == 0:
            return "horizontal"
        small = resize_max_side(binary, 1200)
        _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
        char_size = self.estimate_char_size(small)
        length = max(3, int(char_size * 1.2))

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        horizontal_count = cv2.connectedComponents(cv2.dilate(small, horizontal_kernel))[0]
        vertical_count = cv2.connectedComponents(cv2.dilate(small, vertical_kernel))[0]

        return "vertical" if vertical_count < horizontal_count * 0.85 else "horizontal"

    def analyze_layout(self, binary: np.ndarray, orientation: Orientation) -> list[TextBlock]:
        """Phát hiện các khối văn bản (đoạn / cột) và sắp xếp theo thứ tự đọc tiếng Nhật.

        - 横書き: giãn nở mạnh theo chiều ngang để nối ký tự thành dòng, nhẹ theo chiều dọc
          để nối các dòng gần nhau thành đoạn.
        - 縦書き: ngược lại, giãn nở theo chiều dọc để nối ký tự thành cột.
        """
        char_size = self.estimate_char_size(binary)
        along = max(3, int(char_size * 2.0))  # Nối các ký tự trong cùng dòng/cột
        across = max(1, int(char_size * 0.8))  # Nối các dòng/cột kề nhau
        kernel_size = (along, across) if orientation == "horizontal" else (across, along)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < self.config.min_block_area or min(w, h) < 5:
                continue
            boxes.append((x, y, w, h))

        boxes = self._merge_overlapping(boxes)
        ordered = self._sort_reading_order(boxes, orientation, tolerance=char_size)
        return [TextBlock(x, y, w, h, order=index) for index, (x, y, w, h) in enumerate(ordered)]

    @staticmethod
    def _merge_overlapping(
        boxes: list[tuple[int, int, int, int]], gap: int = 2
    ) -> list[tuple[int, int, int, int]]:
        """Gộp các hộp chồng lấn nhau (lặp tới khi không còn cặp nào cần gộp)."""
        merged = list(boxes)
        changed = True
        while changed:
            changed = False
            result: list[tuple[int, int, int, int]] = []
            while merged:
                x, y, w, h = merged.pop()
                index = 0
                while index < len(merged):
                    ox, oy, ow, oh = merged[index]
                    overlap = (
                        x - gap < ox + ow and ox - gap < x + w
                        and y - gap < oy + oh and oy - gap < y + h
                    )
                    if overlap:
                        nx1, ny1 = min(x, ox), min(y, oy)
                        nx2, ny2 = max(x + w, ox + ow), max(y + h, oy + oh)
                        x, y, w, h = nx1, ny1, nx2 - nx1, ny2 - ny1
                        merged.pop(index)
                        changed = True
                    else:
                        index += 1
                result.append((x, y, w, h))
            merged = result
        return merged

    @staticmethod
    def _sort_reading_order(
        boxes: list[tuple[int, int, int, int]], orientation: Orientation, tolerance: int
    ) -> list[tuple[int, int, int, int]]:
        """Sắp xếp khối theo thứ tự đọc chuẩn tiếng Nhật.

        - 横書き: trên -> dưới, trong cùng một hàng thì trái -> phải.
        - 縦書き: phải -> trái (theo cạnh phải của cột), trong cùng cột thì trên -> dưới.
        """
        if not boxes:
            return []
        groups: list[list[tuple[int, int, int, int]]] = []

        if orientation == "horizontal":
            for box in sorted(boxes, key=lambda b: b[1]):
                if groups and box[1] - groups[-1][0][1] <= tolerance:
                    groups[-1].append(box)
                else:
                    groups.append([box])
            return [box for group in groups for box in sorted(group, key=lambda b: b[0])]

        for box in sorted(boxes, key=lambda b: -(b[0] + b[2])):
            right_edge = box[0] + box[2]
            anchor = groups[-1][0] if groups else None
            if anchor is not None and (anchor[0] + anchor[2]) - right_edge <= tolerance:
                groups[-1].append(box)
            else:
                groups.append([box])
        return [box for group in groups for box in sorted(group, key=lambda b: b[1])]

    # ------------------------------------------------------------------
    # Trực quan hóa
    # ------------------------------------------------------------------
    @staticmethod
    def draw_blocks(image: np.ndarray, blocks: list[TextBlock]) -> np.ndarray:
        """Vẽ khung các khối văn bản kèm số thứ tự đọc lên ảnh (dùng cho UI/debug)."""
        canvas = ensure_bgr(image.copy())
        thickness = max(2, canvas.shape[1] // 600)
        for block in blocks:
            x1, y1, x2, y2 = block.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 90, 255), thickness)
            cv2.putText(
                canvas, str(block.order + 1), (x1 + 4, max(20, y1 + 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 30, 30), thickness, cv2.LINE_AA,
            )
        return canvas
