from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

import network2


class MnistModelService:
    #Обёртка над моделью network2 для загрузки весов, центрирования и распознавания цифр
    def __init__(
        self,
        weights_path: Union[str, Path] = "network2.json",
        image_size: tuple[int, int] = (28, 28),
        auto_invert: bool = True,
        light_bg_threshold: float = 127.5,
        center_digit: bool = True,
        digit_box_size: int = 20,
        fg_threshold: float = 15.0,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.image_size = image_size
        self.auto_invert = auto_invert
        self.light_bg_threshold = light_bg_threshold
        self.center_digit = center_digit
        self.digit_box_size = digit_box_size
        self.fg_threshold = fg_threshold

        self.net = network2.load(str(self.weights_path))

    def _load_grayscale(self, image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        # Открывает изображение и переводит его в оттенки серого без потери пропорций
        if isinstance(image_input, Image.Image):
            img = image_input.copy()
        elif isinstance(image_input, bytes):
            img = Image.open(BytesIO(image_input))
        else:
            img = Image.open(Path(image_input))

        if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, rgba).convert("L")
        else:
            img = img.convert("L")

        return img

    @staticmethod
    def _shift_2d(arr: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
        #Сдвигает 2D-массив на целое число пикселей с заполнением пустых краёв нулями
        h, w = arr.shape
        shifted = np.zeros_like(arr)

        src_y_start = max(0, -shift_y)
        src_y_end = min(h, h - shift_y)
        dst_y_start = max(0, shift_y)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)

        src_x_start = max(0, -shift_x)
        src_x_end = min(w, w - shift_x)
        dst_x_start = max(0, shift_x)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        if src_y_end > src_y_start and src_x_end > src_x_start:
            shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = arr[
                src_y_start:src_y_end, src_x_start:src_x_end
            ]

        return shifted

    def _center_and_resize(self, arr: np.ndarray) -> np.ndarray:
        """
        Обрезает цифру по границам, вписывает в digit_box_size с сохранением пропорций,
        размещает на холсте image_size и выравнивает центр масс по центру кадра.
        """
        target_h, target_w = self.image_size
        mask = arr > self.fg_threshold

        # Если холст пустой — просто приводит к нужному размеру
        if not np.any(mask):
            pil_empty = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
            return np.asarray(
                pil_empty.resize((target_w, target_h), resample=Image.Resampling.LANCZOS),
                dtype=np.float64,
            )

        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max() + 1
        x_min, x_max = xs.min(), xs.max() + 1

        cropped = arr[y_min:y_max, x_min:x_max]
        crop_h, crop_w = cropped.shape

        # Масштабирует так, чтобы большая сторона равнялась digit_box_size
        max_side = max(crop_h, crop_w)
        scale = self.digit_box_size / float(max_side)
        new_w = max(1, int(round(crop_w * scale)))
        new_h = max(1, int(round(crop_h * scale)))

        pil_crop = Image.fromarray(np.clip(cropped, 0, 255).astype(np.uint8), mode="L")
        resized_crop = np.asarray(
            pil_crop.resize((new_w, new_h), resample=Image.Resampling.LANCZOS),
            dtype=np.float64,
        )

        # помещает масштабированную цифру на пустой холст target_h x target_w
        canvas = np.zeros((target_h, target_w), dtype=np.float64)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized_crop

        # вычисляет центр масс яркости и сдвигает к центру кадра
        total_mass = float(np.sum(canvas))
        if total_mass > 0:
            y_indices, x_indices = np.indices((target_h, target_w))
            com_y = float(np.sum(y_indices * canvas)) / total_mass
            com_x = float(np.sum(x_indices * canvas)) / total_mass

            center_y = (target_h - 1) / 2.0
            center_x = (target_w - 1) / 2.0

            shift_y = int(round(center_y - com_y))
            shift_x = int(round(center_x - com_x))
            canvas = self._shift_2d(canvas, shift_y=shift_y, shift_x=shift_x)

        return canvas

    def preprocess(self, image_input: Union[str, Path, bytes, Image.Image]) -> np.ndarray:
        #Преобразует изображение в вектор формы (784, 1) со значениями в [0.0, 1.0]
        gray_img = self._load_grayscale(image_input)
        arr = np.asarray(gray_img, dtype=np.float64)

        if self.auto_invert:
            corners = np.array([
                arr[0, 0],
                arr[0, -1],
                arr[-1, 0],
                arr[-1, -1],
            ])
            if float(np.mean(corners)) > self.light_bg_threshold:
                arr = 255.0 - arr

        if self.center_digit:
            arr = self._center_and_resize(arr)
        else:
            pil_resized = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").resize(
                self.image_size, resample=Image.Resampling.LANCZOS
            )
            arr = np.asarray(pil_resized, dtype=np.float64)

        normalized = np.clip(arr / 255.0, 0.0, 1.0)
        total_pixels = self.image_size[0] * self.image_size[1]
        return normalized.reshape((total_pixels, 1))

    def predict(self, image_input: Union[str, Path, bytes, Image.Image]) -> dict:
        #Выполняет прямой проход через сеть и возвращает предсказание и оценки по классам

        x = self.preprocess(image_input)
        output = self.net.feedforward(x)

        raw_scores = np.asarray(output, dtype=np.float64).reshape(-1)
        predicted_digit = int(np.argmax(raw_scores))

        shifted = raw_scores - np.max(raw_scores)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores)

        return {
            "digit": predicted_digit,
            "confidence": float(probs[predicted_digit]),
            "probabilities": {str(i): float(probs[i]) for i in range(len(probs))},
            "raw_scores": [float(v) for v in raw_scores],
        }


if __name__ == "__main__":
    service = MnistModelService(weights_path="network2.json", center_digit=True)
    print("Модель загружена, центрирование включено. Размеры слоёв:", getattr(service.net, "sizes", "не указаны"))