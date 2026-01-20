# Drone Localization Evaluation Pack (dataset_v1)

Этот пакет предназначен для **воспроизводимого прогона наших сохранённых моделей** по готовому датасету `dataset_v1`
и получения **единых артефактов** для магистерской:

- `predictions_*.csv` — предсказания модели
- `report_*.csv` — таблица `true/pred/error` по каждому кадру
- `metrics_*.json` — агрегированные метрики (MAE/MSE/RMSE + pos error)
- `*_hist.png`, `*_cdf.png` — распределение ошибки позиционирования
- `model_comparison.xlsx` — итоговая сводная таблица (включая ручные результаты YOLO из `manual_results.json`)

> ⚠️ Пакет **не содержит датасет и веса**. Их нужно положить в соответствующие папки (см. ниже).

---

## 0) Структура папок (что куда класть)

После распаковки архива структура такая:

```
drone_localization_pack/
  dataset_v1/
    images/{train,val,test}/
    labels/{train,val,test}/_{split}_annotations.csv
  weights/
  results/
  scripts/
  manual_results.json
```

### Датасет

В датасете ожидается структура:

- `dataset_v1/images/train/*.png`
- `dataset_v1/images/val/*.png`
- `dataset_v1/images/test/*.png`
- `dataset_v1/labels/train/_train_annotations.csv`
- `dataset_v1/labels/val/_val_annotations.csv`
- `dataset_v1/labels/test/_test_annotations.csv`

### Формат CSV (единый для всех)

Колонки **строго** (как у тебя в Excel):

`filename, lat, lon, alt, x, y, z, w`

Пример строки:

`img_16202.png, 5.67195845, -3.11980271, 1.5, -0.0016745, -0.25948498, 0.00387561, -0.96573794`

---

## 1) Вариант A (рекомендуется): Google Colab

### 1.1 Подготовка
1) Открой Colab.
2) Залей архив `drone_localization_pack.zip` в Colab (или на Google Drive и смонтируй).
3) Распакуй архив (пример):

```bash
!unzip -q drone_localization_pack.zip
%cd drone_localization_pack
```

4) Положи датасет в папку `dataset_v1/` (можно через Drive).
5) Положи веса в `weights/`:
- `weights/best_model_18_100_128_2.pth`
- `weights/best_model_34_100_128.pth`
- `weights/best_geo_model_new.pth`

### 1.2 Установка зависимостей
В Colab обычно уже есть `torch/torchvision`. Ставим остальное:

```bash
!pip -q install -r requirements_base.txt
```

---

## 2) Вариант B: Windows ноутбук (локальный запуск)

### 2.1 Подготовка окружения
Рекомендуется Python 3.10+.

Пример через venv:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements_base.txt
```

Далее нужно установить **PyTorch** (CPU или CUDA) — проще всего по официальной инструкции PyTorch (версия зависит от твоей видеокарты).

После установки PyTorch проверь:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3) Последовательный запуск (одинаково для Colab и Windows)

> Все команды выполняются из корня папки `drone_localization_pack/`.

### Шаг 1 — аудит датасета (проверка структуры/CSV/NaN/отсутствующих файлов)

```bash
python scripts/0_dataset_audit.py --dataset_root dataset_v1 --outdir results/dataset_audit
```

Что получится:
- `results/dataset_audit/dataset_audit_summary.json`
- `results/dataset_audit/dataset_audit_table.csv`
- `results/dataset_audit/dataset_audit_table.xlsx`

---

### Шаг 2 — ResNet18 (7D: lat/lon/alt + quaternion)

```bash
python scripts/1_eval_resnet18_7d.py   --dataset_root dataset_v1   --split test   --weights weights/best_model_18_100_128_2.pth   --outdir results/resnet18_7d   --batch_size 16
```

Что получится:
- `results/resnet18_7d/predictions_test.csv`
- `results/resnet18_7d/report_test.csv`
- `results/resnet18_7d/metrics_test.json`
- графики: `resnet18_7d_test_*.(png)`

---

### Шаг 3 — ResNet34 (2D: lat/lon)

```bash
python scripts/2_eval_resnet34_2d.py   --dataset_root dataset_v1   --split test   --weights weights/best_model_34_100_128.pth   --outdir results/resnet34_2d   --batch_size 16
```

Что получится:
- `results/resnet34_2d/predictions_test.csv`
- `results/resnet34_2d/report_test.csv`
- `results/resnet34_2d/metrics_test.json`
- графики: `resnet34_2d_test_*.(png)`

---

### Шаг 4 — ResNeXt50_32x4d (3D: lat/lon/alt)

По умолчанию скрипт **фитит StandardScaler по train+val**, как в нашем обучении.

```bash
python scripts/3_eval_resnext50_3d.py   --dataset_root dataset_v1   --split test   --weights weights/best_geo_model_new.pth   --outdir results/resnext50_3d   --batch_size 16
```

Что получится:
- `results/resnext50_3d/predictions_test.csv`
- `results/resnext50_3d/report_test.csv`
- `results/resnext50_3d/metrics_test.json`
- графики: `resnext50_3d_test_*.(png)`

> Если твоя модель **вдруг** обучалась без scaler’а, добавь `--no_scaler`.

---

### Шаг 5 — сводная таблица (Excel) + добавление YOLO из manual_results.json

```bash
python scripts/4_aggregate_results.py   --results_root results   --manual manual_results.json   --out_xlsx results/model_comparison.xlsx   --out_csv results/model_comparison.csv
```

Что получится:
- `results/model_comparison.xlsx` (3 вкладки: auto_results / manual_results / combined)
- `results/model_comparison.csv`

---

## 4) Важные замечания про метрики (чтобы правильно интерпретировать)

- В отчётах считается `pos_error_2d = sqrt((lat_err)^2 + (lon_err)^2)`.
- Единицы измерения **такие же, как в твоих колонках lat/lon**.
  В AgroTechSim обычно это локальные координаты (часто метры), тогда `pos_error_2d` можно напрямую считать **ошибкой в метрах**.

---

## 5) Если не хватает памяти / очень медленно

Попробуй уменьшить batch size:

- `--batch_size 8` или `--batch_size 4`

И/или явно поставить CPU:

- `--device cpu`

---

## 6) (Опционально) дообучение ResNet34 (если захочешь)

Этот шаг **не обязателен** для воспроизведения метрик по уже готовым весам.

```bash
python scripts/optional_train_resnet34_2d.py --dataset_root dataset_v1 --epochs 10 --batch_size 32 --outdir results/train_resnet34_2d
```

После обучения можно прогнать evaluation:

```bash
python scripts/2_eval_resnet34_2d.py --dataset_root dataset_v1 --split test --weights results/train_resnet34_2d/best_resnet34_2d.pth --outdir results/resnet34_2d_retrained
```

---

## 7) Что в итоге прикладывать к научной работе (артефакты)

Рекомендуемый набор файлов/картинок:

- `results/model_comparison.xlsx` — итоговая таблица
- `results/*/metrics_test.json` — подтверждение метрик
- `results/*/resnet*_cdf.png` и `*_hist.png` — графики распределения ошибки
- `results/*/report_test.csv` — при необходимости для доп. анализа
- `manual_results.json` — как протокол ручных результатов (например YOLO11l-obb на LinQ H)
