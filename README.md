# Drone Localization Pack — воспроизводимое тестирование моделей

Репозиторий предназначен для **воспроизводимого прогона сохраненных ResNet/ResNeXt моделей** по изображению
и получения единых артефактов (predictions/report/metrics/plots) для магистерской работы.

Важно: **датасет и веса не входят в репозиторий** — их нужно скачать и положить в соответствующие папки.

## Состав репозитория

- `dataset_v1/` — структура датасета (images/labels).
- `scripts/` — скрипты аудита, оценки моделей и агрегации результатов.
- `results/` — примеры выходных артефактов (CSV/JSON/PNG/XLSX).
- `requirements_base.txt` — зависимости без PyTorch.
- `requirements_full.txt` — зависимости + PyTorch/torchvision.
- `requirements.lock` — "полный слепок" окружения Colab (опционально).

## Требования

- Python 3.10+.
- PyTorch (CPU или CUDA).
- В Colab: `gdown` и `p7zip-full` для скачивания/распаковки архива.

## Формат датасета

Ожидаемая структура:

```
dataset_v1/
  images/{train,val,test}/*.png
  labels/{train,val,test}/_{split}_annotations.csv
```

CSV строго содержит колонки:

```
filename, lat, lon, alt, x, y, z, w
```

Пример строки:

```
img_16202.png, 5.67195845, -3.11980271, 1.5, -0.0016745, -0.25948498, 0.00387561, -0.96573794
```

Если у вас есть только `labels/*.txt`, используйте `scripts/00_build_annotations_from_txt.py`.

## Быстрый старт (Google Colab)

1) Откройте Colab и загрузите/распакуйте репозиторий в `/content/drone_localization_pack`.

2) Установите зависимости:

```bash
pip -q install -r requirements_base.txt
```

Если PyTorch отсутствует или нужен CPU-only:

```bash
pip -q install -r requirements_full.txt
```

3) Скачайте датасет (Google Drive) и распакуйте:

```bash
pip -q install gdown
gdown --folder 1FFvvjTX_EcXBPGPc5z2pG1y4U_nIaMvi

apt-get update -y
apt-get install -y p7zip-full

7z x /content/datasets/dataset_v5.7z.001 -o/content/extracted_dataset
```

4) Объедините распакованный датасет с `dataset_v1`:

```bash
rsync -av --update /content/extracted_dataset/dataset_v1/ /content/drone_localization_pack/dataset_v1/
```

5) Соберите аннотации из `.txt` (если нужно):

```bash
python scripts/00_build_annotations_from_txt.py \
  --dataset_root dataset_v1 \
  --image_ext .png \
  --splits train val test \
  --overwrite
```

6) Аудит датасета:

```bash
python scripts/0_dataset_audit.py \
  --dataset_root dataset_v1 \
  --outdir results/dataset_audit
```

7) Прогоны моделей:

```bash
python scripts/1_eval_resnet18_auto.py \
  --dataset_root dataset_v1 \
  --split test \
  --weights weights/best_model_18_100_128_2.pth \
  --outdir results/resnet18_2d \
  --batch_size 32 \
  --device cuda

python scripts/2_eval_resnet34_auto.py \
  --dataset_root dataset_v1 \
  --split test \
  --weights weights/best_model_34_100_128.pth \
  --outdir results/resnet34_7d \
  --batch_size 32 \
  --device cuda

python scripts/3_eval_resnext50_3d.py \
  --dataset_root dataset_v1 \
  --split test \
  --weights weights/best_geo_model_new.pth \
  --outdir results/resnext50_3d \
  --batch_size 16 \
  --num_workers 2 \
  --device cuda
```

8) Сводная таблица (авто-результаты + ручные, если есть `manual_results.json`):

```bash
python scripts/4_aggregate_results_v2.py \
  --results_root results \
  --manual manual_results.json \
  --out_xlsx results/model_comparison_v3.xlsx \
  --out_csv results/model_comparison_v3.csv
```

## Локальный запуск (Windows)

1) Создайте виртуальное окружение и установите зависимости:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements_base.txt
```

2) Установите PyTorch по официальной инструкции (CPU/CUDA).

3) Скопируйте датасет в `dataset_v1/` и веса в `weights/`.

4) Запустите те же команды из раздела Colab (пункты 5–8).

## Артефакты на выходе

- `predictions_*.csv` — предсказания.
- `report_*.csv` — таблица `true/pred/error` по кадрам.
- `metrics_*.json` — агрегированные метрики (MAE/MSE/RMSE + pos error).
- `*_hist.png`, `*_cdf.png` — распределение ошибки.
- `model_comparison_*.xlsx` / `model_comparison_*.csv` — итоговая сводная таблица.

## Скрипты

- `scripts/00_build_annotations_from_txt.py` — сбор CSV из `labels/*.txt`.
- `scripts/0_dataset_audit.py` — аудит структуры и целостности.
- `scripts/1_eval_resnet18_auto.py` — оценка ResNet18 (авто-выводность 2/3/7).
- `scripts/2_eval_resnet34_auto.py` — оценка ResNet34 (авто-выводность 2/3/7).
- `scripts/3_eval_resnext50_3d.py` — оценка ResNeXt50 (lat/lon/alt).
- `scripts/4_aggregate_results_v2.py` — сбор результатов в Excel/CSV.

Альтернативные/фиксированные варианты (на случай совместимости):
- `scripts/1_eval_resnet18_7d.py`
- `scripts/2_eval_resnet34_2d.py`
- `scripts/4_aggregate_results.py`

## Notebook

Для воспроизводимого запуска используйте ноутбук:

- `drone_localization_pipeline.ipynb`
