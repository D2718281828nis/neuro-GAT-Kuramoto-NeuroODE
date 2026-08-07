# Colab notebooks / Ноутбуки Colab

## English

Each notebook below is self-contained: open it directly in Google Colab (via
its "Open in Colab" badge) or run it locally with Jupyter. Every notebook
clones only what it needs, imports the real modules from `hmb_kuramoto_ode`
and `examples/` (nothing here is reimplemented), and produces its own plots.
All seven were executed end-to-end while this set was written, so the
committed `.ipynb` files already contain real outputs and figures — open
them on GitHub to see the plots without running anything.

| Notebook | What it shows |
| --- | --- |
| [`01_message_passing.ipynb`](01_message_passing.ipynb) | Plain mean aggregation on the real hierarchical rhythm graph; watches a signal spread hop by hop. |
| [`02_gat_attention.ipynb`](02_gat_attention.ipynb) | `EdgeAttention`'s segment softmax, trained on a toy classification task; attention weight by edge type before/after training. |
| [`03_grand_diffusion.ipynb`](03_grand_diffusion.ipynb) | The GRAND diffusion term from `KuramotoGrandVectorField`, in isolation; feature-variance decay over diffusion steps. |
| [`04_kuramoto_sync.ipynb`](04_kuramoto_sync.ipynb) | Classic Kuramoto phase coupling; order parameter vs. coupling strength, phase snapshots, then the same dynamics through the project's RK4 solver. |
| [`05_neural_ode.ipynb`](05_neural_ode.ipynb) | Euler vs. RK4 convergence order, backpropagation through every solver stage, and the real `KuramotoGrandVectorField` integrated end to end. |
| [`06_synthetic_multitask.ipynb`](06_synthetic_multitask.ipynb) | The full `HierarchicalKuramotoODE` trained on synthetic data across all three heads (graph/node/link), plus a Kuramoto/GRAND ablation comparison. |
| [`07_stew_real_experiment.ipynb`](07_stew_real_experiment.ipynb) | Real STEW windows, subject-disjoint split, GAT vs. full-model training, ROC/confusion-matrix results — the Colab counterpart of `examples/stew_real_experiment.py`. |

Notebooks 01-06 do a sparse, blobless git clone so they never download the
bundled STEW dataset; 07 needs the real recordings, so it clones the full
repository (or accepts your own STEW copy mounted from Google Drive instead —
see its second cell). Because they run on a Colab CPU, 06 and 07 deliberately
train on a small slice of data for a few epochs — genuinely fast, not
genuinely tuned. Each notebook's closing cell points to the full protocol
(`configs/stew_full.yaml`, `hmb_kuramoto_ode.cli cross-validate`) for anyone
who wants real numbers rather than a live demo.

`generate_notebooks.py` regenerates all seven `.ipynb` files from the cell
sources defined in that script (the same pattern as
`reports/generate_architecture.py`). After editing it, re-execute whichever
notebooks changed (e.g. `jupyter nbconvert --to notebook --execute --inplace
06_synthetic_multitask.ipynb`) so the committed files keep real outputs
instead of stale or empty ones.

## Русский

Каждый ноутбук ниже самодостаточен: открывайте его прямо в Google Colab (по
значку «Open in Colab») или запускайте локально в Jupyter. Каждый ноутбук
клонирует только необходимое, использует настоящие модули из
`hmb_kuramoto_ode` и `examples/` (ничего здесь не переписано заново) и сам
строит свои графики. Все семь были полностью выполнены при подготовке этого
набора, поэтому закоммиченные файлы `.ipynb` уже содержат реальные выводы и
рисунки — их можно посмотреть прямо на GitHub, ничего не запуская.

| Ноутбук | Что показывает |
| --- | --- |
| [`01_message_passing.ipynb`](01_message_passing.ipynb) | Простое усреднение по реальному иерархическому графу ритмов; распространение сигнала шаг за шагом. |
| [`02_gat_attention.ipynb`](02_gat_attention.ipynb) | Segment softmax в `EdgeAttention`, обученный на игрушечной задаче классификации; вес внимания по типу ребра до/после обучения. |
| [`03_grand_diffusion.ipynb`](03_grand_diffusion.ipynb) | Член диффузии GRAND из `KuramotoGrandVectorField` в изоляции; спад дисперсии признаков по шагам диффузии. |
| [`04_kuramoto_sync.ipynb`](04_kuramoto_sync.ipynb) | Классическая связь фаз Курамото; параметр порядка в зависимости от силы связи, снимки фаз, затем та же динамика через RK4-решатель проекта. |
| [`05_neural_ode.ipynb`](05_neural_ode.ipynb) | Порядок сходимости Euler и RK4, обратное распространение через каждый этап решателя и реальный `KuramotoGrandVectorField`, проинтегрированный целиком. |
| [`06_synthetic_multitask.ipynb`](06_synthetic_multitask.ipynb) | Полная `HierarchicalKuramotoODE`, обученная на синтетических данных по всем трём головам (граф/узел/связь), плюс сравнение абляций Курамото/GRAND. |
| [`07_stew_real_experiment.ipynb`](07_stew_real_experiment.ipynb) | Реальные окна STEW, разбиение без пересечения испытуемых, обучение GAT и полной модели, ROC и матрица ошибок — аналог `examples/stew_real_experiment.py` для Colab. |

Ноутбуки 01–06 выполняют разреженное клонирование без файловых блобов, поэтому
никогда не скачивают бандл STEW; 07 нужны настоящие записи, поэтому он
клонирует полный репозиторий (либо принимает собственную копию STEW,
смонтированную из Google Drive — см. вторую ячейку). Поскольку они работают на
CPU в Colab, 06 и 07 намеренно обучаются на небольшом срезе данных за
несколько эпох — это осознанно быстро, а не тщательно настроено. Последняя
ячейка каждого ноутбука указывает на полный протокол (`configs/stew_full.yaml`,
`hmb_kuramoto_ode.cli cross-validate`) для тех, кому нужны настоящие цифры, а
не живая демонстрация.

`generate_notebooks.py` пересобирает все семь файлов `.ipynb` из исходников
ячеек, определённых в этом скрипте (по тому же принципу, что и
`reports/generate_architecture.py`). После его правки заново выполните
изменившиеся ноутбуки (например, `jupyter nbconvert --to notebook --execute
--inplace 06_synthetic_multitask.ipynb`), чтобы в закоммиченных файлах
оставались настоящие выводы, а не устаревшие или пустые.
