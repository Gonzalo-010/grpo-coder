# Coding AI local

Qwen2.5-Coder + GRPO + LoRA en una GPU NVIDIA bajo WSL2. Todo local: sin APIs externas.

**Estudio con resultados y gráficas:** [docs/INVESTIGACION.md](docs/INVESTIGACION.md) (13 variantes de entrenamiento comparadas de forma pareada en una RTX 5070 de 12 GB).

## Instalación (WSL2, Ubuntu)

```bash
sudo apt install bubblewrap
python3 -m venv ~/.venvs/project && source ~/.venvs/project/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx: CUDA >= 12.8
pip install -r requirements.txt
```

Todos los comandos, desde la carpeta del proyecto y con el entorno activado (fuera de él `python` no existe y
`python3` del sistema no tiene torch).

## Comandos

```bash
python tests/run_tests.py [-k texto]      # SKIP = no verificado en esta máquina (falta GPU, torch o bwrap)
python -m problems.validate               # dataset: 0 errores antes de entrenar
python train.py eval --split val [--checkpoint DIR] [--n 16] [--out DIR]
python train.py phase2 [--fresh] [--run DIR] [--init-from CKPT]   # GRPO + LoRA; reanuda del último checkpoint válido
python train.py improve                   # phase2 + currículo + grupos de reparación
python train.py fix --iters 3 [--checkpoint .../checkpoints/best]
python train.py distill [--teacher Qwen/Qwen2.5-Coder-1.5B-Instruct] [--k 8] [--epochs 2]   # SFT con soluciones verificadas
python train.py genbench [--checkpoint DIR]   # sólo mide: generación eager vs CUDA graphs (caché estática)
python train.py humaneval [--checkpoint DIR] [--n 8]   # benchmark externo (HumanEval, 164) en el sandbox
python agent.py /ruta/al/repo "tarea" [--test "python -m pytest -x -q"]
python app.py                             # panel en http://localhost:8501
python -m experiments.run PLAN            # brazos comparables (experiments/plans/*.yaml) + informe pareado
python -m experiments.calibrate --n 8     # dificultad de cada problema para el modelo de partida
python -m experiments.worker              # cola: ejecuta los trabajos que se dejen en runs/jobs/queue/
```

Ctrl+C (o «Detener y guardar» en el panel) guarda un checkpoint del último step completo; un segundo Ctrl+C corta.
Un fallo (OOM no recuperable, excepción) también deja un checkpoint de emergencia del último estado completo.

## Dataset

`problems/problems.py`: los 19 originales intactos (`core`). `problems/catalog/`: 183 más con `split`, `category`
(14, todas en cada split), `family` (cada familia vive en un único split) y `version` (huella de enunciado y tests).
Con `dataset.sets: [core, catalog]` (por defecto) hay 102 de train, 50 de val y 50 de test; val y test nunca entran
en el entrenamiento. `python -m problems.validate` comprueba por problema determinismo, referencia, tests suficientes,
diversidad de salidas (un predictor constante no puede acertar > 60 % ni en los ocultos ni en el generador), mutation
score >= 0.8, que ninguna entrada oculta aparezca en el prompt o en el feedback y que la versión O(n²) de los de
optimización falle; y globalmente duplicados, familias repartidas y problemas casi iguales entre splits.

## Entrenamiento (training/)

- `logprob.py`: log-probabilidades sólo de los tokens de respuesta, con la cabeza aplicada por trozos y recalculada
  en el backward: no se materializan los logits (B, T, 151936). En la RTX 5070, 4 x (200+384) tokens: 0.87 GB de
  pico extra frente a 3.99 GB con logits completos.
- `grpo.py`: un único trainer. `grpo.updates_per_batch` (U) pasadas por lote y `grpo.minibatches`; con U=1 el ratio
  es exactamente 1. La referencia (KL) se calcula una vez por lote. Dropout a 0 (on-policy). Un OOM en un
  micro-batch reduce `micro_bs` a la mitad y rehace ese minibatch; un gradiente no finito no se aplica.
- `model/vram.py`: `vram.budget_gb: auto` limita la memoria del proceso a la VRAM libre menos `headroom_gb`. En WSL,
  pasarse no da OOM sino que pagina en la RAM (steps 10-30x más lentos); con el límite, el exceso es un OOM que el
  trainer sí sabe manejar.
- `metrics.jsonl`: por step, tiempos por fase (`t_sample`, `t_gen`, `t_score_wait`, `t_repair`, `t_ref`, `t_old`,
  `t_update`), tokens/s, VRAM reservada, RAM/swap/CPU (`sys_*`) y avisos `slow` / `vram_paging`.
- `checkpoint.py` (formato 2): escritura atómica, `MANIFEST.json` con sha256 de cada fichero (un checkpoint
  truncado o corrupto se descarta al reanudar), `best/` reemplazado sin quedar a medias, optimizador, RNG y
  metadatos: arquitectura, crecimiento, linaje, hash del código y de la config.

## Crecimiento del modelo (model/growth.py)

`growth.enabled: true` añade bloques Transformer nuevos repartidos por la profundidad (cada uno detrás de una capa
original, copiando sus pesos y con `o_proj`/`down_proj` a cero: la salida del modelo es bit a bit la misma al
crecer). Los bloques nuevos se entrenan completos en fp32 con su propio grupo de optimizador (`growth.lr`, warmup
desde el step en que nacen); las capas originales siguen con LoRA. `growth.schedule` admite crecer en varios
momentos. La referencia del KL ignora los bloques nuevos (es el modelo original). Los checkpoints guardan los
eventos de crecimiento y se reconstruyen al cargar (`train.py eval`, panel, `fix`, reanudación). Desactivado por
defecto: el `config_hash` de un config sin crecimiento no cambia.

## Experimentos (experiments/)

Un plan define brazos sobre la misma config base: mismos datos, semilla, nº de steps y evaluación final (pareada
por problema contra el primer brazo y contra el modelo de partida: delta de pass@1 con IC bootstrap y test de
signos). `experiments/plans/`: `smoke` (comprueba la tubería) y `main` (los 13 brazos del estudio: A baseline, U2
dos pasadas por lote, A_lr2 su control, lr5, Abat, DS, BIN, el control RND, B crecimiento, KD/RFT/KD_RL destilación y
C modelo de 1.5B; tabla 1 de [docs/INVESTIGACION.md](docs/INVESTIGACION.md)). Salidas en `runs/exp/<plan>/` (`report.md`). Relanzar el mismo
comando continúa donde se quedó; `--steps N` alarga todos los brazos reanudando sus checkpoints. Además de claves de
config, un brazo admite `init_from: <run o checkpoint>` (entrena partiendo de esos pesos) y `eval_checkpoint: <run o
checkpoint>` (no entrena: sólo evalúa, p. ej. un modelo destilado).

## Destilación (training/distill.py)

`train.py distill`: un profesor (por defecto el propio modelo: "RFT") genera k soluciones por problema de train, el
sandbox se queda con las que pasan todos los tests (hasta `--per-problem` distintas) y el alumno hace SFT con ellas
(el mismo trainer con ventaja +1, KL 0 y log-probs a T=1 = NLL por token). Deja un checkpoint normal para `eval` y
para `phase2 --init-from`. Motivo: el RL sube pass@1 pero no pass@16 (afina, no amplía); destilar un modelo más
capaz es la vía estándar para que un modelo pequeño aprenda lo que no sabe.

## Evaluación

`train.py eval`: greedy + pass@1/5/10/16 con IC 95 % bootstrap, desglose por nivel y categoría, errores por tipo,
compilación y truncado. Durante el entrenamiento, `eval.during_train: [val]` evalúa cada `eval_every_steps` y lo
compara con el modelo de partida (cacheado en `runs/eval_cache`): `val_delta_pass@1`, su IC y el test de signos
están en `metrics.jsonl`. `eval.select_best: val` elige `best/` por val pass@1. ¿Mejora de verdad?:
`python -m evaluation.compare A.json B.json`. Test, una sola vez al final.

Un checkpoint se evalúa (y lo usan fix, el agente y el panel) con LoRA SIN fusionar, la misma cuenta que en el
entrenamiento. El código original lo fusionaba en los pesos bf16, y el redondeo atenuaba la actualización y le
añadía ruido (ver `model/infer.py`); el efecto medido en pass@1 fue nulo, pero así la evaluación es exacta por
construcción (a cambio de ~1,8x más tiempo de evaluación). `experiments.run` repite las de la versión antigua y el
informe compara las dos.

## Resto

- `fix`: genera, ejecuta en el sandbox y devuelve al modelo el primer caso visible que falla (de los ocultos sólo
  cuántos fallan). `fix.feedback: counterexample` y `fix.engine: v2` son variantes medibles con
  `python train.py fix --compare --chains 4`.
- `improve`: currículo por aprendibilidad y grupos de reparación (`improve.curriculum: v2` es experimental).
- Agente: trabaja en una copia (`runs/agent/<id>/work`), tests en el sandbox sin red; si una edición empeora los
  tests vuelve al mejor estado. `agent.patience` (parar tras N rondas sin mejorar) viene desactivado: en el A/B
  resolvía menos repos. Tu repo sólo cambia al aceptar.
- Panel: tareas, cadena de intentos, agente, entrenamiento, métricas (también de `runs/exp`), checkpoints y
  ajustes (en `runs/ui`, no tocan `config.yaml`). Escucha en 127.0.0.1; si Windows no abre `localhost:8501`,
  `python app.py --host 0.0.0.0`.
- `config.yaml` es la única fuente; `model`, `dtype`, `lora.*`, `grpo.lr`, `grpo.weight_decay` (y `growth.*` si está
  activo) definen el estado del modelo: si cambian, el entrenamiento empieza un run nuevo.
