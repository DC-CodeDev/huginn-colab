# Calibración del prechequeo de memoria (Huginn / Colab)

Modelo lineal provisional de estimación antes de encolar un trabajo (`POST /jobs`):

```text
estimado_bytes = BASE_MB * 1024 * 1024 + k * vóxeles_remuestreados_1_5mm
```

Constantes configurables por variables de entorno (calibrados con 1 punto, revisar):

- `HUGINN_MEMORY_BASE_MB` (default `DEFAULT_MEMORY_BASE_MB = 4000.0` MB): base fija estimada para runtime Python + PyTorch + CUDA + modelos.
- `HUGINN_MEMORY_ESTIMATION_K` (default `DEFAULT_MEMORY_ESTIMATION_K = 150.0` bytes/vóxel): costo marginal por vóxel remuestreado a `1.5 mm` por eje.

En cada corrida, `run_segmentation_worker` imprime en la salida de la celda (`[volume] ...`) y guarda en `diagnostics.volume` la forma del volumen, espaciado (`spacing_mm`), vóxeles originales y vóxeles remuestreados a `1.5 mm` (sin datos de paciente) para sumar nuevos puntos a esta tabla.

## Puntos medidos

| # | Fecha / Nota | Dimensiones `(X, Y, Z)` | Espaciado `(mm)` | Vóxeles originales | Vóxeles remuestreados `1.5 mm` | `validating` pico RSS | `locating_l3` pico RSS (duración) | `segmenting_tissues` pico RSS (duración) | `packaging` pico RSS | VRAM máx usada | RAM total runtime | ¿Entró? |
| - | ------------ | ----------------------- | ---------------- | ------------------ | ------------------------------ | --------------------- | --------------------------------- | ---------------------------------------- | -------------------- | -------------- | ----------------- | ------- |
| 1 | Plan gratis (1er punto) | `(?, ?, 421)` *(421 cortes)* | `(?, ?, 1.3)` | *pendiente* | *pendiente* | `0.66 GB` | `7.36 GB` (`56 s`) | `6.76 GB` (`50 s`) | `3.70 GB` | `3.20 GB` | `~12.7 GB` | Sí |

### Observaciones del punto 1
- El pico de RAM de CPU ocurrió en `locating_l3` (tarea `total` con `roi_subset`), no en `segmenting_tissues`.
- La VRAM usada máxima fue `3.20 GB` (no es el cuello de botella en T4 de 15 GB).
- Se observa una base fija de `~3.7 GB` (torch + CUDA + modelos) al finalizar en `packaging`.
