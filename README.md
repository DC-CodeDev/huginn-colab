# Huginn v2 — Servicio de Segmentación en Google Colab

Servicio FastAPI + TotalSegmentator (`2.18.0`) pensado para ejecutarse en Google Colab con GPU T4 y exponerse mediante un túnel HTTPS de ngrok.

## Requisitos en Google Colab

1. **GPU activa**: `Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU T4`.
2. **Secretos de Colab** (ícono de llave en la barra lateral izquierda):
   - `NGROK_AUTHTOKEN` (obligatorio): authtoken de tu cuenta de ngrok.
   - `HUGINN_API_TOKEN` (recomendado): token Bearer compartido con la app. Si no se define, se genera uno temporal por sesión.
   - `HUGINN_TOTALSEG_LICENSE` (opcional): licencia académica para `tissue_4_types`. Sin ella, el servicio funciona en modo degradado (solo L3).
   - `NGROK_DOMAIN` (opcional): dominio fijo de ngrok (también puede ingresarse en el formulario del notebook).
   - `GITHUB_TOKEN` (solo si este repositorio es privado): Personal Access Token con permiso de lectura (`contents:read`).

## Uso rápido

1. Abrí `main.ipynb` en Google Colab.
2. En la **Celda 1**, completá los parámetros del formulario:
   - `REPO_URL`: URL HTTPS de este repositorio (ej. `https://github.com/<usuario>/huginn-colab.git`).
   - `RAMA_O_ETIQUETA`: rama o tag a clonar (por defecto `main`).
   - `NGROK_DOMAIN`: opcional, tu dominio fijo de ngrok.
   - `APP_URL`: opcional, URL base de la app web (para imprimir el enlace directo).
3. Ejecutá **Entorno de ejecución > Ejecutar todo**.
4. La primera vez, Colab pedirá permiso para acceder a los Secretos configurados.
5. Al finalizar, copiá el enlace directo (si definiste `APP_URL`) o el código de conexión `HGN1-...` impreso al final de la salida.

## Pruebas locales

```bash
pytest colab/tests
```

## Celda de diagnóstico (opcional)

El notebook incluye una celda opcional de diagnóstico después de la instalación que imprime las versiones de Python, CUDA, `torch`, `numpy` y `pip freeze`. Solo es necesario ejecutarla si se requiere ajustar las versiones exactas de `requirements.txt` en un entorno nuevo de Colab.

## Publicar solo `colab/` en un repositorio independiente

Para extraer únicamente la carpeta `colab/` sin el resto del proyecto ni su historial previo:

```bash
# Desde la raíz del repositorio principal
git subtree split --prefix=colab -b colab-standalone

# Crear un repositorio remoto vacío en GitHub (ej. huginn-colab) y luego:
git push https://github.com/<TU_USUARIO>/huginn-colab.git colab-standalone:main
```

Para sincronizar actualizaciones futuras desde `HEAD:colab` hacia el repositorio público, ejecutá `./scripts/publicar-colab.sh` (o con `--dry-run` para simular sin push).

## Extensiones aditivas al Contrato 2.0

Los siguientes campos son opcionales y retrocompatibles con clientes 2.0:

- `GET /health`:
  - `ram_total_mb` (`number | null`): RAM total del sistema en MiB.
  - `ram_available_mb` (`number | null`): RAM disponible del sistema en MiB.
- `GET /jobs/{job_id}`:
  - `diagnostics` (`object | null`): resumen de consumo de memoria por etapa (`stages.<etapa>` con `duration_seconds`, `samples`, `peak_rss_mb`, `peak_sys_used_mb`, `min_sys_avail_mb`, `peak_vram_used_mb`, `peak_vram_allocated_mb`), sin datos de paciente.
- `POST /jobs`:
  - Prechequeo de RAM antes de procesar usando `HUGINN_MEMORY_ESTIMATION_K` (por defecto `20.0`, a calibrar con mediciones reales). Si la memoria estimada supera la RAM disponible, rechaza con HTTP `422`, `error_code: "ERR_INSUFFICIENT_MEMORY"` y `stage: "validating"`.
