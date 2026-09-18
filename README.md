# Optimización AP — Adaptación de Markowitz a Accounts Payable

## Objetivo

Este proyecto demuestra cómo la lógica de optimización de Markowitz puede adaptarse a un problema empresarial de **asignación de capacidad operativa limitada** en Accounts Payable (AP).

No se trata de una aplicación de inversión ni de una recomendación financiera. El “portafolio” representa una distribución hipotética de la capacidad mensual de un equipo de AP entre seis actividades.

## Actividades

El modelo utiliza exactamente estas seis actividades:

1. Procesamiento de facturas estándar
2. Resolución de excepciones
3. Revisión de facturas sin PO
4. Revisión de facturas de alto monto
5. Seguimiento de facturas pendientes
6. Conciliación / revisión de statements de proveedores

## Analogía con Markowitz

| Markowitz financiero | Adaptación a AP |
|---|---|
| Activo | Actividad de Accounts Payable |
| Capital invertido | Capacidad operativa disponible |
| Rendimiento | Rendimiento / eficiencia operativa |
| Volatilidad | Riesgo o variabilidad operativa |
| Portafolio | Distribución de capacidad entre actividades |
| Frontera eficiente | Combinaciones eficientes entre rendimiento y riesgo |

La aplicación busca mostrar cómo cambia el equilibrio entre eficiencia y variabilidad cuando se distribuye la capacidad de forma diferente.

## Datos sintéticos

La aplicación genera automáticamente aproximadamente **1,000 registros de facturas sintéticas**, distribuidos durante 12 meses.

Cada registro contiene:

- Fecha
- Mes
- Actividad
- Monto de factura
- Tiempo de procesamiento
- Indicador de excepción
- Indicador de cumplimiento de PO
- Indicador de pago a tiempo
- Horas de procesamiento

Los datos se generan con una semilla fija para que los resultados sean reproducibles.

**Importante:** los datos son simulados. No contienen información real, confidencial o propietaria de Honeywell.

## Las siete variables principales

El análisis descriptivo y de correlación utiliza:

1. Volumen de facturas
2. Monto promedio
3. Tiempo de procesamiento
4. Tasa de excepciones
5. Cumplimiento de PO
6. Pago a tiempo
7. Horas de procesamiento

Su función es principalmente descriptiva:

- **Volumen de facturas:** carga de trabajo.
- **Monto promedio:** magnitud económica de las facturas.
- **Tiempo de procesamiento:** eficiencia / complejidad.
- **Tasa de excepciones:** fricción o variabilidad operativa.
- **Cumplimiento de PO:** calidad / eficiencia del proceso.
- **Pago a tiempo:** cumplimiento del objetivo de procesamiento.
- **Horas de procesamiento:** consumo de capacidad.

No es necesario introducir las siete variables directamente dentro de la fórmula de Markowitz.

## Rendimiento operativo

El rendimiento se define como un índice compuesto:

- **40% Pago a tiempo**
- **30% Cumplimiento de PO**
- **30% Eficiencia de procesamiento**

El tiempo de procesamiento tiene una dirección inversa: menor tiempo es mejor. Por ello se transforma primero a una escala normalizada donde:

> mayor eficiencia = mejor rendimiento

La eficiencia de procesamiento se obtiene invirtiendo y normalizando el tiempo de procesamiento entre las seis actividades.

El índice final queda en una escala comparable entre actividades.

## Riesgo operativo

El riesgo no representa volatilidad financiera.

Se define como incertidumbre / variabilidad operacional:

- **40% Tasa de excepciones**
- **30% Variabilidad del tiempo de procesamiento**
- **30% Variabilidad de horas de procesamiento**

Los componentes se normalizan antes de combinarlos para evitar que una variable domine únicamente por estar expresada en una escala diferente.

Una actividad con mayor riesgo operativo presenta, dentro del escenario sintético, más fricción o un comportamiento menos predecible y puede requerir capacidad variable.

## Metodología de Markowitz adaptada

### 1. Rendimiento esperado

Para cada actividad se calcula el rendimiento operativo esperado a partir del índice compuesto.

### 2. Riesgo individual

Cada actividad recibe un índice de riesgo operativo entre 0 y 1.

### 3. Correlaciones

Se calcula el co-movimiento de la eficiencia operativa mensual entre las seis actividades.

Esto permite representar que dos actividades pueden variar de forma relacionada, en lugar de asumir que sus riesgos son completamente independientes.

### 4. Matriz de covarianza

Se construye una matriz de covarianza adaptada:

- La diagonal se relaciona con el riesgo operativo de cada actividad.
- Las correlaciones se estiman a partir del comportamiento mensual sintético.
- La matriz se proyecta a una matriz positiva semidefinida para permitir una optimización estable.

No es una matriz de covarianza financiera.

### 5. Simulación

La aplicación genera miles de combinaciones de pesos.

Cada portafolio cumple:

- peso >= 0
- suma de pesos = 100%

Cada combinación representa una posible distribución de la capacidad del equipo.

Para cada combinación se calcula:

- Rendimiento esperado
- Riesgo operativo
- Índice de eficiencia ajustada por riesgo

### 6. Portafolio de mínima varianza

Se utiliza optimización restringida para encontrar la combinación con el menor riesgo operativo, manteniendo:

- pesos no negativos
- suma de pesos = 100%

### 7. Eficiencia ajustada por riesgo

Se utiliza:

`Rendimiento operativo / Riesgo operativo`

La aplicación lo llama:

> **Índice de eficiencia ajustada por riesgo**

Es una adaptación académica para comparar eficiencia operativa en relación con riesgo.

**No utiliza una tasa libre de riesgo financiera y no debe interpretarse como un Sharpe Ratio financiero tradicional.**

## Frontera eficiente

La aplicación calcula una frontera aproximada mediante optimización restringida para diferentes niveles objetivo de rendimiento.

El gráfico muestra:

- nube de portafolios simulados
- frontera eficiente
- portafolio base
- portafolio de mínima varianza
- portafolio de eficiencia ajustada por riesgo

La frontera representa diferentes trade-offs entre rendimiento operativo y riesgo operativo.

## Portafolio base

Se utiliza una asignación inicial transparente para facilitar la comparación:

| Actividad | Base |
|---|---:|
| Procesamiento de facturas estándar | 42% |
| Resolución de excepciones | 16% |
| Revisión de facturas sin PO | 10% |
| Revisión de facturas de alto monto | 8% |
| Seguimiento de facturas pendientes | 12% |
| Conciliación / revisión de statements de proveedores | 12% |

La asignación suma 100%.

Esta base es un supuesto didáctico y no representa la distribución real de capacidad de ninguna organización.

## Dashboard

La aplicación Streamlit contiene:

1. Resumen de KPIs
2. Análisis de actividades
3. Definición de rendimiento y riesgo
4. Matriz de correlación
5. Heatmap
6. Selector interactivo de variables
7. Scatter plot
8. Correlación de Pearson
9. Interpretación automática
10. Simulación de portafolios
11. Frontera eficiente
12. Comparación Base vs. Mínima Varianza vs. Eficiencia Ajustada por Riesgo
13. Tabla y gráfica de asignaciones
14. Limitaciones del modelo
15. Carga opcional de CSV

## Carga opcional de CSV

Si no se carga un archivo, la aplicación utiliza automáticamente datos sintéticos.

Para cargar un CSV, las columnas requeridas son exactamente:

```text
Fecha
Monto de factura
Tiempo de procesamiento
Indicador de excepción
Indicador de cumplimiento de PO
Indicador de pago a tiempo
Horas de procesamiento
Actividad
```

Los tres indicadores deben estar codificados como 0/1.

La columna `Actividad` debe utilizar las seis actividades definidas por el proyecto.

No cargues información confidencial, personal o propietaria.

## Instalación

Se recomienda Python 3.10+.

Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Desde la carpeta del proyecto:

```bash
streamlit run app.py
```

Streamlit abrirá la aplicación en el navegador.

## Archivos

```text
app.py
requirements.txt
README.md
```

## Tecnologías

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy

No se requieren librerías adicionales.

## Limitaciones

Este proyecto es una demostración académica de optimización de capacidad.

Entre sus principales limitaciones:

- Los datos son sintéticos.
- Los resultados dependen de los supuestos de generación de datos.
- El modelo no conoce restricciones reales de personal, habilidades, SLA, prioridades o controles.
- La matriz de covarianza es una adaptación operacional, no financiera.
- La frontera eficiente muestra trade-offs matemáticos; no determina por sí sola una asignación operativa definitiva.
- El índice de eficiencia ajustada por riesgo es una métrica académica creada para este caso.
- La correlación no demuestra causalidad.
- Los pesos optimizados no deben interpretarse como instrucciones automáticas para operar un proceso de AP.

## Privacidad

La aplicación está diseñada para funcionar sin datos empresariales reales.

Los datos sintéticos se generan localmente en la sesión de Streamlit. Si se utiliza la función de carga de CSV, el usuario debe asegurarse de que el archivo no contenga información confidencial, propietaria o personal.

## Uso responsable

El modelo debe utilizarse como herramienta de análisis y aprendizaje para explorar la asignación de capacidad.

Las decisiones reales de operación deben considerar adicionalmente:

- demanda real
- capacidad disponible
- habilidades del equipo
- SLAs
- prioridades del negocio
- controles internos
- segregación de funciones
- criticidad de proveedores
- restricciones regulatorias
- información histórica de calidad

