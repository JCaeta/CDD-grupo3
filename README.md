# Pipeline ETL - Limpieza y Preparación de Datos
## Asignación 2: Proceso ETL Completo

Este repositorio contiene un pipeline ETL (Extraer, Transformar, Cargar) completo diseñado para la **Asignación 2: Limpieza y Preparación de Datos**. El proyecto implementa un proceso colaborativo de limpieza de datos con responsabilidades individuales para diferentes aspectos de los datos.

## 🎯 Descripción del Proyecto

El objetivo es crear un dataset limpio y listo para modelado a partir de datos de consumo energético y meteorológicos. El proyecto está estructurado para simular un entorno de equipo donde cada persona es responsable de columnas específicas de datos.

### Estructura del Equipo

| Persona | Responsabilidad | Columnas |
|---------|-----------------|----------|
| **Persona 1** | Datos Temporales | `Appliances`, `date` |
| **Persona 2** | Sistema de Iluminación | `lights` |
| **Persona 3** | Sensores de Temperatura Interna | `T1`, `T2`, `T3`, `T4`, `T5`, `T6`, `T7`, `T8`, `T9` |
| **Persona 4** | Sensores de Humedad Interna | `RH_1`, `RH_2`, `RH_3`, `RH_4`, `RH_5`, `RH_6`, `RH_7`, `RH_8`, `RH_9` |
| **Persona 5** | Datos Meteorológicos Externos (Parte 1) | `T_out`, `RH_out`, `Tdewpoint` |
| **Persona 6** | Datos Meteorológicos Externos (Parte 2) | `Pressure`, `Wind speed`, `Visibility` |

## 📁 Estructura de Archivos

```
├── dataset.csv                           # Dataset original (requerido)
├── etl_main.py                          # Orquestador principal del pipeline ETL
├── etl_person1_temporal.py              # Persona 1: Análisis de datos temporales
├── etl_person2_lighting.py              # Persona 2: Análisis del sistema de iluminación
├── etl_person3_temp_sensors.py          # Persona 3: Análisis de sensores de temperatura
├── etl_person4_humidity_sensors.py      # Persona 4: Análisis de sensores de humedad
├── etl_person5_weather_part1.py         # Persona 5: Análisis meteorológico parte 1
├── etl_person6_weather_part2.py         # Persona 6: Análisis meteorológico parte 2
├── streamlit_etl_app.py                 # Interfaz web de Streamlit
├── run_etl.py                           # Ejecutor simple de línea de comandos
├── INSTRUCCIONES.md                     # Este archivo
└── requirements.txt                     # Dependencias de Python
```

## 🛠️ Instalación

### Prerrequisitos

1. **Python 3.8 o superior**
2. **Paquetes de Python requeridos** (instalar vía requirements.txt)

### Pasos de Instalación

1. **Descargar o clonar** este repositorio
2. **Crear entorno virtual** (recomendado):
   ```bash
   # Crear entorno virtual
   python3 -m venv venv_etl
   
   # Activar entorno virtual
   source venv_etl/bin/activate  # En Linux/Mac
   # O en Windows: venv_etl\Scripts\activate
   ```
3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Asegurar que el dataset** se llame `dataset.csv` y esté ubicado en el directorio del proyecto

### Formato del Dataset Requerido

Su archivo `dataset.csv` debe contener las siguientes columnas:
- `date` - Información de fecha y hora
- `Appliances` - Consumo energético de electrodomésticos
- `lights` - Consumo energético de iluminación
- `T1` a `T9` - Lecturas de sensores de temperatura
- `RH_1` a `RH_9` - Lecturas de sensores de humedad
- `T_out`, `RH_out`, `Tdewpoint` - Datos meteorológicos externos parte 1
- `Pressure`, `Windspeed` (o `Wind speed`), `Visibility` - Datos meteorológicos externos parte 2

## 🚀 Ejecución del Pipeline ETL

Tiene **tres opciones** para ejecutar el pipeline ETL:

### Opción 1: Interfaz Web de Streamlit (Recomendada)

Lanzar la interfaz web interactiva:

```bash
streamlit run streamlit_etl_app.py
```

**Características:**
- Interfaz web interactiva
- Seguimiento de progreso en tiempo real
- Visualización de datos
- Secciones de análisis individual
- Descarga del dataset limpio
- Reportes comprehensivos

### Opción 2: Línea de Comandos (Simple)

Ejecutar el pipeline ETL directamente:

```bash
python run_etl.py
```

**Características:**
- Salida de línea de comandos
- Seguimiento de progreso
- Generación automática de reportes
- Ejecución ligera

### Opción 3: Script de Python (Avanzado)

Importar y usar en su propio código Python:

```python
from etl_main import ETLPipeline

# Inicializar pipeline
etl = ETLPipeline('dataset.csv')

# Ejecutar proceso ETL completo
etl.load_data()
etl.run_individual_analyses()
etl.integrate_external_data()
etl.save_final_dataset('datos_limpios.csv')
```

## 📊 Análisis por Persona - Explicación Simple

### 👤 Persona 1: Datos Temporales
**¿Qué hace?**
- Convierte las fechas al formato correcto
- Busca huecos en el tiempo (datos faltantes)
- Elimina valores imposibles de energía (negativos)
- Crea nuevas características: hora del día, día de la semana, temporada

**¿Por qué es importante?**
Los datos temporales son la base para entender patrones de consumo energético a lo largo del tiempo.

### 💡 Persona 2: Sistema de Iluminación
**¿Qué hace?**
- Verifica que los valores de iluminación sean válidos
- Rellena datos faltantes con métodos inteligentes
- Clasifica la intensidad: Apagado, Bajo, Medio, Alto
- Calcula eficiencia energética

**¿Por qué es importante?**
La iluminación representa una parte significativa del consumo energético del hogar.

### 🌡️ Persona 3: Sensores de Temperatura Interna
**¿Qué hace?**
- Verifica temperaturas razonables (entre 0°C y 50°C)
- Encuentra sensores que miden lo mismo (correlación)
- Usa sensores funcionales para estimar datos faltantes
- Calcula temperatura promedio y confort térmico

**¿Por qué es importante?**
La temperatura interna afecta directamente el consumo energético de calefacción y refrigeración.

### 💧 Persona 4: Sensores de Humedad Interna
**¿Qué hace?**
- Verifica que la humedad esté entre 0% y 100%
- Identifica zonas de confort (40-60%)
- Calcula riesgo de moho en áreas muy húmedas
- Evalúa uniformidad de humedad

**¿Por qué es importante?**
La humedad afecta el confort y puede influir en el uso de deshumidificadores y aire acondicionado.

### 🌤️ Persona 5: Datos Meteorológicos (Parte 1)
**¿Qué hace?**
- Verifica relaciones físicas (punto de rocío ≤ temperatura)
- Calcula índice de calor y temperatura aparente
- Categoriza condiciones climáticas
- Crea índices de confort exterior

**¿Por qué es importante?**
El clima exterior determina las necesidades energéticas del hogar.

### 🌪️ Persona 6: Datos Meteorológicos (Parte 2)
**¿Qué hace?**
- Valida presión atmosférica y detecta unidades
- Procesa velocidad del viento (debe ser positiva)
- Identifica condiciones de tormenta
- Calcula estabilidad meteorológica

**¿Por qué es importante?**
La presión, viento y visibilidad complementan el análisis meteorológico para predicciones energéticas.

## 🔧 Estrategias de Limpieza de Datos

### Manejo de Valores Faltantes
1. **Huecos cortos (≤3 valores)**: Rellenar hacia adelante
2. **Huecos medianos**: Imputación KNN usando variables correlacionadas
3. **Huecos largos**: Mediana estacional o imputación estadística

### Tratamiento de Valores Atípicos
1. **Valores inválidos**: Eliminar valores físicamente imposibles
2. **Valores extremos**: Limitar a rangos razonables
3. **Restricciones físicas**: Aplicar conocimiento del dominio

### Ingeniería de Características
- **Características temporales**: Hora, día de la semana, temporada
- **Características agregadas**: Promedios, rangos, desviaciones
- **Características categóricas**: Niveles de confort, categorías climáticas
- **Métricas derivadas**: Índices de eficiencia, confort y estabilidad

## 📈 Resultados Esperados

### Salidas Principales
1. **`dataset_final_cleaned.csv`** - Dataset limpio y listo para modelado
2. **`etl_processing_log.txt`** - Log completo del procesamiento
3. **Reporte de calidad de datos** - Métricas y validaciones

### Mejoras del Dataset
- ✅ **Sin valores faltantes** (o decisiones documentadas)
- ✅ **Sin valores inválidos** (dentro de límites físicos/lógicos)
- ✅ **Tipos de datos consistentes** y formato adecuado
- ✅ **Conjunto de características mejorado** para modelado
- ✅ **Consistencia temporal** y manejo adecuado de fechas

### Nuevas Características Creadas
- **Temporales**: Hora, día de la semana, temporada, indicadores de fin de semana
- **Agregadas**: Temperaturas/humedad promedio, rangos, medidas de uniformidad
- **Confort**: Índices de confort térmico, de humedad y meteorológico
- **Eficiencia**: Eficiencia de iluminación, ratios de utilización energética
- **Meteorológicas**: Índice de calor, temperatura aparente, indicadores de sistemas meteorológicos

## 🎨 Para Presentación PowerPoint

### Diapositiva 1: Título del Proyecto
**Pipeline ETL para Limpieza de Datos Energéticos**
- Procesamiento colaborativo de 6 personas
- Dataset de consumo energético y datos meteorológicos
- Objetivo: Dataset limpio y listo para modelado

### Diapositiva 2: División del Trabajo
**6 Responsabilidades Individuales:**
1. 📅 **Temporal** - Fechas y consumo de electrodomésticos
2. 💡 **Iluminación** - Sistema de luces
3. 🌡️ **Temperatura** - 9 sensores internos
4. 💧 **Humedad** - 9 sensores internos
5. 🌤️ **Meteorología 1** - Temperatura, humedad y punto de rocío externos
6. 🌪️ **Meteorología 2** - Presión, viento y visibilidad

### Diapositiva 3: Proceso ETL
**3 Fases Principales:**
- **E**xtraer: Cargar y validar datos originales
- **T**ransformar: Limpieza individual + ingeniería de características
- **L**oad (Cargar): Dataset final limpio y documentado

### Diapositiva 4: Técnicas de Limpieza
**Estrategias Aplicadas:**
- ✅ Validación de restricciones físicas
- ✅ Imputación inteligente de valores faltantes
- ✅ Detección y tratamiento de valores atípicos
- ✅ Ingeniería de características basada en dominio

### Diapositiva 5: Resultados Obtenidos
**Mejoras del Dataset:**
- Sin valores faltantes o inválidos
- +20 nuevas características creadas
- Validación de consistencia física
- Documentación completa del proceso

### Diapositiva 6: Herramientas Utilizadas
**Tecnologías:**
- 🐍 **Python** - Lenguaje principal
- 📊 **Pandas/Numpy** - Manipulación de datos
- 📈 **Streamlit** - Interfaz web interactiva
- 🤖 **Scikit-learn** - Imputación avanzada
- 📋 **Logging completo** - Trazabilidad

## 🔍 Solución de Problemas Comunes

### Errores Frecuentes

1. **"Módulo no encontrado"**
   - Verificar que el entorno virtual esté activado: `source venv_etl/bin/activate`
   - Verificar que todos los archivos `.py` estén en el mismo directorio
   - Instalar paquetes: `pip install -r requirements.txt`

2. **"Dataset no encontrado"**
   - Asegurar que `dataset.csv` existe en el directorio del proyecto
   - Verificar nombre y formato del archivo

3. **Problemas de memoria**
   - Usar muestreo para datasets grandes
   - Procesar datos en lotes si es necesario

4. **Problemas con el entorno virtual**
   - Verificar que Python 3.8+ esté instalado: `python3 --version`
   - Recrear el entorno si es necesario: `rm -rf venv_etl && python3 -m venv venv_etl`
   - Activar siempre antes de trabajar: `source venv_etl/bin/activate`

## 📞 Soporte

Si encuentra problemas:
1. **Revisar la sección de solución de problemas**
2. **Examinar los logs de procesamiento** para detalles de error
3. **Probar con muestras pequeñas** para identificar cuellos de botella
4. **Verificar formato de datos** y requisitos

---

## 🎯 Resumen Ejecutivo

Este pipeline ETL demuestra:
- **Flujos de trabajo colaborativos** en ciencia de datos
- **Técnicas comprehensivas de evaluación** de calidad de datos
- **Estrategias de limpieza específicas** del dominio energético
- **Ingeniería de características** para datos energéticos y meteorológicos
- **Mejores prácticas de documentación** y reproducibilidad

**¡El sistema está listo para producir un dataset limpio y de alta calidad para sus análisis energéticos!** 🚀

### Contacto del Proyecto
- **Repositorio**: CDD-grupo3
- **Propietario**: JCaeta
- **Rama**: main
- **Fecha**: 28 de septiembre de 2025