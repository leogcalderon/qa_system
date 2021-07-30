# QA System
Sistema Question-Answering entrenado con datos aumentados.

## Modelo
El modelo utilizado es el resultado del entrenamiento de un modelo robusto para QA propuesto como **[PROYECTO FINAL](https://github.com/leogcalderon/qa_system/tree/main/project)** en el curso de **[APRENDIZAJE PROFUNDO](https://datitos.github.io/curso-aprendizaje-profundo/)** dictado por los docentes de Datitos.

El modelo propuesto es DistilBert junto con dos capas lineales. Este fue entrenado con datasets de dominio general, con datasets aumentados de dominio específico y entrenamiento adversario.

El modelo recibe como entrada una pregunta y un contexto, el cual puede o no estar la respuesta. Esta entrada es consumida por un modelo DistilBert pre-entrenado, cuya salida es enviada a dos capas lineales, una para predecir el indice del token del contexto donde comienza la respuesta, y otra para predecir el indice del token del contexto donde termina la respuesta.

**El funcionamiento del siguiente script puede describirse en los siguientes pasos:**
1. Busqueda de la pregunta en Google Search de N articulos
2. Parseo de HTML para obtener los contextos.
3. Rankear articulos para seleccionar el mejor.
4. Extraer la respuesta del contexto con la salida del modelo.

# Probar QA System

1. Instalar las dependencias
```bash
pip install -r requirements.txt
```

2. Descargar modelo en el directorio `/model` desde [aqui](https://drive.google.com/drive/folders/1QD2orVj-T1XP87m3RNbUXNrwnAbwQvZV?usp=sharing)

3. Iniciar el servidor
```bash
uvicorn main:app
```

4. En el navegador ir a http://127.0.0.1:8000/docs
5. Seleccionar el endpoint `/predict`, luego `Try it out`, escribir la pregunta en el campo `question` y por ultimo `execute`. Será respondido con un JSON conteniendo la pregunta y la respuesta del modelo.
