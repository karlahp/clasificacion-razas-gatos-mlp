# Clasificación de razas de gatos con un MLP (Oxford-IIIT Pet)

**EP1 – Técnicas Avanzadas de Machine Learning I (TLY1101) · Duoc UC**
**Integrantes:** Karla Hoch, Sebastián Huaiquimilla

Este repositorio contiene un perceptrón multicapa (MLP) que clasifica la **raza de un gato** a partir de una fotografía, y un análisis crítico de su desempeño según qué tan parecidas son las razas entre sí.

---

## Contenido del repositorio

```
├── README.md                 Informe técnico (este documento)
├── notebooks/
│   └── EP1_TLY1101_MLP_Razas_Gatos_final_v2.ipynb   Notebook completo y ejecutable
├── data/                     Sin datos: el notebook descarga el dataset oficial
├── models/
│   └── mlp_gatos_parecidas.keras           Modelo de referencia entrenado
├── images/                   Figuras generadas por el notebook
└── resultados/               Tablas (.csv) generadas por el notebook
```

---

## 1. Descripción del problema de negocio

Una clínica veterinaria quiere agilizar el registro de pacientes. A partir de una fotografía de un gato, el sistema propone su **raza** para prellenar la ficha clínica (la raza condiciona controles preventivos y riesgos conocidos), y el personal confirma o corrige. Es una herramienta de **apoyo**, no de decisión.

## 2. Objetivos del proyecto

**General:** construir un MLP que clasifique la raza de un gato a partir de una fotografía, y evaluar críticamente su desempeño y limitaciones según el parecido visual entre razas.

**Específicos:**
1. Explorar y preparar los subconjuntos (calidad de datos, redimensionamiento, normalización, división estratificada).
2. Cuantificar con datos que el subconjunto PARECIDAS es realmente más parecido que DISTINTAS.
3. Implementar y justificar un MLP base, y compararlo con variantes (aumento de datos, learning rate, neuronas, profundidad, activación, batch, estandarización, dropout) usando varias semillas.
4. Evaluar con Accuracy, Precision, Recall, F1-Score y matriz de confusión, y analizar aciertos, errores y razas confundidas.
5. Contrastar el desempeño entre PARECIDAS y DISTINTAS, e identificar las limitaciones del MLP en imágenes.

## 3. Definición de KPIs

La línea base del azar con 5 clases balanceadas es **20 % de accuracy**. Los KPIs se fijaron antes de entrenar y no se modificaron después de ver los resultados.

| KPI | Meta | Medición |
|---|---|---|
| Principal: F1-Score macro en el set de prueba | ≥ 0,40 (el doble del azar) | Promedio de 5 semillas |
| Secundario: recall mínimo entre las 5 razas | ≥ 0,20 | Promedio de 5 semillas; significa que ninguna raza queda ignorada |

## 4. Descripción de las fuentes de datos

**Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (Parkhi et al., 2012): 37 razas de perros y gatos, unas 200 imágenes por raza, con variabilidad de fondo, postura e iluminación y anotaciones oficiales.

**Justificación de la elección:**
1. **Coherencia con el negocio:** son fotografías de mascotas con la raza como etiqueta, que es lo que necesita la clínica. Los otros datasets propuestos (escenarios naturales, enfermedades de plantas, Los Simpson) no se relacionan con este problema.
2. **Dificultad realista para un MLP:** la variabilidad de fondo, tamaño del animal e iluminación permite observar las limitaciones de un MLP sobre píxeles.
3. **Permite construir subconjuntos controlados por raza**, necesarios para estudiar el efecto del parecido visual entre clases.

**Cifras:** se usan las **7.349 imágenes con anotación oficial** (`list.txt`). La carpeta descargada trae 7.390 archivos `.jpg`; las 41 imágenes sin anotación se excluyen.

**Variante propia del proyecto.** "Perro vs. gato" es el ejemplo que da la pauta, por lo que se descartó. La variante combina:

- **Problema multiclase de 5 razas de gatos**, en lugar de clasificación binaria.
- **Dos subconjuntos disjuntos, ambos solo de gatos** (así la especie no cambia entre ellos):
  - **PARECIDAS:** Birman, Ragdoll, Siamese, Persian, Maine_Coon. Birman, Ragdoll y Siamese comparten el patrón *colorpoint*; Persian y Maine_Coon se incluyen por su pelaje largo, un parecido más débil.
  - **DISTINTAS:** Bombay, Sphynx, Russian_Blue, Bengal, Abyssinian. Cada raza difiere de las otras en color, textura o forma.
- **Cantidad limitada y pareja de imágenes por clase:** 184 por raza (la menor disponible entre las 10 razas, Bombay), sorteadas con semilla propia.
- **Semilla propia** (`SEED_SPLIT = 2604`) y división estratificada **70/15/15** (644 / 138 / 138 imágenes por subconjunto).
- **Comparación entre subconjuntos:** toda la experimentación se hace sobre PARECIDAS; la configuración elegida se aplica después a DISTINTAS sin reajustarla.

**Hipótesis (escrita antes de entrenar):** el MLP logrará un mayor F1 macro en DISTINTAS que en PARECIDAS, porque un MLP sobre píxeles aplanados se apoya sobre todo en color y textura global y no en la forma local de rasgos.

## 5. Preparación y análisis exploratorio de los datos (EDA)

Las decisiones de preprocesamiento se toman mirando **solo el set de entrenamiento**.

**Hallazgos del EDA:**
- **Balance:** cada subconjunto tiene 644 imágenes de entrenamiento, con 128 o 129 por raza. Como las clases están parejas por diseño, no se aplicó `class_weight`.
- **Dimensiones** (1.288 imágenes de entrenamiento): anchos de 114 a 1.725 px (media 438, mediana 500) y altos de 108 a 1.600 px (media 392, mediana 375). Las imágenes no son cuadradas (mediana 500×375), por lo que redimensionar a 64×64 las comprime horizontalmente; la deformación afecta a todas por igual.
- **Calidad:** 0 imágenes corruptas. Todas están en RGB salvo una en RGBA, que `load_img` convierte al cargarla.
- **Dificultades visuales:** poses muy variadas, fondos con mucho contenido, iluminación distinta y animales que ocupan una parte pequeña de la imagen.
- **Similitud entre razas** (medida antes de entrenar, a menor valor más parecidas):

| Medida (promedio de los 10 pares de razas) | PARECIDAS | DISTINTAS |
|---|---|---|
| Distancia entre histogramas de color | 0,0854 | 0,1178 |
| Distancia entre imágenes promedio | 0,0631 | 0,0830 |

El par más parecido de PARECIDAS (Birman – Siamese, 0,0404) está casi a la mitad de distancia que el de DISTINTAS (Bengal – Abyssinian, 0,0743). Estas medidas capturan solo color y apariencia global, por lo que son un indicador parcial.

**Preprocesamiento aplicado:**
1. Carga en RGB (3 canales).
2. Redimensionamiento a 64×64 (12.288 valores de entrada). La comparación con 96×96 y 128×128 no mejoró el resultado (F1 de validación 0,376 y 0,341 contra 0,397).
3. Normalización a [0, 1] (`Rescaling`) y estandarización por canal con media y desviación calculadas solo con entrenamiento.
4. Aumento de datos solo en entrenamiento (volteo horizontal, rotación hasta ~20° y desplazamientos de hasta 10 %).

## 6. Metodología utilizada (CRISP-DM)

| Fase | Dónde se aborda en el notebook |
|---|---|
| 1. Comprensión del negocio | Sección 1: problema, objetivos, KPIs, hipótesis |
| 2. Comprensión de los datos | Secciones 2 y 3: carga, variante, EDA, calidad de datos |
| 3. Preparación de los datos | Secciones 2 y 4: muestreo, división, redimensionamiento, normalización, aumento, similitud entre razas |
| 4. Modelado | Secciones 5, 6 y 7: arquitectura, experimentos, selección, entrenamiento y curvas |
| 5. Evaluación | Secciones 8, 9 y 10: métricas en prueba, errores, contraste entre subconjuntos, KPIs |
| 6. Despliegue | No se despliega. El modelo se guarda en `models/` y se proponen mejoras (CNN, transfer learning) en la sección 11 |

**Experimentación.** Se parte de un MLP base (M0) y se cambia un factor a la vez, con 3 semillas por configuración y reportando media ± desviación estándar, siempre sobre PARECIDAS y en validación. Criterio de selección fijado antes de ejecutar: mayor F1 macro de validación; entre las configuraciones dentro de una desviación estándar de la mejor, la de menos parámetros.

**Configuración final:** 1 capa oculta de 128 neuronas con ReLU, dropout 0,3, salida softmax de 5 neuronas, pérdida `sparse_categorical_crossentropy`, Adam con learning rate 1e-4, batch 32, estandarización por canal, aumento de datos, `EarlyStopping` sobre `val_loss` con paciencia 20 y entrada de 64×64.

## 7. Resultados y conclusiones

Resultados en el **set de prueba**, media ± desviación estándar de 5 semillas:

| Subconjunto | Accuracy | F1 macro | Recall mínimo | KPI principal (F1 ≥ 0,40) | KPI secundario (recall ≥ 0,20) |
|---|---|---|---|---|---|
| PARECIDAS | 0,365 ± 0,041 | 0,363 ± 0,041 | 0,254 ± 0,026 | No se cumple | Se cumple |
| DISTINTAS | 0,490 ± 0,036 | 0,487 ± 0,037 | 0,360 ± 0,059 | Se cumple | Se cumple |

- **Lo que más importó fue cómo se entrena:** estandarizar las entradas y bajar el learning rate llevaron el F1 de validación de 0,134 (M0) a 0,397. Tamaño de la capa, profundidad y batch no mostraron un efecto claro sobre una base que aprende; quitar el dropout o el aumento de datos aumentó el sobreajuste.
- **Hipótesis PARECIDAS vs. DISTINTAS:** se cumplió (diferencia de F1 macro de +0,124, unas 1,6 veces la suma de las desviaciones entre semillas). La hipótesis secundaria (confusión dominante Birman–Siamese) no se cumplió: en PARECIDAS el modelo usa a **Ragdoll como clase por defecto**.
- **Razas mejor reconocidas:** las de color liso y distintivo (Bombay, Russian_Blue). Bengal y Abyssinian se confunden entre sí, y son el par más cercano de DISTINTAS.
- **Sobreconfianza:** la confianza promedio (≈ 0,60) es mucho mayor que el accuracy (0,33), por lo que la probabilidad del modelo no sirve para filtrar errores.
- **Impacto en el negocio:** en PARECIDAS el modelo se equivoca en 2 de cada 3 imágenes. No es utilizable como apoyo en producción sin confirmación humana.

**Limitaciones del MLP en imágenes:** al aplanar la imagen se pierde la estructura espacial (orejas, hocico, patrón de la cara); los parámetros crecen con la resolución (1,57 millones para solo 644 imágenes de entrenamiento); no hay invarianza a traslaciones; y los resultados son sensibles a la semilla y a los hiperparámetros.

**Limitaciones del estudio:** una sola partición de datos y 138 imágenes de prueba; búsqueda de un factor a la vez; learning rate y tamaño de capa ajustados a 64×64; y **la independencia del set de prueba no es total**: la configuración final se eligió solo con validación, pero la ronda 2 de experimentos se diseñó tras ver un resultado en prueba de una ejecución anterior (sección 6.2 del notebook).

**Mejoras futuras:** redes convolucionales (CNN) o transfer learning; calibración de probabilidades; validación cruzada estratificada; búsqueda aleatoria de hiperparámetros; más imágenes por raza e inclusión de gatos mestizos.

## 8. Consideraciones éticas

- **Origen y uso de los datos:** el Oxford-IIIT Pet se usa con fines académicos y se cita su fuente. Algunas fotos pueden mostrar personas o lugares reconocibles al fondo, que no se usan ni se analizan.
- **Privacidad en un uso real:** en una clínica, las fotos y la ficha quedan asociadas a los dueños. Habría que pedir consentimiento y guardar las imágenes de forma segura.
- **Sesgo y representatividad:** el dataset solo tiene razas definidas, con fotos de internet, y no incluye gatos mestizos, que son la mayoría en la práctica. Los errores no se reparten parejo entre razas.
- **Riesgo de uso:** un error confirmado sin mirar podría llevar a registrar mal una ficha o a sugerir controles inadecuados. Por eso se plantea solo como apoyo, con confirmación del personal.
- **Transparencia:** se reportan también los resultados que no cumplen los KPIs, junto con sus limitaciones.

## 9. Cómo reproducir

1. Abrir `notebooks/EP1_TLY1101_MLP_Razas_Gatos_final_v2.ipynb` en [Google Colab](https://colab.research.google.com/) (se recomienda GPU).
2. Ir a `Entorno de ejecución > Ejecutar todo`. El notebook descarga el dataset oficial y ejecuta todo el pipeline, y genera las carpetas `images/`, `models/` y `resultados/`.
3. Las semillas están fijadas (`SEED_SPLIT` para el muestreo y la partición, y listas de semillas para la inicialización) y se activa el modo determinista de TensorFlow. Una celda de la sección 5 verifica que dos entrenamientos con la misma semilla dan resultados idénticos. Aun así, otra versión de TensorFlow o distinto hardware puede dar resultados levemente distintos, por eso los resultados se reportan como media ± desviación estándar de varias semillas.
4. Tiempo estimado: unos 87 entrenamientos en total; pueden ser varias decenas de minutos en GPU y bastante más en CPU.

## Referencia

Parkhi, O. M., Vedaldi, A., Zisserman, A. y Jawahar, C. V. (2012). *Cats and Dogs*. IEEE Conference on Computer Vision and Pattern Recognition.
