# Resultado del entrenamiento — 2026-11-16 18:33

Modelo evaluado: `/home/juandi/orienta-ft-17b/final` (estado: **completo**, ultimo paso `4075/4075`).

## Las cifras

| Metrica | Qwen3-1.7B entrenado | Qwen3-1.7B sin entrenar | Ronda 1 (0.6B) | Umbral |
|---|---|---|---|---|
| Precision factual | **17.5%** | 3.1% | 16.5% | >= 80% |
| Contaminacion cruzada | 7.0% | 5.0% | 8.5% | < 5% |
| Programas inventados | 2 | 0 | 3 | 0 |
| Precios dados | 0 | 0 | 0 | 0 |
| Formato no hablable | 1 | 19 | 1 | 0 |

Las tres cifras se miden en fp16 con las mismas 200 preguntas apartadas y el mismo
codigo, asi que son comparables entre si. La cifra de aceptacion real se mide sobre el
GGUF cuantizado, que todavia no existe.

## Como leerlo

- **Entrenado vs sin entrenar** dice si nuestros datos aportaron. Si son parecidos, el
  problema esta en el dataset, no en el tamano del modelo.
- **Entrenado vs ronda 1** dice si pasar de 0.6B a 1.7B con mas datos y mas epocas valio.

## Fallos concretos

Los ejemplos de respuestas falladas estan al final de `logs/quiz_ft17b.log`,
con la pregunta, la respuesta y los terminos que se esperaban. Son lo mas util para
decidir el siguiente paso.

## Entrenamiento

```
{'train_runtime': '7733', 'train_samples_per_second': '4.293', 'train_steps_per_second': '0.268', 'train_loss': '1.259', 'epoch': '5'}
{'loss': '1.193', 'grad_norm': '15.25', 'learning_rate': '6.063e-10', 'epoch': '4.982'}
```

Pausas termicas durante la corrida: 8.
Corte de luz a las 15:13 reanudado desde `checkpoint-2000` (9 min perdidos).

---

## Diagnostico posterior (18:50): la cifra no se puede leer como exactitud

Guarde las 200 respuestas (`logs/respuestas_ft17b.jsonl`) y re-corri el quiz: dio 17.5%
identico, asi que es determinista y las respuestas guardadas son las que produjeron la cifra.
Despues las examine. Tres hallazgos, dos de ellos contra lo que supuse primero:

**1. La metrica es recall de redaccion, no exactitud.** `precision` = fraccion de las palabras
de contenido de la respuesta de referencia que aparecen en la del modelo. 96 de 147 preguntas
puntuadas sacaron **cero exacto**, y entre ellas hay respuestas verificablemente correctas:

| Pregunta | Modelo | Referencia | Puntaje |
|---|---|---|---|
| ¿Hay un taller llamado Seminario Taller del Diseño Industrial II? | Sí, hay Seminario Taller del Diseño Industrial II | Sí, en el quinto semestre | **0** |
| ¿Hay temas como Neurociencias en las clases? | Sí, hay Neurociencias en el programa | Sí, te hablan en el tercer ciclo... | **0** |

**2. Pero el modelo tambien se equivoca de verdad.** No todo es la metrica. Hay confusion grave
entre programas, que el detector de contaminacion cruzada (7.0%) no atrapa:

| Pregunta | Programa | Respuesta del modelo |
|---|---|---|
| ¿En qué consiste ese plan de estudios? | Arquitectura | *Fundamentos de Ingeniería Electrónica y Telecomunicaciones, Circuitos Eléctricos...* |
| cómo se llama el título del programa | Psicología | *ingeniero en informática* |
| y qué otros sistemas industriales puedo diseñar | Ing. Mecánica | *automatización y control* (es la respuesta de Mecatrónica) |

En una muestra de 17 respuestas de cero, aproximadamente la mitad son errores reales y la otra
mitad son correctas penalizadas por redaccion. No es ni "la metrica esta mal" ni "el modelo no
sabe": son las dos cosas a la vez, y por eso la cifra no sirve para decidir.

**3. El quiz contiene preguntas imposibles.** 35% son seguimientos arrancados de su conversacion
("¿en qué consiste *ese* plan de estudios?", "que hay en el sexto semestre") y algunas son ruido
de STT convertido en pregunta ("Esa parte de 'que el tiene gusta por' se cortó un poquito,
podrías repetirla"). Sin su contexto, nadie puede responderlas.

Dos hipotesis mias que los datos **refutaron**, anotadas para no repetirlas:
- "La metrica castiga la brevedad": no. El modelo responde 12.7 palabras y la referencia 13.1.
- "Falla porque la pregunta no nombra el programa": no. Las que lo nombran sacan 14.3% y las que
  no, 18.7%.

## Recomendacion

**No entrenar otra ronda todavia.** Un modelo mas grande o mas epocas cuestan horas de GPU y
ahora mismo no podriamos distinguir una mejora de un empeoramiento: la medicion no separa
"correcto dicho con otras palabras" de "equivocado". Arreglar la evaluacion primero es mas
barato que otra corrida y decide las siguientes.

Orden sugerido:
1. **Juez LLM** sobre las 200 respuestas ya guardadas, con el `qwen2.5:14b` que esta aqui. No
   gasta GPU de generacion: las respuestas existen. Da la primera cifra de exactitud real.
2. **Limpiar el quiz** de los seguimientos sin ancla y del ruido de STT, o devolverles su
   contexto. Hoy miden lo imposible.
3. Con una cifra confiable, decidir si el problema es el dataset (lo mas probable: la confusion
   entre programas apunta a que los hechos no estan amarrados a su carrera) o el tamano.

---

## Juicio del maestro (19:30): exactitud real 26.5%, tasa de error 46.5%

`finetune/judge.py` juzgo las 200 respuestas guardadas con `qwen2.5:14b` contra el **texto
fuente**, sin mostrarle nunca la respuesta de referencia. Antes de usarlo verifique que
discrimina: aprueba el caso correcto que la metrica puntuaba con cero y rechaza los tres errores
graves, incluido un precio inventado que le inyecte a proposito.

| | |
|---|---|
| correctas | 41 |
| incompletas (ciertas, no responden lo preguntado) | 42 |
| **incorrectas (contradicen la fuente o inventan)** | **72** |
| imposibles | 2 |
| sin juicio | 0 |

Sobre 155 juzgables: **exactitud estricta 26.5%**, exactitud laxa 53.5%, **tasa de error 46.5%**.

### La metrica vieja no sirve para ordenar corridas

Cruzando las dos mediciones sobre las 147 respuestas que tienen ambas:

| Veredicto del juez | n | Puntaje promedio de la metrica vieja |
|---|---|---|
| correcta | 39 | 33.1% |
| **incorrecta** | 64 | **14.4%** |
| incompleta (pero cierta) | 42 | **8.5%** |

**Las respuestas incorrectas puntuaron mas alto que las ciertas-pero-incompletas.** La metrica
correlaciona debilmente con la verdad y esta invertida en el medio: 16 de 39 respuestas
correctas sacaron cero, y 12 de 64 incorrectas sacaron mas de 25%. No puede decidir si una
corrida es mejor que otra.

### El modo de falla dominante

De los 72 errores, el patron mas comun no es vaguedad sino **inventar especificos plausibles**:
materias atribuidas al programa equivocado, nombres de cursos casi correctos, detalles que no
estan en la fuente.

    [Diseño Industrial] las materias mencionadas no corresponden al programa en el sexto semestre
    [Derecho]           el nombre del curso en el texto es 'Teoría general del proceso y justicia digital'
    [Contaduría]        el horario no es flexible, sino presencial en dia y noche

Esto obliga a mirar con sospecha el cambio principal de la ronda 2: las pasadas `esp*` exigian
nombrar los sustantivos propios de la fuente **literalmente**. Es posible que le hayamos
ensenado el **habito de sonar especifico** sin ensenarle los especificos, que es peor que la
vaguedad de la ronda 1 — un robot vago confunde, uno que inventa nombres de materias con
seguridad desinforma a un aspirante.

**Prueba pendiente para confirmarlo:** correr el quiz con `--guardar` sobre el modelo de la
ronda 1 (0.6B, ~12 min) y juzgarlo igual. Si su tasa de error es menor que 46.5%, el empuje de
especificidad fue contraproducente y hay que rehacer esa parte del dataset.

### Correccion a un hallazgo anterior

Arriba dije que 35% de las preguntas son imposibles; el juez marco solo 2. **Las dos cifras no
son comparables**: al juez se le da el texto fuente, asi que para el la pregunta siempre es
respondible — mide si la respuesta concuerda con la fuente, no si el modelo tenia el contexto.
El 35% salio de un heuristico de regex sobre pronombres ("ese plan", "esa parte") y sigue en pie
como problema del conjunto de preguntas, pero no esta confirmado por el juez.

---

## Configuracion final (23:30): Q8 + BM25 k=5 + ponderacion del programa

| Configuracion | Estricta | Laxa | Tasa de error |
|---|---|---|---|
| Afinado solo, fp16 | 26.5% | 53.5% | 46.5% |
| Afinado + BM25, fp16 | 53.5% | 80.3% | 19.7% |
| Afinado + BM25, Q4 | 41.0% | 65.4% | 34.6% |
| Afinado + BM25, Q8, k=3 | 46.5% | 74.2% | 25.8% |
| Afinado + BM25, Q8, k=8 | 47.4% | 75.6% | 24.4% |
| **Afinado + BM25, Q8, k=5, peso 3** | **50.6%** | **80.1%** | **19.9%** |

Ponderar el nombre del programa tres veces en el indice recupero toda la perdida de la
cuantizacion y del motor: 1.8 GB iguala a 3.4 GB en tasa de error. Lo que decidio no fue el
tamano del modelo.

**Mas documentos no compra exactitud.** k=3, k=5 y k=8 quedan dentro de ~4 puntos (seis
respuestas en n=156) mientras el recall del recuperador sube de 61.9% a 84.9%. El cuello no es
falta de informacion sino que el modelo no la aprovecha toda. Eso descarta "entregarle mas
documentos" como camino de mejora, y apunta al prompt o al modelo.

Queda sin cumplirse un requisito duro: 3 programas inventados de 200 (umbral 0). Los precios
siguen en 0 de 200 en las cinco configuraciones medidas.
