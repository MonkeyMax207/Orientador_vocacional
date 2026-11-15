# Plantilla de chat del entrenamiento

Contrato entre `finetune/train.py` y el `Modelfile` de Ollama. Si estas dos cadenas
difieren en un solo token, la calidad se cae sin ningun error visible.

Modelo: `Qwen/Qwen3-0.6B`  ·  `enable_thinking=False`

Mensajes de prueba: [{"role": "system", "content": "SISTEMA"}, {"role": "user", "content": "USUARIO"}, {"role": "assistant", "content": "ASISTENTE"}]

```
<|im_start|>system
SISTEMA<|im_end|>
<|im_start|>user
USUARIO<|im_end|>
<|im_start|>assistant
ASISTENTE<|im_end|>

```
