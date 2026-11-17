"""Prueba el animatronico sin adivinar: busca el puerto, saluda y recorre los gestos.

    python3 probar_oso.py            # busca /dev/ttyUSB* y /dev/ttyACM*
    python3 probar_oso.py /dev/ttyUSB0
"""
import glob, sys, time
sys.path.insert(0, "/home/juandi/Orientador_vocacional")
import esp32

puerto = sys.argv[1] if len(sys.argv) > 1 else ""
if not puerto:
    encontrados = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if not encontrados:
        raise SystemExit("No hay ningun puerto serie. ¿Esta enchufado el ESP32?\n"
                          "Tras enchufarlo: ls /dev/ttyUSB* /dev/ttyACM*")
    puerto = encontrados[0]
    print(f"usando {puerto} (habia {len(encontrados)})")

oso = esp32.Animatronico(puerto)
if not oso.conectado:
    raise SystemExit("No se pudo abrir el puerto. Si dice 'permiso denegado': "
                      "sudo usermod -aG dialout $USER  y volver a entrar.")

def paso(descripcion, accion, espera=1.5):
    print(f"  {descripcion}")
    accion()
    time.sleep(espera)

print("\n-- ojos --")
paso("reposo: ojos abiertos, mirada al centro, boca cerrada", oso.reposo)
paso("parpadeo", oso.parpadear, 1.0)
paso("cerrar los ojos", lambda: oso.parpados(100))
paso("abrirlos", lambda: oso.parpados(0))
paso("mirar a la izquierda", lambda: oso.mirar(10, 50))
paso("mirar a la derecha", lambda: oso.mirar(90, 50))
paso("mirar arriba", lambda: oso.mirar(50, 15))
paso("volver al centro", lambda: oso.mirar(50, 50))

print("\n-- boca: recorrido completo --")
for nivel in (0, 25, 50, 75, 100, 50, 0):
    print(f"  boca {nivel}%"); oso.boca(nivel); time.sleep(0.5)

print("\n-- boca: como se movera al hablar (envolvente de una frase) --")
for nivel in (0, 22, 77, 100, 55, 13, 88, 66, 16, 0) * 2:
    oso.boca(nivel); time.sleep(0.046)
time.sleep(0.5)

print("\n-- gestos con nombre --")
for gesto in ("saludar", "asentir", "inclinar_cabeza", "pensar", "despedir"):
    paso(gesto, lambda g=gesto: oso.animacion(g), 2.0)

paso("reposo", oso.reposo)
oso.cerrar()
print("\nlisto")
