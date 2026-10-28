"""
generar_mensajes_cab.py
-----------------------
Genera data-v3.0.0: el conjunto de mensajes con el que se ajusta el router de
intención (mDeBERTa-v3) del chatbot de Componentes Automotrices del Bajío (CAB).

Por qué existe esta versión
  En la Actividad 5 el router se probó con 50 mensajes escritos a mano y su
  exactitud bajó a 0.525 en inglés y 0.20 en español. Dos causas: los datos de
  Bitext están en inglés y casi todos los mensajes de estatus traían el marcador
  {{Order Number}}, mientras que los de fecha nunca lo traían (atajo). La
  profesora recomendó agregar mensajes con terminología de CAB, sobre todo en
  español, antes de integrar el modelo en la API.

Qué hace
  1. Toma la muestra balanceada de Bitext de la A4 (2,000 mensajes en inglés,
     400 por intención) y reemplaza los marcadores por valores realistas. El
     número de orden se agrega también a mensajes de fecha, queja y otro para que
     "traer un número de OC" ya no delate la intención.
  2. Genera 2,000 mensajes simulados de CAB (80 % español, 20 % inglés) a partir
     de plantillas con vocabulario de la industria: OC, número de parte, planta,
     embarque, línea de producción, armadora.
     Los 135 mensajes de estatus de Bitext que preguntan por el "ETA" se
     reetiquetan como fecha_entrega, porque en CAB el ETA es la fecha estimada.
  3. Agrega ruido realista: minúsculas, sin acentos, sin signos de
     interrogación, errores de dedo y abreviaturas de WhatsApp.
  4. Elimina duplicados y divide en entrenamiento / validación / prueba
     (80/10/10), estratificado por intención, fuente e idioma, con semilla 42.
  5. Escribe las huellas SHA-256 en datos/mensajes_cab/versiones.json.

Importante: los mensajes de CAB son simulados (la empresa es ficticia). Por eso la
evaluación honesta del modelo se hace con la prueba externa escrita a mano
(prueba_externa_v3.csv), que no sale de estas plantillas.

Uso:
    python entrenamiento/generar_mensajes_cab.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEMILLA = 42
VERSION = "data-v3.0.0"
RAIZ = Path(__file__).resolve().parents[1]
DIR = RAIZ / "datos" / "mensajes_cab"
BITEXT = DIR / "bitext_muestra_a4.csv"
CLASES = ["estatus_orden", "fecha_entrega", "queja", "contactar_ejecutivo", "otro"]
CAB_POR_CLASE = {"es": 320, "en": 80}

rng = random.Random(SEMILLA)


# ---------------------------------------------------------------------------
# Valores para llenar las plantillas
# ---------------------------------------------------------------------------
def numero_oc() -> str:
    """Número de orden de compra con el formato de SAP de las armadoras (45 + 8 dígitos)."""
    return f"45{rng.randint(0, 99_999_999):08d}"


def oc_es() -> str:
    n = numero_oc()
    return rng.choice([f"OC {n}", f"OC-{n}", f"OC{n}", f"#{n}", n, f"orden {n}", f"PO {n}"])


def oc_en() -> str:
    n = numero_oc()
    return rng.choice([f"PO {n}", f"PO#{n}", f"PO-{n}", f"#{n}", n, f"order {n}", f"OC-{n}"])


PRODUCTOS_ES = ["arneses", "estampados", "maquinados", "plásticos inyectados", "ensambles",
                "soportes de motor", "bujes", "tornillería", "brackets", "molduras"]
PRODUCTOS_EN = ["wire harnesses", "stampings", "machined parts", "injection molded parts",
                "assemblies", "engine mounts", "bushings", "brackets", "fasteners"]
DIAS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "fin de mes", "la próxima semana",
           "el 15", "el día 30", "mañana"]
DIAS_EN = ["monday", "tuesday", "wednesday", "thursday", "friday", "end of month", "next week",
           "the 15th", "tomorrow"]
PLANTAS_CLIENTE = ["nuestra planta de Silao", "la planta de Aguascalientes", "Saltillo", "Ramos Arizpe",
                   "Salamanca", "Toluca", "Puebla", "la planta de Laredo", "nuestro almacén de Celaya"]
NUM_PARTE = lambda: f"NP {rng.randint(1000, 9999)}-{rng.choice('ABCDEFGH')}{rng.randint(10, 99)}"  # noqa: E731

SALUDOS_ES = ["", "", "", "Hola, ", "Buen día, ", "Buenas tardes, ", "Buenos días. ", "Qué tal, ",
              "Hola equipo CAB, ", "Hola buenas, ", "Saludos. "]
CIERRES_ES = ["", "", "", " gracias", " saludos", " por favor", " es urgente", " quedo atento",
              " gracias!", " de antemano gracias", " porfa"]
SALUDOS_EN = ["", "", "", "Hi, ", "Hello, ", "Good morning, ", "Hi team, ", "Hey, "]
CIERRES_EN = ["", "", "", " thanks", " please", " thank you", " asap", " regards"]

PROBLEMAS_ES = ["llegaron piezas dañadas", "el embarque llegó tarde otra vez", "nos mandaron un número de parte equivocado",
                "faltaron piezas en la entrega", "el chofer no respetó el horario de recibo",
                "nadie contesta mis correos", "las piezas venían sin certificado de calidad",
                "el material llegó oxidado", "el empaque venía roto", "nos pararon la línea por su retraso",
                "llegó menos cantidad de la que pedimos", "el transportista dejó el material en otra planta"]
PROBLEMAS_EN = ["the parts arrived damaged", "the shipment was late again", "you sent the wrong part number",
                "parts were missing from the delivery", "nobody answers my emails",
                "the packaging was broken", "our line stopped because of your delay",
                "the material arrived rusty"]

# ---------------------------------------------------------------------------
# Plantillas por intención. {oc} = número de OC, {prod} = producto, {dia} = día.
# ---------------------------------------------------------------------------
PLANTILLAS_ES = {
    "estatus_orden": [
        "¿cómo va la {oc}?", "¿cuál es el estatus de la {oc}?", "¿ya se embarcó la {oc}?",
        "¿en qué etapa está mi pedido {oc}?", "necesito saber si la {oc} ya salió de planta",
        "¿ya surtieron la {oc}?", "¿me confirman si la {oc} ya está en tránsito?",
        "¿dónde viene el embarque de la {oc}?", "quiero rastrear mi pedido {oc}", "estatus {oc}",
        "¿qué pasó con el pedido de {prod} {oc}?", "¿ya está lista la producción de la {oc}?",
        "revisar estatus de la {oc}", "¿la {oc} sigue en producción?", "¿me dan seguimiento a la {oc}?",
        "¿cómo vamos con la {oc}?", "¿hay avance en la {oc}?", "¿el transportista ya recogió la {oc}?",
        "¿mi pedido de {prod} ya va en camino?", "¿en dónde está mi orden de {prod}?",
        "¿ya salió el camión con la {oc}?", "¿cómo va mi pedido?", "¿mi orden ya se procesó?",
        "¿en qué va la orden de compra {oc}?", "¿ya tienen guía de la {oc}?",
        "consulta de estatus de la {oc}", "¿me pueden decir cómo va el pedido de {prod}?",
    ],
    "fecha_entrega": [
        "¿cuándo llega la {oc}?", "¿qué día me entregan la {oc}?", "¿cuál es la fecha de entrega de la {oc}?",
        "¿cuál es el ETA de la {oc}?", "¿para cuándo tienen la {oc}?", "¿la {oc} llega antes del {dia}?",
        "¿en cuántos días recibimos la {oc}?", "necesito la fecha estimada de llegada de la {oc} a {planta}",
        "¿qué día llegan las piezas de la {oc}?", "¿se va a retrasar la entrega de la {oc}?",
        "¿mantienen la fecha compromiso de la {oc}?", "¿me confirman la fecha de arribo de la {oc}?",
        "¿llegará a tiempo la {oc} para mi línea del {dia}?", "¿cuánto tarda en llegar la {oc}?",
        "¿para qué fecha está programada la entrega de {prod}?", "¿cuándo nos llegan los {prod} de la {oc}?",
        "¿cuándo llega mi pedido?", "fecha de entrega de la {oc}", "ETA {oc}",
        "¿la entrega de {prod} sigue para el {dia}?", "¿a qué fecha se movió la entrega de la {oc}?",
        "¿cuándo recibiremos los {prod}?", "¿en qué fecha llega el material a {planta}?",
        "¿me pueden dar la fecha estimada de la {oc}?", "¿para cuándo queda la {oc}?",
    ],
    "queja": [
        "quiero poner una queja, {problema}", "{problema}, esto es inaceptable", "estoy muy molesto, {problema}",
        "es la tercera vez que {problema}", "pésimo servicio, {problema}",
        "quiero levantar una queja formal porque {problema}", "no es posible que {problema}",
        "exijo una explicación, {problema}", "{problema} en la {oc}, quiero reportarlo",
        "muy mal servicio con la {oc}, {problema}", "estoy inconforme, {problema}",
        "quiero reportar un problema: {problema}", "otra vez lo mismo, {problema}",
        "necesito levantar una reclamación, {problema}",
    ],
    "contactar_ejecutivo": [
        "quiero hablar con mi ejecutivo de cuenta", "comuníquenme con una persona", "¿me pueden pasar con un asesor?",
        "prefiero hablar con alguien de servicio al cliente", "¿hay un teléfono para hablar con un humano?",
        "pásenme con alguien de logística", "no quiero hablar con un bot", "necesito que mi ejecutivo me llame",
        "¿quién es mi ejecutivo asignado? necesito contactarlo", "que me marquen por favor",
        "necesito atención de una persona", "¿me comunican con el área comercial?",
        "quiero hablar con alguien sobre la {oc}", "¿me puede atender un ejecutivo?",
        "¿cómo contacto a servicio al cliente?", "necesito hablar con el gerente de cuenta",
        "transfiéranme con un agente", "mejor pásame con un humano",
    ],
    "otro": [
        "necesito la factura de la {oc}", "¿cómo cancelo la {oc}?", "quiero cambiar la cantidad de piezas de la {oc}",
        "¿cuál es su cuenta para transferencia?", "necesito una nota de crédito de la {oc}",
        "¿me pueden cotizar {prod}?", "quiero dar de alta un usuario en el portal",
        "¿cómo cambio la dirección de entrega?", "¿tienen precio especial por volumen?",
        "olvidé mi contraseña del portal", "¿cuáles son sus términos de pago?", "quiero agregar piezas a la {oc}",
        "¿me envían el complemento de pago de la {oc}?", "necesito la ficha técnica de los {prod}",
        "¿pueden enviarme el certificado de calidad del {np}?", "quiero un reembolso de la {oc}",
        "¿aceptan pago con tarjeta?", "necesito actualizar mis datos fiscales",
        "¿cómo me registro como nuevo cliente?", "quiero cancelar el pedido de {prod}",
        "¿me reenvían el XML de la factura?", "¿hacen descuentos por pronto pago?",
    ],
}

PLANTILLAS_EN = {
    "estatus_orden": [
        "what's the status of {oc}?", "has {oc} shipped yet?", "where is my order {oc}?", "track {oc} please",
        "is {oc} still in production?", "can you check the status of our {prod} order?",
        "did the carrier pick up {oc}?", "any update on {oc}?", "status of {oc}", "is my order on its way?",
    ],
    "fecha_entrega": [
        "when will {oc} arrive?", "what is the delivery date for {oc}?", "what's the ETA for {oc}?",
        "will {oc} arrive before {dia}?", "how many days until {oc} gets here?", "when do we get the {prod}?",
        "is the delivery of {oc} still on schedule?", "expected arrival date for {oc}?",
        "will {oc} be late?", "when is {oc} due at our plant?",
    ],
    "queja": [
        "i want to file a complaint, {problema}", "{problema}, this is unacceptable", "terrible service, {problema}",
        "i'm very upset, {problema}", "this is the third time {problema}", "{problema} on {oc}, i want to report it",
    ],
    "contactar_ejecutivo": [
        "i want to talk to my account manager", "connect me with a person please", "can i speak to a human?",
        "please have my sales rep call me", "transfer me to customer service", "i don't want to talk to a bot",
        "who is my account executive? i need to reach them", "i need to talk to someone about {oc}",
    ],
    "otro": [
        "i need the invoice for {oc}", "how do i cancel {oc}?", "can you quote {prod}?",
        "i forgot my portal password", "what are your payment terms?", "i want to add parts to {oc}",
        "can i get a credit note for {oc}?", "do you offer volume discounts?", "how do i update my tax info?",
        "please send the quality certificate for {np}",
    ],
}


# ---------------------------------------------------------------------------
# Ruido realista
# ---------------------------------------------------------------------------
def quitar_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def error_de_dedo(texto: str) -> str:
    palabras = texto.split()
    candidatas = [i for i, p in enumerate(palabras) if len(p) > 4 and p.isalpha()]
    if not candidatas:
        return texto
    i = rng.choice(candidatas)
    p = list(palabras[i])
    j = rng.randrange(len(p) - 1)
    p[j], p[j + 1] = p[j + 1], p[j]
    palabras[i] = "".join(p)
    return " ".join(palabras)


ABREVIATURAS = [(r"\bpor favor\b", "xfa"), (r"\bque\b", "q"), (r"\bporque\b", "pq"), (r"\bpara\b", "pa"),
                (r"\bgracias\b", "grax"), (r"\bplease\b", "pls"), (r"\byou\b", "u"), (r"\bthanks\b", "thx")]


def agregar_ruido(texto: str) -> str:
    if rng.random() < 0.45:
        texto = texto.lower()
    if rng.random() < 0.30:
        texto = quitar_acentos(texto)
    if rng.random() < 0.30:
        texto = texto.replace("¿", "").replace("?", "")
    if rng.random() < 0.15:
        texto = error_de_dedo(texto)
    if rng.random() < 0.12:
        for patron, reemplazo in ABREVIATURAS:
            texto = re.sub(patron, reemplazo, texto)
    return re.sub(r"\s+", " ", texto).strip()


def llenar(plantilla: str, idioma: str) -> str:
    es = idioma == "es"
    valores = {
        "oc": oc_es() if es else oc_en(),
        "prod": rng.choice(PRODUCTOS_ES if es else PRODUCTOS_EN),
        "dia": rng.choice(DIAS_ES if es else DIAS_EN),
        "planta": rng.choice(PLANTAS_CLIENTE),
        "problema": rng.choice(PROBLEMAS_ES if es else PROBLEMAS_EN),
        "np": NUM_PARTE(),
    }
    texto = plantilla.format(**valores)
    saludo = rng.choice(SALUDOS_ES if es else SALUDOS_EN)
    cierre = rng.choice(CIERRES_ES if es else CIERRES_EN)
    if saludo:
        texto = saludo + texto[0].lower() + texto[1:]
    return agregar_ruido(texto + cierre)


def generar_cab() -> pd.DataFrame:
    filas = []
    for idioma, plantillas in (("es", PLANTILLAS_ES), ("en", PLANTILLAS_EN)):
        for clase in CLASES:
            vistos, intentos = set(), 0
            while len(vistos) < CAB_POR_CLASE[idioma] and intentos < 50_000:
                intentos += 1
                plantilla = rng.choice(plantillas[clase])
                mensaje = llenar(plantilla, idioma)
                if mensaje.lower() not in vistos:
                    vistos.add(mensaje.lower())
                    filas.append({"mensaje": mensaje, "intencion": clase, "idioma": idioma,
                                  "fuente": "cab_simulado", "plantilla": plantilla})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Bitext (A4) con marcadores reemplazados
# ---------------------------------------------------------------------------
RELLENOS = {
    "{{Person Name}}": lambda: rng.choice(["Laura", "Carlos Méndez", "Ana", "Jorge"]),
    "{{Account Type}}": lambda: rng.choice(["premium", "business", "standard"]),
    "{{Account Category}}": lambda: rng.choice(["supplier", "business", "corporate"]),
    "{{Refund Amount}}": lambda: rng.choice(["1,250", "830", "4,900"]),
    "{{Currency Symbol}}": lambda: "$",
    "{{Delivery City}}": lambda: rng.choice(["Querétaro", "Saltillo", "Celaya"]),
}


def preparar_bitext() -> pd.DataFrame:
    df = pd.read_csv(BITEXT)
    salida = []
    for _, fila in df.iterrows():
        texto = fila["mensaje"]
        intencion = fila["intencion"]
        # En Bitext "ver el ETA de mi orden" es track_order (estatus). En CAB el ETA es la
        # fecha estimada de llegada, así que esos 135 mensajes se reetiquetan como fecha_entrega.
        reetiquetado = intencion == "estatus_orden" and re.search(r"\beta\b", texto, re.IGNORECASE) is not None
        if reetiquetado:
            intencion = "fecha_entrega"
        while "{{Order Number}}" in texto:
            texto = texto.replace("{{Order Number}}", oc_en(), 1)
        for marcador, funcion in RELLENOS.items():
            while marcador in texto:
                texto = texto.replace(marcador, funcion(), 1)
        # El número de orden ya no debe delatar la intención: se agrega a otras clases.
        if intencion != "estatus_orden" and not re.search(r"\d{6,}", texto) and rng.random() < 0.35:
            texto = texto.rstrip(" ?.!") + rng.choice([" for ", " on ", " regarding "]) + oc_en()
        salida.append({"mensaje": re.sub(r"\s+", " ", texto).strip(), "intencion": intencion,
                       "idioma": "en", "fuente": "bitext_a4",
                       "plantilla": "reetiquetado: ETA = fecha en CAB" if reetiquetado else ""})
    return pd.DataFrame(salida)


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 16), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> None:
    bitext = preparar_bitext()
    cab = generar_cab()
    todo = pd.concat([bitext, cab], ignore_index=True)
    antes = len(todo)
    todo["_clave"] = todo["mensaje"].str.lower().str.strip()
    todo = todo.drop_duplicates("_clave").drop(columns="_clave").reset_index(drop=True)
    print(f"Duplicados eliminados: {antes - len(todo)}")

    estrato = todo["intencion"] + "|" + todo["fuente"] + "|" + todo["idioma"]
    entrena, temp = train_test_split(todo, test_size=0.20, random_state=SEMILLA, stratify=estrato)
    valida, prueba = train_test_split(temp, test_size=0.50, random_state=SEMILLA,
                                      stratify=temp["intencion"] + "|" + temp["fuente"] + "|" + temp["idioma"])

    archivos = {}
    for nombre, parte in (("entrenamiento", entrena), ("validacion", valida), ("prueba", prueba)):
        parte = parte.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
        parte.insert(0, "id", range(1, len(parte) + 1))
        ruta = DIR / f"{nombre}.csv"
        parte.to_csv(ruta, index=False)
        archivos[ruta.name] = {
            "filas": len(parte),
            "por_intencion": parte["intencion"].value_counts().reindex(CLASES).astype(int).to_dict(),
            "por_idioma": parte["idioma"].value_counts().astype(int).to_dict(),
            "sha256": sha256(ruta),
        }
        print(f"{nombre:14s} {len(parte):5d} mensajes  {archivos[ruta.name]['por_idioma']}")

    externa = DIR / "prueba_externa_v3.csv"
    if externa.exists():
        ext = pd.read_csv(externa)
        archivos[externa.name] = {"filas": len(ext), "por_idioma": ext["idioma"].value_counts().to_dict(),
                                  "sha256": sha256(externa),
                                  "descripcion": "Escrita a mano, no sale de las plantillas. Solo para la evaluación final."}

    manifiesto = {
        "version": VERSION,
        "fecha_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "semilla": SEMILLA,
        "generador": "entrenamiento/generar_mensajes_cab.py",
        "origen": {
            "bitext_muestra_a4.csv": {"sha256": sha256(BITEXT), "filas": 2000,
                                      "descripcion": "data-v1.0.0 (A4): muestra balanceada de Bitext, CDLA-Sharing-1.0"},
        },
        "clases": CLASES,
        "archivos": archivos,
    }
    (DIR / "versiones.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Manifiesto escrito en datos/mensajes_cab/versiones.json")


if __name__ == "__main__":
    main()
