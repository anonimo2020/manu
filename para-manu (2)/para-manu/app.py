import json, os, random, re, time, unicodedata
from collections import defaultdict, deque

import requests
from flask import Flask, jsonify, render_template, request

from prompt import SYSTEM_PROMPT

app = Flask(__name__)
BASE = os.path.dirname(__file__)
MENSAJES = json.load(open(os.path.join(BASE, "frases_pequena_manu_2000.json"), encoding="utf-8"))["messages"]
CRISIS_RESP = json.load(open(os.path.join(BASE, "crisis.json"), encoding="utf-8"))["crisis"]
POR_EMOCION = defaultdict(list)
for _m in MENSAJES:
    POR_EMOCION[_m["emotion"]].append(_m)

# Qué emociones del JSON se usan para cada momento del día (frase al abrir la página).
# Mañana y noche tienen saludos propios; la tarde usa mensajes de cariño y ánimo.
MOMENTOS = {
    "manana": ["mañana"],
    "tarde": ["carino", "sorpresa", "alegria", "motivacion", "autoestima"],
    "noche": ["noche"],
}
# Respaldo (cuando no responde ninguna IA): emoción detectada -> emociones del JSON.
RESPALDO = {
    "tristeza": ["tristeza"],
    "cansancio": ["carino", "apoyo"],
    "estres": ["ansiedad"],
    "extrano": ["amor", "carino"],
    "feliz": ["alegria"],
    "general": ["carino", "amor"],
}

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Los patrones se aplican sobre texto sin tildes ni mayúsculas (ver normalizar).
# Todos llevan \b al inicio y al final para no coincidir dentro de otras palabras
# ("normal" no es "mal", "solamente" no es "sola").
NEGADO = (
    r"\bno\b(?:\s+\w+){0,3}?\s+(?:bien|feliz|contenta|contento|alegre|genial|animada|animado)\b"
    r"|\b(?:nada|poco)\s+(?:bien|feliz|animada|animado)\b"
)
EMOCIONES = {
    "tristeza": r"\b(?:triste\w*|llor\w*|deprim\w*|vacia|desanim\w*|mal|fatal|horrible|pesimo|sola|soledad)\b",
    "cansancio": r"\b(?:cansad\w*|cansanci\w*|agotad\w*|agotamiento|exhaust\w*|no puedo mas|(?:tengo|con|mucho|me da|me dio) sueno)\b",
    "estres": r"\b(?:estres\w*|agobi\w*|ansied\w*|ansios\w*|nervios\w*|preocup\w*|presion)\b",
    "extrano": r"\b(?:te extrano|te echo de menos|echo de menos|me haces falta|me hace falta)\b",
    "feliz": r"\b(?:feliz|felices|content[ao]s?|alegre|alegria|genial|emocionad[ao]|increible|bien)\b",
}
# Frases de riesgo: se revisan ANTES de la IA y del límite de uso. Es mejor
# pasarse de cuidadosa que quedarse corta.
CRISIS = (
    r"\b(?:suicid\w*|quitarme la vida|quiero morir(?:me)?|quisiera morir(?:me)?|me quiero matar|matarme"
    r"|no quiero (?:seguir )?viviendo|ya no quiero vivir|no quiero seguir (?:aqui|asi|con (?:esto|todo))"
    r"|acabar con (?:todo|mi vida)|hacerme dano|lastimarme|cortarme|quiero desaparecer"
    r"|no vale la pena vivir|mejor sin mi)\b"
)
hits = defaultdict(deque)


def limitado(ip, maximo=30, ventana=3600):
    q, ahora = hits[ip], time.time()
    while q and ahora - q[0] > ventana:
        q.popleft()
    if len(q) >= maximo:
        return True
    q.append(ahora)
    return False


def normalizar(texto):
    t = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def en_crisis(texto):
    return bool(re.search(CRISIS, normalizar(texto)))


def elegir(emociones, vistos=(), previas=()):
    """Elige un mensaje al azar sin repetir los ya vistos. Devuelve (mensaje, reinicio)."""
    pool = [m for e in emociones for m in POR_EMOCION[e]]
    libres = [
        m for m in pool
        if m["id"] not in vistos and not any(m["message"] in t for t in previas)
    ]
    return random.choice(libres or pool), not libres


def frase_de_respaldo(texto, previas=()):
    t = normalizar(texto)
    if re.search(NEGADO, t):
        emo = "tristeza"
    else:
        emo = next((e for e, patron in EMOCIONES.items() if re.search(patron, t)), "general")
    m, _ = elegir(RESPALDO[emo], previas=previas)
    return m["message"] + (f"\n\n{m['follow_up']}" if m["follow_up"] else "")


def gemini(msgs):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": GEMINI_KEY},
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["text"]}]}
                for m in msgs
            ],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 300,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        },
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def groq(msgs):
    chat = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": "user" if m["role"] == "user" else "assistant", "content": m["text"]} for m in msgs
    ]
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        json={"model": GROQ_MODEL, "messages": chat, "temperature": 0.9, "max_tokens": 300},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


@app.get("/")
def inicio():
    return render_template("index.html")


@app.post("/api/frase")
def frase():
    d = request.get_json(silent=True) or {}
    momento = d.get("momento") if d.get("momento") in MOMENTOS else "manana"
    m, reinicio = elegir(MOMENTOS[momento], vistos=set(d.get("vistas", [])))
    return jsonify(id=m["id"], frase=m["message"], seguir=m["follow_up"], reinicio=reinicio)


@app.post("/api/chat")
def chat():
    d = request.get_json(silent=True) or {}
    msgs = [
        {"role": m.get("role"), "text": str(m.get("text", ""))[:500]}
        for m in d.get("mensajes", [])[-8:]
        if m.get("role") in ("user", "assistant") and m.get("text")
    ]
    if not msgs or msgs[-1]["role"] != "user":
        return jsonify(error="mensaje vacío"), 400
    previas = [m["text"] for m in msgs if m["role"] == "assistant"]
    if en_crisis(msgs[-1]["text"]):
        return jsonify(respuesta=random.choice(CRISIS_RESP), origen="crisis")
    if limitado(request.headers.get("X-Forwarded-For", request.remote_addr)):
        return jsonify(respuesta="Descansa un ratito y me cuentas más luego, ¿sí?", origen="limite")
    for nombre, clave, fn in (("gemini", GEMINI_KEY, gemini), ("groq", GROQ_KEY, groq)):
        if clave:
            try:
                return jsonify(respuesta=fn(msgs), origen=nombre)
            except Exception as e:
                app.logger.warning("%s falló: %s", nombre, e)
    return jsonify(respuesta=frase_de_respaldo(msgs[-1]["text"], previas), origen="frases")


if __name__ == "__main__":
    app.run(debug=True)
