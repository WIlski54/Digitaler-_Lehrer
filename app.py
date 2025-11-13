# app.py (ehemals server.py)
# -*- coding: utf-8 -*-
import os
import json
import base64
import logging
import datetime
import re
from io import BytesIO
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

# PATCH 1: 'render_template' hinzugefügt
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# ---------- KI: Text (Gemini) ----------
import google.generativeai as genai
# PATCH 4: Import für Safety Settings
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ---------- KI: Vertex Imagen (HTTP über google-auth) ----------
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

# ---------- Bild-Nachbearbeitung ----------
from PIL import Image


# ======================================================================
# Grund-Setup
# ======================================================================
load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ki-lehrer")

# ----------------------------------------------------------------------
# Verzeichnisse
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
GEN_DIR = STATIC_DIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Text-KI (Gemini) – robuste Initialisierung mit Auto-Select
# (unverändert)
# ======================================================================
VERSION_RE = re.compile(r"(\d+)\.(\d+)")

def _get_api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Kein API-Key gesetzt (GEMINI_API_KEY oder GOOGLE_API_KEY).")
    return key

def _version_tuple(name: str) -> tuple[int, int]:
    m = VERSION_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

def _list_text_models() -> list[str]:
    names: list[str] = []
    for m in genai.list_models():
        try:
            if "generateContent" in m.supported_generation_methods:
                names.append(m.name)  # z. B. "gemini-2.5-flash"
        except Exception:
            pass
    return names

def pick_gemini_model(prefer: str = "fast") -> str:
    """
    prefer: "fast" (flash bevorzugen) oder "quality" (pro bevorzugen).
    Nimmt sonst das beste verfügbare Textmodell.
    """
    forced = (os.getenv("GEMINI_MODEL") or "").strip()
    if forced:
        return forced

    available = _list_text_models()
    if not available:
        raise RuntimeError("Keine Text-Modelle via list_models() gefunden. "
                           "Bitte 'google-generativeai' aktualisieren.")

    if prefer == "quality":
        cand = [n for n in available if "pro" in n]
        cand = cand or available
    else:
        cand = [n for n in available if "flash" in n]
        cand = cand or available

    cand.sort(key=_version_tuple, reverse=True)
    return cand[0]

GEMINI_MODEL = None
GEMINI_MODEL_NAME = None

GENERATION_CONFIG = {
    "max_output_tokens": 512,
    "temperature": 0.6,
}

def init_gemini(prefer: str = "fast") -> None:
    global GEMINI_MODEL, GEMINI_MODEL_NAME
    try:
        api_key = _get_api_key()
        genai.configure(api_key=api_key)
        name = pick_gemini_model(prefer=prefer)
        GEMINI_MODEL = genai.GenerativeModel(name)
        GEMINI_MODEL_NAME = name
        logger.info("Gemini initialisiert: %s", name)
    except Exception as e:
        logger.exception("Gemini Init-Fehler: %s", e)
        GEMINI_MODEL = None
        GEMINI_MODEL_NAME = None

def reselect_gemini(prefer: str = "fast") -> None:
    logger.info("Gemini Re-Select angestoßen …")
    init_gemini(prefer=prefer)

# Initial einmal starten
init_gemini(prefer="fast")

TUTOR_PROMPT = """
[Rolle]
Du bist ein geduldiger, fachlich korrekter Tutor für Sekundarstufe I/II in NRW.
Du erklärst klar und knapp – mit korrekten Fachbegriffen – und arbeitest sehr schülerorientiert.

[Dialog-Regeln]
- Antworte natürlich, knüpfe an die letzte Schüleräußerung an.
- Stelle höchstens **eine** gezielte Rückfrage pro Antwort.
- Max. **8 Sätze** oder nummerierte, kurze Schritte.
- Gib am Ende (wenn passend) **eine** Mini-Übungsfrage oder handlungsorientierten nächsten Schritt.

[Didaktik]
- Fachbegriffe korrekt, kurz erläutert.
- Denk-/Rechenschritte knapp nachvollziehbar.
- Altersangemessene, motivierende Sprache.

[Format]
- Nutze bei Bedarf Markdown (fett, Listen, Formeln).
- Rechen-/Begriffsarbeit strukturiert.
"""

# ======================================================================
# Vertex AI / Imagen (Service-Account & Endpunkt)
# ======================================================================
# >>>> Diese Werte an deine Umgebung anpassen <<<<
GCP_PROJECT_ID       = os.getenv("GCP_PROJECT_ID", "ki-lehrer-468715")
GCP_LOCATION         = os.getenv("GCP_LOCATION", "us-central1")
IMAGEN_MODEL         = os.getenv("IMAGEN_MODEL", "imagen-3.0-generate-002")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "/home/DrWK2/service_account_key.json") # Fallback für lokales Testen

# PATCH 2: NEUE UMGEBUNGSVARIABLE FÜR RENDER
GCP_SERVICE_ACCOUNT_JSON = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

IMAGEN_ENDPOINT = (
    f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/"
    f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/publishers/google/models/{IMAGEN_MODEL}:predict"
)

VERTEX_CREDENTIALS = None
VERTEX_SESSION = None

# --- PATCH 2: ANGEPASSTE CREDENTIAL-LOGIK FÜR RENDER ---
try:
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    if GCP_SERVICE_ACCOUNT_JSON:
        # Methode 1: Aus Umgebungsvariable (Für Render.com)
        logger.info("Versuche Vertex-Login via GCP_SERVICE_ACCOUNT_JSON (Render)...")
        service_account_info = json.loads(GCP_SERVICE_ACCOUNT_JSON)
        VERTEX_CREDENTIALS = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=scopes
        )
        logger.info("Vertex-Login via Environment-JSON erfolgreich.")
    elif os.path.exists(SERVICE_ACCOUNT_FILE):
        # Methode 2: Aus Datei (Für lokales Testen / PythonAnywhere)
        logger.info("Versuche Vertex-Login via SERVICE_ACCOUNT_FILE (Lokal)...")
        VERTEX_CREDENTIALS = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=scopes
        )
        logger.info("Vertex-Login via Datei erfolgreich.")
    else:
        logger.warning("Keine Vertex-Credentials gefunden (weder ENV noch Datei). /draw funktioniert nicht.")

    if VERTEX_CREDENTIALS:
        VERTEX_SESSION = AuthorizedSession(VERTEX_CREDENTIALS)
        logger.info("Vertex/Imagen AuthorizedSession initialisiert.")

except Exception as e:
    logger.exception("Vertex Init-Fehler: %s", e)
# --- ENDE PATCH ---


# ======================================================================
# Visuelle Hilfen - Konfiguration
# (unverändert)
# ======================================================================
VISUAL_TRIGGERS = {
    'Biologie': {
        'aufbau': { 'pflanzenzelle': { 'prompt': 'plant cell cross section labeled educational diagram style', 'labels': ['Zellwand', 'Zellmembran', 'Zellkern', 'Chloroplast', 'Vakuole', 'Mitochondrium', 'Endoplasmatisches Retikulum'], 'auto_show': True }, 'tierzelle': { 'prompt': 'animal cell cross section labeled educational diagram', 'labels': ['Zellmembran', 'Zellkern', 'Mitochondrium', 'Golgi-Apparat', 'Endoplasmatisches Retikulum', 'Lysosom'], 'auto_show': True }, 'blatt': { 'prompt': 'leaf cross section microscopic view labeled diagram', 'labels': ['Kutikula', 'Epidermis', 'Palisadengewebe', 'Schwammgewebe', 'Leitbündel', 'Spaltöffnung'], 'auto_show': True }, 'herz': { 'prompt': 'human heart anatomy cross section educational diagram', 'labels': ['Rechter Vorhof', 'Linker Vorhof', 'Rechte Kammer', 'Linke Kammer', 'Aorta', 'Lungenvene'], 'auto_show': True } },
        'prozess': { 'photosynthese': { 'prompt': 'photosynthesis process diagram chloroplast light reactions', 'auto_show': True }, 'zellteilung': { 'prompt': 'cell division mitosis stages diagram educational', 'auto_show': True }, 'verdauung': { 'prompt': 'human digestive system diagram labeled educational', 'labels': ['Mund', 'Speiseröhre', 'Magen', 'Dünndarm', 'Dickdarm', 'Leber', 'Bauchspeicheldrüse'], 'auto_show': True } }
    },
    'Chemie': { 'aufbau': { 'atom': { 'prompt': 'atom structure diagram protons neutrons electrons orbital model', 'labels': ['Atomkern', 'Proton', 'Neutron', 'Elektron', 'Elektronenschale'], 'auto_show': True }, 'periodensystem': { 'prompt': 'periodic table of elements educational poster style', 'auto_show': False }, 'molekül': { 'prompt': 'water molecule H2O 3D structure diagram', 'auto_show': True } }, 'reaktion': { 'säure-base': { 'prompt': 'acid base reaction diagram pH scale educational', 'auto_show': True }, 'redox': { 'prompt': 'redox reaction electron transfer diagram', 'auto_show': True } } },
    'Physik': { 'schema': { 'stromkreis': { 'prompt': 'simple electric circuit diagram battery bulb switch labeled', 'labels': ['Batterie', 'Schalter', 'Glühbirne', 'Leitung', 'Widerstand'], 'auto_show': True }, 'hebel': { 'prompt': 'lever physics diagram fulcrum force educational', 'labels': ['Drehpunkt', 'Kraftarm', 'Lastarm', 'Kraft', 'Last'], 'auto_show': True }, 'optik': { 'prompt': 'light refraction lens diagram ray tracing educational', 'labels': ['Linse', 'Brennpunkt', 'Hauptachse', 'Gegenstand', 'Bild'], 'auto_show': True } } },
    'Mathematik': { 'geometrie': { 'dreieck': { 'prompt': 'triangle geometry diagram angles sides labeled mathematical', 'labels': ['Seite a', 'Seite b', 'Seite c', 'Winkel α', 'Winkel β', 'Winkel γ'], 'auto_show': True }, 'kreis': { 'prompt': 'circle geometry diagram radius diameter circumference', 'labels': ['Mittelpunkt', 'Radius', 'Durchmesser', 'Umfang', 'Sehne'], 'auto_show': True } } }
}

# (Spezialfälle aus check_for_visual_aid, damit /improve_prompt sie finden kann)
SPECIAL_CASES = {
    'dna': { 'prompt': 'DNA double helix structure diagram educational', 'topic': 'DNA-Struktur', 'auto_show': True, 'supports_labeling': True, 'labels': ['Adenin', 'Thymin', 'Guanin', 'Cytosin', 'Zucker-Phosphat-Rückgrat'] },
    'mikroskop': { 'prompt': 'microscope parts labeled educational diagram', 'topic': 'Mikroskop', 'auto_show': True, 'supports_labeling': True, 'labels': ['Okular', 'Tubus', 'Objektiv', 'Objekttisch', 'Kondensor', 'Lichtquelle'] },
    'auge': { 'prompt': 'human eye anatomy cross section educational diagram', 'topic': 'Auge', 'auto_show': True, 'supports_labeling': True, 'labels': ['Hornhaut', 'Iris', 'Pupille', 'Linse', 'Netzhaut', 'Sehnerv'] }
}
# Synonyme aus build_image_prompt
SYNONYMS = { "becherglas": "laboratory beaker", "messzylinder": "graduated cylinder", "erlenmeyerkolben": "erlenmeyer flask", "pipette": "pipette", "chloroplast": "chloroplast", "blattquerschnitt": "cross section of a leaf", }


# ======================================================================
# Hilfsfunktionen – Größen, Speichern, Prompt-Aufbereitung
# (unverändert)
# ======================================================================
def _safe_size(size_str: str) -> Tuple[int, int]:
    s = (size_str or "1024x1024").strip().split()[0]
    try:
        w, h = s.lower().split("x")
        w, h = int(w), int(h)
        w = max(256, min(2048, w))
        h = max(256, min(2048, h))
        return w, h
    except Exception:
        return 1024, 1024

def _extract_b64(pred_json: Dict[str, Any]) -> str | None:
    if not isinstance(pred_json, dict):
        return None
    p = None
    if "predictions" in pred_json and isinstance(pred_json["predictions"], list) and pred_json["predictions"]:
        p = pred_json["predictions"][0]
    b64 = (p or {}).get("bytesBase64Encoded")
    if b64:
        return b64
    b64 = (p or {}).get("content") or (p or {}).get("image") or pred_json.get("content")
    if b64:
        if isinstance(b64, str) and b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        return b64
    b64 = (p or {}).get("b64_json")
    if b64:
        return b64
    return None

def _save_png_from_b64_to_static(b64: str, transparent: bool) -> Tuple[str, Path]:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = GEN_DIR / f"img_{ts}.png"
    raw = base64.b64decode(b64)
    if not transparent:
        with open(out, "wb") as f:
            f.write(raw)
        return f"/static/generated/{out.name}", out
    try:
        img = Image.open(BytesIO(raw)).convert("RGBA")
        datas = img.getdata()
        new_data = []
        for px in datas:
            if px[0] > 245 and px[1] > 245 and px[2] > 245:
                new_data.append((px[0], px[1], px[2], 0))
            else:
                new_data.append(px)
        img.putdata(new_data)
        img.save(out, "PNG")
        return f"/static/generated/{out.name}", out
    except Exception:
        with open(out, "wb") as f:
            f.write(raw)
        return f"/static/generated/{out.name}", out

def build_image_prompt(user_prompt: str, style: str = "clipart", accuracy: str = "fast") -> str:
    p = user_prompt.strip()
    if style == "clipart":
        p = (
            f"{p}. Black outline vector clipart, simple, minimal, flat, high contrast, "
            f"solid fills, no background, no text, no watermark"
        )
    else:
        p = f"{p}. photorealistic, studio lighting, sharp focus, no text, no watermark"
    if accuracy == "careful":
        p += ". focus on shape accuracy and label fidelity, avoid extra parts, ensure correct proportions"
    else:
        p += ". balanced detail"
    
    lower = user_prompt.lower()
    for de, en in SYNONYMS.items():
        if de in lower:
            p += f". ({en})"
            break
    return p

def check_for_visual_aid(question: str, answer: str, subject: str) -> Optional[Dict[str, Any]]:
    q_lower = question.lower()
    structure_keywords = ['aufbau', 'struktur', 'bestandteil', 'teil', 'schicht', 'aussehen', 'wie sieht', 'zeig', 'erkläre mir']
    process_keywords = ['ablauf', 'prozess', 'vorgang', 'funktioniert', 'geschieht', 'passiert']
    has_structure_question = any(kw in q_lower for kw in structure_keywords)
    has_process_question = any(kw in q_lower for kw in process_keywords)
    if not (has_structure_question or has_process_question):
        return None
    if subject in VISUAL_TRIGGERS:
        for category, items in VISUAL_TRIGGERS[subject].items():
            for key, config in items.items():
                if key in q_lower or key.replace('-', ' ') in q_lower:
                    result = {
                        'prompt': config['prompt'],
                        'topic': key.replace('-', ' ').title(),
                        'auto_show': config.get('auto_show', False),
                        'supports_labeling': 'labels' in config
                    }
                    if 'labels' in config:
                        result['labels'] = config['labels']
                    return result

    for key, config in SPECIAL_CASES.items():
        if key in q_lower:
            return config
    return None


# ======================================================================
# Text-KI Helper – sicher generieren mit Auto-Re-Select
# ======================================================================

# --- PATCH 4: Safety Settings definieren ---
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- PATCH 4: Komplette Funktion ersetzen ---
def gemini_generate(prompt: str, retries: int = 2) -> str:
    global GEMINI_MODEL
    last_err = None
    for attempt in range(retries + 1):
        try:
            if not GEMINI_MODEL:
                init_gemini(prefer="fast")
                if not GEMINI_MODEL:
                    raise RuntimeError("Text-KI nicht initialisiert.")
            
            # --- PATCH START ---
            resp = GEMINI_MODEL.generate_content(
                prompt, 
                generation_config=GENERATION_CONFIG,
                safety_settings=SAFETY_SETTINGS  # <-- HIER IST DIE ÄNDERUNG
            )
            
            # Robusterer Text-Zugriff, um den Fehler von eben zu verhindern
            if resp.parts:
                return resp.text
            else:
                # Loggen, warum es leer war
                finish_reason = resp.candidates[0].finish_reason if resp.candidates else "UNKNOWN"
                logger.warning("Gemini-Antwort hatte keine 'parts', finish_reason: %s", finish_reason)
                # Den Fehler, den du gesehen hast, manuell auslösen, damit der Retry-Loop ihn fängt
                raise ValueError(f"Leere Antwort von KI, Grund: {finish_reason}")
            # --- PATCH ENDE ---

        except Exception as e:
            msg = str(e)
            last_err = e
            # Bei Model-404/400 sofort neu wählen
            if "not found" in msg or "is not supported" in msg or "404" in msg or "400" in msg:
                logger.warning("Gemini Modellfehler (%s) – Re-Select & Retry (%d/%d)", msg, attempt+1, retries)
                reselect_gemini(prefer="fast")
                continue
            # Rate Limit: kurzer Backoff wäre denkbar (hier weggelassen)
            logger.exception("Gemini generate_content Fehler: %s", e)
            break
    raise last_err if last_err else RuntimeError("Unbekannter Fehler bei Gemini.")


# ======================================================================
# Routen
# ======================================================================

# --- PATCH 1: Route zum Ausliefern der HTML-Seite ---
@app.route("/")
def index():
    """Liefert die Haupt-HTML-Datei aus."""
    # Sucht automatisch im 'templates' Ordner nach der Datei
    return render_template('KILehrer.html')
# --- ENDE NEUE ROUTE ---


@app.route("/ping")
def ping():
    return jsonify({"ok": True, "model": GEMINI_MODEL_NAME})

@app.route("/ask", methods=["POST"])
def ask():
    """
    Erweiterte /ask Route mit automatischer Visual Aid Erkennung
    (unverändert)
    """
    data = request.get_json(silent=True) or {}
    user_question = (data.get("question") or "").strip()
    subject = (data.get("subject") or "Chemie").strip()
    grade = (data.get("grade") or "Klasse 9").strip()
    request_visual = data.get("request_visual", False)

    if not user_question:
        return jsonify({"answer": "Bitte gib eine Frage ein."}), 400

    final_prompt = f"""{TUTOR_PROMPT}

Kontext: Fach={subject}, Stufe={grade}, Sprache=Deutsch.
Schüler: "{user_question}"
Tutor:
"""

    try:
        answer_text = gemini_generate(final_prompt)
        response = {"answer": answer_text}

        if request_visual:
            visual_aid = check_for_visual_aid(user_question, answer_text, subject)
            if visual_aid:
                response['visual_aid'] = visual_aid
                if visual_aid.get('supports_labeling') and 'labels' in visual_aid:
                    response['interactive_elements'] = [{
                        'type': 'labeling_exercise',
                        'instruction': f'Beschrifte die Teile: {visual_aid["topic"]}',
                        'terms': visual_aid['labels']
                    }]

        return jsonify(response)

    except Exception as e:
        logger.exception("Fehler /ask: %s", e)
        return jsonify({"answer": "Entschuldigung, es gab ein Problem mit der KI-Verbindung."}), 500


# ---------------------------------------
# /draw  (robuste Version mit Logging)
# (unverändert)
# ---------------------------------------
@app.route("/draw", methods=["POST"])
def draw():
    if not VERTEX_SESSION:
        return jsonify({"error": "Bild-KI nicht initialisiert (Service-Account/Session fehlt)."}), 500

    try:
        data = request.get_json(silent=True) or {}
        prompt       = (data.get("prompt") or "").strip()
        style        = (data.get("style")  or "clipart").strip().lower()
        size_str     = (data.get("size")   or "1024x1024")
        transparent  = bool(data.get("transparent"))
        accuracy     = (data.get("accuracy") or "fast").strip().lower()

        if not prompt:
            return jsonify({"error": "Bitte eine Bildbeschreibung angeben."}), 400

        w, h = _safe_size(size_str)
        prompt_final = build_image_prompt(prompt, style=style, accuracy=accuracy)

        payload = {
            "instances": [{
                "prompt": prompt_final,
                "imageDimensions": {"width": w, "height": h},
                "sampleCount": 1
            }],
            "parameters": {}
        }

        logger.info("DRAW payload: size=%sx%s, style=%s, accuracy=%s, prompt=%r",
                    w, h, style, accuracy, prompt_final[:180])

        vr = VERTEX_SESSION.post(IMAGEN_ENDPOINT, json=payload, timeout=120)

        if vr.status_code != 200:
            logger.error("Vertex error %s: %s", vr.status_code, vr.text[:1000])
            return jsonify({"error": f"Vertex API {vr.status_code}: {vr.text[:500]}"}), 500

        jr = vr.json()
        b64 = _extract_b64(jr)
        if not b64:
            logger.error("Keine Base64-Nutzlast im Prediction-Objekt: %s", jr)
            return jsonify({"error": "Antwort ohne Bilddaten (predictions/bytesBase64Encoded nicht gefunden)."}), 500

        url, _pfad = _save_png_from_b64_to_static(b64, transparent=transparent)

        return jsonify({
            "image_url": url,
            "alt": prompt_final,
            "w": w, "h": h
        })

    except Exception as e:
        logger.exception("DRAW crashed: %s", e)
        return jsonify({"error": f"Serverfehler: {e.__class__.__name__}: {e}"}), 500


# --- PATCH 3: Fehlende Route /improve_prompt ---
@app.route("/improve_prompt", methods=["POST"])
def improve_prompt():
    """
    Prüft, ob für den Input ein spezielles, optimiertes Prompt existiert.
    Wird von der UI für die "🎯 Optimiert"-Anzeige genutzt.
    """
    data = request.get_json(silent=True) or {}
    user_input = (data.get("input") or "").strip().lower()
    
    if not user_input:
        return jsonify({"has_enhancement": False})

    # Prüfe Synonyme
    for de, en in SYNONYMS.items():
        if de in user_input:
            return jsonify({
                "has_enhancement": True,
                "enhanced_prompts": {"primary": f"{en} ({de}), simple vector clipart"}
            })

    # Prüfe Visual Triggers
    for subject, categories in VISUAL_TRIGGERS.items():
        for category, items in categories.items():
            for key, config in items.items():
                if key in user_input or key.replace('-', ' ') in user_input:
                    return jsonify({
                        "has_enhancement": True,
                        "enhanced_prompts": {"primary": config['prompt']}
                    })

    # Prüfe Spezialfälle
    for key, config in SPECIAL_CASES.items():
        if key in user_input:
            return jsonify({
                "has_enhancement": True,
                "enhanced_prompts": {"primary": config['prompt']}
            })

    # Nichts gefunden
    return jsonify({"has_enhancement": False})
# --- ENDE NEUE ROUTE ---


# ---------------------------------------
# /review_annotations  (mit Kontext-Awareness)
# (unverändert)
# ---------------------------------------
@app.route("/review_annotations", methods=["POST"])
def review_annotations():
    if not GEMINI_MODEL and not GEMINI_MODEL_NAME:
        init_gemini(prefer="quality")
        if not GEMINI_MODEL:
            return jsonify({"error": "Text-KI nicht initialisiert."}), 500

    data = request.get_json(silent=True) or {}
    image_b64 = (data.get("image_b64") or "")
    annotations = data.get("annotations") or []
    task = (data.get("task") or "Beschrifte die Abbildung").strip()
    subject = (data.get("subject") or "Biologie").strip()
    grade = (data.get("grade") or "Klasse 9").strip()
    context = data.get("context", "")
    expected_terms = data.get("expected_terms") or []

    sys_prompt = f"""
Du bist ein strenger, aber faire*r Fachlehrer*in ({subject}, {grade}). 
Du bekommst eine Liste von Etiketten (Beschriftungen), die auf einem Bild platziert wurden.
{f"Kontext zum Bild: {context}" if context else ""}
Aufgabe: Prüfe die fachliche Korrektheit und gib präzises, kurzes Feedback.
WICHTIG: Beziehe dich NUR auf das aktuelle Bild und dessen Kontext!

Gib **ausschließlich** gültiges JSON zurück im Format:
{{
  "overall": "Kurzes Gesamtfeedback in 1-2 Sätzen.",
  "score": 0-100,
  "per_label": [
     {{ "id": "<id>", "text": "<label>", "correct": true/false, "feedback": "ein Satz" }}
  ],
  "missing": ["Begriff1", "Begriff2"],
  "wrong": ["falscherBegriff1"]
}}
"""

    labels_text = "\n".join([f"- ({it.get('id','?')}): {it.get('text','').strip()}" for it in annotations])
    user_prompt = f"""
Aufgabe: {task}
{f"Bildinhalt: {context}" if context else ""}
Erwartete Begriffe (optional): {', '.join(expected_terms) if expected_terms else '—'}

Beschriftungen:
{labels_text}

Bitte **nur** das JSON ausgeben. Beziehe dich auf das AKTUELLE Bild!
"""

    try:
        text = gemini_generate(sys_prompt + "\n\n" + user_prompt)
        
        j = {}
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                j = json.loads(text[start:end+1])
            else:
                raise Exception("Kein JSON gefunden")
        except Exception:
             j = {"overall": "Konnte das JSON nicht parsen.", "score": 0, "per_label": [], "missing": [], "wrong": []}

        j.setdefault("overall", "")
        j.setdefault("score", None)
        j.setdefault("per_label", [])
        j.setdefault("missing", [])
        j.setdefault("wrong", [])

        return jsonify(j)

    except Exception as e:
        logger.exception("review_annotations Fehler: %s", e)
        return jsonify({"error": f"Serverfehler: {e.__class__.__name__}: {e}"}), 500


# ---------------------------------------
# /check_visual_aid - Endpoint für Visual Aid Prüfung
# (unverändert)
# ---------------------------------------
@app.route("/check_visual_aid", methods=["POST"])
def check_visual_aid_endpoint():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    subject = (data.get("subject") or "Biologie").strip()
    if not question:
        return jsonify({"needs_visual": False})
    visual_aid = check_for_visual_aid(question, "", subject)
    if visual_aid:
        return jsonify({"needs_visual": True, "visual_aid": visual_aid})
    else:
        return jsonify({"needs_visual": False})


# ======================================================================
# Main (angepasst für Render)
# ======================================================================
if __name__ == "__main__":
    # Nutzt den PORT von Render oder 5000 lokal
    port = int(os.environ.get('PORT', 5000))
    # Debug auf False setzen für Produktion, True nur für lokales Testen
    app.run(host="0.0.0.0", port=port, debug=False)