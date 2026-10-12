"""Phase V - turn forecasts into stakeholder-specific advisories.

Two engines: a deterministic template engine (offline, always factually consistent) and an
optional LLM engine (Anthropic API) whose output is verified against the same facts pack.
Every advisory is a DRAFT until a human reviewer approves it (see views/advisories.py)."""
import json
import os
import re

import pandas as pd

from .config import ALERT_LEVELS, CITY

STAKEHOLDERS = ["Citizens", "Farmers", "Health agencies and hospitals", "Municipal and disaster authorities"]
LANGS = {"English": "en", "हिन्दी": "hi", "मराठी": "mr"}

MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "hi": ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
    "mr": ["जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून", "जुलै", "ऑगस्ट", "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"],
}


def confidence_label(lead: int) -> str:
    return "high" if lead <= 2 else ("moderate" if lead <= 4 else "low")


def build_facts(city: str, fc: pd.DataFrame, asof, aws_hot: dict | None = None, lookahead: int = 5) -> dict:
    """Grounding facts for one city from its 7-day forecast frame (columns of data.forecast_at)."""
    f = fc[(fc["city"] == city) & (fc["h"] <= lookahead)].sort_values("h")
    worst = int(f["alert"].max())
    asof = pd.Timestamp(asof)
    if worst >= 1:
        first = f[f["alert"] >= worst].iloc[0]
        peak = f.loc[f["tmax_pred"].idxmax()]
    else:
        first = f.iloc[0]
        peak = f.loc[f["tmax_pred"].idxmax()]
    n_days = int((f["alert"] >= max(worst, 1)).sum()) if worst >= 1 else 0
    facts = dict(
        city=city, state=CITY[city]["state"], region=CITY[city]["region"], issue_date=asof.strftime("%Y-%m-%d"),
        alert_level=worst, alert_colour=ALERT_LEVELS[worst]["colour"], alert_name=ALERT_LEVELS[worst]["name"],
        first_alert_date=(asof + pd.Timedelta(days=int(first["h"]))).strftime("%Y-%m-%d"),
        first_alert_lead_days=int(first["h"]), alert_days=n_days,
        peak_date=(asof + pd.Timedelta(days=int(peak["h"]))).strftime("%Y-%m-%d"),
        peak_tmax_c=round(float(peak["tmax_pred"]), 1), peak_tmax_low_c=round(float(peak["tmax_lo"]), 1),
        peak_tmax_high_c=round(float(peak["tmax_hi"]), 1),
        departure_from_normal_c=round(float(peak["tmax_pred"] - peak["normal_tgt"]), 1),
        normal_tmax_c=round(float(peak["normal_tgt"]), 1),
        p_heatwave_peak_pct=int(round(float((f["p2"] + f["p3"]).max()) * 100)),
        p_severe_peak_pct=int(round(float(f["p3"].max()) * 100)),
        confidence=confidence_label(int(first["h"])), lookahead_days=lookahead,
        population_million=CITY[city]["pop"],
    )
    if aws_hot:
        facts["hottest_station"] = aws_hot["name"]
        facts["hottest_station_temp_c"] = round(float(aws_hot["temp_c"]), 1)
    return facts


def _d(date_str: str, lang: str) -> str:
    d = pd.Timestamp(date_str)
    return f"{d.day} {MONTHS[lang][d.month - 1]}"


# --------------------------------------------------------------------------- templates
ACTIONS = {
    "Citizens": {
        1: ["Plan outdoor work and travel for early morning or evening.", "Carry water and drink regularly through the day.",
            "Wear light, loose, light-coloured cotton clothing and cover your head."],
        2: ["Avoid going out in the sun between 11:00 and 16:00.",
            "Drink water often, even if you are not thirsty. Oral rehydration solution (ORS), lemon water and buttermilk help.",
            "Wear light, loose cotton clothes and cover your head with a cap or cloth.",
            "Never leave children, elderly people or pets in a parked vehicle.",
            "Check on elderly neighbours and people living alone."],
        3: ["Stay indoors between 11:00 and 16:00 unless it is essential to go out.",
            "Drink water often. Use ORS, lemon water or buttermilk. Avoid tea, coffee, alcohol and sugary drinks.",
            "Keep rooms cool: use curtains, fans and wet cloths. Take cool showers.",
            "Never leave children, elderly people or pets in a parked vehicle.",
            "High fever, confusion, dizziness or no sweating is an emergency: move the person to shade, cool them with "
            "wet cloths and call 112."],
    },
    "Farmers": {
        1: ["Schedule field work for early morning and late afternoon.", "Check that irrigation and drinking water for livestock are ready."],
        2: ["Do field work before 10:30 or after 16:30 and give workers shaded rest and drinking water.",
            "Irrigate in the early morning or evening and mulch to hold soil moisture.",
            "Provide shade, fresh water and ventilation for livestock and poultry.",
            "Harvest produce in the cool of the morning and store it in shade."],
        3: ["Stop open-field labour during the hottest hours and keep workers hydrated.",
            "Irrigate lightly and frequently in the evening or night; apply mulch; use shade nets for nurseries and vegetables.",
            "Move livestock to shade, give cool water several times a day and avoid transport in the afternoon.",
            "Postpone sowing, transplanting and spraying until temperatures fall."],
    },
    "Health agencies and hospitals": {
        1: ["Review heat-illness readiness: ORS, IV fluids, cooling equipment.", "Remind outreach teams to brief vulnerable groups."],
        2: ["Keep heat-stroke beds, ice packs and cooling equipment ready and stock ORS and IV fluids.",
            "Brief ambulance services and community health workers; target the elderly, children, pregnant women and outdoor workers.",
            "Start daily surveillance of heat-related illness presentations."],
        3: ["Activate the heat-stroke protocol: dedicated cooling wards, extended emergency staffing and ambulance pre-positioning.",
            "Push SMS and field advisories to high-risk groups and outdoor workers.",
            "Report heat-related illness and deaths daily to the district authority.",
            "Check power backup at all hospitals and primary health centres."],
    },
    "Municipal and disaster authorities": {
        1: ["Review the heat action plan and confirm contacts with health, water and power departments."],
        2: ["Open shaded rest areas and extra drinking-water points in busy areas.",
            "Advise employers to shift outdoor work away from 12:00 to 15:00.",
            "Confirm water-tanker and power supply plans; prepare public messaging."],
        3: ["Open cooling centres and extend drinking-water points, prioritising the hottest neighbourhoods.",
            "Enforce a midday pause for outdoor labour and consider adjusting school and office hours.",
            "Coordinate with the power utility to protect hospitals and water pumping.",
            "Issue public alerts through SMS, radio, TV and loudspeakers and review the situation daily."],
    },
}

CITIZEN_I18N = {
    "hi": dict(
        head={1: "गर्मी की निगरानी: {city} में तापमान बढ़ने की संभावना",
              2: "लू की चेतावनी: {city} में {date} से लू की संभावना",
              3: "गंभीर लू की चेतावनी: {city} में {date} से भीषण गर्मी का खतरा"},
        none="{city} में अगले {n} दिनों में लू की संभावना कम है। सामान्य सावधानी रखें।",
        sit="अनुमानित अधिकतम तापमान {date} को लगभग {t}°C रहने की संभावना है (संभावित सीमा {lo}–{hi}°C)। लू की संभावना {p}% तक है।",
        actions={
            1: ["बाहर के काम सुबह या शाम के समय करें।", "पानी साथ रखें और दिन भर पीते रहें।",
                "हल्के रंग के ढीले सूती कपड़े पहनें और सिर ढकें।"],
            2: ["दोपहर 11 बजे से शाम 4 बजे के बीच धूप में बाहर जाने से बचें।",
                "प्यास न लगने पर भी बार-बार पानी पिएं; ओआरएस, नींबू पानी या छाछ लें।",
                "हल्के रंग के ढीले सूती कपड़े पहनें और सिर को टोपी या गमछे से ढकें।",
                "बच्चों, बुजुर्गों और पालतू जानवरों को खड़ी गाड़ी में कभी न छोड़ें।",
                "अकेले रहने वाले बुजुर्गों और पड़ोसियों का हालचाल लें।"],
            3: ["बहुत ज़रूरी न हो तो दोपहर 11 बजे से शाम 4 बजे तक घर के अंदर रहें।",
                "बार-बार पानी, ओआरएस, नींबू पानी या छाछ पिएं; चाय, कॉफी और शराब से बचें।",
                "कमरे ठंडे रखें और ठंडे पानी से नहाएं।",
                "बच्चों, बुजुर्गों और पालतू जानवरों को खड़ी गाड़ी में कभी न छोड़ें।",
                "तेज बुखार, चक्कर, भ्रम या पसीना बंद होने पर व्यक्ति को छाया में ले जाएं, गीले कपड़े से ठंडा करें और तुरंत 112 पर कॉल करें।"],
        }),
    "mr": dict(
        head={1: "उष्णतेवर लक्ष: {city} मध्ये तापमान वाढण्याची शक्यता",
              2: "उष्णतेच्या लाटेचा इशारा: {city} मध्ये {date} पासून लाटेची शक्यता",
              3: "तीव्र उष्णतेच्या लाटेचा इशारा: {city} मध्ये {date} पासून तीव्र उष्णतेचा धोका"},
        none="{city} मध्ये पुढील {n} दिवसांत उष्णतेच्या लाटेची शक्यता कमी आहे. नेहमीची काळजी घ्या.",
        sit="{date} रोजी कमाल तापमान सुमारे {t}°C राहण्याची शक्यता आहे (संभाव्य श्रेणी {lo}–{hi}°C). लाटेची शक्यता {p}% पर्यंत आहे.",
        actions={
            1: ["बाहेरची कामे सकाळी किंवा संध्याकाळी करा.", "सोबत पाणी ठेवा आणि दिवसभर पीत रहा.",
                "सैलसर, फिकट रंगाचे सुती कपडे घाला आणि डोके झाका."],
            2: ["सकाळी 11 ते दुपारी 4 या वेळेत उन्हात बाहेर पडणे टाळा.",
                "तहान नसली तरी वारंवार पाणी प्या; ओआरएस, लिंबू सरबत किंवा ताक घ्या.",
                "सैलसर, फिकट रंगाचे सुती कपडे घाला आणि डोके टोपी किंवा रुमालाने झाका.",
                "लहान मुले, वृद्ध आणि पाळीव प्राण्यांना उभ्या असलेल्या वाहनात कधीही सोडू नका.",
                "एकटे राहणाऱ्या वृद्धांची आणि शेजाऱ्यांची चौकशी करा."],
            3: ["अत्यावश्यक नसल्यास सकाळी 11 ते दुपारी 4 या वेळेत घरातच रहा.",
                "वारंवार पाणी, ओआरएस, लिंबू सरबत किंवा ताक प्या; चहा, कॉफी आणि मद्य टाळा.",
                "खोल्या थंड ठेवा आणि थंड पाण्याने आंघोळ करा.",
                "लहान मुले, वृद्ध आणि पाळीव प्राण्यांना उभ्या असलेल्या वाहनात कधीही सोडू नका.",
                "खूप ताप, चक्कर, गोंधळ किंवा घाम थांबल्यास व्यक्तीला सावलीत न्या, ओल्या कापडाने थंड करा आणि ताबडतोब 112 वर संपर्क साधा."],
        }),
}


def template_advisory(stakeholder: str, facts: dict, lang: str = "en") -> dict:
    lvl = facts["alert_level"]
    city = facts["city"]
    unc = {"high": "Confidence is high (short lead time).", "moderate": "Confidence is moderate; the forecast may shift in the next update.",
           "low": "Confidence is low at this lead time. Treat this as an early heads-up and watch for updates."}[facts["confidence"]]
    if lang in CITIZEN_I18N and stakeholder == "Citizens":
        t = CITIZEN_I18N[lang]
        date = _d(facts["first_alert_date"], lang)
        if lvl == 0:
            return dict(title=t["none"].format(city=city, n=facts["lookahead_days"]), summary="", actions=[], uncertainty="")
        title = t["head"][lvl].format(city=city, date=date)
        summary = t["sit"].format(date=_d(facts["peak_date"], lang), t=facts["peak_tmax_c"], lo=facts["peak_tmax_low_c"],
                                  hi=facts["peak_tmax_high_c"], p=facts["p_heatwave_peak_pct"])
        return dict(title=title, summary=summary, actions=t["actions"][lvl], uncertainty="")
    if lvl == 0:
        return dict(title=f"No heatwave expected in {city} in the next {facts['lookahead_days']} days",
                    summary="Temperatures are forecast to stay below heatwave criteria. Keep normal heat precautions.",
                    actions=[], uncertainty="")
    head = {1: f"Heat watch for {city}: temperatures are rising",
            2: f"Heatwave alert for {city} from {_d(facts['first_alert_date'], 'en')}",
            3: f"Severe heatwave warning for {city} from {_d(facts['first_alert_date'], 'en')}"}[lvl]
    summary = (f"The maximum temperature is forecast to reach about {facts['peak_tmax_c']}°C on {_d(facts['peak_date'], 'en')} "
               f"(likely range {facts['peak_tmax_low_c']} to {facts['peak_tmax_high_c']}°C), which is "
               f"{facts['departure_from_normal_c']:+.1f}°C against the normal of {facts['normal_tmax_c']}°C. "
               f"The chance of a heatwave reaches {facts['p_heatwave_peak_pct']}%, and of a severe heatwave "
               f"{facts['p_severe_peak_pct']}%. The first alert day is {facts['first_alert_lead_days']} day(s) away.")
    if facts.get("hottest_station"):
        summary += (f" Local sensors already show up to {facts['hottest_station_temp_c']}°C at {facts['hottest_station']}.")
    return dict(title=head, summary=summary, actions=ACTIONS[stakeholder][lvl], uncertainty=unc)


def render_md(a: dict) -> str:
    out = [f"**{a['title']}**"]
    if a["summary"]:
        out.append(a["summary"])
    if a["actions"]:
        out.append("\n".join(f"- {x}" for x in a["actions"]))
    if a["uncertainty"]:
        out.append(f"*{a['uncertainty']}*")
    return "\n\n".join(out)


# --------------------------------------------------------------------------- LLM engine (optional)
def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_advisory(stakeholder: str, facts: dict, lang: str = "en", model: str | None = None) -> str:
    import requests
    model = model or os.environ.get("HEATWAVE_LLM_MODEL", "claude-sonnet-5-5")
    language = {"en": "English", "hi": "Hindi", "mr": "Marathi"}[lang]
    system = ("You write public heatwave advisories for an Indian meteorological early warning service. "
              "Use ONLY the facts in the JSON. Do not invent numbers, places, dates or phone numbers (112 is allowed as "
              "the national emergency number). Keep it under 170 words: one headline line, two sentences of situation, "
              "4 to 5 short action bullets for the stated audience, and one sentence on forecast confidence. "
              "Plain, calm language. No emojis.")
    user = f"Audience: {stakeholder}\nLanguage: {language}\nFacts:\n{json.dumps(facts, indent=2)}"
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=40,
                      headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                               "content-type": "application/json"},
                      json={"model": model, "max_tokens": 600, "system": system,
                            "messages": [{"role": "user", "content": user}]})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])


# --------------------------------------------------------------------------- factual consistency check
def _numbers(obj) -> set:
    out = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out |= {f"{obj:g}", f"{abs(obj):g}", f"{obj:.1f}", f"{abs(obj):.1f}", str(int(round(abs(obj))))}
    elif isinstance(obj, str):
        out |= set(re.findall(r"\d+(?:\.\d+)?", obj))
        d = re.match(r"(\d{4})-(\d{2})-(\d{2})$", obj)
        if d:
            out |= {str(int(d.group(3))), str(int(d.group(2)))}
    return out


ALLOWED_GENERIC = {"11", "16", "4", "10", "30", "12", "15", "112", "1", "2", "3", "5", "7", "170"}


def consistency_check(text: str, facts: dict) -> dict:
    """Every number in the advisory must trace back to the facts pack (or be a fixed protocol number)."""
    allowed = _numbers(facts) | ALLOWED_GENERIC
    found = [n for n in re.findall(r"\d+(?:\.\d+)?", text) if set(n) - {"0"}]
    bad = sorted({n for n in found if n not in allowed and n.lstrip("0") not in allowed})
    return dict(checked=len(found), unverified=bad, ok=not bad)
