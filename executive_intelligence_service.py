from __future__ import annotations
from typing import Any, Dict
import math

def num(v):
    try:
        if v is None or isinstance(v,bool): return None
        x=float(v); return x if math.isfinite(x) else None
    except Exception: return None

def label(k): return str(k).replace("_"," ").title()

def trend(rows):
    if len(rows)<2:return None
    key=next((k for k in ("revenue","sales","profit") if k in rows[-1]),None)
    if not key:key=next((k for k,v in rows[-1].items() if num(v) is not None),None)
    if not key:return None
    a,b=num(rows[-2].get(key)),num(rows[-1].get(key))
    if a is None or b is None:return None
    c=b-a
    return {"metric":key,"previous":a,"latest":b,"change":c,"change_pct":round(c/a*100,2) if a else None,"direction":"up" if c>0 else "down" if c<0 else "flat"}

def build_executive_intelligence(result:Dict[str,Any])->Dict[str,Any]:
    d=result.get("dashboard") or {}; kpis=d.get("kpis") or {}; q=d.get("quality") or {}; tr=d.get("trend") or []
    signals=[]; opportunities=[]; tc=trend(tr)
    if tc: signals.append({"type":"trend","severity":"positive" if tc["direction"]=="up" else "attention" if tc["direction"]=="down" else "neutral","title":f"{label(tc['metric'])} trend","detail":f"Latest period changed {abs(tc['change_pct'] or 0):.2f}% versus the previous period.","evidence":tc})
    margin=next((num(kpis[k]) for k in ("profit_margin","margin") if k in kpis),None)
    if margin is not None and margin<0: signals.append({"type":"financial","severity":"critical","title":"Negative margin","detail":"The current dataset reports a negative margin.","evidence":{"profit_margin":margin}})
    score=num(q.get("score"))
    if score is not None and score<80: signals.append({"type":"quality","severity":"attention","title":"Data quality requires review","detail":f"Current data-quality score is {score:.1f}.","evidence":{"quality_score":score,"grade":q.get("grade")}})
    for key,dim in [("category_performance","category"),("region_performance","region"),("product_performance","product"),("channel_performance","channel")]:
        rows=d.get(key) or []
        if not rows: continue
        metric="profit" if "profit" in rows[0] else ("sales" if "sales" in rows[0] else ("revenue" if "revenue" in rows[0] else None))
        if not metric: continue
        vals=[]
        for r in rows:
            name=r.get(dim,r.get(dim.title(),r.get("name"))); v=num(r.get(metric))
            if name is not None and v is not None: vals.append((name,v))
        if vals:
            top=max(vals,key=lambda x:x[1])
            opportunities.append({"type":"performance","title":f"Leading {dim}","detail":f"{top[0]} has the highest {metric} in the available {dim} performance data.","evidence":{"dimension":dim,"name":top[0],"metric":metric,"value":top[1]}})
    ek=[{"key":k,"label":label(k),"value":v} for k,v in kpis.items() if num(v) is not None]
    parts=[]
    if tc: parts.append(f"{label(tc['metric'])} is {tc['direction']} in the latest period"+(f" by {abs(tc['change_pct']):.2f}%." if tc["change_pct"] is not None else "."))
    if margin is not None: parts.append(f"Reported margin is {margin:.2f}%.")
    if score is not None: parts.append(f"Data quality score is {score:.1f}.")
    if opportunities: parts.append(opportunities[0]["detail"])
    if not parts: parts.append("The current dataset has been analyzed, but there is insufficient evidence for an executive summary.")
    return {"title":"NEXUS Executive Intelligence","summary":" ".join(parts),"kpis":ek,"signals":signals,"opportunities":opportunities,"trend":tr,"quality":q,"domain":d.get("domain") or result.get("module"),"dataset":{"filename":result.get("filename"),"rows":(result.get("shape") or {}).get("rows"),"columns":(result.get("shape") or {}).get("columns")}}
