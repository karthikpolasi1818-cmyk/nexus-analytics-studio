import pandas as pd
import numpy as np

ALIASES={
"transaction_id":["transaction_id","transactionid","txn_id","txn"],
"customer_id":["customer_id","customerid","cust_id","client_id"],
"transaction_date":["transaction_date","date","timestamp","datetime","transaction_time"],
"amount":["amount","transaction_amount","txn_amount","value","transaction_value"],
"fraud":["fraud","is_fraud","fraud_flag","fraudulent","fraud_status"],
"risk_score":["risk_score","risk","fraud_score","risk_rating"],
"status":["status","transaction_status"],
"transaction_type":["transaction_type","txn_type","type","payment_type"],
"merchant":["merchant","merchant_name","merchant_id"],
"country":["country","billing_country","location_country"],
"device":["device","device_type","device_id"],
"channel":["channel","payment_channel","transaction_channel"],
"chargeback":["chargeback","chargeback_flag","is_chargeback"],
}

def _norm(v): return "".join(c.lower() for c in str(v).strip() if c.isalnum())

def _find_column(df,aliases):
    normalized={_norm(c):c for c in df.columns}
    for a in aliases:
        if _norm(a) in normalized: return normalized[_norm(a)]
    for c in df.columns:
        nc=_norm(c)
        for a in aliases:
            na=_norm(a)
            if len(na)>=4 and (nc.startswith(na) or nc.endswith(na)): return c
    return None

def detect_fraud_fields(df): return {k:_find_column(df,v) for k,v in ALIASES.items()}

def _to_flag(s):
    if pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s,errors="coerce").fillna(0).gt(0).astype(int)
    return s.astype(str).str.strip().str.lower().isin(["yes","y","true","1","fraud","fraudulent","chargeback","high"]).astype(int)

def prepare_fraud_data(df):
    w=df.copy(); fields=detect_fraud_fields(w)
    for logical,source in fields.items():
        if source and logical not in w.columns: w[logical]=w[source]
    if "amount" in w: w["amount"]=pd.to_numeric(w["amount"],errors="coerce")
    if "risk_score" in w: w["risk_score"]=pd.to_numeric(w["risk_score"],errors="coerce")
    if "transaction_date" in w: w["transaction_date"]=pd.to_datetime(w["transaction_date"],errors="coerce")
    if "fraud" in w: w["fraud_flag"]=_to_flag(w["fraud"])
    elif "chargeback" in w: w["fraud_flag"]=_to_flag(w["chargeback"])
    elif "risk_score" in w: w["fraud_flag"]=w["risk_score"].fillna(0).ge(70).astype(int)
    else: w["fraud_flag"]=0
    return w,fields

def calculate_fraud_kpis(df):
    total=len(df); count=int(df["fraud_flag"].sum())
    r={"transactions":total,"fraud_transactions":count,"fraud_rate":count/total*100 if total else 0.0,"total_amount":None,"fraud_amount":None,"fraud_amount_rate":None,"average_transaction":None}
    if "amount" in df:
        a=pd.to_numeric(df["amount"],errors="coerce"); r["total_amount"]=float(a.sum()); r["average_transaction"]=float(a.mean())
        fa=float(a.where(df["fraud_flag"].eq(1),0).sum()); r["fraud_amount"]=fa
        r["fraud_amount_rate"]=fa/r["total_amount"]*100 if r["total_amount"] else 0.0
    return r

def fraud_by_type(df):
    if "transaction_type" not in df: return pd.DataFrame()
    return df.groupby("transaction_type",dropna=False)["fraud_flag"].agg(["sum","count"]).reset_index().rename(columns={"sum":"Fraud Transactions","count":"Transactions"}).assign(Fraud_Rate=lambda x:x["Fraud Transactions"]/x["Transactions"]*100).sort_values("Fraud_Rate",ascending=False)

def fraud_by_merchant(df):
    if "merchant" not in df: return pd.DataFrame()
    x=df.groupby("merchant",dropna=False).agg(Transactions=("fraud_flag","size"),Fraud_Transactions=("fraud_flag","sum")).reset_index()
    x["Fraud Rate"]=x["Fraud_Transactions"]/x["Transactions"]*100
    return x.sort_values("Fraud_Transactions",ascending=False)

def fraud_by_country(df):
    if "country" not in df: return pd.DataFrame()
    x=df.groupby("country",dropna=False).agg(Transactions=("fraud_flag","size"),Fraud_Transactions=("fraud_flag","sum")).reset_index()
    x["Fraud Rate"]=x["Fraud_Transactions"]/x["Transactions"]*100
    return x.sort_values("Fraud Rate",ascending=False)

def risk_distribution(df): return df[["risk_score"]].dropna() if "risk_score" in df else pd.DataFrame()

def daily_fraud_trend(df):
    if "transaction_date" not in df: return pd.DataFrame()
    x=df.dropna(subset=["transaction_date"]).copy()
    if x.empty: return pd.DataFrame()
    x["Date"]=x["transaction_date"].dt.date
    return x.groupby("Date").agg(Transactions=("fraud_flag","size"),Fraud_Transactions=("fraud_flag","sum")).reset_index()

def high_risk_transactions(df,threshold=70):
    if "risk_score" not in df: return pd.DataFrame()
    cols=[c for c in ["transaction_id","customer_id","transaction_date","amount","risk_score","transaction_type","merchant","country","device","channel","fraud"] if c in df]
    return df[df["risk_score"].ge(threshold)][cols].sort_values("risk_score",ascending=False) if cols else pd.DataFrame()

def generate_fraud_insights(df):
    k=calculate_fraud_kpis(df); out=[f"Analyzed {k['transactions']:,} transactions with {k['fraud_transactions']:,} flagged as potentially fraudulent.",f"Observed fraud rate is {k['fraud_rate']:.2f}%."]
    if k["fraud_amount"] is not None: out.append(f"Flagged transactions represent {k['fraud_amount']:,.2f} in transaction value.")
    t=fraud_by_type(df)
    if not t.empty:
        r=t.iloc[0]; out.append(f"{r['transaction_type']} has the highest observed fraud rate at {r['Fraud_Rate']:.2f}% among transaction types.")
    return out

def fraud_summary(df):
    return {"kpis":calculate_fraud_kpis(df),"by_type":fraud_by_type(df),"by_merchant":fraud_by_merchant(df),"by_country":fraud_by_country(df),"risk_distribution":risk_distribution(df),"daily_trend":daily_fraud_trend(df),"high_risk":high_risk_transactions(df),"insights":generate_fraud_insights(df)}
