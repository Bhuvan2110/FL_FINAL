"""
AI Agent Chat API — step-by-step prediction feature collector and conversational helper.
"""
import os
import re
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_current_user
from app.api.predict import _load_model
from app.db.supabase_client import get_supabase
from app.ml.logistic_regression import predict_proba
from app.ml.preprocessing import apply_scaler, compute_input_hash

router = APIRouter(prefix="/agent", tags=["agent"])


class ChatMessage(BaseModel):
    role: str  # user | assistant
    content: str


class AgentChatRequest(BaseModel):
    model_id: str | None = None
    messages: list[ChatMessage]
    current_feature: str | None = None
    collected_features: dict[str, Any] = {}


def clean_string(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "").replace(".", "")


def generate_gemini_response(prompt: str, api_key: str) -> str | None:
    """Call Gemini 1.5 Flash to generate a conversational response."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.post(url, json=payload, headers=headers)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"⚠️ Gemini API call failed: {e}")
    return None


@router.post("/chat")
async def agent_chat(body: AgentChatRequest, user: dict = Depends(get_current_user)):
    sb = get_supabase()
    api_key = os.environ.get("GEMINI_API_KEY")

    # 1. Start: If no model is selected
    if not body.model_id:
        # Fetch models to help the user choose
        from app.api.predict import list_available_models
        models = await list_available_models(user)
        
        welcome_text = (
            "Hello! I am your FL Predictive Analytics AI Agent. 🤖\n\n"
            "I can guide you through running predictions using our trained Federated Learning models. "
            "To begin, please select one of the models listed below:"
        )
        
        return {
            "message": welcome_text,
            "options": [{"value": m["id"], "label": f"{m['algorithm'].upper()} (v{m['version']})"} for m in models],
            "current_feature": None,
            "collected_features": {},
            "prediction": None
        }

    # Load model configuration
    try:
        model_data = _load_model(body.model_id, sb)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {e!s}")

    weights = model_data["weights"]
    bias = model_data["bias"]
    scalers = model_data["scalers"]
    feature_names = model_data.get("feature_names", [])
    encoders = model_data.get("encoders", {})

    # 2. Get last user message
    last_msg = body.messages[-1].content if body.messages else ""
    
    # Check if user wants to reset/restart
    if clean_string(last_msg) in ["reset", "restart", "startover", "clear"]:
        first_feat = feature_names[0] if feature_names else None
        welcome_back = f"Let's start over. Collecting features for model. First, what is the value for **{first_feat}**?"
        options = []
        if first_feat in encoders:
            options = [{"value": c, "label": c} for c in encoders[first_feat]]
        return {
            "message": welcome_back,
            "options": options,
            "current_feature": first_feat,
            "collected_features": {},
            "prediction": None
        }

    collected = dict(body.collected_features)
    current_feat = body.current_feature
    error_msg = None

    # 3. Process previous feature input if there was a pending question
    if current_feat and current_feat in feature_names:
        # Extract value from last_msg
        val_str = last_msg.strip()
        
        # Try to parse
        if current_feat in encoders:
            # Categorical feature
            allowed = encoders[current_feat]
            # Try exact or case-insensitive match
            matched = None
            for cat in allowed:
                if cat.lower() == val_str.lower():
                    matched = cat
                    break
            if matched:
                collected[current_feat] = matched
            else:
                error_msg = f"Sorry, '{val_str}' is not a valid choice for **{current_feat}**. Please select one of the options below:"
        else:
            # Numeric feature
            # Extract number using simple regex
            num_match = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            if num_match:
                try:
                    # Take the first number found
                    collected[current_feat] = float(num_match[0])
                except ValueError:
                    error_msg = f"I couldn't parse a number from '{val_str}'. Please enter a numeric value for **{current_feat}**:"
            else:
                error_msg = f"Please enter a valid numeric value for **{current_feat}**:"

    # 4. Find next missing feature
    next_feat = None
    for fn in feature_names:
        if fn not in collected:
            next_feat = fn
            break

    # If there was an error, re-ask the current feature
    if error_msg:
        options = []
        if current_feat in encoders:
            options = [{"value": c, "label": c} for c in encoders[current_feat]]
        return {
            "message": error_msg,
            "options": options,
            "current_feature": current_feat,
            "collected_features": collected,
            "prediction": None
        }

    # 5. Case A: Still collecting features
    if next_feat:
        msg = f"Got it. Next, what is the value for **{next_feat}**?"
        
        # If Gemini is online, ask it to make the question sound more natural
        if api_key:
            prompt = (
                f"You are a friendly AI clinical assistant helper. We are collecting inputs to make a prediction using a machine learning model.\n"
                f"So far, we have collected: {collected}.\n"
                f"The next feature we need is: '{next_feat}'. If this feature is categorical, we have these allowed options: {encoders.get(next_feat, 'None')}.\n"
                f"Write a short, professional, and friendly response asking the user for this feature. Keep it concise (1-2 sentences)."
            )
            gemini_msg = generate_gemini_response(prompt, api_key)
            if gemini_msg:
                msg = gemini_msg

        options = []
        if next_feat in encoders:
            options = [{"value": c, "label": c} for c in encoders[next_feat]]

        return {
            "message": msg,
            "options": options,
            "current_feature": next_feat,
            "collected_features": collected,
            "prediction": None
        }

    # 6. Case B: All features collected, run prediction!
    raw_row = []
    for name in feature_names:
        val = collected.get(name, 0.0)
        if name in encoders:
            if isinstance(val, str):
                val_str = val.strip()
                if val_str in encoders[name]:
                    val_float = float(encoders[name].index(val_str))
                else:
                    val_float = 0.0
            else:
                val_float = float(val)
        else:
            val_float = float(val)
        raw_row.append(val_float)

    normalized_row = apply_scaler(raw_row, scalers)
    prob = predict_proba(weights, bias, [normalized_row])[0]
    pred = 1 if prob >= 0.5 else 0
    input_hash = compute_input_hash(collected)
    class_label = "Positive" if pred == 1 else "Negative"

    # Store prediction history in database
    sb.table("predictions").insert({
        "user_id": user["id"],
        "model_id": body.model_id,
        "input_hash": input_hash,
        "output": pred,
        "confidence": round(prob, 4),
    }).execute()

    prediction_result = {
        "prediction": pred,
        "confidence": round(prob, 4),
        "class_label": class_label,
        "input_hash": input_hash,
    }

    # Build details summary
    inputs_md = "\n".join([f"- **{k}**: `{v}`" for k, v in collected.items()])
    summary_message = (
        f"All features collected! Running prediction... 🔮\n\n"
        f"### Prediction Result: **{class_label}**\n"
        f"- **Confidence**: `{round(prob * 100, 2)}%`\n"
        f"- **Prediction ID**: `{input_hash}`\n\n"
        f"#### Collected Inputs:\n{inputs_md}\n\n"
        f"Would you like to run another prediction or try a different model?"
    )

    # Use Gemini to summarize the prediction results in detail if online
    if api_key:
        prompt = (
            f"You are a friendly AI clinical assistant helper. We just ran a logistic regression prediction model using federated learning weights.\n"
            f"The collected features: {collected}.\n"
            f"Prediction outcome is: {class_label} (Confidence: {round(prob * 100, 2)}%).\n"
            f"Explain these results to the user in a professional, clear, and reassuring way. Highlight what the features mean and how the confidence is calculated. "
            f"List the collected inputs clearly as bullet points, and ask if they would like to do another run or try a different model."
        )
        gemini_summary = generate_gemini_response(prompt, api_key)
        if gemini_summary:
            summary_message = gemini_summary

    return {
        "message": summary_message,
        "options": [
            {"value": "restart", "label": "Run again with same model"},
            {"value": "switch_model", "label": "Switch model"}
        ],
        "current_feature": None,
        "collected_features": collected,
        "prediction": prediction_result
    }
