from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    system_prompt = (
        "You are NASA Mission Intelligence, a careful assistant for historical "
        "NASA mission documents. Answer using the retrieved context when it is "
        "available. If the context does not contain enough evidence, say so and "
        "offer the best limited answer you can. Be concise, factual, and identify "
        "relevant missions or source material when helpful."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({
            "role": "system",
            "content": f"Retrieved mission context:\n\n{context}"
        })

    for message in conversation_history[-10:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    base_url = "https://openai.vocareum.com/v1" if openai_key.startswith("voc") else None
    client = OpenAI(api_key=openai_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=700
    )

    return response.choices[0].message.content or ""
