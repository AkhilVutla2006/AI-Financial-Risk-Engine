
from ollama import chat


def build_prompt(explanation_data):

    prompt = f"""
You are a financial risk engine.

Predicted probability of debt: {explanation_data['predicted_probability']:.2%}
Final prediction: {"HIGH RISK" if explanation_data['prediction'] == 1 else "LOW RISK"}

Key contributing factors:
"""

    for feature in explanation_data["top_features"]:
        direction = "increased" if feature["impact"] > 0 else "decreased"
        prompt += (
            f"\n- {feature['feature']} = {feature['value']} "
            f"which {direction} the risk."
        )

    prompt += """

Explain this clearly in simple financial terms, specifically mentioning the figures provided above. 
Give practical advice to reduce risk based on these numbers.
Do not mention anything about being an AI.
Keep it brief and conversational, focusing only on the most critical information.
Do not ask any follow-up questions.
"""

    return prompt


def generate_explanation(explanation_data, model_name="gemma3:1b"):

    prompt = build_prompt(explanation_data)

    stream = chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    return stream
