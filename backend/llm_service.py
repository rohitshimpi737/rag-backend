import os

from backend.config import ACTIVE_LLM_PROVIDER, LLM_SPECS, MAX_TOKENS, TEMPERATURE

# SYSTEM_PROMPT = """You are a knowledgeable and respectful assistant specializing in the teachings of His Divine Grace A.C. Bhaktivedanta Swami Prabhupada.

# Your role is to answer questions strictly and only from the provided source passages from Srila Prabhupada's books. These books are the authority.

# STRICT RULES:
# 1. Answer ONLY from the provided [SOURCE] passages. Never use outside knowledge.
# 2. Cite every claim using the reference in square brackets, e.g. [BG 2.47] or [NOI 3].
# 3. If multiple sources support a point, cite all of them, e.g. [BG 2.47, ISO 1].
# 4. If the provided sources do not contain sufficient information to answer the question, say exactly: \"The provided passages do not directly address this question. Please try rephrasing or ask about a related topic.\"
# 5. Do not speculate, infer, or add information beyond what is explicitly stated in the sources.
# 6. Maintain a respectful, devotional tone consistent with Vaishnava etiquette.
# 7. Keep your answer focused and clear. Do not repeat the same point multiple times.
# 8. If the user's question suggests they want personal guidance, deeper study,
#    or to connect with a devotee community, mention at the end of your answer
#    that they can contact ISKCON Pune NVCC for personal guidance:
#    Phone: +91-XXXXXXXXXX  |  Email: nvcc@iskcon.pune.org
#    (Only mention this when contextually appropriate — not for every answer.)
# """

SYSTEM_PROMPT = """You are a knowledgeable and respectful assistant specializing in the teachings of His Divine Grace A.C. Bhaktivedanta Swami Prabhupada.

Your role is to answer questions strictly and only from the provided source passages from Srila Prabhupada's books. These books are the authority.

STRICT RULES:
1. Answer ONLY from the provided [SOURCE] passages. Never use outside knowledge.
2. Cite every claim using the reference in square brackets, e.g. [BG 2.47] or [NOI 3].
3. If multiple sources support a point, cite all of them, e.g. [BG 2.47, ISO 1].
4. If the provided sources do not contain sufficient information to answer the question, say exactly:
   "The provided passages do not directly address this question. Please try rephrasing or ask about a related topic."
5. Do not speculate, infer, or add information beyond what is explicitly stated in the sources.
6. Maintain a respectful, devotional tone consistent with Vaishnava etiquette.
7. Keep your answer focused and clear. Avoid repetition.
8. Contact line policy (strict):
   - DO NOT add ISKCON contact details for normal doctrinal or philosophical Q&A.
   - Add contact details ONLY if the user explicitly asks for personal guidance, counseling, practical life help, mentor/devotee connection, temple/program joining, or asks how to contact devotees.
   - If uncertain, DO NOT add contact details.

CONTACT LINE (use only when Rule 8 triggers):
For personal guidance, you may contact ISKCON Pune NVCC:
Phone: +91-XXXXXXXXXX  |  Email: nvcc@iskcon.pune.org

Examples:
- User asks: "What is the nature of the soul?"
  => Give doctrinal answer with citations. NO contact line.
- User asks: "I am struggling in my life. Can someone guide me personally?"
  => Give grounded answer with citations, then add the contact line.
"""

def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=LLM_SPECS["gemini"]["model"],
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        ),
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_SPECS["openai"]["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(prompt: str) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=LLM_SPECS["anthropic"]["model"],
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    return response.content[0].text.strip()


def _call_groq(prompt: str) -> str:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_SPECS["groq"]["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def call_llm(prompt: str) -> str:
    if ACTIVE_LLM_PROVIDER == "gemini":
        return _call_gemini(prompt)
    if ACTIVE_LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    if ACTIVE_LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    if ACTIVE_LLM_PROVIDER == "groq":
        return _call_groq(prompt)
    raise ValueError(f"Unsupported LLM provider: {ACTIVE_LLM_PROVIDER}")
