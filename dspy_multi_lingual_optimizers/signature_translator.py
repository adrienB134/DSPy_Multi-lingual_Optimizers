import os

import dsp
import dspy

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

lm = dspy.OpenAI(
    model="gpt-3.5-turbo",
    max_tokens=3800,
    api_key=os.getenv("OPENAI_KEY"),
)


class TranslatorSignature(dspy.Signature):
    """Translate the text input in the target language"""

    text_input = dspy.InputField(desc="text to translate")
    target_language = dspy.InputField()
    output = dspy.OutputField(desc="just the translation")


class TranslatorModule(dspy.Module):
    def __init__(self, language: str) -> None:
        self.language = language
        self.translator = dspy.Predict(TranslatorSignature)

    def forward(self, text_input: str) -> str:
        return self.translator(
            text_input=text_input, target_language=self.language
        ).output


class SignatureTranslator:
    def __init__(self, target_language: str, lm: dsp.LM = lm):
        self.lm = lm
        self.lm.drop_prompt_from_output = True
        dspy.settings.configure(lm=lm)
        self.translator = TranslatorModule(target_language)

    def translate_signature(self, signature: dspy.Signature) -> dspy.Signature:
        for field in signature.fields:
            text = signature.model_fields[f"{field}"].json_schema_extra["desc"]
            if not text[0] == "$":
                text = self.translator(text)
            signature.model_fields[f"{field}"].json_schema_extra["desc"] = text
            signature.__doc__ = self.translator(signature.instructions)
        return signature
