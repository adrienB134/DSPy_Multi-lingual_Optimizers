import dspy


class Test(dspy.Signature):
    """you are a signature my bro"""

    text_input = dspy.InputField(desc="thing to translate")
    target_language = dspy.InputField()
    reponse = dspy.OutputField(desc="translated text")


class MIPRO(dspy.Module):
    def __init__(self):
        self.test = Test

    def forward(self):
        print(self.test)
