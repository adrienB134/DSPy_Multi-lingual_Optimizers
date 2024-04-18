import dspy
from signature_translator import SignatureTranslator
from dspy.teleprompt.mipro_optmizer import (
    BasicGenerateInstruction,
    BasicGenerateInstructionWithDataObservations,
    BasicGenerateInstructionWithExamples,
    BasicGenerateInstructionWithExamplesAndDataObservations,
    ObservationSummarizer,
    DatasetDescriptor,
    DatasetDescriptorWithPriorObservations,
)
from typing import Optional
import dsp


class MIPROTranslated(MIPRO):
    def __init__(
        self,
        language,
        translator_lm: Optional[dsp.LM],
        metric,
        prompt_model=None,
        task_model=None,
        teacher_settings={},
        num_candidates=10,
        init_temperature=1.0,
        verbose=False,
        track_stats=True,
        view_data_batch_size=10,
    ):
        translator = SignatureTranslator(language)
        translator.translate_signature(BasicGenerateInstruction)
        translator.translate_signature(BasicGenerateInstructionWithDataObservations)
        translator.translate_signature(BasicGenerateInstructionWithExamples)
        translator.translate_signature(
            BasicGenerateInstructionWithExamplesAndDataObservations
        )
        translator.translate_signature(ObservationSummarizer)
        translator.translate_signature(DatasetDescriptor)
        translator.translate_signature(DatasetDescriptorWithPriorObservations)
        super().__init__(
            self,
            metric,
            prompt_model=prompt_model,
            task_model=task_model,
            teacher_settings=teacher_settings,
            num_candidates=num_candidates,
            init_temperature=init_temperature,
            verbose=verbose,
            track_stats=track_stats,
            view_data_batch_size=view_data_batch_size,
        )
