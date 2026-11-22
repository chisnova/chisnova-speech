"""AsrStreamConformer(stream_conformer arch) 를 C++ 런타임용으로 내보내는 진입점.

    PYTHONPATH=$PWD python3 exporter/stream_conformer_cpp_exporter.py stream_conformer

<model_path>_cpp_export/ 폴더를 새로 만들어 두 가지를 쓴다:
  - 텐서별 raw float32 바이너리 — exporter/model_saver.py 의 타입 디스패치 재사용
  - model.json manifest — 모듈 트리 + 추론에 필요한 하이퍼파라미터

추론 파라미터(ctc_time_reduction, stream_left_context, blank/sos/eos id 등)는
학습 config.json 과 모델·추론 래퍼 인스턴스에서 파생한다 — manifest 에
하드코딩된 매직값을 두지 않기 위함. 이 값들은 model 서브트리에 들어가며,
C++ 대응 클래스(cpp/decoder/asr_stream_conformer.{h,cc})가 생성자에서 읽는다.
"""
import argparse
import json
import math
import os

import torch

from app.asr_stream_conformer import AsrStreamConformer
from exporter.model_saver import save_model
from util.inference_wrapper import InferenceWrapper
from util.stream_conformer_inference import StreamConformerInference


class StreamConformerCppExporter:
    """학습된 AsrStreamConformer 체크포인트를 C++ 로딩 포맷으로 변환한다."""

    EXPORT_DIR_SUFFIX = "_cpp_export"
    MANIFEST_NAME = "model.json"
    MANIFEST_FORMAT_VERSION = 1
    CHECKPOINT_NAME = "asr_model.pt"
    LEXICON_DICT_NAME = "full.dict"   # model/data/full.dict — C++ scorer 용 발음사전
    PHONE_LIST_NAME = "phone.txt"
    FUNCTION_WORD_LIST_NAME = "function_words.txt"     # 점수 하한을 적용할 기능어 목록
    PROPER_NOUN_LIST_NAME = "proper_nouns.txt"         # 점수 하한을 적용할 고유명사 목록
    UNGRADED_WORD_LIST_NAME = "ungraded_words.txt"      # 채점 제외 낱말 목록(마지막 수단, 아주 작게 유지)

    def __init__(self, model_path, export_path=None):
        self.model_path = model_path
        self.export_path = (export_path
                            or model_path.rstrip(os.sep) + self.EXPORT_DIR_SUFFIX)
        self.model_config = self.load_model_config()
        self.model = self.load_trained_model()
        # blank/bos/eos 규약을 여기서 하드코딩하지 않고 실제 추론 래퍼에서 읽는다
        self.inference_wrapper = InferenceWrapper(self.model)

    def load_model_config(self):
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path) as config_file:
            return json.load(config_file)["model"]

    def load_trained_model(self):
        model = AsrStreamConformer.from_config(self.model_config)
        checkpoint_path = os.path.join(self.model_path, self.CHECKPOINT_NAME)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return model

    def resolve_vocab_path(self):
        return (self.model_config.get("vocab_path")
                or StreamConformerInference.DEFAULT_VOCAB_PATH)

    def collect_inference_params(self):
        """C++ 추론이 재현해야 하는 파라미터 — 학습 config 와 모델에서 파생한다."""
        return {
            "ctc_time_reduction": self.model.ctc_time_reduction,
            "stream_left_context": self.model_config.get("stream_left_context"),
            "vocab_size": self.model.vocab_size,
            "blank_id": self.inference_wrapper.blank_id,
            "sos_id": self.inference_wrapper.bos_id,
            "eos_id": self.inference_wrapper.eos_id,
        }

    def build_manifest(self, module_tree):
        recognition_weight = float(self.model_config.get("recognition_weight", 0.5))
        if not math.isfinite(recognition_weight) or not 0.0 <= recognition_weight <= 1.0:
            raise ValueError("recognition_weight must be finite and between 0 and 1")
        proper_noun_floor = float(self.model_config.get("proper_noun_floor", 90.0))
        if not math.isfinite(proper_noun_floor) or not 0.0 <= proper_noun_floor <= 100.0:
            raise ValueError("proper_noun_floor must be finite and between 0 and 100")
        vocab_path = self.resolve_vocab_path()
        vocab_file_stem = os.path.splitext(os.path.basename(vocab_path))[0]
        phone_vocab_size = self.model.phn_decoder.tout_emb.out_features
        return {"model_path": {
            "format_version": self.MANIFEST_FORMAT_VERSION,
            "dtype": "float32",
            "model_type": self.model_config["arch"],
            "vocab_name": os.path.basename(vocab_path),
            "vocab_model": vocab_file_stem + ".model",
            "vocab_size": self.model.vocab_size,
            "phone_size": phone_vocab_size,
            "phone_name": self.PHONE_LIST_NAME,
            "dict_name": self.LEXICON_DICT_NAME,
            # 발음 점수를 낼 때 음소 점수와 낱말 점수를 섞는 비율(음소 쪽 몫).
            # C++ Scorer::mergeScore 가 읽는다. 학습 config 에 없으면 종전 값 0.6.
            # 이 항목이 없는 옛 model_state 도 엔진이 기본값으로 처리한다.
            "phone_weight": float(self.model_config.get("phone_weight", 0.6)),
            # 기능어(관사·전치사·be동사 …) 점수의 하한. 0 이면 적용하지 않는다.
            # 강세 박자 언어에서 이런 낱말은 약해지는 것이 정상이라, 또박또박 읽지
            # 않았다고 감점하면 실제 발음 습관과 어긋난다. 고유명사 하한과 같은 90 을 쓴다
            # (2026-09 콘텐츠 검수: 85 였을 때 정상 약형 14개가 검토 목록에 남았다).
            "function_word_list": self.FUNCTION_WORD_LIST_NAME,
            "function_word_floor": float(self.model_config.get("function_word_floor", 90.0)),
            # 고유명사(지명·인명·외래어) 점수의 하한. 0 이면 적용하지 않는다.
            # 채점 모델이 학습에서 거의 보지 못한 낱말은 참조 음원조차 낮게 나오므로,
            # 원점수가 하한의 절반 이상일 때만 이 값으로 받쳐 준다(C++ Scorer::mergeScore).
            "proper_noun_list": self.PROPER_NOUN_LIST_NAME,
            "proper_noun_floor": proper_noun_floor,
            # 채점 제외 낱말 목록. 목록의 낱말은 점수를 남기되 문장 평균·최저 낱말에서 뺀다.
            # 모델이 학습으로도 배우지 못한 형태(문자명 "AA")에만 쓰는 마지막 수단이다.
            "ungraded_word_list": self.UNGRADED_WORD_LIST_NAME,
            # 2패스 인식 결과가 타겟과 어긋난 만큼 총점을 깎는 정도(0~1). 0 이면 끈다.
            "recognition_weight": recognition_weight,
            "model": dict(module_tree, **self.collect_inference_params()),
        }}

    def export(self):
        os.makedirs(self.export_path, exist_ok=True)
        module_tree = save_model(self.model, self.model_config["arch"],
                                 self.export_path)
        manifest = self.build_manifest(module_tree)
        manifest_path = os.path.join(self.export_path, self.MANIFEST_NAME)
        with open(manifest_path, "w") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent="\t")
        exported_file_count = len(os.listdir(self.export_path))
        print("[cpp-export] {} -> {} (파일 {:,}개, manifest: {})".format(
            self.model_path, self.export_path, exported_file_count, manifest_path))
        return manifest_path


def parseArguments():
    parser = argparse.ArgumentParser(
        description="AsrStreamConformer 체크포인트를 C++ 로딩 포맷으로 export")
    parser.add_argument("model_path", type=str,
                        help="config.json 과 asr_model.pt 가 있는 모델 폴더")
    parser.add_argument("--export_path", type=str, default=None,
                        help="출력 폴더 (기본: <model_path>_cpp_export)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArguments()
    exporter = StreamConformerCppExporter(args.model_path,
                                          export_path=args.export_path)
    exporter.export()
