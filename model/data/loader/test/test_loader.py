#!/usr/bin/env python3
"""data.loader 패키지 동작 확인 — 실제 DB 를 읽어 값을 눈으로 보고 검증한다.

케이스마다 파일이 하나(units/)이고, 파일 안은 step 함수로 나뉜다.

  units/common.py            경로/검증집계/출력/로더생성 (케이스가 맨 먼저 임포트)
  units/case_01_manifest.py  카탈로그 길이 분포 / 샤드 분포 / 필드 정합성
  units/case_02_record.py    저장소 원본 레코드 몇 개를 눈으로 확인
  units/case_03_sample.py    무작위 표본을 읽어 통계로 이상 탐지
  units/case_04_batch.py     배치 텐서 모양·dtype·마스크 경계·패딩
  units/case_05_shuffle.py   seed 재현성 / 커버리지 / 에폭 재셔플 / chunk 정렬 효과
  units/case_06_augment.py   FeatureTransform 적용 전/후와 패딩 보존
  units/case_07_worker.py    num_workers 0 vs N 결과 일치 + 속도

사용 예
  python3 test_loader.py                                   # dev, 전체 케이스
  python3 test_loader.py --case batch
  python3 test_loader.py --db /home/chisnova/data/records/train --usage_ratio 0.001
  python3 units/case_04_batch.py --batch_size 32           # 케이스 단독 실행
"""
import os
import sys

# 케이스 파일은 units/ 에 있다 — 임포트 전에 경로를 잡아 준다.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "units"))

import common          # sys.path 를 여기서 정리한다 — 반드시 먼저 임포트

import case_01_manifest
import case_02_record
import case_03_sample
import case_04_batch
import case_05_shuffle
import case_06_augment
import case_07_worker

CASES = {
    "manifest": case_01_manifest.run,
    "record": case_02_record.run,
    "sample": case_03_sample.run,
    "batch": case_04_batch.run,
    "shuffle": case_05_shuffle.run,
    "augment": case_06_augment.run,
    "worker": case_07_worker.run,
}


def main():
    args = common.build_arg_parser(CASES, description=__doc__).parse_args()
    sp_processor = common.load_sentencepiece(args.vocab)
    loader = common.build_loader(args, sp_processor)
    try:
        for case_name in (list(CASES) if args.case == "all" else [args.case]):
            CASES[case_name](loader, args, sp_processor)
    finally:
        loader.close()
    return common.summarize()


if __name__ == "__main__":
    sys.exit(main())
