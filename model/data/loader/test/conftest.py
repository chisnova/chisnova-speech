"""pytest 수집 전에 units/ 를 임포트 경로에 올린다.

케이스 파일끼리 `from common import ...` 로 서로를 부르므로 units/ 가 sys.path 에
있어야 한다. pytest 가 파일 디렉토리를 자동으로 넣어 주긴 하지만, 실행 위치나
import 모드에 따라 달라져서 여기서 못박는다.
"""
import os
import sys

_UNITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "units")
if _UNITS_DIR not in sys.path:
    sys.path.insert(0, _UNITS_DIR)
