"""공개 배포용 보호장치: 선택적 비밀번호 + 일일 분석 횟수 제한."""
import hmac
import threading
from datetime import date

import streamlit as st

from utils.config import get_setting

DEFAULT_DAILY_LIMIT = 100


@st.cache_resource
def _usage() -> dict:
    return {"day": date.today(), "count": 0, "lock": threading.Lock()}


def daily_limit() -> int:
    try:
        return int(get_setting("DAILY_LIMIT", str(DEFAULT_DAILY_LIMIT)))
    except ValueError:
        return DEFAULT_DAILY_LIMIT


def try_consume() -> bool:
    """오늘 사용 가능 횟수가 남아 있으면 1회 차감하고 True. 서버 재시작 시 초기화된다."""
    u = _usage()
    with u["lock"]:
        if u["day"] != date.today():
            u["day"], u["count"] = date.today(), 0
        if u["count"] >= daily_limit():
            return False
        u["count"] += 1
        return True


def require_password() -> None:
    """APP_PASSWORD가 설정된 경우에만 비밀번호를 요구한다. 통과하지 못하면 앱을 중단한다."""
    expected = get_setting("APP_PASSWORD")
    if not expected or st.session_state.get("authed"):
        return
    st.title("📄 PDF 분석")
    pw = st.text_input("비밀번호", type="password")
    if pw:
        if hmac.compare_digest(pw.encode(), expected.encode()):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
    st.stop()
