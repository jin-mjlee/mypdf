"""설정 로딩: st.secrets -> 환경변수(.env) 순서."""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gpt-6.1-terra"


def _get(key: str) -> str | None:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        # secrets.toml이 없으면 예외가 날 수 있다
        pass
    return os.getenv(key) or None


def get_api_key() -> str | None:
    return _get("OPENAI_API_KEY")


def get_model() -> str:
    return _get("OPENAI_MODEL") or DEFAULT_MODEL


def get_setting(key: str, default: str | None = None) -> str | None:
    return _get(key) or default
