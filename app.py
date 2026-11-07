from datetime import datetime

import streamlit as st
from openai import OpenAI

from utils.analyzer import ANALYSIS_TYPES, AnalysisError, analyze, apply_limits
from utils.config import get_api_key, get_model
from utils.guard import daily_limit, require_password, try_consume
from utils.pdf_reader import read_pdf

st.set_page_config(page_title="PDF 분석", page_icon="📄", layout="wide")

# 파일 크기(20MB)는 .streamlit/config.toml의 server.maxUploadSize가 제한한다.
MAX_FILE_MB = 20

require_password()

# 모델이 만든 마크다운 제목이 페이지 제목보다 커지지 않도록 결과 영역 제목만 축소
st.markdown(
    """
    <style>
    [data-testid="stTabs"] h1 { font-size: 1.5rem; }
    [data-testid="stTabs"] h2 { font-size: 1.3rem; }
    [data-testid="stTabs"] h3 { font-size: 1.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

api_key = get_api_key()
model = get_model()
st.session_state.setdefault("history", [])

st.title("📄 PDF 분석")
st.caption("PDF를 올리고 분석 유형을 고르면, AI가 문서를 읽고 요약·질의응답·비교를 해 드립니다.")

with st.sidebar:
    st.header("ℹ️ 정보")
    st.caption(f"모델: `{model}`")
    if not api_key:
        st.warning("OPENAI_API_KEY가 없습니다. 로컬은 .env, 배포는 Streamlit Secrets에 설정하세요.")
    st.caption("업로드한 문서의 내용은 분석을 위해 OpenAI API로 전송됩니다.")


# 개인정보 보호를 위해 추출 결과 캐시는 1시간, 최대 20개까지만 유지한다
@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def _extract(name: str, data: bytes):
    return read_pdf(name, data)


def _unique_names(names: list[str]) -> list[str]:
    """같은 이름의 파일이 여러 개면 (2), (3)을 붙여 구분한다."""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


# ---------------------------------------------------------------- 1. 업로드
st.subheader("1. 파일 업로드")
files = st.file_uploader(
    f"PDF 파일을 끌어다 놓거나 선택하세요 (파일당 최대 {MAX_FILE_MB}MB)",
    type=["pdf"],
    accept_multiple_files=True,
)

results = []  # 분석 가능한 파일
if not files:
    with st.container(border=True):
        st.markdown("**아직 업로드된 파일이 없습니다.**")
        st.markdown(
            "1. 위에서 PDF를 업로드합니다\n"
            "2. 아래에서 분석 유형을 고릅니다\n"
            "3. **분석 시작**을 누릅니다"
        )
else:
    labels = _unique_names([f.name for f in files])
    for f, label in zip(files, labels):
        data = f.getvalue()
        with st.container(border=True):
            head, badge = st.columns([5, 1], vertical_alignment="center")
            head.markdown(f"**{label}**")
            res = _extract(label, data)
            if res.ok:
                badge.badge("분석 가능", color="green")
            elif res.page_count:
                badge.badge("텍스트 없음", color="orange")
            else:
                badge.badge("읽기 실패", color="red")
            st.caption(f"📑 {res.page_count:,}쪽 · 🔤 {len(res.text):,}자 · 💾 {len(data) / 1024 / 1024:.1f} MB")
            if res.warning:
                st.warning(res.warning)
            if res.ok:
                results.append(res)
                with st.expander("추출 텍스트 미리보기"):
                    preview = res.text[:3000]
                    st.text(preview + ("\n…" if len(res.text) > 3000 else ""))

# ---------------------------------------------------------------- 2. 분석
st.subheader("2. 분석")
types = [k for k in ANALYSIS_TYPES if k != "compare" or len(results) >= 2]
atype = st.radio("분석 유형", types, format_func=ANALYSIS_TYPES.get, horizontal=True)
if len(results) < 2:
    st.caption("파일 간 비교는 분석 가능한 PDF가 2개 이상일 때 선택할 수 있습니다.")
question = ""
if atype == "qa":
    question = st.text_area("질문", placeholder="문서에 대해 궁금한 점을 입력하세요")

disabled = not results or not api_key or (atype == "qa" and not question.strip())
if not results:
    hint = "PDF를 업로드하면 분석할 수 있습니다."
elif not api_key:
    hint = "API 키를 설정하면 분석할 수 있습니다."
elif atype == "qa" and not question.strip():
    hint = "질문을 입력하면 분석을 시작할 수 있습니다."
else:
    hint = f"{ANALYSIS_TYPES[atype]} · 파일 {len(results)}개"
run_col, hint_col = st.columns([1, 3], vertical_alignment="center")
run = run_col.button("분석 시작", type="primary", disabled=disabled, use_container_width=True)
hint_col.caption(hint)

tab_result, tab_history = st.tabs(["분석 결과", "히스토리"])
output = tab_result.container()


def _save(text: str, docs: dict, note: str = ""):
    st.session_state.history.append(
        {
            "type": ANALYSIS_TYPES[atype] + note,
            "question": question.strip() if atype == "qa" else "",
            "files": list(docs),
            "result": text,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )


if run:
    with output:
        if not try_consume():
            st.error(f"오늘의 분석 횟수 한도({daily_limit()}회)를 모두 사용했습니다. 내일 다시 시도해 주세요.", icon="🚫")
        else:
            docs, warns = apply_limits({r.name: r.pages for r in results})
            for w in warns:
                st.warning(w)
            status = st.empty()
            collected: list[str] = []
            try:
                client = OpenAI(api_key=api_key)
                gen = analyze(client, model, atype, docs, question, progress=lambda m: status.info(m, icon="⏳"))
                # 첫 조각이 올 때까지(map 단계 포함) status가 진행 상황을 보여준다
                first = next(gen, None)
                status.empty()
                if first is None:
                    st.error("모델이 빈 응답을 반환했습니다. 다시 시도해 주세요.")
                else:
                    st.markdown(f"##### {ANALYSIS_TYPES[atype]}")

                    def _all():
                        collected.append(first)
                        yield first
                        for piece in gen:
                            collected.append(piece)
                            yield piece

                    st.write_stream(_all())
                    _save("".join(collected), docs)
            except AnalysisError as e:
                status.empty()
                st.error(str(e), icon="🚫")
                if collected:  # 스트리밍 도중 실패: 받은 부분까지 보관
                    _save("".join(collected), docs, " (중단됨)")
            except Exception as e:
                status.empty()
                st.error(f"예상치 못한 오류가 발생했습니다: {type(e).__name__}", icon="🚫")
                if collected:
                    _save("".join(collected), docs, " (중단됨)")
elif st.session_state.history:
    last = st.session_state.history[-1]
    with output:
        st.markdown(f"##### {last['type']}")
        st.caption(f"{last['time']} · {', '.join(last['files'])}")
        st.markdown(last["result"])
else:
    with output:
        st.info("분석 결과가 여기에 표시됩니다.", icon="💡")

# ---------------------------------------------------------------- 결과 다운로드 / 히스토리
if st.session_state.history:
    last = st.session_state.history[-1]
    md = f"# {last['type']}\n\n- 파일: {', '.join(last['files'])}\n"
    if last["question"]:
        md += f"- 질문: {last['question']}\n"
    md += f"\n{last['result']}\n"
    with tab_result:
        st.download_button("⬇️ 결과 다운로드 (.md)", md, file_name="analysis.md", mime="text/markdown")
    with tab_history:
        for h in reversed(st.session_state.history):
            with st.container(border=True):
                st.markdown(f"**{h['type']}** · {h['time']}")
                st.caption(", ".join(h["files"]))
                if h["question"]:
                    st.markdown(f"질문: {h['question']}")
                with st.expander("결과 보기"):
                    st.markdown(h["result"])
else:
    with tab_history:
        st.info("아직 분석 기록이 없습니다.", icon="🕘")
