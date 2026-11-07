"""OpenAI 기반 PDF 분석. 긴 문서는 청크별로 핵심을 뽑은 뒤(map) 통합 분석(reduce)한다."""
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import OpenAI

ANALYSIS_TYPES = {
    "summary": "전체 요약",
    "key_points": "핵심 포인트",
    "qa": "자유 질문",
    "compare": "파일 간 비교",
}

MAX_CHARS_PER_FILE = 150_000  # 파일당 처리 상한 (초과분은 잘라냄)
MAX_CHARS_TOTAL = 300_000  # 전체 처리 상한
DIRECT_LIMIT = 40_000  # 이 이하면 청크 분할 없이 바로 분석
CHUNK_SIZE = 12_000
MAX_WORKERS = 5  # 청크 병렬 호출 수

SYSTEM_BASE = (
    "당신은 PDF 문서 분석 도우미입니다. 반드시 제공된 문서 내용만 근거로 한국어로 답하세요. "
    "문서에 없는 내용은 추측하지 말고 '문서에 없음'이라고 말하세요. "
    "가능하면 근거가 되는 파일명과 페이지 표시(예: [p.3])를 함께 적으세요. "
    "마크다운 제목은 ### 이하 수준만 사용하고 #, ##는 쓰지 마세요."
)

TASK_PROMPTS = {
    "summary": "아래 문서 전체를 읽고 핵심 내용을 구조적으로 요약하세요. 한 문단 개요 뒤에 주제별 요약을 마크다운으로 정리하세요.",
    "key_points": "아래 문서의 핵심 포인트를 중요도 순으로 불릿 리스트로 정리하세요. 수치, 날짜, 조건 등 구체적 정보를 포함하세요.",
    "qa": "아래 문서를 근거로 다음 질문에 답하세요.\n\n질문: {question}",
    "compare": "아래 여러 문서를 비교하세요. '공통점', '차이점', '종합 의견' 순서의 마크다운으로 정리하세요.",
}


class AnalysisError(Exception):
    """사용자에게 그대로 보여줘도 되는 오류 메시지를 담는다."""


def friendly_error(e: Exception) -> AnalysisError:
    if isinstance(e, openai.AuthenticationError):
        return AnalysisError("API 키가 올바르지 않습니다. OPENAI_API_KEY를 확인하세요.")
    if isinstance(e, openai.PermissionDeniedError):
        return AnalysisError("이 API 키로는 해당 모델/기능에 접근할 수 없습니다.")
    if isinstance(e, openai.NotFoundError):
        return AnalysisError("모델을 찾을 수 없습니다. OPENAI_MODEL 값이 올바른지, 계정에서 사용 가능한 모델인지 확인하세요.")
    if isinstance(e, openai.RateLimitError):
        return AnalysisError("요청 한도를 초과했거나 크레딧이 부족합니다. 잠시 후 다시 시도하세요.")
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return AnalysisError("OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인하고 다시 시도하세요.")
    if isinstance(e, openai.BadRequestError):
        return AnalysisError(f"요청이 거부되었습니다(문서가 너무 길거나 잘못된 요청): {getattr(e, 'message', e)}")
    if isinstance(e, openai.APIStatusError):
        return AnalysisError(f"OpenAI 서버 오류가 발생했습니다 (HTTP {e.status_code}). 잠시 후 다시 시도하세요.")
    return AnalysisError(f"분석 중 알 수 없는 오류가 발생했습니다: {type(e).__name__}")


def apply_limits(documents: dict[str, list[str]]) -> tuple[dict[str, str], list[str]]:
    """페이지별 텍스트를 [p.N] 표시가 붙은 문자열로 합치고 길이 상한을 적용한다."""
    warnings: list[str] = []
    out: dict[str, str] = {}
    total = 0
    for name, pages in documents.items():
        text = "\n".join(f"[p.{i}] {t}" for i, t in enumerate(pages, 1) if t)
        if len(text) > MAX_CHARS_PER_FILE:
            text = text[:MAX_CHARS_PER_FILE]
            warnings.append(f"'{name}'이(가) 길어 앞부분 {MAX_CHARS_PER_FILE:,}자만 분석합니다.")
        remaining = MAX_CHARS_TOTAL - total
        if remaining <= 0:
            warnings.append(f"전체 분량 상한({MAX_CHARS_TOTAL:,}자)을 넘어 '{name}'은(는) 분석에서 제외했습니다.")
            continue
        if len(text) > remaining:
            text = text[:remaining]
            warnings.append(f"전체 분량 상한 때문에 '{name}'은(는) 일부만 분석합니다.")
        total += len(text)
        out[name] = text
    return out, warnings


def _chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    parts, buf = [], ""
    for line in text.split("\n"):
        while len(line) > size:  # 한 줄이 너무 길면 강제 분할
            if buf:
                parts.append(buf)
                buf = ""
            parts.append(line[:size])
            line = line[size:]
        if len(buf) + len(line) + 1 > size and buf:
            parts.append(buf)
            buf = ""
        buf += line + "\n"
    if buf.strip():
        parts.append(buf)
    return parts


def _complete(client: OpenAI, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""


def _map_chunk(client, model, name, chunk, focus) -> str:
    return _complete(
        client,
        model,
        SYSTEM_BASE,
        f"다음은 문서 '{name}'의 일부입니다. {focus}을 빠짐없이 간결한 노트로 추출하세요. "
        f"관련 내용이 없으면 '해당 없음'이라고만 쓰세요. [p.N] 표시는 유지하세요.\n\n{chunk}",
    )


def _condense_all(client, model, documents, analysis_type, question, progress) -> dict[str, str]:
    """긴 문서를 청크로 나눠 병렬로 핵심만 뽑는다(map). 짧은 문서는 그대로 둔다."""
    focus = f"질문 '{question}'과 관련된 내용" if analysis_type == "qa" else "핵심 내용(수치·날짜·조건 포함)"
    tasks = [
        (name, i, chunk)
        for name, text in documents.items()
        if len(text) > DIRECT_LIMIT
        for i, chunk in enumerate(_chunks(text))
    ]
    notes: dict[tuple[str, int], str] = {}
    if tasks:
        ex = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        try:
            futures = {ex.submit(_map_chunk, client, model, n, c, focus): (n, i) for n, i, c in tasks}
            for done, f in enumerate(as_completed(futures), 1):
                notes[futures[f]] = f.result()
                if progress:  # 진행 표시는 호출 스레드(메인)에서만 한다
                    progress(f"문서를 나눠 분석하는 중… ({done}/{len(tasks)})")
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    out = {}
    for name, text in documents.items():
        if len(text) > DIRECT_LIMIT:
            idx = sorted(i for (n, i) in notes if n == name)
            out[name] = "\n".join(notes[(name, i)] for i in idx)
        else:
            out[name] = text
    return out


def analyze(
    client: OpenAI,
    model: str,
    analysis_type: str,
    documents: dict[str, str],
    question: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """응답 텍스트 조각을 yield 한다. 실패 시 AnalysisError를 raise 한다."""
    if analysis_type == "qa" and not (question or "").strip():
        raise AnalysisError("질문을 입력해 주세요.")
    if analysis_type == "compare" and len(documents) < 2:
        raise AnalysisError("파일 간 비교는 PDF가 2개 이상 필요합니다.")
    if not documents:
        raise AnalysisError("분석할 문서가 없습니다.")

    try:
        condensed = _condense_all(client, model, documents, analysis_type, question, progress)
        body = "\n\n".join(f"=== 파일: {n} ===\n{t}" for n, t in condensed.items())
        task = TASK_PROMPTS[analysis_type].format(question=(question or "").strip())
        if progress:
            progress("답변 생성 중…")
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_BASE},
                {"role": "user", "content": f"{task}\n\n{body}"},
            ],
            stream=True,
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content
    except AnalysisError:
        raise
    except Exception as e:
        raise friendly_error(e) from e
