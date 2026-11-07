"""PDF 텍스트 추출. 파일은 디스크에 저장하지 않고 메모리에서만 처리한다."""
from dataclasses import dataclass, field
from io import BytesIO

from pypdf import PdfReader


@dataclass
class PdfResult:
    name: str
    pages: list[str] = field(default_factory=list)
    page_count: int = 0
    warning: str | None = None  # 값이 있으면 사용자에게 보여줄 경고/오류
    ok: bool = False  # 텍스트를 하나라도 추출했는지

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)


def read_pdf(name: str, data: bytes) -> PdfResult:
    """PDF 바이트에서 페이지별 텍스트를 추출한다. 예외를 던지지 않는다."""
    result = PdfResult(name=name)
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            # 빈 비밀번호로 열리는 PDF만 지원
            if reader.decrypt("") == 0:
                result.warning = "암호로 보호된 PDF라 읽을 수 없습니다."
                return result
        result.page_count = len(reader.pages)
        for page in reader.pages:
            try:
                result.pages.append((page.extract_text() or "").strip())
            except Exception:
                result.pages.append("")
    except Exception:
        result.warning = "PDF를 읽을 수 없습니다. 손상되었거나 PDF 형식이 아닐 수 있습니다."
        result.pages = []
        result.page_count = 0
        return result

    if not result.text.strip():
        result.warning = "텍스트를 추출할 수 없습니다. 스캔본(이미지) PDF일 수 있으며 현재는 지원하지 않습니다."
        return result

    empty = sum(1 for t in result.pages if not t)
    if empty:
        result.warning = f"{empty}쪽은 텍스트가 없어(이미지 등) 분석에서 제외됩니다."
    result.ok = True
    return result
