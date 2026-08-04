from app.onboarding import (
    COMPANIES_DIR,
    OnboardingIntake,
    generate_company_context,
    slugify,
    unique_company_slug,
    write_company_context,
)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, messages):
        return _FakeResponse(self._content)


def test_slugify_lowercases_and_replaces_specials():
    assert slugify("Acme Corp!") == "acme-corp"
    assert slugify("  Тест Компания  ") == "тест-компания"


def test_slugify_empty_falls_back_to_company():
    assert slugify("!!!") == "company"


def test_unique_company_slug_appends_suffix_on_collision(tmp_path, monkeypatch):
    monkeypatch.setattr("app.onboarding.COMPANIES_DIR", tmp_path)
    (tmp_path / "acme").mkdir()
    (tmp_path / "acme-2").mkdir()

    assert unique_company_slug("Acme") == "acme-3"


def test_unique_company_slug_no_collision(tmp_path, monkeypatch):
    monkeypatch.setattr("app.onboarding.COMPANIES_DIR", tmp_path)

    assert unique_company_slug("Fresh Co") == "fresh-co"


def test_generate_company_context_fills_three_sections():
    intake = OnboardingIntake(
        company_name="Acme",
        product_description="B2B SaaS platform",
        business_model="B2B",
        key_metrics="activation, retention",
        chat_notes=["Long sales cycle, ~3 months"],
    )
    llm = _FakeLLM("Продукт: SaaS\n\nМетрики: activation\n\nСпецифика: long cycle")

    result = generate_company_context(intake, llm)

    assert "## Продукт" in result
    assert "## Ключевые метрики" in result
    assert "## Специфика домена" in result
    assert "SaaS" in result


def test_generate_company_context_pads_missing_sections():
    intake = OnboardingIntake(company_name="Acme")
    llm = _FakeLLM("Только один абзац")

    result = generate_company_context(intake, llm)

    assert result.count("не указано") == 2


def test_write_company_context_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.onboarding.COMPANIES_DIR", tmp_path)

    path = write_company_context("acme", "## Продукт\n\ntext\n")

    assert path == tmp_path / "acme" / "company_context.md"
    assert path.read_text(encoding="utf-8") == "## Продукт\n\ntext\n"
