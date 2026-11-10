import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from app.chat_history import SlidingWindowHistory, TokenCounter
from app.config import Settings
from app.llm.client import LLMClient
from app.prompts.support_chat import get_support_chat_prompt
from app.prompts.template import PromptTemplate
from app.services.chat_service import ChatService
from app.services.conversation_summary_service import ConversationSummaryService
from app.services.tool_loop import BoundedToolLoop
from app.tools.default_registry import build_default_tool_registry
from app.tools.execution import ToolExecutor
from evals.chat_checks import run_suite
from evals.chat_models import TurnEvalResult

PROMPT_VERSIONS = (
    "1.0.0",
    "1.1.0",
    "1.2.0",
    "1.3.0",
)
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_chat_service(
    settings: Settings,
    llm_client: LLMClient,
    prompt: PromptTemplate,
) -> tuple[ChatService, ToolExecutor]:
    tool_registry = build_default_tool_registry()
    tool_executor = ToolExecutor(tool_registry)
    tool_loop = BoundedToolLoop(
        client=llm_client,
        registry=tool_registry,
        executor=tool_executor,
        max_steps=settings.tool_max_steps,
    )

    token_counter = TokenCounter()
    history_policy = SlidingWindowHistory(
        token_counter=token_counter,
        input_token_budget=settings.chat_input_token_budget,
    )
    summary_service = ConversationSummaryService(llm_client)

    chat_service = ChatService(
        tool_loop=tool_loop,
        history_policy=history_policy,
        summary_service=summary_service,
        prompt=prompt,
    )
    return chat_service, tool_executor


def save_report(
    settings: Settings,
    prompt: PromptTemplate,
    results: list[TurnEvalResult],
) -> Path:
    created_at = datetime.now(UTC)
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at": created_at.isoformat(),
        "model": settings.model,
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_output_tokens,
        "chat_input_token_budget": settings.chat_input_token_budget,
        "prompt": {
            "prompt_id": prompt.prompt_id,
            "version": prompt.version,
            "body": prompt.body,
        },
        "results": [asdict(result) for result in results],
    }

    report_path = RESULTS_DIR / (
        f"{prompt.prompt_id}-{prompt.version}-{timestamp}.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def count_passed_cases(
    results: list[TurnEvalResult],
) -> tuple[int, int]:
    case_statuses: dict[str, bool] = {}

    for result in results:
        previous_status = case_statuses.get(result.case_id, True)
        case_statuses[result.case_id] = previous_status and result.passed

    passed = sum(case_statuses.values())
    return passed, len(case_statuses)


def print_summary(
    prompt: PromptTemplate,
    results: list[TurnEvalResult],
    report_path: Path,
) -> None:
    passed, total = count_passed_cases(results)
    print(f"\nИтог для {prompt.prompt_id}@{prompt.version}: {passed}/{total}")
    print(f"Отчет: {report_path}")


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return

    llm_client = LLMClient(settings)
    summaries: dict[str, tuple[int, int]] = {}

    try:
        for version in PROMPT_VERSIONS:
            prompt = get_support_chat_prompt(version)
            chat_service, tool_executor = build_chat_service(
                settings=settings,
                llm_client=llm_client,
                prompt=prompt,
            )

            try:
                print(f"\n=== {prompt.prompt_id}@{prompt.version} ===")
                results = run_suite(chat_service)
                report_path = save_report(
                    settings,
                    prompt,
                    results,
                )
                print_summary(prompt, results, report_path)
                summaries[version] = count_passed_cases(results)
            finally:
                tool_executor.close()
    finally:
        llm_client.close()

    print("\nСравнение автоматических проверок:")
    for version, (passed, total) in summaries.items():
        print(f"- {version}: {passed}/{total} сценариев")


if __name__ == "__main__":
    main()
