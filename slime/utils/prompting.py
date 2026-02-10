from __future__ import annotations

MATH_VERIFICATION_USER_PROMPT = (
    "You will be given a math problem and a candidate solution.\n"
    "Your task is to verify whether the candidate solution is correct.\n"
    "You MUST call the code_interpreter tool at least once before giving the final answer.\n"
    "Use the tool to check calculations or edge cases.\n\n"
    "**Problem**\n"
    "{problem}\n\n"
    "**Candidate Solution**\n"
    "{candidate_solution}\n\n"
    "After using the tool, output only one line: \\boxed{{1}} if the solution is correct, otherwise \\boxed{{0}}."
)

MATH_VERIFICATION_USER_PROMPT_PURE = (
    "You will be given a math problem and a candidate solution.\n"
    "Your task is to verify whether the candidate solution is correct.\n"
    "Reason step by step and check calculations carefully.\n\n"
    "**Problem**\n"
    "{problem}\n\n"
    "**Candidate Solution**\n"
    "{candidate_solution}\n\n"
    "Output only one line: \\boxed{{1}} if the solution is correct, otherwise \\boxed{{0}}."
)

CODE_VERIFICATION_USER_PROMPT = (
    "You will be given a programming problem and a candidate Python solution.\n"
    "Your task is to verify whether the candidate solution is correct for all valid inputs.\n"
    "You MUST call the code_interpreter tool at least once before giving the final answer.\n"
    "Use the tool to test edge cases and validate the logic.\n\n"
    "**Problem**\n"
    "{problem}\n\n"
    "**Candidate Python Solution**\n"
    "{candidate_solution}\n\n"
    "After using the tool, output only one line: \\boxed{{1}} if the solution is correct, otherwise \\boxed{{0}}."
)

CODE_VERIFICATION_USER_PROMPT_PURE = (
    "You will be given a programming problem and a candidate Python solution.\n"
    "Your task is to verify whether the candidate solution is correct for all valid inputs.\n"
    "Reason step by step, check corner cases carefully, and do not provide fixes.\n\n"
    "**Problem**\n"
    "{problem}\n\n"
    "**Candidate Python Solution**\n"
    "{candidate_solution}\n\n"
    "Output only one line: \\boxed{{1}} if the solution is correct, otherwise \\boxed{{0}}."
)


def build_math_verification_user_prompt(
    problem: str, candidate_solution: str, *, require_tool: bool = True
) -> str:
    template = MATH_VERIFICATION_USER_PROMPT if require_tool else MATH_VERIFICATION_USER_PROMPT_PURE
    return template.format(
        problem=problem,
        candidate_solution=candidate_solution,
    )


def build_code_verification_user_prompt(
    problem: str, candidate_solution: str, *, require_tool: bool = True
) -> str:
    template = CODE_VERIFICATION_USER_PROMPT if require_tool else CODE_VERIFICATION_USER_PROMPT_PURE
    return template.format(
        problem=problem,
        candidate_solution=candidate_solution,
    )
