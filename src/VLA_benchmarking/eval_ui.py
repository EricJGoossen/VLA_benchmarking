def update_status(status: str) -> None:
    """Update the current status message shown to the user."""
    
    print(status)


def get_score_input(max_step_score=None, max_recall_score=None) -> dict:
    """Get score input from the user after a rollout completes."""
    
    score_results = {}
    score_results["success"] = _prompt_yn("Was the task successful?")
    score_results["step_score"] = _prompt_int("Enter step score", min_val=0, max_val=max_step_score)
    score_results["recall_score"] = _prompt_int("Enter recall score", min_val=0, max_val=max_recall_score)
    score_results["comments"] = input("Any additional comments? ").strip()
    return score_results


def start_rollout(instruction: str, rollout_num: int) -> None:
    """Notify the user that a rollout is starting with the given instruction."""
    
    update_status(f"Starting rollout {rollout_num} with instruction: '{instruction}'")
    input("Press Enter to begin the rollout...")


def start_demo(instruction: str, demo_num: int, num_demos: int) -> None:
    """Notify the operator that a teleop demo is starting with the given instruction."""

    update_status(f"\n=== Demo {demo_num} ({demo_num + 1} of {num_demos}): '{instruction}' ===")
    update_status("  Hold GRIP to move the arm, TRIGGER to close the gripper, click the JOYSTICK to re-align directions.")
    update_status("  When done, press A on the controller to SAVE the demo, or B to DISCARD it and redo it.")
    update_status("  (Ctrl+C stops recording and asks whether to keep the demo.)")
    input("Set up the scene, then press Enter to start recording...")


def confirm_keep_demo(demo_num: int) -> bool:
    """Ask whether to keep a demo whose recording was stopped with Ctrl+C."""

    return _prompt_yn(f"Keep demo {demo_num} as recorded so far?")


def get_test_instruction() -> str:
    """Prompt the user to enter a test instruction for the open-loop test interface."""
    
    instruction = input("Enter instruction for test rollout (or 'exit' to quit): ").strip()
    if instruction.lower() == "exit":
        raise KeyboardInterrupt
    return instruction


def _prompt_int(prompt: str, default=None, min_val=None, max_val=None) -> int:
    """
    Prompt the user for an integer, retrying until valid input is given.

    If default is given, a blank response returns default without validation
    against min_val/max_val.
    """
    while True:
        response = input(f"{prompt}: ").strip()
        if not response and default is not None:
            return default
        try:
            value = int(response)
        except ValueError:
            print(f"  Invalid input '{response}' — please enter a whole number.")
            continue
        if min_val is not None and value < min_val:
            print(f"  Value must be at least {min_val}.")
            continue
        if max_val is not None and value > max_val:
            print(f"  Value must be at most {max_val}.")
            continue
        return value


def _prompt_yn(prompt: str, default=None) -> bool:
    """
    Prompt the user for a yes/no answer, retrying until valid input is given.

    If default is given, a blank response returns default.
    """
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if not response and default is not None:
            return default
        if response in ("y", "n"):
            return response == "y"
        print(f"  Invalid input '{response}' — please enter 'y' or 'n'.")
