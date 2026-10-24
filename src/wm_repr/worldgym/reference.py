# The real-world success counts are those reported in the WorldGym paper (arXiv 2506.00613).
"""WorldGym reference success counts and compound-task aggregation."""

TRIALS_PER_TASK = 10

REAL_WORLD_OPENVLA = {
    # ood/easy
    "ood/easy/put_eggplant_into_pot_or_pan--easy_robot_init_pos": 10,
    "ood/easy/put_eggplant_into_pot_or_pan": 10,
    "ood/easy/put_cup_from_counter_or_drying_rack_into_sink": 7,
    "ood/easy/put_carrot_on_plate": 8,
    # ood/hard/visual_generalization
    "ood/hard/visual_generalization/put_eggplant_into_pot_or_pan--with_clutter": 7.5,
    "ood/hard/visual_generalization/put_yellow_corn_on_pink_plate--with_clutter": 9,
    # ood/hard/motion_generalization
    "ood/hard/motion_generalization/lift_eggplant--extreme_init_pos": 7.5,
    "ood/hard/motion_generalization/put_carrot_on_plate--table_height": 4.5,
    # ood/hard/physical_generalization
    "ood/hard/physical_generalization/flip_pot_upright_which_is_in_sink": 8,
    "ood/hard/physical_generalization/lift_AAA_battery": 7,
    # ood/hard/semantic_generalization
    "ood/hard/semantic_generalization/lift_white_tape": 1,
    "ood/hard/semantic_generalization/move_skull_into_drying_rack": 5,
    "ood/hard/semantic_generalization/take_purple_grapes_out_of_pot": 4,
    "ood/hard/semantic_generalization/stack_blue_cup_on_pink_cup": 4.5,
    # language_conditioning (compound tasks — the WorldGym paper reports jointly, we split to 2 keys)
    "language_conditioning/put_eggplant_or_red_bottle_into_pot/put_eggplant_into_pot": 7.5,
    "language_conditioning/put_eggplant_or_red_bottle_into_pot/put_red_bottle_into_pot": 7.5,
    "language_conditioning/lift_red_chili_pepper_or_cheese/lift_cheese": 10,
    "language_conditioning/lift_red_chili_pepper_or_cheese/lift_red_chili_pepper": 10,
    "language_conditioning/put_blue_cup_or_pink_cup_on_plate/put_blue_cup_on_plate": 9.5,
    "language_conditioning/put_blue_cup_or_pink_cup_on_plate/put_pink_cup_on_plate": 9.5,
}

REAL_WORLD_OCTO = {
    "ood/easy/put_eggplant_into_pot_or_pan--easy_robot_init_pos": 5,
    "ood/easy/put_eggplant_into_pot_or_pan": 1,
    "ood/easy/put_cup_from_counter_or_drying_rack_into_sink": 1,
    "ood/easy/put_carrot_on_plate": 0,
    "ood/hard/visual_generalization/put_eggplant_into_pot_or_pan--with_clutter": 3.5,
    "ood/hard/visual_generalization/put_yellow_corn_on_pink_plate--with_clutter": 4,
    "ood/hard/motion_generalization/lift_eggplant--extreme_init_pos": 0.5,
    "ood/hard/motion_generalization/put_carrot_on_plate--table_height": 1,
    "ood/hard/physical_generalization/flip_pot_upright_which_is_in_sink": 6,
    "ood/hard/physical_generalization/lift_AAA_battery": 0,
    "ood/hard/semantic_generalization/lift_white_tape": 0,
    "ood/hard/semantic_generalization/move_skull_into_drying_rack": 0,
    "ood/hard/semantic_generalization/take_purple_grapes_out_of_pot": 0,
    "ood/hard/semantic_generalization/stack_blue_cup_on_pink_cup": 0,
    "language_conditioning/put_eggplant_or_red_bottle_into_pot/put_eggplant_into_pot": 4,
    "language_conditioning/put_eggplant_or_red_bottle_into_pot/put_red_bottle_into_pot": 4,
    "language_conditioning/lift_red_chili_pepper_or_cheese/lift_cheese": 2.5,
    "language_conditioning/lift_red_chili_pepper_or_cheese/lift_red_chili_pepper": 2.5,
    "language_conditioning/put_blue_cup_or_pink_cup_on_plate/put_blue_cup_on_plate": 5.5,
    "language_conditioning/put_blue_cup_or_pink_cup_on_plate/put_pink_cup_on_plate": 5.5,
}

REAL_WORLD_RT1 = {
    "ood/easy/put_eggplant_into_pot_or_pan--easy_robot_init_pos": 1,
    "ood/easy/put_eggplant_into_pot_or_pan": 0,
    "ood/easy/put_cup_from_counter_or_drying_rack_into_sink": 1,
    "ood/easy/put_carrot_on_plate": 1,
    "ood/hard/visual_generalization/put_eggplant_into_pot_or_pan--with_clutter": 1,
    "ood/hard/visual_generalization/put_yellow_corn_on_pink_plate--with_clutter": 1,
    "ood/hard/motion_generalization/lift_eggplant--extreme_init_pos": 3,
    "ood/hard/motion_generalization/put_carrot_on_plate--table_height": 2,
    "ood/hard/physical_generalization/flip_pot_upright_which_is_in_sink": 2,
    "ood/hard/physical_generalization/lift_AAA_battery": 0,
    "ood/hard/semantic_generalization/lift_white_tape": 3,
    "ood/hard/semantic_generalization/move_skull_into_drying_rack": 1,
    "ood/hard/semantic_generalization/take_purple_grapes_out_of_pot": 6,
    "ood/hard/semantic_generalization/stack_blue_cup_on_pink_cup": 0.5,
    "language_conditioning/put_eggplant_or_red_bottle_into_pot/put_eggplant_into_pot": 2.5,
    "language_conditioning/put_eggplant_or_red_bottle_into_pot/put_red_bottle_into_pot": 2.5,
    "language_conditioning/lift_red_chili_pepper_or_cheese/lift_cheese": 1.5,
    "language_conditioning/lift_red_chili_pepper_or_cheese/lift_red_chili_pepper": 1.5,
    "language_conditioning/put_blue_cup_or_pink_cup_on_plate/put_blue_cup_on_plate": 5,
    "language_conditioning/put_blue_cup_or_pink_cup_on_plate/put_pink_cup_on_plate": 5,
}

REAL_WORLD = {
    "openvla": REAL_WORLD_OPENVLA,
    "octo": REAL_WORLD_OCTO,
    "rt1": REAL_WORLD_RT1,
}

COMPOUND_TASKS = {
    "language_conditioning/put_eggplant_or_red_bottle_into_pot/put_eggplant_into_pot":
        "Put {Eggplant, Red Bottle} into Pot",
    "language_conditioning/put_eggplant_or_red_bottle_into_pot/put_red_bottle_into_pot":
        "Put {Eggplant, Red Bottle} into Pot",
    "language_conditioning/lift_red_chili_pepper_or_cheese/lift_cheese":
        "Lift {Cheese, Red Chili Pepper}",
    "language_conditioning/lift_red_chili_pepper_or_cheese/lift_red_chili_pepper":
        "Lift {Cheese, Red Chili Pepper}",
    "language_conditioning/put_blue_cup_or_pink_cup_on_plate/put_blue_cup_on_plate":
        "Put {Blue Cup, Pink Cup} on Plate",
    "language_conditioning/put_blue_cup_or_pink_cup_on_plate/put_pink_cup_on_plate":
        "Put {Blue Cup, Pink Cup} on Plate",
}

def per_task_wm_stats(data: dict) -> dict[str, dict]:
    """Aggregate per-trial scores into per-task (trials, successes) stats."""
    tasks: dict[str, dict] = {}
    for r in data["per_trial"]:
        key = r["task_key"]
        if key not in tasks:
            tasks[key] = {"trials": 0, "successes": 0.0}
        tasks[key]["trials"] += 1
        tasks[key]["successes"] += float(r["score"])
    return tasks

def aggregate_policy(data: dict, policy: str) -> tuple[list[float], list[float], list[str]]:
    """Return (real_pct, wm_pct, labels) aligned by compound-task grouping."""
    if policy not in REAL_WORLD:
        raise ValueError(
            f"No real-world data for policy '{policy}'. "
            f"Available: {sorted(REAL_WORLD)}."
        )
    real_table = REAL_WORLD[policy]
    wm_tasks = per_task_wm_stats(data)

    aggregated: dict[str, dict[str, float]] = {}
    for task_key, wm_stats in wm_tasks.items():
        if task_key not in real_table:
            print(f"  {task_key}: no matching real-world data (skipped)")
            continue
        label = COMPOUND_TASKS.get(task_key, task_key.split("/")[-1])
        real_succ = real_table[task_key]
        if label not in aggregated:
            aggregated[label] = {"real_s": 0, "real_t": 0, "wm_s": 0, "wm_t": 0}
        aggregated[label]["real_s"] += real_succ
        aggregated[label]["real_t"] += TRIALS_PER_TASK
        aggregated[label]["wm_s"] += wm_stats["successes"]
        aggregated[label]["wm_t"] += wm_stats["trials"]

    real_pct, wm_pct, labels = [], [], []
    for label, a in sorted(aggregated.items()):
        real_pct.append(100.0 * a["real_s"] / a["real_t"])
        wm_pct.append(100.0 * a["wm_s"] / a["wm_t"])
        labels.append(label)
    return real_pct, wm_pct, labels
