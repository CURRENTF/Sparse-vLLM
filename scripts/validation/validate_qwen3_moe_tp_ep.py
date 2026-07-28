"""Validate Qwen3MoE outer attention TP with MoE TP x EP."""

if __package__:
    from .validate_qwen3_moe_pure_tp import main
else:
    from validate_qwen3_moe_pure_tp import main


if __name__ == "__main__":
    main()
