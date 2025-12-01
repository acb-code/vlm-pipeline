def wandb_safe_log(data: dict):
    """
    Log to wandb only if wandb is enabled, initialized, and active.
    Otherwise, do nothing.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(data)
    except Exception:
        pass