#!/usr/bin/env python
"""Shared wandb utilities for training scripts."""

import wandb
from contextlib import contextmanager
from tr_config import config


@contextmanager
def wandb_run(project_name: str, group: str | None, tags: list[str] | None = None,  extra_config: dict | None = None):
    """Context manager for wandb runs."""
    if not config.wandb:
        yield None
        return
        
    wandb_config = extra_config or {}
    
    with wandb.init(
        entity=config.wandb.entity,
        project=project_name,
        group=group,
        tags=tags,
        config=wandb_config,
    ) as run:
        yield run


def get_wandb_report_to():
    """Get the report_to parameter for training configs."""
    return "wandb" if config.wandb else "none"