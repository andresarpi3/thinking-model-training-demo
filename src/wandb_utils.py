#!/usr/bin/env python
"""Shared wandb utilities for training scripts."""

import wandb
from contextlib import contextmanager
from tr_config import config


@contextmanager
def wandb_run(project_name: str, tags: list, extra_config: dict = None):
    """Context manager for wandb runs."""
    if not config.wanddb:
        yield None
        return
        
    wandb_config = extra_config or {}
    
    with wandb.init(
        entity=config.wanddb.entity,
        project=project_name,
        tags=tags,
        config=wandb_config,
    ) as run:
        yield run


def get_wandb_report_to():
    """Get the report_to parameter for training configs."""
    return "wandb" if config.wanddb else "none"