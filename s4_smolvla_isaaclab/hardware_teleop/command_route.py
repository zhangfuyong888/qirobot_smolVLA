"""Reviewed SDK command routes for standalone and leg-deploy teleoperation."""

from __future__ import annotations

from dataclasses import replace

from hardware_teleop.config_loader import HardwareTeleopConfig


LEG_DEPLOY_COMMAND_TOPIC = "/lowcmd"
LEG_DEPLOY_MODE_CTRL = 5
LEG_DEPLOY_POLICY_MODE_CTRL = 1
LEG_DEPLOY_PUBLISHER = "/qi_topic_converter"
LEG_DEPLOY_COMMAND_MAX_AGE_S = 0.25


def configure_command_route(
    config: HardwareTeleopConfig,
    *,
    with_leg_deploy: bool,
) -> tuple[HardwareTeleopConfig, tuple[str, ...]]:
    """Select one of the two reviewed SDK command routes."""
    if not with_leg_deploy:
        return config, ()
    return (
        replace(
            config,
            hardware=replace(
                config.hardware,
                arm_command_topic=LEG_DEPLOY_COMMAND_TOPIC,
                arm_command_mode_ctrl=LEG_DEPLOY_MODE_CTRL,
            ),
        ),
        (LEG_DEPLOY_PUBLISHER,),
    )
