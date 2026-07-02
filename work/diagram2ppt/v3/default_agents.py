"""Default reconstruction agent registration."""
from __future__ import annotations

from .agents.chart import ChartAgent
from .agents.connector import ConnectorAgent
from .agents.cross_panel_bridge import CrossPanelBridgeAgent
from .agents.formula import FormulaAgent
from .agents.icon import IconAgent
from .agents.action_cards import ActionCardAgent
from .agents.auditor_cards import AuditorCardAgent
from .agents.bottom_mini_surface import BottomMiniSurfaceAgent
from .agents.layout import LayoutAgent
from .agents.pipeline_context import PipelineContextAgent
from .agents.failure_summary import FailureSummaryAgent
from .agents.procedural_surface import ProceduralSurfaceAgent
from .agents.shape import ShapeAgent
from .agents.style import StyleAgent
from .agents.surface import SurfaceAgent
from .agents.template_slot import TemplateSlotAgent
from .agents.text import TextAgent
from .agents.text_layout import TextLayoutAgent
from .agents.vectorize import VectorizeAgent


def register_default_agents(planner) -> None:
    """Register the standard native reconstruction specialist agents."""
    planner.register_agent(TextAgent())
    planner.register_agent(TextLayoutAgent())
    planner.register_agent(TemplateSlotAgent())
    planner.register_agent(ActionCardAgent())
    planner.register_agent(AuditorCardAgent())
    planner.register_agent(BottomMiniSurfaceAgent())
    planner.register_agent(FailureSummaryAgent())
    planner.register_agent(PipelineContextAgent())
    planner.register_agent(CrossPanelBridgeAgent())
    planner.register_agent(LayoutAgent())
    planner.register_agent(ShapeAgent())
    planner.register_agent(StyleAgent())
    planner.register_agent(ProceduralSurfaceAgent())
    planner.register_agent(SurfaceAgent())
    planner.register_agent(FormulaAgent())
    planner.register_agent(ConnectorAgent())
    planner.register_agent(IconAgent())
    planner.register_agent(ChartAgent())
    planner.register_agent(VectorizeAgent())
