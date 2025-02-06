import logging
import pathlib

from .gene.symbols import AdaptiveMaxPool1DSymbol
from .gene.symbols import LinearSymbol
from .gene.symbols import RepeatStartSymbol
from .gene.symbols import SimpleSymbol
from .gene.symbols import SymbolType
from .run_model import run_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
run_model(
    datasets_name="MNIST",
    output_file=pathlib.Path("random_model.json"),
    symbols=[
        SimpleSymbol(type=SymbolType.BRANCH_STOP),
        AdaptiveMaxPool1DSymbol(out_features=3369),
        SimpleSymbol(type=SymbolType.REPEAT_END),
        LinearSymbol(bias=True, out_features=1960),
        SimpleSymbol(type=SymbolType.RELU),
        SimpleSymbol(type=SymbolType.RELU),
        SimpleSymbol(type=SymbolType.BRANCH_START),
        SimpleSymbol(type=SymbolType.REPEAT_END),
        LinearSymbol(bias=False, out_features=3595),
        AdaptiveMaxPool1DSymbol(out_features=1211),
        AdaptiveMaxPool1DSymbol(out_features=4000),
        SimpleSymbol(type=SymbolType.DEACTIVATE),
        SimpleSymbol(type=SymbolType.DEACTIVATE),
        RepeatStartSymbol(times=2),
        SimpleSymbol(type=SymbolType.SOFTMAX),
        SimpleSymbol(type=SymbolType.DEACTIVATE),
        SimpleSymbol(type=SymbolType.TANH),
        SimpleSymbol(type=SymbolType.ACTIVATE),
        AdaptiveMaxPool1DSymbol(out_features=3506),
        LinearSymbol(bias=False, out_features=3256),
        SimpleSymbol(type=SymbolType.REPEAT_END),
        SimpleSymbol(type=SymbolType.ACTIVATE),
        SimpleSymbol(type=SymbolType.BRANCH_START),
        SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
        SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
        SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
        SimpleSymbol(type=SymbolType.ACTIVATE),
        SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
        SimpleSymbol(type=SymbolType.REPEAT_END),
        SimpleSymbol(type=SymbolType.ADAPTIVE_AVGPOOL1D),
        SimpleSymbol(type=SymbolType.LEAKY_RELU),
        SimpleSymbol(type=SymbolType.ADAPTIVE_AVGPOOL1D),
        SimpleSymbol(type=SymbolType.BRANCH_START),
        SimpleSymbol(type=SymbolType.BRANCH_SEGMENT_MARKER),
        SimpleSymbol(type=SymbolType.DEACTIVATE),
    ],
)
