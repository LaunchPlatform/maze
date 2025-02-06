import logging
import pathlib

from .gene.symbols import AdaptiveMaxPool1DSymbol
from .gene.symbols import LinearSymbol
from .gene.symbols import SimpleSymbol
from .gene.symbols import SymbolType
from .run_model import run_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
run_model(
    datasets_name="MNIST",
    output_file=pathlib.Path("handcraft_model_with_maxpool_upscale.json"),
    symbols=[
        AdaptiveMaxPool1DSymbol(out_features=4096),
        LinearSymbol(bias=True, out_features=1024),
        SimpleSymbol(type=SymbolType.RELU),
        AdaptiveMaxPool1DSymbol(out_features=4096),
        LinearSymbol(bias=True, out_features=1024),
        SimpleSymbol(type=SymbolType.RELU),
    ],
)
