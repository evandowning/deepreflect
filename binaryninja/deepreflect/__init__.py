from binaryninja import PluginCommand

from .display import display_all, display_highlight
from .sort import sort_score, sort_size, sort_callee
from .label import modify_label

# Register plugin options
PluginCommand.register("DeepReflect\\1. Show ALL Functions", "Outputs all functions stored in database to BinaryNinja console", display_all)
PluginCommand.register("DeepReflect\\2. Show Highlighted Functions", "Outputs highlighted functions stored in database to BinaryNinja console", display_highlight)
PluginCommand.register_for_function("DeepReflect\\3. Modify Function Label", "Modify function's label", modify_label)
PluginCommand.register("DeepReflect\\4. Sort Function Score", "Sort highlighted functions by score", sort_score)
PluginCommand.register("DeepReflect\\5. Sort Function Size", "Sort highlighted functions by number of basic blocks", sort_size)
PluginCommand.register("DeepReflect\\6. Sort Function Callees", "Sort highlighted functions by number of callees", sort_callee)
