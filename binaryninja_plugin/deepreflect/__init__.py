from binaryninja import PluginCommand

from .display import display_all, display_highlight
from .sort import sort_score, sort_size, sort_callee
from .label import start_time_label, add_label, remove_label

# Register plugin options
PluginCommand.register("DeepReflect\\Display\\1. Show ALL Functions", "Outputs all functions stored in database to BinaryNinja console", display_all)
PluginCommand.register("DeepReflect\\Display\\2. Show Highlighted Functions", "Outputs highlighted functions stored in database to BinaryNinja console", display_highlight)

PluginCommand.register_for_function("DeepReflect\\Label\\1. Start Time Function Label", "Add start time for this function", start_time_label)
PluginCommand.register_for_function("DeepReflect\\Label\\2. Add Function Label", "Add function's label", add_label)
PluginCommand.register_for_function("DeepReflect\\Label\\3. Remove Function Label", "Remove function's label", remove_label)

PluginCommand.register("DeepReflect\\Sort\\1. Sort Function Score", "Sort highlighted functions by score", sort_score)
PluginCommand.register("DeepReflect\\Sort\\2. Sort Function Size", "Sort highlighted functions by number of basic blocks", sort_size)
PluginCommand.register("DeepReflect\\Sort\\3. Sort Function Callees", "Sort highlighted functions by number of callees", sort_callee)
