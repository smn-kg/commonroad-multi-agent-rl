from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.visualization.mp_renderer import MPRenderer
from pathlib import Path

"""
XML → video renderer for CommonRoad scenarios.
"""

BASE_DIR = Path(__file__).resolve().parent
xml_path = BASE_DIR / "outputs" / "simulated_scenarios" / "run_0001.xml"

out_dir = BASE_DIR / "outputs" / "videos"
out_dir.mkdir(parents=True, exist_ok=True)

scenario, _ = CommonRoadFileReader(xml_path).open()


draw_params = MPDrawParams()
draw_params.time_begin = 0
draw_params.time_end = 50
draw_params.dynamic_obstacle.draw_shape = True
draw_params.dynamic_obstacle.draw_icon = True
draw_params.dynamic_obstacle.show_label = False



sid = str(scenario.scenario_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
out_path = out_dir / f"{sid}.mp4"

rnd = MPRenderer()

rnd.create_video([scenario], str(out_path), draw_params=draw_params)
