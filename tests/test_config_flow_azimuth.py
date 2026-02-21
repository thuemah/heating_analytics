"""Test config flow azimuth step via AST analysis."""
import ast
import os

def test_azimuth_step_in_code():
    """Verify that the source code defines step=5 for azimuth."""
    file_path = "custom_components/heating_analytics/config_flow.py"

    with open(file_path, "r") as f:
        tree = ast.parse(f.read())

    found_azimuth_step = False

    for node in ast.walk(tree):
        # Look for Call to NumberSelectorConfig
        if isinstance(node, ast.Call):
            # Check if it's NumberSelectorConfig (can be selector.NumberSelectorConfig)
            is_config = False
            if isinstance(node.func, ast.Attribute) and node.func.attr == "NumberSelectorConfig":
                is_config = True
            elif isinstance(node.func, ast.Name) and node.func.id == "NumberSelectorConfig":
                is_config = True

            if is_config:
                # Check keywords
                args = {kw.arg: kw.value for kw in node.keywords}

                # We are looking for the one with unit_of_measurement="°" or max=360
                is_azimuth = False
                if "unit_of_measurement" in args and isinstance(args["unit_of_measurement"], ast.Constant) and args["unit_of_measurement"].value == "°":
                    is_azimuth = True

                if is_azimuth:
                    # Check step
                    if "step" in args:
                        step_val = args["step"]
                        if isinstance(step_val, ast.Constant):
                            if step_val.value == 5:
                                found_azimuth_step = True
                            else:
                                print(f"Found Azimuth but step is {step_val.value}")
                        else:
                            print(f"Found Azimuth but step is not constant: {step_val}")

    assert found_azimuth_step, "Did not find NumberSelectorConfig with unit='°' and step=5 in config_flow.py"
